import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure
import numpy as np
from typing import Union
import math


def angle_aware_resample(
    angles: torch.Tensor,
    slices: torch.Tensor,
    query_angles: torch.Tensor,
    radians: bool = False,
) -> torch.Tensor:
    """
    Angle-aware two-neighbor linear interpolation over angles.

    For each query angle, after sorting the base angles, the function finds
    the immediate lower and upper neighbors and linearly blends the two
    corresponding slices. Queries are clamped to the data range; at the
    boundaries or when neighbors coincide (duplicate angles), this reduces to
    nearest-neighbor (the lower slice). Inputs may be in degrees or radians.

    Args:
        angles (Tensor): [T] base angles (not necessarily sorted). Floating dtype
            (degrees if ``radians=False``; radians otherwise).
        slices (Tensor): [T, C, H, W] values at the given angles. Floating dtype
            (e.g., float16/float32/bfloat16). Device determines computation device.
        query_angles (Tensor): [R] target angles to resample at. Floating dtype
            (degrees if ``radians=False``; radians otherwise).
        radians (bool): If False, inputs are degrees and converted to radians.

    Returns:
        Tensor: [R, C, H, W] linearly interpolated slices. Dtype: ``slices.dtype``;
        device: ``slices.device``.

    Notes:
        - Internally sorts ``angles`` and aligns ``slices`` along T.
        - Uses searchsorted to locate the bracketing neighbors per query.
        - Queries are clamped to the [min, max] angle range.
    """
    assert angles.dim() == 1, "angles must be [T]"
    assert slices.dim() == 4, "slices must be [T, C, H, W]"
    assert query_angles.dim() == 1, "query_angles must be [R]"
    T = angles.shape[0]
    Ts, C, H, W = slices.shape
    R = query_angles.shape[0]
    assert Ts == T, "Time dims mismatch between angles and slices"

    device, dtype = slices.device, slices.dtype

    # ensure consistent device and type
    angles = angles.to(device=device, dtype=dtype)  # [T]
    query_angles = query_angles.to(device=device, dtype=dtype)  # [R]
    if not radians:
        angles = angles * torch.pi / 180
        query_angles = query_angles * torch.pi / 180

    # Sort base angles and align slices accordingly along T
    angles_sorted, sort_idx = torch.sort(angles, dim=0)  # [T]
    sort_idx_exp = sort_idx[:, None, None, None].expand(T, C, H, W)
    slices_sorted = torch.gather(slices, dim=0, index=sort_idx_exp)  # [T, C, H, W]

    # Clamp queries to [min, max] to avoid out-of-range
    min_a = angles_sorted[0]  # scalar
    max_a = angles_sorted[-1]  # scalar
    query_angles = torch.clamp(query_angles, min_a, max_a)  # [R]

    # For each query, find insertion point on the right
    idx_right = torch.searchsorted(
        angles_sorted, query_angles, right=True
    )  # [R] in [0..T]
    idx_left = idx_right - 1  # [R]
    idx_left = idx_left.clamp(0, T - 1)
    idx_right = idx_right.clamp(0, T - 1)

    # Gather neighbor angles
    lower = angles_sorted[idx_left]  # [R]
    upper = angles_sorted[idx_right]  # [R]

    # Gather neighbor slices along T
    idx_left_4d = idx_left[:, None, None, None].expand(R, C, H, W)
    idx_right_4d = idx_right[:, None, None, None].expand(R, C, H, W)
    lower_slices = torch.gather(slices_sorted, 0, idx_left_4d)  # [R, C, H, W]
    upper_slices = torch.gather(slices_sorted, 0, idx_right_4d)  # [R, C, H, W]

    # Compute weights for ascending angles: lower <= q <= upper
    difference = upper - lower
    eps = torch.tensor(1e-8, device=device, dtype=dtype)
    denom = difference.clamp_min(eps)
    w_upper = (query_angles - lower) / denom
    w_lower = (upper - query_angles) / denom

    # If neighbors collapse (difference ~ 0), fall back to lower slice
    same = difference <= 1e-8
    w_lower = w_lower.masked_fill(same, 1.0)
    w_upper = w_upper.masked_fill(same, 0.0)

    w_lower = w_lower[:, None, None, None]  # [R, 1, 1, 1]
    w_upper = w_upper[:, None, None, None]
    out = lower_slices * w_lower + upper_slices * w_upper  # [R, C, H, W]
    return out


"""
--------------------------------------------------------------------------------
reconstruct_removed_hw
--------------------------------------------------------------------------------
"""


def _gaussian2d(h, w, sigma_scale=0.125):
    ref = max(h, w)
    sigma = max(1e-6, ref * sigma_scale)
    y = np.arange(h)
    x = np.arange(w)
    gy = np.exp(-((y - (h - 1) / 2) ** 2) / (2 * sigma**2))
    gx = np.exp(-((x - (w - 1) / 2) ** 2) / (2 * sigma**2))
    W = torch.from_numpy(gy[:, None] * gx[None, :]).float()
    return W / W.max()


@torch.no_grad()
def reconstruct_removed_hw(
    model,
    kept_slices,  # (T, H, W), float
    kept_angles_deg,  # (T,), degrees
    target_angles_deg,  # (Q,),  degrees
    patch_size,  # int or (ph, pw)
    stride,  # int or (sy, sx)
    sigma_scale=0.125,
):
    device = next(model.parameters()).device
    slices = kept_slices.to(device).float().contiguous()
    angles = kept_angles_deg.to(device).float().contiguous()
    query_angles = target_angles_deg.to(device).float().contiguous()

    T_keep, H, W = slices.shape
    if isinstance(patch_size, int):
        ph = pw = patch_size
    else:
        ph, pw = patch_size
    if isinstance(stride, int):
        sy = sx = stride
    else:
        sy, sx = stride
    assert ph <= H and pw <= W, f"patch_size {(ph,pw)} must fit {(H,W)}"

    # masks: all valid (no temporal padding in val)
    Q = int(query_angles.shape[0])

    # precompute weights
    w_patch = _gaussian2d(ph, pw, sigma_scale).to(device)  # (ph, pw)

    pred_sum = torch.zeros((Q, H, W), dtype=torch.float32, device=device)
    w_sum = torch.zeros((H, W), dtype=torch.float32, device=device)

    # tiling grid (cover end)
    ys = list(range(0, max(1, H - ph + 1), max(1, sy)))
    if not ys or ys[-1] + ph < H:
        ys.append(max(0, H - ph))
    xs = list(range(0, max(1, W - pw + 1), max(1, sx)))
    if not xs or xs[-1] + pw < W:
        xs.append(max(0, W - pw))

    for y0 in ys:
        for x0 in xs:
            kept_patch = slices[:, y0 : y0 + ph, x0 : x0 + pw]  # (T, ph, pw)
            x = kept_patch  # (T, ph, pw)
            ak = angles  # (T)  (degrees)
            aq = query_angles  # (Q)   (degrees)

            y, _ = model(x, ak, aq, is_train=False)
            if y.dim() != 3:
                raise RuntimeError(f"Unexpected model output shape: {tuple(y.shape)}")

            pred_sum[:, y0 : y0 + ph, x0 : x0 + pw] += y * w_patch
            w_sum[y0 : y0 + ph, x0 : x0 + pw] += w_patch

    return pred_sum / w_sum.clamp_min(1e-8)  # (T_rem, H, W)


def reconstruct_angle_linear(
    slices_THW,
    angles_deg_T,
    target_step_deg,
):
    # Accept (1,T,...) or (T,...)
    if slices_THW.dim() == 4 and slices_THW.shape[0] == 1:
        slices_THW = slices_THW[0]
    if angles_deg_T.dim() == 2 and angles_deg_T.shape[0] == 1:
        angles_deg_T = angles_deg_T[0]

    x_all = slices_THW.float().contiguous()  # (T,H,W)

    # normalize slices to [0,1]
    x_all = x_all.float() / 255.0
    a_all = angles_deg_T.float().contiguous()  # (T,)
    assert x_all.dim() == 3 and a_all.dim() == 1 and a_all.numel() == x_all.shape[0]

    # Sort by angle
    a_all, perm = torch.sort(a_all)  # (T,)
    x_all = x_all[perm]  # (T,H,W)
    x_all = x_all.unsqueeze(1)  # (T,1,H,W)
    T, C, H, W = x_all.shape

    # Determine target range from data if not provided
    data_min = float(a_all.min().item())
    data_max = float(a_all.max().item())
    amin = data_min
    amax = data_max
    if not (amin < amax):
        raise ValueError(f"angle_min ({amin}) must be < angle_max ({amax})")

    steps = max(1, int(round((amax - amin) / target_step_deg)))
    grid_deg = amin + torch.arange(steps + 1, dtype=torch.float32) * target_step_deg
    N = grid_deg.numel()

    out = (angle_aware_resample(a_all, x_all, grid_deg, radians=False)*255.0)  # (N,1,H,W)
    return out, grid_deg


@torch.no_grad()
def reconstruct_angle_sr(
    model,
    slices_THW,  # (T,H,W) or (1,T,H,W)
    angles_deg_T,  # (T,)   or (1,T)
    target_step_deg=1.0,
    zero_one=False,
    angle_min=None,  # if None -> use data min
    angle_max=None,  # if None -> use data max
    patch_size=None,  # int or (ph,pw); if None, no tiling
    stride_hw=None,  # int or (sy,sx); if None and patch_size set, defaults to patch_size (no overlap)
    sigma_scale=0.125,
):
    """
    Super-resolve along angle using a pairwise model:
      - For each adjacent measured pair (a_k, a_{k+1}), generate predictions at
        uniform query angles g in that interval with step ``target_step_deg``.
      - The model is called with two slices [prev, next] and a relative angle r in [0,1].
      - The final output is a uniform grid over [angle_min, angle_max] including endpoints.

    Returns:
      pred_uniform: [N,H,W] tensor in the same normalization ([-1,1] or [0,1])
      grid_deg:     [N] tensor of degrees corresponding to pred_uniform slices
    """
    device = next(model.parameters()).device

    # Accept (1,T,...) or (T,...)
    if slices_THW.dim() == 4 and slices_THW.shape[0] == 1:
        slices_THW = slices_THW[0]
    if angles_deg_T.dim() == 2 and angles_deg_T.shape[0] == 1:
        angles_deg_T = angles_deg_T[0]

    x_all = slices_THW.to(device).float().contiguous()  # (T,H,W)
    a_all = angles_deg_T.to(device).float().contiguous()  # (T,)
    assert x_all.dim() == 3 and a_all.dim() == 1 and a_all.numel() == x_all.shape[0]

    # Normalize slices
    if zero_one:
        x_all = x_all / 255.0
    else:
        x_all = x_all / 255.0 * 2.0 - 1.0

    # Sort by absolute angle
    a_all, perm = torch.sort(a_all)
    x_all = x_all[perm]
    T, H, W = x_all.shape

    # Determine target range and grid
    data_min = float(a_all.min().item())
    data_max = float(a_all.max().item())
    amin = data_min if angle_min is None else float(angle_min)
    amax = data_max if angle_max is None else float(angle_max)
    if not (amin < amax):
        raise ValueError(f"angle_min ({amin}) must be < angle_max ({amax})")

    steps = max(1, int(round((amax - amin) / float(target_step_deg))))
    grid_deg = amin + torch.arange(steps + 1, device=device, dtype=torch.float32) * float(
        target_step_deg
    )  # [N]
    N = int(grid_deg.numel())

    # Allocate output grid and (shared) patch weights if tiling
    pred_uniform = torch.zeros((N, H, W), dtype=torch.float32, device=device)

    do_tiling = patch_size is not None
    if do_tiling:
        if isinstance(patch_size, int):
            ph = pw = int(patch_size)
        else:
            ph, pw = int(patch_size[0]), int(patch_size[1])
        if isinstance(stride_hw, int) or stride_hw is None:
            sy = sx = int(stride_hw if stride_hw is not None else ph)
        else:
            sy, sx = int(stride_hw[0]), int(stride_hw[1])
        assert ph <= H and pw <= W, f"patch_size {(ph,pw)} must fit {(H,W)}"

        w_patch = _gaussian2d(ph, pw, sigma_scale).to(device)  # (ph,pw)

        # Tiling grid (cover end)
        ys = list(range(0, max(1, H - ph + 1), max(1, sy)))
        if not ys or ys[-1] + ph < H:
            ys.append(max(0, H - ph))
        xs = list(range(0, max(1, W - pw + 1), max(1, sx)))
        if not xs or xs[-1] + pw < W:
            xs.append(max(0, W - pw))

    # Iterate adjacent pairs
    for k in range(T - 1):
        a0 = float(a_all[k].item())
        a1 = float(a_all[k + 1].item())
        if not (a1 > a0 + 1e-8):
            continue  # skip degenerate/duplicate

        x0 = x_all[k]  # [H,W]
        x1 = x_all[k + 1]  # [H,W]

        # Determine which grid points lie in this interval
        # For all but the last pair: [a0, a1); for the last pair include right end.
        include_right = (k == T - 2)
        left_idx = int(np.ceil((max(a0, amin) - amin) / float(target_step_deg)))
        right_idx = int(np.floor((min(a1, amax) - amin) / float(target_step_deg)))
        if not include_right and right_idx >= left_idx:
            # drop right endpoint if exactly on grid to avoid duplication
            r_exact = abs((a1 - amin) / float(target_step_deg) - round((a1 - amin) / float(target_step_deg))) < 1e-8
            if r_exact:
                right_idx -= 1

        left_idx = max(0, min(left_idx, N - 1))
        right_idx = max(-1, min(right_idx, N - 1))
        if right_idx < left_idx:
            continue

        # Generate predictions for each grid index in this pair
        for gi in range(left_idx, right_idx + 1):
            g = float(amin + gi * float(target_step_deg))

            # If exactly at measured endpoints, copy them
            if abs(g - a0) < 1e-8:
                if do_tiling:
                    # Assemble from x0 via weighted patches
                    pred_sum = torch.zeros((H, W), dtype=torch.float32, device=device)
                    w_sum = torch.zeros((H, W), dtype=torch.float32, device=device)
                    for y0_ in ys:
                        for x0_ in xs:
                            patch = x0[y0_ : y0_ + ph, x0_ : x0_ + pw]
                            pred_sum[y0_ : y0_ + ph, x0_ : x0_ + pw] += patch * w_patch
                            w_sum[y0_ : y0_ + ph, x0_ : x0_ + pw] += w_patch
                    pred_uniform[gi] = pred_sum / w_sum.clamp_min(1e-8)
                else:
                    pred_uniform[gi] = x0
                continue
            if abs(g - a1) < 1e-8:
                if do_tiling:
                    pred_sum = torch.zeros((H, W), dtype=torch.float32, device=device)
                    w_sum = torch.zeros((H, W), dtype=torch.float32, device=device)
                    for y0_ in ys:
                        for x0_ in xs:
                            patch = x1[y0_ : y0_ + ph, x0_ : x0_ + pw]
                            pred_sum[y0_ : y0_ + ph, x0_ : x0_ + pw] += patch * w_patch
                            w_sum[y0_ : y0_ + ph, x0_ : x0_ + pw] += w_patch
                    pred_uniform[gi] = pred_sum / w_sum.clamp_min(1e-8)
                else:
                    pred_uniform[gi] = x1
                continue

            # Relative position in (0,1)
            rel = (g - a0) / (a1 - a0)
            rel_t = torch.tensor([rel], dtype=torch.float32, device=device)

            if do_tiling:
                pred_sum = torch.zeros((H, W), dtype=torch.float32, device=device)
                w_sum = torch.zeros((H, W), dtype=torch.float32, device=device)
                for y0_ in ys:
                    for x0_ in xs:
                        patch0 = x0[y0_ : y0_ + ph, x0_ : x0_ + pw]
                        patch1 = x1[y0_ : y0_ + ph, x0_ : x0_ + pw]
                        pair = torch.stack([patch0, patch1], dim=0)  # [2,ph,pw]
                        y = model(pair, rel_t)
                        # Normalize output shape to [ph,pw]
                        if isinstance(y, (tuple, list)):
                            y = y[0]
                        if y.dim() == 2:
                            y_hw = y
                        elif y.dim() == 3:
                            y_hw = y[0]
                        elif y.dim() == 4:
                            y_hw = y[0, 0]
                        else:
                            raise RuntimeError(
                                f"Unexpected model output shape: {tuple(y.shape)}"
                            )
                        pred_sum[y0_ : y0_ + ph, x0_ : x0_ + pw] += y_hw * w_patch
                        w_sum[y0_ : y0_ + ph, x0_ : x0_ + pw] += w_patch
                pred_uniform[gi] = pred_sum / w_sum.clamp_min(1e-8)
            else:
                pair_full = torch.stack([x0, x1], dim=0)  # [2,H,W]
                y = model(pair_full, rel_t)
                if isinstance(y, (tuple, list)):
                    y = y[0]
                if y.dim() == 2:
                    y_hw = y
                elif y.dim() == 3:
                    y_hw = y[0]
                elif y.dim() == 4:
                    y_hw = y[0, 0]
                else:
                    raise RuntimeError(f"Unexpected model output shape: {tuple(y.shape)}")
                pred_uniform[gi] = y_hw

    return pred_uniform, grid_deg

def PSNR(x, y, zero_one=False):
    mse = torch.mean((x - y) ** 2)
    if mse == 0:
        return float("inf")
    data_range = 2.0 if not zero_one else 1.0
    return (20 * torch.log10(data_range / torch.sqrt(mse))).item()

def SSIM_slicewise(x, y, zero_one=False):
    assert x.shape == y.shape
    assert x.dim() == 3  # (T,H,W)

    x = x.unsqueeze(1)
    y = y.unsqueeze(1)

    data_range = 2.0 if not zero_one else 1.0
    ssim = StructuralSimilarityIndexMeasure(data_range=data_range).to(x.device)

    # torch ssim expects (B,C,H,W)
    return ssim(x, y).item()
