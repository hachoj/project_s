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


def _gaussian1d(n, sigma_scale=0.25):
    if n <= 1:
        return torch.ones(n, dtype=torch.float32)
    sigma = max(1e-6, n * sigma_scale)
    x = np.arange(n, dtype=np.float32)
    c = (n - 1) / 2.0
    w = np.exp(-((x - c) ** 2) / (2 * sigma**2))
    w /= w.max()
    return torch.from_numpy(w.astype(np.float32))


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
    zero_one=False,
    angle_norm=3,
    arc_deg=15.0,
    stride_deg=8.0,
    target_step_deg=1.0,
    angle_min=None,  # if None -> use data min
    angle_max=None,  # if None -> use data max
    patch_size=(56, 56),
    stride_hw=(48, 48),
    angle_sigma_scale=0.25,
    min_keep_per_win=2,
    reconstruct_removed_hw_fn=None,
):
    assert (
        reconstruct_removed_hw_fn is not None
    ), "Pass reconstruct_removed_hw via reconstruct_removed_hw_fn"
    device = next(model.parameters()).device

    # Accept (1,T,...) or (T,...)
    if slices_THW.dim() == 4 and slices_THW.shape[0] == 1:
        slices_THW = slices_THW[0]
    if angles_deg_T.dim() == 2 and angles_deg_T.shape[0] == 1:
        angles_deg_T = angles_deg_T[0]

    slices = slices_THW.to(device).float().contiguous()  # (T,H,W)

    if zero_one:
        slices = slices.float() / 255.0
    else:
        slices = slices.float() / 255.0 * 2.0 - 1.0

    angles = angles_deg_T.to(device).float().contiguous()  # (T,)
    assert slices.dim() == 3 and angles.dim() == 1 and angles.numel() == slices.shape[0]

    # Sort by angle
    angles, perm = torch.sort(angles)  # (T,)
    slices = slices[perm]  # (T,H,W)
    T, H, W = slices.shape

    # Determine target range from data if not provided
    data_min = float(angles.min().item())
    data_max = float(angles.max().item())
    angle_min = data_min if angle_min is None else float(angle_min)
    angle_max = data_max if angle_max is None else float(angle_max)

    if not (angle_min < angle_max):
        raise ValueError(f"angle_min ({angle_min}) must be < angle_max ({angle_max})")

    # Build uniform grid over [amin, amax]
    # Ensure we include the endpoint if it lands exactly on the grid
    steps = max(1, int(round((angle_max - angle_min) / target_step_deg)))
    grid_deg = (
        angle_min
        + torch.arange(steps + 1, device=device, dtype=torch.float32) * target_step_deg
    )
    N = grid_deg.numel()

    pred_sum = torch.zeros((N, H, W), dtype=torch.float32, device=device)
    w_sum = torch.zeros((N,), dtype=torch.float32, device=device)

    # Window starts; ensure last touches end
    if arc_deg <= 0 or stride_deg <= 0:
        raise ValueError("arc_deg and stride_deg must be > 0")

    num_windows = max(
        1, int(np.floor((angle_max - angle_min - arc_deg) / stride_deg)) + 1
    )
    window_starts = (
        angle_min
        + torch.arange(num_windows, device=device, dtype=torch.float32) * stride_deg
    )
    if window_starts.numel() == 0 or window_starts[-1] + arc_deg < angle_max - 1e-6:
        window_starts = torch.cat(
            [window_starts, torch.tensor([angle_max - arc_deg], device=device)]
        )

    for window_start in window_starts:
        window_end = window_start + arc_deg

        # Measured slices inside [ws, we)
        keep_mask = (angles >= window_start) & (angles < window_end)
        Ti = int(keep_mask.sum().item())
        if Ti < min_keep_per_win:
            continue

        kept_slices = slices[keep_mask]  # (Ti,H,W)
        kept_angles = angles[keep_mask]  # (Ti,)

        # Target angles from global grid within [ws, we] (inclusive end if aligned)
        query_mask = (grid_deg >= window_start) & (grid_deg <= window_end + 1e-6)
        if not torch.any(query_mask):
            continue
        query_angles = grid_deg[query_mask]  # (Q,)
        Q = query_angles.numel()

        # normalize angles within window
        min_angle = float(kept_angles.min().item())
        max_angle = min_angle + arc_deg
        assert max_angle >= float(
            kept_angles.max().item()
        ), f"arc_deg is too small: min_angle: {min_angle}, max_angle: {max_angle}, kept_angles.max(): {kept_angles.max()}"
        kept_angles = (kept_angles - min_angle) / (
            max_angle - min_angle
        ) * np.pi / (angle_norm/2) - np.pi / angle_norm
        query_angles = (query_angles - min_angle) / (
            max_angle - min_angle
        ) * np.pi / (angle_norm/2) - np.pi / angle_norm

        # Spatial tiling per window → (Q,H,W)
        pred_RHW = reconstruct_removed_hw_fn(
            model=model,
            kept_slices=kept_slices,
            kept_angles_deg=kept_angles,
            target_angles_deg=query_angles,
            patch_size=patch_size,
            stride=stride_hw,
            sigma_scale=0.125,
        )  # (R,H,W)

        # Angle-domain blending within window
        w_ang = _gaussian1d(Q, sigma_scale=angle_sigma_scale).to(device)  # (R,)
        g_idx = torch.nonzero(query_mask, as_tuple=False).squeeze(1)  # (R,)
        pred_sum[g_idx] += pred_RHW * w_ang[:, None, None]
        w_sum[g_idx] += w_ang

    pred_uniform = pred_sum / w_sum.clamp_min(1e-8)[:, None, None]  # (N,H,W)
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
