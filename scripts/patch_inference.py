import os
import numpy as np
import torch
import yaml
import SimpleITK as sitk
from typing import Sequence, Union

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.model import ProjectI
from model.utils import (
    angle_aware_resample,
    reconstruct_removed_hw,
    reconstruct_angle_sr,
)


def _to_uint8(vol_f32: np.ndarray) -> np.ndarray:
    vol = (vol_f32) * 255.0
    return np.clip(np.rint(vol), 0, 255).astype(np.uint8)


def save_three_niftis_from_patch(
    model: torch.nn.Module,
    input_npz_path: str,
    output_prefix: str,
    patch_size: Union[int, Sequence[int]],
    stride: Union[int, Sequence[int]],
    target_step_deg: float = 1.0,
) -> None:
    os.makedirs(os.path.dirname(output_prefix) or ".", exist_ok=True)

    with np.load(input_npz_path, allow_pickle=False) as f:
        slices = f["slices"].astype(np.float32)  # (T,H,W), normalized [0,1]
        angles = f["angles"].astype(np.float32)  # (T,)
        avg_diff = float(f["average_angle_difference"])  # scalar

    T, H, W = slices.shape
    if T < 2:
        raise ValueError("Patch must contain at least 2 slices.")

    amin = float(angles.min())
    amax = float(angles.max())
    if not (amax > amin):
        raise ValueError("Angle range must be non-zero.")

    # Build uniform angle grid
    steps = max(1, int(round((amax - amin) / float(target_step_deg))))
    grid_deg = amin + np.arange(steps + 1, dtype=np.float32) * float(target_step_deg)
    N = grid_deg.size

    # 1) Original volume on uniform grid via linear interpolation
    a_t = torch.from_numpy(angles).float()
    x_t = torch.from_numpy(slices).float().unsqueeze(1)  # [T,1,H,W]
    q_t = torch.from_numpy(grid_deg).float()

    with torch.no_grad():
        orig_rchw = angle_aware_resample(a_t, x_t, q_t, radians=False)  # [N,1,H,W]
    orig_vol = orig_rchw[:, 0].cpu().numpy()  # [N,H,W], in [0,1]

    # 2) Randomly remove slices (ensure at least one)
    remove_n = max(1, min(T - 1, int(np.ceil(T * 0.25))))
    perm = np.random.permutation(T)
    idx_remove = np.sort(perm[:remove_n])
    idx_keep = np.sort(perm[remove_n:])

    removed_angles = angles[idx_remove]
    removed_gt = slices[idx_remove]
    kept_slices = slices[idx_keep]
    kept_angles = angles[idx_keep]

    # Prepare removed GT volume with only available removed slices (no empty slots)
    sort_idx_removed = np.argsort(removed_angles)
    removed_angles_sorted = removed_angles[sort_idx_removed]
    removed_gt_sorted = removed_gt[sort_idx_removed]

    # 3) Reconstruct removed slices from kept slices at removed angles
    kept_slices_t = torch.from_numpy(kept_slices).float()
    kept_angles_t = torch.from_numpy(kept_angles).float()
    removed_angles_t = torch.from_numpy(removed_angles).float()

    pred_removed = (
        reconstruct_removed_hw(
            model=model,
            kept_slices=kept_slices_t,
            kept_angles_deg=kept_angles_t,
            target_angles_deg=removed_angles_t,
            patch_size=patch_size,
            stride=stride,
        )
        .cpu()
        .numpy()
    )  # [R,H,W], in [0,1]

    # Sort predictions to match ascending removed angles
    pred_removed_sorted = pred_removed[sort_idx_removed]

    # Convert to uint8 for visualization
    orig_u8 = _to_uint8(orig_vol)
    gt_removed_u8 = _to_uint8(removed_gt_sorted)
    pred_removed_u8 = _to_uint8(pred_removed_sorted)

    # Save as NIfTI with z-spacing = target_step_deg using SimpleITK
    img_orig = sitk.GetImageFromArray(orig_u8)  # array shape: [z,y,x]
    img_orig.SetSpacing((1.0, 1.0, float(target_step_deg)))
    sitk.WriteImage(img_orig, f"{output_prefix}_original.nii.gz")

    img_gt = sitk.GetImageFromArray(gt_removed_u8)
    img_gt.SetSpacing((1.0, 1.0, float(target_step_deg)))
    sitk.WriteImage(img_gt, f"{output_prefix}_removed_gt.nii.gz")

    img_pred = sitk.GetImageFromArray(pred_removed_u8)
    img_pred.SetSpacing((1.0, 1.0, float(target_step_deg)))
    sitk.WriteImage(img_pred, f"{output_prefix}_removed_pred.nii.gz")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"--------------------------------")
    print(f"Loading config file...")
    cfg = yaml.safe_load(open("configs/patch_inference_config.yaml"))

    # model params
    embd_dim = cfg["embd_dim"]
    num_harmonics = cfg["num_harmonics"]

    decoder_depth = cfg["decoder_depth"]
    decoder_hidden_size = cfg["decoder_hidden_size"]

    FiLM = cfg["FiLM"]

    encoder_type = cfg["encoder_type"]

    # metaformer params
    metaformer_internal_dim = cfg["metaformer_internal_dim"]
    metaformer_depth = cfg["metaformer_depth"]
    metaformer_pool_size = cfg["metaformer_pool_size"]
    metaformer_num_heads = cfg["metaformer_num_heads"]
    metaformer_mlp_ratio = cfg["metaformer_mlp_ratio"]
    metaformer_stem_type = cfg["metaformer_stem_type"]

    # paths
    model_path = cfg["model_path"]
    patch_path = cfg["patch_path"]
    output_prefix = cfg["output_prefix"]

    # data params
    patch_size = cfg["patch_size"]
    stride = cfg["stride"]

    # reconstruction params
    target_step_deg = cfg["target_step_deg"]
    sampled_step_deg = cfg.get("sampled_step_deg", None)
    arc_deg = cfg.get("arc_deg", 30)
    stride_deg = cfg.get("stride_deg", 30)

    print(f"Config file loaded successfully")
    print(f"--------------------------------")
    print(f"Building model...")

    model = ProjectI(
        num_harmonics=num_harmonics,
        embd_dim=embd_dim,
        decoder_depth=decoder_depth,
        decoder_hidden_size=decoder_hidden_size,
        use_film=FiLM,
        encoder_type=encoder_type,
        patch_size=patch_size,
        metaformer_internal_dim=metaformer_internal_dim,
        metaformer_depth=metaformer_depth,
        metaformer_pool_size=metaformer_pool_size,
        metaformer_num_heads=metaformer_num_heads,
        metaformer_mlp_ratio=metaformer_mlp_ratio,
        metaformer_stem_type=metaformer_stem_type,
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location="cpu")["model_state"])
    model.eval()

    print(f"Model built successfully")
    print(f"--------------------------------")
    print(f"Processing patch -> NIfTI volumes...")

    save_three_niftis_from_patch(
        model=model,
        input_npz_path=patch_path,
        output_prefix=output_prefix,
        patch_size=patch_size,
        stride=stride,
        target_step_deg=target_step_deg,
    )

    # Optional: save a fourth volume reconstructed by the model at fixed angular spacing
    if sampled_step_deg is not None:
        with np.load(patch_path, allow_pickle=False) as f:
            slices = f["slices"].astype(np.float32)  # [0,1]
            angles = f["angles"].astype(np.float32)
        # Use the same SR pipeline as inference.py, but on this patch
        slices_255 = (slices) * 255.0  # back to 0..255 as expected by SR util
        slices_t = torch.from_numpy(slices_255).float().to(device)
        angles_t = torch.from_numpy(angles).float().to(device)
        with torch.no_grad():
            pred_uniform, grid_deg = reconstruct_angle_sr(
                model,
                slices_THW=slices_t,
                angles_deg_T=angles_t,
                arc_deg=arc_deg,
                stride_deg=stride_deg,
                target_step_deg=float(sampled_step_deg),
                patch_size=patch_size,
                stride_hw=stride,
                reconstruct_removed_hw_fn=reconstruct_removed_hw,
            )
        sr_u8 = _to_uint8(pred_uniform.cpu().numpy())  # [N,H,W]
        img_smpl = sitk.GetImageFromArray(sr_u8)
        img_smpl.SetSpacing((1.0, 1.0, float(sampled_step_deg)))
        sitk.WriteImage(img_smpl, f"{output_prefix}_sr_{sampled_step_deg}deg.nii.gz")

    print(f"Saved:")
    print(f"  {output_prefix}_original.nii.gz")
    print(f"  {output_prefix}_removed_gt.nii.gz")
    print(f"  {output_prefix}_removed_pred.nii.gz")
    if sampled_step_deg is not None:
        print(f"  {output_prefix}_sr_{sampled_step_deg}deg.nii.gz")
    print(f"--------------------------------")
