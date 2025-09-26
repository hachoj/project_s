
#!/usr/bin/env python3
import os
import sys
from typing import Sequence, Tuple, Union

import numpy as np
import SimpleITK as sitk
import torch
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.model import ProjectI
from model.utils import angle_aware_resample, reconstruct_angle_linear, reconstruct_angle_sr


def _to_uint8(vol_f32: np.ndarray) -> np.ndarray:
    vol = np.clip(np.rint(vol_f32), 0.0, 255.0)
    return vol.astype(np.uint8)


def _ensure_pair(value: Union[int, Sequence[int]]) -> Tuple[int, int]:
    if isinstance(value, int):
        return value, value
    if isinstance(value, Sequence) and len(value) == 2:
        return int(value[0]), int(value[1])
    raise ValueError(f"Expected scalar or length-2 sequence, got {value!r}")


def _save_volume(array_u8: np.ndarray, path: str) -> None:
    img = sitk.GetImageFromArray(array_u8)
    img.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(img, path)


def _build_uniform_original(
    slices_unit: np.ndarray,
    angles_deg: np.ndarray,
    target_step_deg: float,
) -> Tuple[np.ndarray, np.ndarray]:
    amin = float(angles_deg.min())
    amax = float(angles_deg.max())
    if not (amax > amin):
        raise ValueError("Angle range must be non-zero.")
    steps = max(1, int(round((amax - amin) / float(target_step_deg))))
    grid_deg = amin + np.arange(steps + 1, dtype=np.float32) * float(target_step_deg)

    angles_t = torch.from_numpy(angles_deg).float()
    slices_t = torch.from_numpy(slices_unit).float().unsqueeze(1)  # [T,1,H,W]
    grid_t = torch.from_numpy(grid_deg).float()

    with torch.no_grad():
        resampled = angle_aware_resample(angles_t, slices_t, grid_t, radians=False)
    return resampled[:, 0].cpu().numpy(), grid_deg


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    print("--------------------------------")
    print("Loading config files...")

    with open("configs/model/model_config.yaml") as f:
        model_cfg = yaml.safe_load(f)
    with open("configs/model/encoder.yaml") as f:
        encoder_cfg = yaml.safe_load(f)
    with open("configs/model/decoder_config.yaml") as f:
        decoder_cfg = yaml.safe_load(f)
    with open("configs/inference/patch_inference_config.yaml") as f:
        patch_cfg = yaml.safe_load(f)

    model_path = patch_cfg["model_path"]
    patch_path = patch_cfg["patch_path"]
    output_prefix = patch_cfg["output_prefix"]
    target_step_deg = float(patch_cfg["target_step_deg"])
    just_sr = bool(patch_cfg.get("just_sr", False))

    patch_size = patch_cfg.get("patch_size", model_cfg["patch_size"])
    stride = patch_cfg.get("stride", model_cfg["stride"])
    patch_size = _ensure_pair(patch_size)
    stride = _ensure_pair(stride)

    zero_one = bool(model_cfg.get("zero_one", False))

    print("Config files loaded successfully")
    print("--------------------------------")
    print("Building model...")

    decoder_cfg = dict(decoder_cfg)

    model = ProjectI(
        embd_dim=model_cfg["embd_dim"],
        time_steps=model_cfg["time_steps"],
        encoder_config=encoder_cfg,
        decoder_config=decoder_cfg,
    ).to(device)
    model = torch.compile(model)

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()

    print("Model built successfully")
    print("--------------------------------")
    print("Loading patch data...")

    with np.load(patch_path, allow_pickle=False) as f:
        slices_255 = f["slices"].astype(np.float32)
        angles_deg = f["angles"].astype(np.float32)

    print("Patch data loaded successfully")
    print("--------------------------------")

    slices_unit = slices_255 / 255.0

    os.makedirs(os.path.dirname(output_prefix) or ".", exist_ok=True)

    if not just_sr:
        print("Saving uniform original volume...")
        orig_unit, grid_orig = _build_uniform_original(slices_unit, angles_deg, target_step_deg)
        orig_u8 = _to_uint8(orig_unit * 255.0)
        _save_volume(orig_u8, f"{output_prefix}_original.nii.gz")
        print("Original volume saved")
        print("--------------------------------")

        print("Running linear interpolation baseline...")
        slices_t = torch.from_numpy(slices_255).float()
        angles_t = torch.from_numpy(angles_deg).float()
        with torch.no_grad():
            linear_pred, linear_grid = reconstruct_angle_linear(
                slices_THW=slices_t,
                angles_deg_T=angles_t,
                target_step_deg=target_step_deg,
            )
        linear_u8 = _to_uint8(linear_pred[:, 0].cpu().numpy())
        _save_volume(linear_u8, f"{output_prefix}_linear_{target_step_deg}deg.nii.gz")
        print("Linear baseline saved")
        print("--------------------------------")

    print("Running SR inference on patch...")
    with torch.no_grad():
        slices_sr = torch.from_numpy(slices_255).float().to(device)
        angles_sr = torch.from_numpy(angles_deg).float().to(device)
        sr_pred, sr_grid = reconstruct_angle_sr(
            model,
            slices_THW=slices_sr,
            angles_deg_T=angles_sr,
            target_step_deg=target_step_deg,
            zero_one=zero_one,
            patch_size=patch_size,
            stride_hw=stride,
        )

    if zero_one:
        sr_unit = sr_pred.clamp(0.0, 1.0)
    else:
        sr_unit = ((sr_pred + 1.0) * 0.5).clamp(0.0, 1.0)
    sr_u8 = _to_uint8(sr_unit.cpu().numpy() * 255.0)
    _save_volume(sr_u8, f"{output_prefix}_sr_{target_step_deg}deg.nii.gz")

    print("SR inference completed")
    print("Saved:")
    if not just_sr:
        print(f"  {output_prefix}_original.nii.gz")
        print(f"  {output_prefix}_linear_{target_step_deg}deg.nii.gz")
    print(f"  {output_prefix}_sr_{target_step_deg}deg.nii.gz")
    print("--------------------------------")
