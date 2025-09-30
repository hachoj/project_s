import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from torchmetrics.image import StructuralSimilarityIndexMeasure
import SimpleITK as sitk

torch.set_float32_matmul_precision("high")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.model import ProjectI
from model.reconstruction import extract_slices
from model.utils import angle_aware_resample, PSNR


def _ensure_pair(value: Sequence[int]) -> Tuple[int, int]:
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError("Expected sequence of length 2 for patch/stride.")
        return int(value[0]), int(value[1])
    return int(value), int(value)


def _normalize_slices(slices: torch.Tensor, zero_one: bool) -> torch.Tensor:
    if zero_one:
        return slices / 255.0
    return slices / 255.0 * 2.0 - 1.0


def _gaussian2d(ph: int, pw: int, sigma_scale: float = 0.125) -> torch.Tensor:
    ref = max(ph, pw)
    sigma = max(1e-6, ref * sigma_scale)
    y = torch.arange(ph, dtype=torch.float32)
    x = torch.arange(pw, dtype=torch.float32)
    gy = torch.exp(-((y - (ph - 1) / 2.0) ** 2) / (2.0 * sigma**2))
    gx = torch.exp(-((x - (pw - 1) / 2.0) ** 2) / (2.0 * sigma**2))
    w = gy.unsqueeze(1) * gx.unsqueeze(0)
    return w / w.max().clamp_min(1e-8)


def _build_tiling_grid(H: int, W: int, ph: int, pw: int, sy: int, sx: int) -> Tuple[List[int], List[int]]:
    ys = list(range(0, max(1, H - ph + 1), max(1, sy)))
    if not ys or ys[-1] + ph < H:
        ys.append(max(0, H - ph))
    xs = list(range(0, max(1, W - pw + 1), max(1, sx)))
    if not xs or xs[-1] + pw < W:
        xs.append(max(0, W - pw))
    return ys, xs


def _save_volume_uint8(array_u8: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    img = sitk.GetImageFromArray(array_u8)
    img.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(img, path)


def _patient_to_dir(image_root: str, patient_id: str) -> str:
    if "_" not in patient_id:
        raise ValueError(
            f"Patient identifier '{patient_id}' must include label prefix (e.g., positive_UF179)"
        )
    label, case = patient_id.split("_", 1)
    return os.path.join(image_root, label, case)


def _predict_removed_slices(
    model: ProjectI,
    kept_slices: torch.Tensor,
    kept_angles: torch.Tensor,
    query_angles: torch.Tensor,
    patch_size: Tuple[int, int],
    stride: Tuple[int, int],
) -> torch.Tensor:
    ph, pw = patch_size
    sy, sx = stride

    angles_sorted, order = torch.sort(kept_angles)
    slices_sorted = kept_slices[order]

    device = kept_slices.device
    Q = int(query_angles.shape[0])
    if Q == 0:
        return torch.empty((0, slices_sorted.shape[1], slices_sorted.shape[2]), device=device)

    H, W = slices_sorted.shape[1:]
    ys, xs = _build_tiling_grid(H, W, ph, pw, sy, sx)
    w_patch = _gaussian2d(ph, pw).to(device)

    preds = torch.zeros((Q, H, W), dtype=torch.float32, device=device)

    for qi in range(Q):
        angle_q = query_angles[qi]
        idx_right = torch.searchsorted(angles_sorted, angle_q, right=True).item()
        idx_right = min(idx_right, angles_sorted.numel() - 1)
        idx_left = max(idx_right - 1, 0)

        a_left = angles_sorted[idx_left]
        a_right = angles_sorted[idx_right]
        slice_left = slices_sorted[idx_left]
        slice_right = slices_sorted[idx_right]

        if idx_left == idx_right or torch.isclose(a_left, a_right, atol=1e-6):
            preds[qi] = slice_left
            continue

        gap = (a_right - a_left).item()
        r_val = float((angle_q - a_left).item() / max(gap, 1e-6))
        r_val = float(min(max(r_val, 0.0), 1.0))
        rel_t = torch.tensor([[r_val]], dtype=torch.float32, device=device)
        delta_t = torch.tensor([[gap]], dtype=torch.float32, device=device)
        mid_t = torch.tensor([[(a_right + a_left).item() * 0.5]], dtype=torch.float32, device=device)

        acc = torch.zeros((H, W), dtype=torch.float32, device=device)
        weights = torch.zeros((H, W), dtype=torch.float32, device=device)

        for y0 in ys:
            y1 = y0 + ph
            for x0 in xs:
                x1 = x0 + pw
                patch_l = slice_left[y0:y1, x0:x1]
                patch_r = slice_right[y0:y1, x0:x1]
                pair = torch.stack([patch_l, patch_r], dim=0).unsqueeze(0)
                y_pred = model(pair, rel_t, delta_t, mid_t)
                if isinstance(y_pred, (list, tuple)):
                    y_pred = y_pred[0]
                if y_pred.dim() == 4:
                    patch = y_pred[0, 0]
                elif y_pred.dim() == 3:
                    patch = y_pred[0]
                elif y_pred.dim() == 2:
                    patch = y_pred
                else:
                    raise RuntimeError(f"Unexpected model output shape: {tuple(y_pred.shape)}")
                acc[y0:y1, x0:x1] += patch * w_patch
                weights[y0:y1, x0:x1] += w_patch

        preds[qi] = acc / weights.clamp_min(1e-8)

    return preds


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate SR model against angle-aware resample on removed slices.")
    parser.add_argument("--inference-config", default="configs/inference/inference_config.yaml", help="Path to inference config.")
    parser.add_argument("--model-config", default="configs/model/model_config.yaml", help="Path to model config.")
    parser.add_argument("--decoder-config", default="configs/model/decoder_config.yaml", help="Path to decoder config.")
    parser.add_argument("--full-eval-config", default="configs/inference/full_eval_config.yaml", help="Optional full-eval config file overriding CLI defaults.")
    parser.add_argument("--every-x-slices", type=int, help="Keep every Nth slice and reconstruct the rest. Overrides config value when provided.")
    parser.add_argument("--device", default=None, help="Override device (cpu or cuda).")
    parser.add_argument("--skip-compile", action="store_true", help="Disable torch.compile for the model.")
    parser.add_argument(
        "--patients",
        nargs="+",
        help="Explicit patient identifiers to process (e.g., positive_UF179). Overrides split info list.",
    )
    parser.add_argument(
        "--split-info",
        default=os.path.join(PROJECT_ROOT, "data/datasets/data_1.8_rev/split_info.json"),
        help="Split info JSON containing a 'val_patients' list.",
    )
    parser.add_argument(
        "--image-root",
        default=None,
        help="Root directory with label subfolders (positive/negative). Defaults to parent of inference image path.",
    )
    return parser


def _load_patient_ids(
    cli_patients: Optional[Sequence[str]],
    cfg_patients: Optional[Sequence[str]],
    split_path: str,
) -> List[str]:
    if cli_patients:
        return list(cli_patients)

    if cfg_patients:
        return list(cfg_patients)

    if os.path.isfile(split_path):
        with open(split_path) as f:
            data = json.load(f)
        patients = data.get("val_patients", [])
        if patients:
            return list(patients)

    return []


def _evaluate_patient(
    patient_id: str,
    patient_dir: str,
    model: ProjectI,
    every_x_slices: int,
    patch_size: Tuple[int, int],
    stride: Tuple[int, int],
    zero_one: bool,
    save_root: str,
    device: torch.device,
) -> Dict[str, float]:
    print(f"Processing patient: {patient_id}")
    extracted = extract_slices(patient_dir)
    if not extracted:
        raise ValueError(f"No slices found in {patient_dir}")

    entries = sorted(extracted.values(), key=lambda v: v["angle"])

    angles = torch.tensor([float(v["angle"]) for v in entries], dtype=torch.float32)
    slices_np = np.stack([v["slice_volume"] for v in entries], axis=0)
    slices = torch.from_numpy(slices_np).float()

    num_slices = angles.numel()
    keep_idx = list(range(0, num_slices, every_x_slices))
    if len(keep_idx) < 2:
        raise ValueError("Need at least two kept slices for interpolation.")
    removed_idx = [i for i in range(num_slices) if i not in keep_idx]
    if not removed_idx:
        raise ValueError("No slices removed; adjust every_x_slices.")

    keep_idx_tensor = torch.tensor(keep_idx, dtype=torch.long)
    removed_idx_tensor = torch.tensor(removed_idx, dtype=torch.long)

    kept_angles = angles[keep_idx_tensor]
    kept_slices = slices[keep_idx_tensor]
    removed_angles = angles[removed_idx_tensor]
    removed_gt = slices[removed_idx_tensor]

    kept_norm = _normalize_slices(kept_slices, zero_one=zero_one).to(device)
    kept_angles = kept_angles.to(device)
    removed_angles = removed_angles.to(device)

    print(f"Total slices: {num_slices}")
    print(f"Kept slices: {len(keep_idx)} | Removed slices: {len(removed_idx)}")
    print("--------------------------------")
    print("Predicting removed slices with SR model...")

    with torch.no_grad():
        sr_pred = _predict_removed_slices(
            model,
            kept_slices=kept_norm,
            kept_angles=kept_angles,
            query_angles=removed_angles,
            patch_size=patch_size,
            stride=stride,
        )

    if zero_one:
        sr_pred = sr_pred.clamp(0.0, 1.0)
    else:
        sr_pred = sr_pred.clamp(-1.0, 1.0)

    sr_pred = sr_pred.unsqueeze(1)

    print("Computing angle-aware baseline for removed slices...")
    with torch.no_grad():
        baseline_pred = angle_aware_resample(
            kept_angles,
            kept_norm.unsqueeze(1),
            removed_angles,
            radians=False,
        )

    if zero_one:
        baseline_pred = baseline_pred.clamp(0.0, 1.0)
    else:
        baseline_pred = baseline_pred.clamp(-1.0, 1.0)

    baseline_pred = baseline_pred.to(sr_pred.device)

    removed_gt_norm = _normalize_slices(removed_gt, zero_one=zero_one).unsqueeze(1).to(sr_pred.device)

    print("--------------------------------")
    print("Evaluating metrics...")

    ssim_metric = StructuralSimilarityIndexMeasure(
        data_range=1.0 if zero_one else 2.0
    ).to(sr_pred.device)

    ssim_values = []
    for i in range(sr_pred.shape[0]):
        value = ssim_metric(sr_pred[i : i + 1], baseline_pred[i : i + 1]).item()
        ssim_values.append(value)
        ssim_metric.reset()

    psnr_sr_vs_linear = PSNR(sr_pred, baseline_pred, zero_one=zero_one)
    psnr_sr_vs_gt = PSNR(sr_pred, removed_gt_norm, zero_one=zero_one)
    psnr_linear_vs_gt = PSNR(baseline_pred, removed_gt_norm, zero_one=zero_one)

    ssim_tensor = torch.tensor(ssim_values)
    ssim_mean = float(ssim_tensor.mean().item())
    ssim_min = float(ssim_tensor.min().item())
    ssim_max = float(ssim_tensor.max().item())

    removed_angles_cpu = removed_angles.cpu().tolist()

    print("Per-slice SSIM (SR vs angle-aware):")
    for angle, value in zip(removed_angles_cpu, ssim_values):
        print(f"  angle {angle:.3f} deg -> SSIM {value:.4f}")

    print("--------------------------------")
    print(f"Average SSIM (SR vs angle-aware): {ssim_mean:.4f}")
    print(f"Min/Max SSIM (SR vs angle-aware): {ssim_min:.4f} / {ssim_max:.4f}")
    print(f"PSNR (SR vs angle-aware): {psnr_sr_vs_linear:.2f} dB")
    print(f"PSNR (SR vs ground truth): {psnr_sr_vs_gt:.2f} dB")
    print(f"PSNR (Angle-aware vs ground truth): {psnr_linear_vs_gt:.2f} dB")

    ssim_metric.reset()
    sr_vs_gt_ssim = ssim_metric(sr_pred, removed_gt_norm).item()
    ssim_metric.reset()
    linear_vs_gt_ssim = ssim_metric(baseline_pred, removed_gt_norm).item()

    print(f"SSIM (SR vs ground truth): {sr_vs_gt_ssim:.4f}")
    print(f"SSIM (Angle-aware vs ground truth): {linear_vs_gt_ssim:.4f}")
    print("--------------------------------")

    save_root = save_root or "."
    save_subdir = os.path.join(save_root, patient_id)
    prefix = os.path.join(save_subdir, f"removed_every_{every_x_slices}")

    print("Saving removed slice volumes (ground truth, linear, SR)...")

    gt_u8 = removed_gt.cpu().numpy().astype(np.uint8)
    baseline_cpu = baseline_pred.squeeze(1).detach().cpu()
    sr_cpu = sr_pred.squeeze(1).detach().cpu()

    if zero_one:
        baseline_u8 = (baseline_cpu.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).numpy()
        sr_u8 = (sr_cpu.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).numpy()
    else:
        baseline_u8 = (((baseline_cpu.clamp(-1.0, 1.0) + 1.0) * 0.5) * 255.0).round().to(torch.uint8).numpy()
        sr_u8 = (((sr_cpu.clamp(-1.0, 1.0) + 1.0) * 0.5) * 255.0).round().to(torch.uint8).numpy()

    _save_volume_uint8(gt_u8, f"{prefix}_gt.nii.gz")
    _save_volume_uint8(baseline_u8, f"{prefix}_linear.nii.gz")
    _save_volume_uint8(sr_u8, f"{prefix}_sr.nii.gz")

    print("Saved:")
    print(f"  {prefix}_gt.nii.gz")
    print(f"  {prefix}_linear.nii.gz")
    print(f"  {prefix}_sr.nii.gz")
    print("================================")

    return {
        "patient": patient_id,
        "num_removed": len(removed_idx),
        "psnr_sr_vs_linear": float(psnr_sr_vs_linear),
        "psnr_sr_vs_gt": float(psnr_sr_vs_gt),
        "psnr_linear_vs_gt": float(psnr_linear_vs_gt),
        "ssim_sr_vs_linear_mean": ssim_mean,
        "ssim_sr_vs_linear_min": ssim_min,
        "ssim_sr_vs_linear_max": ssim_max,
        "ssim_sr_vs_gt": float(sr_vs_gt_ssim),
        "ssim_linear_vs_gt": float(linear_vs_gt_ssim),
        "ssim_values": ssim_values,
    }


def main() -> None:
    parser = _build_argparser()
    args = parser.parse_args()

    full_eval_cfg: Dict[str, object] = {}
    cfg_path = args.full_eval_config
    if cfg_path and os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            loaded = yaml.safe_load(f)
        if isinstance(loaded, dict):
            full_eval_cfg = loaded

    cfg_every = full_eval_cfg.get("every_x_slices") if isinstance(full_eval_cfg, dict) else None
    every_x_slices = args.every_x_slices if args.every_x_slices is not None else cfg_every
    if every_x_slices is None:
        raise ValueError("every_x_slices must be provided via CLI or full-eval config.")
    if int(every_x_slices) <= 1:
        raise ValueError("every_x_slices must be greater than 1 to remove slices.")
    every_x_slices = int(every_x_slices)

    cfg_device = full_eval_cfg.get("device") if isinstance(full_eval_cfg, dict) else None
    device = args.device or cfg_device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("--------------------------------")
    print("Loading config files...")

    with open(args.model_config) as f:
        model_cfg = yaml.safe_load(f)

    if isinstance(model_cfg.get("decoder_config"), dict):
        decoder_cfg = dict(model_cfg["decoder_config"])
    else:
        with open(args.decoder_config) as f:
            decoder_cfg = yaml.safe_load(f)
                
    with open("configs/model/encoder_config.yaml") as f:
        encoder_cfg = yaml.safe_load(f)

    with open(args.inference_config) as f:
        inference_cfg = yaml.safe_load(f)

    cfg_model_path = full_eval_cfg.get("model_path") if isinstance(full_eval_cfg, dict) else None
    model_path = cfg_model_path if cfg_model_path is not None else inference_cfg["model_path"]
    image_path = inference_cfg["image_path"]

    patch_size = _ensure_pair(model_cfg["patch_size"])
    stride = _ensure_pair(model_cfg["stride"])
    zero_one = bool(model_cfg.get("zero_one", False))

    print("Config files loaded successfully")
    print("--------------------------------")
    print("Building model...")

    model = ProjectI(
        embd_dim=model_cfg["embd_dim"],
        encoder_config=encoder_cfg,
        decoder_config=decoder_cfg,
        linres=model_cfg["linres"],
    ).to(device)

    skip_compile_cfg = bool(full_eval_cfg.get("skip_compile", False)) if isinstance(full_eval_cfg, dict) else False
    skip_compile = args.skip_compile or skip_compile_cfg

    if not skip_compile:
        model = torch.compile(model)

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()

    print("Model built successfully")
    print("--------------------------------")

    cfg_image_root = full_eval_cfg.get("image_root") if isinstance(full_eval_cfg, dict) else None
    image_root = args.image_root or cfg_image_root or os.path.dirname(os.path.dirname(image_path.rstrip("/")))

    cfg_split = full_eval_cfg.get("split_info") if isinstance(full_eval_cfg, dict) else None
    split_info_path = args.split_info
    if cfg_split:
        split_info_path = cfg_split

    cfg_patients = full_eval_cfg.get("patients") if isinstance(full_eval_cfg, dict) else None
    patients = _load_patient_ids(args.patients, cfg_patients, split_info_path)

    if not patients:
        label_dir = os.path.basename(os.path.dirname(image_path.rstrip("/")))
        case_dir = os.path.basename(image_path.rstrip("/"))
        patients = [f"{label_dir}_{case_dir}"]

    cfg_save_root = full_eval_cfg.get("save_path") if isinstance(full_eval_cfg, dict) else None
    save_root = cfg_save_root if cfg_save_root is not None else inference_cfg.get("save_path", "")
    if save_root:
        os.makedirs(save_root, exist_ok=True)

    summaries: List[Dict[str, float]] = []

    for patient_id in patients:
        patient_dir = _patient_to_dir(image_root, patient_id)
        if not os.path.isdir(patient_dir):
            print(f"Skipping {patient_id}: directory not found at {patient_dir}")
            continue
        result = _evaluate_patient(
            patient_id=patient_id,
            patient_dir=patient_dir,
            model=model,
            every_x_slices=every_x_slices,
            patch_size=patch_size,
            stride=stride,
            zero_one=zero_one,
            save_root=save_root,
            device=device,
        )
        summaries.append(result)

    if summaries:
        print("######## Aggregate Metrics ########")
        n_patients = len(summaries)
        avg_psnr_sr_vs_linear = sum(d["psnr_sr_vs_linear"] for d in summaries) / n_patients
        avg_psnr_sr_vs_gt = sum(d["psnr_sr_vs_gt"] for d in summaries) / n_patients
        avg_psnr_linear_vs_gt = sum(d["psnr_linear_vs_gt"] for d in summaries) / n_patients
        avg_ssim_sr_vs_linear = sum(d["ssim_sr_vs_linear_mean"] for d in summaries) / n_patients
        avg_ssim_sr_vs_gt = sum(d["ssim_sr_vs_gt"] for d in summaries) / n_patients
        avg_ssim_linear_vs_gt = sum(d["ssim_linear_vs_gt"] for d in summaries) / n_patients
        total_removed = sum(d["num_removed"] for d in summaries)
        all_ssims = [value for d in summaries for value in d["ssim_values"]]
        if all_ssims:
            global_ssim_mean = float(sum(all_ssims) / len(all_ssims))
            global_ssim_min = float(min(all_ssims))
            global_ssim_max = float(max(all_ssims))
        else:
            global_ssim_mean = global_ssim_min = global_ssim_max = float("nan")

        print(f"Patients processed: {n_patients}")
        print(f"Total removed slices: {total_removed}")
        print(f"Average PSNR (SR vs linear): {avg_psnr_sr_vs_linear:.2f} dB")
        print(f"Average PSNR (SR vs GT): {avg_psnr_sr_vs_gt:.2f} dB")
        print(f"Average PSNR (Linear vs GT): {avg_psnr_linear_vs_gt:.2f} dB")
        print(f"Average SSIM (SR vs linear): {avg_ssim_sr_vs_linear:.4f}")
        print(f"Average SSIM (SR vs GT): {avg_ssim_sr_vs_gt:.4f}")
        print(f"Average SSIM (Linear vs GT): {avg_ssim_linear_vs_gt:.4f}")
        print(
            f"Global SSIM stats (SR vs linear): mean {global_ssim_mean:.4f}, "
            f"min {global_ssim_min:.4f}, max {global_ssim_max:.4f}"
        )
        print("###################################")


if __name__ == "__main__":
    main()
