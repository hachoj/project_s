import torch
import torch.nn.functional as F
import yaml

import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.data import build_val_dataloader
from model.model import ProjectI
from model.utils import PSNR, SSIM_slicewise, angle_aware_resample


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--------------------------------")
    print(f"Loading configs file...")

    with open("configs/train/eval_config.yaml") as f:
        eval_cfg = yaml.safe_load(f)

    with open("configs/model/model_config.yaml") as f:
        model_cfg = yaml.safe_load(f)

    # RDN encoder/decoder configs may be embedded or in separate files
    if isinstance(model_cfg.get("encoder_config"), dict):
        encoder_cfg = model_cfg["encoder_config"]
    else:
        with open("configs/model/encoder.yaml") as f:
            encoder_cfg = yaml.safe_load(f)

    if isinstance(model_cfg.get("decoder_config"), dict):
        decoder_cfg = model_cfg["decoder_config"]
    else:
        with open("configs/model/decoder_config.yaml") as f:
            decoder_cfg = yaml.safe_load(f)

    with open("configs/train/val_data_config.yaml") as f:
        val_data_cfg = yaml.safe_load(f)

    # paths
    resume_path = eval_cfg.get("model_resume_path", None)

    zero_one = bool(model_cfg.get("zero_one", False))
    slices_per_pred = model_cfg["slices_per_pred"]

    print(f"Config file loaded successfully")
    print(f"--------------------------------")
    print(f"Building dataloaders...")

    val_data_cfg["patch_size"] = model_cfg["patch_size"]
    val_data_cfg["zero_one"] = zero_one
    val_loader = build_val_dataloader(**val_data_cfg)

    print(f"Dataloaders built successfully")
    print(f"--------------------------------")
    print(f"Building model...")

    decoder_cfg["in_channels"] = slices_per_pred * model_cfg["embd_dim"]
    model = ProjectI(
        embd_dim=model_cfg["embd_dim"],
        encoder_config=encoder_cfg,
        decoder_config=decoder_cfg,
    ).to(device)

    model = torch.compile(model)

    if resume_path and os.path.isfile(resume_path):
        _ckpt = torch.load(resume_path, map_location="cpu")
        if "model_state" in _ckpt:
            model.load_state_dict(_ckpt["model_state"])

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Model built successfully")
    print(f"--------------------------------")

    # Map metric series to desired step axes in W&B
    torch.set_float32_matmul_precision("high")

    print(f"Beginning eval...")

    model.eval()

    val_loss = 0.0
    PSNR_total = 0.0
    SSIM_total = 0.0

    SSIM_total_linear = 0.0
    PSNR_total_linear = 0.0
    val_loss_linear = 0.0

    inp_idx = [0, 2]
    gt_idx = [1]

    with torch.no_grad():
        for (slices, angles) in val_loader:
            slices = slices.to(device)
            angles = angles.to(device)

            delta = (angles[:, 2] - angles[:, 0]).clamp_min(1e-6)
            relative_angle = (angles[:, 1] - angles[:, 0]) / delta
            real_midline = (angles[:, 2] + angles[:, 0]) * 0.5

            conditioning_slices = slices[:, inp_idx, :, :]
            target_slice = slices[:, gt_idx, :, :]

            out = model(
                conditioning_slices,
                relative_angle.unsqueeze(1),
                delta.unsqueeze(1),
                real_midline.unsqueeze(1),
            )

            loss = F.mse_loss(out, target_slice)

            PSNR_total += PSNR(out, target_slice, zero_one=zero_one)
            SSIM_total += SSIM_slicewise(out, target_slice, zero_one=zero_one)
            val_loss += loss.item()


            out_linear = angle_aware_resample(
                angles[0, inp_idx],
                slices[0, inp_idx, :, :].unsqueeze(1),
                angles[0, gt_idx],
                radians=False,
            )

            loss_linear = F.mse_loss(out_linear, slices[:, gt_idx, :, :])
            PSNR_total_linear += PSNR(out_linear, slices[:, gt_idx, :, :], zero_one=zero_one)
            SSIM_total_linear += SSIM_slicewise(out_linear, slices[:, gt_idx, :, :], zero_one=zero_one)
            val_loss_linear += loss_linear.item()

    # Calculate average validation loss for the epoch

    avg_val_loss = val_loss / len(val_loader)
    avg_PSNR = PSNR_total / len(val_loader)
    avg_SSIM = SSIM_total / len(val_loader)

    avg_val_loss_linear = val_loss_linear / len(val_loader)
    avg_PSNR_linear = PSNR_total_linear / len(val_loader)
    avg_SSIM_linear = SSIM_total_linear / len(val_loader)

    print(f"Eval completed")
    print(f"################################")
    print(f"Average Val Loss: {avg_val_loss:.6f}")
    print(f"Average PSNR: {avg_PSNR:.6f}")
    print(f"Average SSIM: {avg_SSIM:.6f}")
    print(f"################################")
    print(f"Average Val Loss (Linear): {avg_val_loss_linear:.6f}")
    print(f"Average PSNR (Linear): {avg_PSNR_linear:.6f}")
    print(f"Average SSIM (Linear): {avg_SSIM_linear:.6f}")
    print(f"################################")
