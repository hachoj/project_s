import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
import wandb
from torch.amp import autocast

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.data import build_train_dataloader, build_val_dataloader
from model.model import ProjectI
from model.utils import reconstruct_angle_sr, reconstruct_removed_hw, PSNR, SSIM_slicewise, angle_aware_resample
from model.reconstruction import extract_slices, reconstruct_volume


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"--------------------------------")
    print(f"Loading configs file...")

    with open("configs/train/eval_config.yaml") as f:
        eval_cfg = yaml.safe_load(f)

    with open("configs/model/model_config.yaml") as f:
        model_cfg = yaml.safe_load(f)

    with open(f"configs/model/{model_cfg['encoder_type']}_config.yaml") as f:
        encoder_cfg = yaml.safe_load(f)

    with open("configs/model/decoder_config.yaml") as f:
        decoder_cfg = yaml.safe_load(f)

    with open("configs/train/train_data_config.yaml") as f:
        train_data_cfg = yaml.safe_load(f)

    with open("configs/train/val_data_config.yaml") as f:
        val_data_cfg = yaml.safe_load(f)

    # paths
    resume_path = eval_cfg.get("model_resume_path", None)

    dtype = (
        torch.bfloat16
        if eval_cfg.get("dtype", "bfloat16") == "bfloat16"
        else torch.float16
    )

    zero_one = eval_cfg.get("zero_one", False)

    # Minimal resume support: single path or None
    start_epoch = 0

    print(f"Config file loaded successfully")
    print(f"--------------------------------")
    print(f"Building dataloaders...")

    val_loader = build_val_dataloader(**val_data_cfg)

    print(f"Dataloaders built successfully")
    print(f"--------------------------------")
    print(f"Building model...")

    model = ProjectI(
        embd_dim=model_cfg["embd_dim"],
        encoder_type=model_cfg["encoder_type"],
        is_attention_resample=model_cfg["is_attention_resample"],
        encoder_config=encoder_cfg,
        decoder_config=decoder_cfg,
    ).to(device)

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

    cyclic_ratio = 0.0

    model.eval()

    val_loss = 0.0
    PSNR_total = 0.0
    SSIM_total = 0.0

    val_loss_linear = 0.0
    PSNR_total_linear = 0.0
    SSIM_total_linear = 0.0

    with torch.no_grad():
        for (
            input_slices,
            input_angles,
            query_slices,
            query_angles,
            full_slices,
            full_angles,
        ) in val_loader:
            input_slices = input_slices.squeeze(0).to(device)
            input_angles = input_angles.squeeze(0).to(device)
            query_slices = query_slices.squeeze(0).to(device)
            query_angles = query_angles.squeeze(0).to(device)
            full_slices = full_slices.squeeze(0).to(device)
            full_angles = full_angles.squeeze(0).to(device)

            out, _ = model(
                input_slices, input_angles, full_angles, is_train=False
            )

            loss = F.l1_loss(out, full_slices)
            PSNR_total += PSNR(out, full_slices, zero_one=zero_one)
            SSIM_total += SSIM_slicewise(out, full_slices, zero_one=zero_one)
            val_loss += loss.item()

            out_linear = angle_aware_resample(
                input_angles,
                input_slices.unsqueeze(1),
                full_angles,
                radians=True,
            ).squeeze(1)
            loss_linear = F.l1_loss(out_linear, full_slices)
            PSNR_total_linear += PSNR(out_linear, full_slices, zero_one=zero_one)
            SSIM_total_linear += SSIM_slicewise(out_linear, full_slices, zero_one=zero_one)
            val_loss_linear += loss_linear.item()

    # Calculate average validation loss for the epoch

    avg_val_loss = val_loss / len(val_loader)
    avg_PSNR = PSNR_total / len(val_loader)
    avg_SSIM = SSIM_total / len(val_loader)

    print(f"Eval completed")
    print(f"--------------------------------")
    print(f"Average Val Loss: {avg_val_loss:.6f}")
    print(f"Average PSNR: {avg_PSNR:.6f}")
    print(f"Average SSIM: {avg_SSIM:.6f}")
    print(f"################################################")
    print(f"Average Val Loss (Linear): {val_loss_linear / len(val_loader):.6f}")
    print(f"Average PSNR (Linear): {PSNR_total_linear / len(val_loader):.6f}")
    print(f"Average SSIM (Linear): {SSIM_total_linear / len(val_loader):.6f}")
    print(f"--------------------------------")