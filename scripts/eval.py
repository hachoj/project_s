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
from model.utils import PSNR, SSIM_slicewise


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

    zero_one = bool(val_data_cfg.get("zero_one", False))

    print(f"Config file loaded successfully")
    print(f"--------------------------------")
    print(f"Building dataloaders...")

    val_data_cfg["patch_size"] = model_cfg["patch_size"]
    val_loader = build_val_dataloader(**val_data_cfg)

    print(f"Dataloaders built successfully")
    print(f"--------------------------------")
    print(f"Building model...")

    model = ProjectI(
        embd_dim=model_cfg["embd_dim"],
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

    model.eval()

    val_loss = 0.0
    PSNR_total = 0.0
    SSIM_total = 0.0

    inp_idx = [0, 2]
    gt_idx = [1]

    with torch.no_grad():
        for (slices, angle) in val_loader:
            slices = slices.to(device)
            angle = angle.to(device)

            out = model(slices[:, inp_idx, :, :], angle)

            target = slices[:, gt_idx, :, :].to(dtype=out.dtype)
            loss = F.mse_loss(out, target)
            out_3d = out.squeeze(1)
            tgt_3d = target.squeeze(1)
            PSNR_total += PSNR(out_3d, tgt_3d, zero_one=zero_one)
            SSIM_total += SSIM_slicewise(out_3d, tgt_3d, zero_one=zero_one)
            val_loss += loss.item()

    # Calculate average validation loss for the epoch

    avg_val_loss = val_loss / len(val_loader)
    avg_PSNR = PSNR_total / len(val_loader)
    avg_SSIM = SSIM_total / len(val_loader)

    print(f"Eval completed")
    print(f"--------------------------------")
    print(f"Average Val Loss: {avg_val_loss:.6f}")
    print(f"Average PSNR: {avg_PSNR:.6f}")
    print(f"Average SSIM: {avg_SSIM:.6f}")
    print(f"--------------------------------")
