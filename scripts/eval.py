import torch
import torch.nn.functional as F
import yaml

from scripts.common import bootstrap_project_root, get_device

# Ensure project root imports (model/, data/, etc.) work reliably
bootstrap_project_root()

from data.data import build_val_dataloader
from model.model import ProjectI
from model.utils import PSNR, SSIM_slicewise, angle_aware_resample


if __name__ == "__main__":
    device = get_device()
    print(f"--------------------------------")
    print(f"Loading configs file...")

    with open("configs/train/eval_config.yaml") as f:
        eval_cfg = yaml.safe_load(f)

    with open("configs/model/model_config.yaml") as f:
        model_cfg = yaml.safe_load(f)

    # RDN encoder/decoder configs
    with open("configs/model/encoder.yaml") as f:
        encoder_cfg = yaml.safe_load(f)

    with open("configs/model/decoder_config.yaml") as f:
        decoder_cfg = yaml.safe_load(f)

    with open("configs/train/val_data_config.yaml") as f:
        val_data_cfg = yaml.safe_load(f)

    # paths
    resume_path = eval_cfg.get("model_resume_path", None)

    zero_one = eval_cfg.get("zero_one", False)

    print(f"Config file loaded successfully")
    print(f"--------------------------------")
    print(f"Building dataloaders...")

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
