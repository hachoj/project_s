import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
import wandb
from torch.amp import autocast

import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.data import build_train_dataloader, build_val_dataloader
from model.model import ProjectI
from model.utils import (
    reconstruct_angle_sr,
    PSNR,
    SSIM_slicewise,
)
from model.reconstruction import extract_slices, reconstruct_volume


def loss_fn(out, full_slices):
    return F.mse_loss(out, full_slices)


# Add this comprehensive monitoring function
def monitor_training_step(model, optimizer, step, log_frequency):
    """Complete training monitoring for wandb"""
    if step % log_frequency != 0:
        return {}

    metrics = {}

    # 4. Log per-parameter gradient norms and histograms (weights only)
    for name, param in model.named_parameters():
        if param.grad is not None and "weight" in name:
            grad_tensor = param.grad.detach()
            metrics[f"gradients/{name}/norm"] = grad_tensor.data.norm(2).item()
            try:
                grad_flat_cpu = grad_tensor.float().cpu().view(-1).numpy()
                metrics[f"gradhist/{name}"] = wandb.Histogram(
                    grad_flat_cpu, num_bins=64
                )
            except Exception:
                pass

    return metrics


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--------------------------------")
    print(f"Loading configs file...")

    with open("configs/train/train_config.yaml") as f:
        train_cfg = yaml.safe_load(f)

    with open("configs/model/model_config.yaml") as f:
        model_cfg = yaml.safe_load(f)

    with open("configs/model/encoder.yaml") as f:
        encoder_cfg = yaml.safe_load(f)

    with open("configs/model/decoder_config.yaml") as f:
        decoder_cfg = yaml.safe_load(f)

    with open("configs/train/train_data_config.yaml") as f:
        train_data_cfg = yaml.safe_load(f)

    with open("configs/train/val_data_config.yaml") as f:
        val_data_cfg = yaml.safe_load(f)

    # paths
    save_dir = train_cfg["checkpoint_save_dir"]
    resume_path = train_cfg.get("model_resume_path", None)

    # training params
    save_frequency = train_cfg["save_frequency"]
    val_frequency = train_cfg["val_frequency"]
    log_frequency = train_cfg["log_frequency"]
    lr = train_cfg["lr"]
    max_halves = train_cfg["max_halves"]
    gradient_accumulation_steps = train_cfg["gradient_accumulation_steps"]
    warmup_steps = train_cfg["warmup_steps"]
    num_epochs = train_cfg["num_epochs"]
    max_grad_norm = train_cfg["max_grad_norm"]
    if max_grad_norm <= 0:
        max_grad_norm = float("inf")

    dtype = (
        torch.bfloat16
        if train_cfg.get("dtype", "bfloat16") == "bfloat16"
        else torch.float16
    )

    # reconstruction params
    image_path = train_cfg["image_path"]
    target_step_deg = train_cfg["target_step_deg"]
    reconstruction_frequency = train_cfg["reconstruction_frequency"]

    # Optional tiling params moved to model_config.yaml
    patch_size = model_cfg["patch_size"]
    stride = model_cfg["stride"]

    zero_one = bool(model_cfg.get("zero_one", False))
    slices_per_pred = int(model_cfg.get("slices_per_pred", 2))

    # Minimal resume support: single path or None
    start_epoch = 0
    wandb_run_id = None

    print(f"Config file loaded successfully")
    print(f"--------------------------------")
    print(f"Building dataloaders...")

    train_data_cfg["patch_size"] = patch_size
    train_data_cfg["zero_one"] = zero_one
    val_data_cfg["patch_size"] = patch_size
    val_data_cfg["zero_one"] = zero_one
    train_loader = build_train_dataloader(**train_data_cfg)
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
    torch.manual_seed(43)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    patience = train_cfg["patience"]
    patience_counter = 0
    halve_counter = 0

    if resume_path and os.path.isfile(resume_path):
        print(f"Resuming from checkpoint: {resume_path}")
        _ckpt = torch.load(resume_path, map_location="cpu")
        if "model_state" in _ckpt:
            model.load_state_dict(_ckpt["model_state"])
        if "optim_state" in _ckpt:
            optimizer.load_state_dict(_ckpt["optim_state"])
        if "halve_counter" in _ckpt:
            halve_counter = _ckpt.get("halve_counter", 0)
        if "patience_counter" in _ckpt:
            patience_counter = _ckpt.get("patience_counter", 0)
        if "best_val_loss" in _ckpt:
            best_val_loss = _ckpt.get("best_val_loss", float("inf"))
        start_epoch = _ckpt.get("epoch", 0)
        wandb_run_id = _ckpt.get("wandb_run_id", None)
        print(f"Resumed state. Starting at epoch {start_epoch}.")
    else:
        print("No checkpoint provided. Starting training from scratch.")

    # model = torch.compile(model)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"Effective batch size: {train_data_cfg['batch_size'] * gradient_accumulation_steps}")

    print(f"Model built successfully")
    print(f"--------------------------------")
    print(f"Initializing WandB...")

    if wandb_run_id:
        wandb.init(
            entity="hachoj-university-of-florida",
            project="project_s",
            config={
                **train_cfg,
                **model_cfg,
                **encoder_cfg,
                **decoder_cfg,
                **train_data_cfg,
                **val_data_cfg,
            },
            id=wandb_run_id,
            resume="allow",
        )
        print(f"WandB resumed successfully (run id: {wandb_run_id})")
    else:
        wandb.init(
            entity="hachoj-university-of-florida",
            project="project_s",
            config={
                **train_cfg,
                **model_cfg,
                **encoder_cfg,
                **decoder_cfg,
                **train_data_cfg,
                **val_data_cfg,
            },
        )
        print(f"WandB initialized successfully")
    print(f"--------------------------------")
    # Map metric series to desired step axes in W&B
    torch.set_float32_matmul_precision("high")

    print(f"Beginning training...")

    inp_idx = [0, 2]
    gt_idx = [1]

    for epoch in range(start_epoch, num_epochs):
        start_time = time.time()

        model.train()

        running_train_loss = 0.0
        micro_count = 0

        # Epoch-level accumulators (per optimizer update)
        epoch_train_loss_sum = 0.0

        # Consolidated end-of-epoch logging payload
        end_of_epoch_log = {}

        for i, (slices, angles) in enumerate(train_loader):
            slices = slices.to(device)
            angles = angles.to(device)

            # Use raw degree offset between the target slice and the previous slice
            relative_angle = angles[:, 1] - angles[:, 0]

            # Only zero gradients at the start of accumulation cycle
            if i % gradient_accumulation_steps == 0:
                optimizer.zero_grad()

            with autocast(device_type=device, dtype=dtype):
                out = model(slices[:, inp_idx, :, :], relative_angle)

            target = slices[:, gt_idx, :, :].to(dtype=out.dtype)
            loss = loss_fn(out, target)

            running_train_loss += loss.item()

            # Scale loss by accumulation steps to maintain effective learning rate
            loss = loss / gradient_accumulation_steps

            loss.backward()
            micro_count += 1

            # Only update parameters and log at the end of accumulation cycle
            if (i + 1) % gradient_accumulation_steps == 0 or (i + 1) == len(
                train_loader
            ):
                # Get training statistics for this step (every log_frequency steps)
                actual_step = (
                    epoch * len(train_loader) + i
                ) // gradient_accumulation_steps
                step_metrics = monitor_training_step(
                    model, optimizer, actual_step, log_frequency
                )

                # Apply linear warmup over the first `warmup_steps` optimizer update steps
                if actual_step < warmup_steps:
                    optimizer.param_groups[0]["lr"] = (
                        lr * (actual_step + 1) / warmup_steps
                    )

                # (Removed cyclic_ratio warmup; not used in this training loop)

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=max_grad_norm
                )
                optimizer.step()

                # Log training metrics per optimizer step using true group averages
                group_count = max(1, micro_count)
                train_log = {
                    "train/loss": running_train_loss / group_count,
                    "train/grad_norm_total": grad_norm.item(),
                    "train/step": actual_step,
                    "train/learning_rate": optimizer.param_groups[0]["lr"],
                }
                train_log.update(step_metrics)
                wandb.log(train_log, step=actual_step)

                # Accumulate for epoch averages
                epoch_train_loss_sum = epoch_train_loss_sum + train_log["train/loss"]

                # Reset group accumulators
                running_train_loss = 0.0
                micro_count = 0

        if (epoch + 1) % val_frequency == 0 or epoch == 0:
            # validation loop
            model.eval()

            val_loss = 0.0
            PSNR_total = 0.0
            SSIM_total = 0.0

            with torch.no_grad():
                for (slices, angles) in val_loader:
                    slices = slices.to(device)
                    angles = angles.to(device)

                    relative_angle = angles[:, 1] - angles[:, 0]

                    out = model(slices[:, inp_idx, :, :], relative_angle)

                    target = slices[:, gt_idx, :, :].to(dtype=out.dtype)
                    loss = loss_fn(out, target)

                    PSNR_total += PSNR(out, target, zero_one=zero_one)
                    SSIM_total += SSIM_slicewise(out, target, zero_one=zero_one)
                    val_loss += loss.item()

            # Calculate average validation loss for the epoch

            avg_val_loss = val_loss / len(val_loader)
            avg_PSNR = PSNR_total / len(val_loader)
            avg_SSIM = SSIM_total / len(val_loader)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience and halve_counter < max_halves:
                optimizer.param_groups[0]["lr"] *= 0.5
                patience_counter = 0
                halve_counter += 1

            # Clear any cached GPU memory used during validation tiling
            if device == "cuda":
                torch.cuda.empty_cache()

            end_of_epoch_log.update(
                {
                    "epoch/val_loss": avg_val_loss,
                    "epoch/best_val_loss": best_val_loss,
                    "epoch/halve_counter": halve_counter,
                    "epoch/patience_counter": patience_counter,
                    "epoch/PSNR": avg_PSNR,
                    "epoch/SSIM": avg_SSIM,
                    "epoch/epoch": epoch + 1,
                }
            )
        if (epoch + 1) % save_frequency == 0:
            os.makedirs(save_dir, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "wandb_run_id": wandb.run.id,
                    "halve_counter": halve_counter,
                    "patience_counter": patience_counter,
                    "best_val_loss": best_val_loss,
                },
                f"{save_dir}/checkpoint_epoch_{epoch + 1}.pt",
            )
        if (epoch + 1) % reconstruction_frequency == 0:
            print(f"Reconstructing volume for epoch {epoch + 1}")
            extracted_slices = extract_slices(image_path)
            # Get the first key in extracted_slices instead of hardcoding
            first_key = next(iter(extracted_slices))
            shape = extracted_slices[first_key]["shape"]
            spacing = extracted_slices[first_key]["spacing"]

            reconstructed_slices = dict()

            slices = []
            angles = []

            for value in extracted_slices.values():
                angles.append(value["angle"])
                slices.append(value["slice_volume"])

            angles = torch.tensor(np.array(angles), dtype=torch.float32)
            slices = torch.tensor(np.array(slices), dtype=torch.float32)

            model.eval()
            with torch.no_grad():
                slices = slices.to(device)  # still in raw [0, 255] uint8
                angles = angles.to(device)  # still in raw degrees

                pred_uniform, grid_deg = reconstruct_angle_sr(
                    model,
                    slices_THW=slices,  # (T,H,W)
                    angles_deg_T=angles,  # (T,)
                    target_step_deg=target_step_deg,
                    zero_one=zero_one,
                    patch_size=patch_size,
                    stride_hw=stride,
                )
                for sr_angle, sr_slices in zip(grid_deg, pred_uniform):
                    reconstructed_slices[sr_angle.item()] = {
                        "angle": sr_angle.item(),
                        "slice_volume": (
                            (sr_slices * 255.0) if zero_one else (sr_slices + 1) * 127.5
                        )
                        .cpu()
                        .numpy()
                        .astype(np.uint8),
                        "shape": shape,
                        "spacing": spacing,
                    }

                reconstructed_volume = reconstruct_volume(reconstructed_slices)

                _mid_slice = reconstructed_volume["volume"][
                    :, :, reconstructed_volume["volume"].shape[2] // 2
                ].transpose(1, 0)
                _mid_slice_uint8 = (_mid_slice * 255.0).astype(np.uint8)
                end_of_epoch_log.update(
                    {
                        "reconstruction/reconstructed_volume": wandb.Image(
                            _mid_slice_uint8
                        ),
                        "epoch/epoch": epoch + 1,
                    }
                )
            print(f"Reconstructed volume for epoch {epoch + 1} saved successfully")
            print(f"--------------------------------")

        end_time = time.time()
        epoch_duration = end_time - start_time
        end_of_epoch_log.update(
            {
                "epoch/duration": epoch_duration,
                "epoch/epoch": epoch + 1,
            }
        )
        wandb.log(end_of_epoch_log, step=actual_step)

    wandb.finish()
    print(f"Training completed successfully")
    print(f"--------------------------------")
