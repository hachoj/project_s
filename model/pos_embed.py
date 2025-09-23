import torch
import torch.nn as nn
import math


class FiLM(nn.Module):
    def __init__(self, num_harmonics, embd_dim):
        super().__init__()
        self.num_harmonics = num_harmonics
        self.embd_dim = embd_dim
        self.gamma_beta = nn.Linear(num_harmonics * 2, embd_dim * 2)

        freqs = torch.arange(1, 1 + num_harmonics, dtype=torch.float32)
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, angles, image_tensor):
        # angles: [1, T]
        angles = angles.to(
            device=image_tensor.device, dtype=image_tensor.dtype
        )  # [1, T]
        freqs = self.freqs.to(
            device=image_tensor.device, dtype=image_tensor.dtype
        )  # [H]

        # Multiply each angle by all harmonics: expand T over H
        angles = angles[..., None] * freqs[None, None, :]  # [1, T, H]s

        # Create sinusoidal features per (1, T)
        harmonics = torch.cat(
            [torch.sin(angles), torch.cos(angles)], dim=-1
        )  # [1, T, 2H]

        # Linear maps 2H -> 2C to produce FiLM params per (1, T)
        gb = self.gamma_beta(harmonics)  # [1, T, 2C]
        gamma, beta = gb.chunk(2, dim=-1)  # [1, T, C] each

        # Broadcast over spatial dims
        gamma = gamma[..., None, None]  # [1, T, C, 1, 1]
        beta = beta[..., None, None]  # [1, T, C, 1, 1]
        return image_tensor * gamma + beta  # [1, T, C, H, W]


class AddativePositionalEmbedding(nn.Module):
    def __init__(self, num_harmonics, embd_dim, mlp_ratio=4, radians=True):
        super().__init__()
        self.num_harmonics = num_harmonics
        self.embd_dim = embd_dim
        self.radians = radians

        self.proj = nn.Sequential(
            nn.Linear(num_harmonics * 2, embd_dim),
            nn.GELU(),
            nn.Linear(embd_dim, embd_dim*mlp_ratio),
            nn.GELU(),
            nn.Linear(embd_dim*mlp_ratio, embd_dim),
        )

        freqs = torch.arange(1, 1 + num_harmonics, dtype=torch.float32)
        self.register_buffer("freqs", freqs)

    def forward(self, angles, image_tensor):
        angles = angles.to(device=image_tensor.device, dtype=image_tensor.dtype)
        freqs = self.freqs.to(device=image_tensor.device, dtype=image_tensor.dtype)

        if not self.radians:
            angles = angles * (torch.pi / 180.0)

        angles = angles[..., None] * freqs[None, None, :]  # [1, T, H]

        harmonics = torch.cat(
            [torch.sin(angles), torch.cos(angles)], dim=-1
        )  # [1, T, 2H]

        pos_embedding = self.proj(harmonics)  # [1, T, C]
        pos_embedding = pos_embedding[..., None, None]  # [1, T, C, 1, 1]
        return image_tensor + pos_embedding  # [1, T, C, H, W]
