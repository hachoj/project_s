import torch
import torch.nn as nn


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm applied over channel dimension for 2D feature maps.

    Expects inputs shaped ``[B, C, H, W]`` and normalizes per spatial location.
    """

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError("LayerNorm2d expects input of shape [B, C, H, W]")
        # Move channels to the last dimension for nn.LayerNorm, then restore layout
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
