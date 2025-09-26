import torch
import torch.nn as nn

from .metaformer import LayerNorm2d


class AttentionResample(nn.Module):
    """
    Pixel-wise cross attention between original and query slices.
    """

    def __init__(self, embd_dim, patch_size, num_heads):
        super().__init__()
        self.query_norm = LayerNorm2d(embd_dim)
        self.key_norm = LayerNorm2d(embd_dim)
        self.attention = nn.MultiheadAttention(
            embd_dim, num_heads=num_heads, batch_first=True
        )
        self.H, self.W = patch_size
        # Stable positional embedding over spatial indices [0 .. H*W-1]
        self.pos_embd = nn.Embedding(self.H * self.W, embd_dim)

    def forward(
        self, original_slices: torch.Tensor, query_slices: torch.Tensor
    ) -> torch.Tensor:
        # original_slices: [T, C, H, W], query_slices: [Q, C, H, W]
        if original_slices.dim() != 4 or query_slices.dim() != 4:
            raise ValueError(
                "original_slices and query_slices must be 4D tensors [T, C, H, W]"
            )
        if query_slices.shape[1:] != original_slices.shape[1:]:
            raise ValueError(
                "original_slices and query_slices must share channel and spatial dimensions"
            )

        T, C, H, W = original_slices.shape
        Q = query_slices.shape[0]

        # Positional embeddings per pixel (bounded magnitude)
        coords = torch.arange(H * W, device=original_slices.device, dtype=torch.long)
        pos_embd = self.pos_embd(coords)  # [HW, C] (float32)
        if pos_embd.shape[1] != C:
            raise ValueError("pos_embd output dimension must match channel dimension C")
        pos_embd = pos_embd.reshape(H, W, C).permute(2, 0, 1).unsqueeze(0)  # [1,C,H,W]
        pos_embd = pos_embd.to(dtype=original_slices.dtype)

        original_slices = original_slices + pos_embd  # [T, C, H, W]
        query_slices = query_slices + pos_embd  # [Q, C, H, W]

        identity = query_slices

        key = self.key_norm(original_slices)
        value = key
        query = self.query_norm(query_slices)

        # Treat each spatial position as an independent batch element for attention.
        key = key.permute(2, 3, 0, 1).reshape(H * W, T, C)
        value = value.permute(2, 3, 0, 1).reshape(H * W, T, C)
        query = query.permute(2, 3, 0, 1).reshape(H * W, Q, C)

        attended, _ = self.attention(query, key, value)
        attended = attended.view(H, W, Q, C).permute(2, 3, 0, 1)

        return identity + attended
