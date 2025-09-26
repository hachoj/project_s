import torch
import torch.nn as nn

from .metaformer import LayerNorm2d


class AttentionResample(nn.Module):
    """Token-level cross attention between context and query slices.

    Accepts inputs with optional leading batch dimension:

    - ``original_slices``: ``[B, T, C, H, W]`` or ``[T, C, H, W]``
    - ``query_slices``: ``[B, Q, C, H, W]`` or ``[Q, C, H, W]``

    Spatial dimensions are flattened so every pixel location in the query can
    attend to every pixel across the context slices.
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
        # Normalize shapes to include batch dimension if necessary
        if original_slices.dim() == 4:
            original_slices = original_slices.unsqueeze(0)
            query_slices = query_slices.unsqueeze(0)
            added_batch_dim = True
        elif original_slices.dim() == 5:
            added_batch_dim = False
        else:
            raise ValueError(
                "original_slices must be a 4D or 5D tensor of shape [T,C,H,W] or [B,T,C,H,W]"
            )

        if query_slices.dim() not in (4, 5):
            raise ValueError(
                "query_slices must be a 4D or 5D tensor of shape [Q,C,H,W] or [B,Q,C,H,W]"
            )
        if added_batch_dim and query_slices.dim() != 5:
            query_slices = query_slices.unsqueeze(0)

        if original_slices.dim() != query_slices.dim():
            raise ValueError("original_slices and query_slices must have matching rank")

        B, T, C, H, W = original_slices.shape
        Bq, Q, Cq, Hq, Wq = query_slices.shape

        if B != Bq:
            raise ValueError("Batch dimension mismatch between original and query slices")
        if (C, H, W) != (Cq, Hq, Wq):
            raise ValueError(
                "Channel/spatial dimensions must match between original and query slices"
            )

        coords = torch.arange(H * W, device=original_slices.device, dtype=torch.long)
        pos_embd = self.pos_embd(coords)  # [HW, C]
        if pos_embd.shape[1] != C:
            raise ValueError("pos_embd output dimension must match channel dimension C")
        pos_embd = pos_embd.reshape(H, W, C).permute(2, 0, 1)
        pos_embd = pos_embd.to(dtype=original_slices.dtype)

        pos_embd = pos_embd.unsqueeze(0).unsqueeze(0)  # [1,1,C,H,W]
        original_slices = original_slices + pos_embd
        query_slices = query_slices + pos_embd

        identity = query_slices

        key = self.key_norm(original_slices.reshape(B * T, C, H, W))
        key = key.view(B, T, C, H, W)
        value = key
        query = self.query_norm(query_slices.reshape(B * Q, C, H, W))
        query = query.view(B, Q, C, H, W)

        key_tokens = key.reshape(B, T, C, H * W).permute(0, 1, 3, 2).reshape(
            B, T * H * W, C
        )
        value_tokens = value.reshape(B, T, C, H * W).permute(0, 1, 3, 2).reshape(
            B, T * H * W, C
        )
        query_tokens = query.reshape(B, Q, C, H * W).permute(0, 1, 3, 2).reshape(
            B, Q * H * W, C
        )

        identity_tokens = identity.reshape(B, Q, C, H * W).permute(0, 1, 3, 2).reshape(
            B, Q * H * W, C
        )

        attended_tokens, _ = self.attention(query_tokens, key_tokens, value_tokens)
        attended_tokens = attended_tokens + identity_tokens

        attended = attended_tokens.view(B, Q, H * W, C).permute(0, 1, 3, 2)
        attended = attended.reshape(B, Q, C, H, W)

        out = attended

        if added_batch_dim:
            out = out.squeeze(0)

        return out
