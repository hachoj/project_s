from typing import Sequence, Tuple, Union

import torch
import torch.nn as nn

from .encoder import RDN as encoder
from .decoder import RDN as decoder
from .angle_embedding import AddativePositionalEmbedding
from .attention_resample import AttentionResample


def _ensure_pair(value: Union[int, Sequence[int]]) -> Tuple[int, int]:
    if isinstance(value, int):
        return value, value
    if isinstance(value, Sequence) and len(value) == 2:
        return int(value[0]), int(value[1])
    raise ValueError("patch_size must be an int or length-2 sequence")


class ProjectI(nn.Module):
    def __init__(
        self,
        embd_dim: int,
        patch_size,
        num_harmonics: int,
        num_heads: int,
        encoder_config,
        decoder_config,
    ) -> None:
        super().__init__()
        patch_hw = _ensure_pair(patch_size)

        self.embd_dim = embd_dim
        self.encoder = encoder(**encoder_config)
        self.decoder = decoder(**decoder_config)
        self.pos_embd = AddativePositionalEmbedding(
            num_harmonics=num_harmonics,
            embd_dim=embd_dim,
            mlp_ratio=4,
            radians=False,
        )
        self.attn_resample = AttentionResample(self.embd_dim, patch_hw, num_heads)

    def forward(
        self,
        context_slices: torch.Tensor,
        context_angles_deg: torch.Tensor,
        query_angles_deg: torch.Tensor,
    ) -> torch.Tensor:
        """Predict query slices from contextual neighbors and their angles.

        Args:
            context_slices: ``[B, T, H, W]`` or ``[T, H, W]`` tensor of input slices.
            context_angles_deg: matching ``[B, T]`` (or ``[T]``) tensor of degrees.
            query_angles_deg: ``[B, Q]`` (or ``[Q]``) tensor for target angles.

        Returns:
            ``[B, Q, H, W]`` (or ``[Q, H, W]``) predictions in the same dtype as inputs.
        """

        if context_slices.dim() not in (3, 4):
            raise ValueError(
                "context_slices must have shape [T,H,W] or [B,T,H,W]"
            )

        was_unbatched = context_slices.dim() == 3
        if was_unbatched:
            context_slices = context_slices.unsqueeze(0)
            context_angles_deg = context_angles_deg.unsqueeze(0)
            query_angles_deg = query_angles_deg.unsqueeze(0)

        if context_angles_deg.dim() == 1:
            context_angles_deg = context_angles_deg.unsqueeze(0)
        if query_angles_deg.dim() == 1:
            query_angles_deg = query_angles_deg.unsqueeze(0)

        if context_slices.dim() != 4:
            raise ValueError("context_slices must be [B,T,H,W] after batching")

        B, T, H, W = context_slices.shape
        if context_angles_deg.shape != (B, T):
            raise ValueError("context_angles_deg must match shape [B,T]")

        Q = query_angles_deg.shape[1]
        device = context_slices.device

        x = context_slices.reshape(B * T, 1, H, W)
        feats = self.encoder(x)
        feats = feats.view(B, T, self.embd_dim, H, W)

        context_angles = context_angles_deg.to(device=feats.device, dtype=feats.dtype)
        query_angles = query_angles_deg.to(device=feats.device, dtype=feats.dtype)

        context_feats = self.pos_embd(context_angles, feats)

        query_feats = torch.zeros(
            (B, Q, self.embd_dim, H, W),
            device=feats.device,
            dtype=feats.dtype,
        )
        query_feats = self.pos_embd(query_angles, query_feats)

        attended = self.attn_resample(context_feats, query_feats)
        attended = attended.view(B * Q, self.embd_dim, H, W)

        decoded = self.decoder(attended)
        decoded = decoded.view(B, Q, -1, H, W).squeeze(2)

        if was_unbatched:
            decoded = decoded.squeeze(0)

        return decoded
