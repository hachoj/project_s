import math
import torch
import torch.nn as nn

from .encoder import RDN as encoder
from .decoder import RDN as decoder
from .angle_embedding import get_angle_embedding


class ProjectI(nn.Module):
    def __init__(
        self,
        embd_dim,
        encoder_config,
        decoder_config,
    ):
        super().__init__()
        self.embd_dim = embd_dim
        self.encoder = encoder(**encoder_config)
        self.decoder = decoder(**decoder_config)

    def forward(self, slices, r, delta, m):
        """
        Forward pass with FiLM conditioning.
        Args:
            slices: [S,H,W] or [B,S,H,W] input slices (S is number of anchors, typically 2)
            r:      normalized position tensor [B] or [B,1]
            delta:  gap between anchors in degrees [B] or [B,1]
            m:      mid-angle in degrees [B] or [B,1]
        Returns:
            out: [B,1,H,W]
        """
        assert slices.dim() in (3, 4), "slices must be [S,H,W] or [B,S,H,W]"
        # Normalize to batched
        if slices.dim() == 3:
            slices = slices.unsqueeze(0)


        # Batched path
        B, S, H, W = slices.shape

        def _prep(t):
            if isinstance(t, (float, int)):
                t = torch.tensor([t], dtype=slices.dtype, device=slices.device)
            t = t if t.dim() > 1 else t.view(B, 1)
            return t
        
        r = _prep(r)
        delta = _prep(delta)
        m = _prep(m)
        # Encode slices
        x = slices.view(B * S, 1, H, W)
        feats = self.encoder(x)  # [B*S,C,H,W]
        feats = feats.view(B, S * self.embd_dim, H, W)

        out = self.decoder(feats, r, delta, m)  # [B,1,H,W]
        return out
