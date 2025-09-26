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

    def forward(self, slices, angle):
        """
        Supports batched and unbatched inputs.

        Args:
            slices: either [S,H,W] (unbatched) or [B,S,H,W] (batched)
            angle: either [1] (unbatched) or [B] (batched), raw degree offset

        Returns:
            out: [1,1,H,W] for unbatched, or [B,1,H,W] for batched
        """
        assert slices.dim() in (3, 4), "slices must be [S,H,W] or [B,S,H,W]"
        if slices.dim() == 3:
            # Unbatched path
            S, H, W = slices.shape
            assert angle.dim() == 1, "Angle should be a 1-D tensor"

            x = slices.unsqueeze(1)  # [S,1,H,W]
            feats = self.encoder(x)  # [S,C,H,W]
            feats = feats.view(1, S*self.embd_dim, H, W)  # [1,S*C,H,W]

            angle_emb = get_angle_embedding(angle, self.embd_dim)  # [1,C]
            angle_emb = angle_emb.view(1, self.embd_dim, 1, 1).expand(1, self.embd_dim, H, W)  # [1,C,H,W]

            out = self.decoder(feats, angle_emb)  # [1,1,H,W]
            return out
        else:
            # Batched path
            B, S, H, W = slices.shape
            assert angle.dim() in (1, 2), "Angle should be [B] or [B,1]"
            angle = angle.view(B)

            # Flatten the pair dimension into batch for encoder
            x = slices.view(B * S, 1, H, W)  # [B*S,1,H,W]
            feats = self.encoder(x)  # [B*S,C,H,W] (angles unused in encoder)
            feats = feats.view(B, S*self.embd_dim, H, W)  # [B,S*C,H,W]

            angle_emb = get_angle_embedding(angle, self.embd_dim)  # [B,C]
            angle_emb = angle_emb.view(B, self.embd_dim, 1, 1).expand(B, self.embd_dim, H, W)  # [B,C,H,W]
            out = self.decoder(feats, angle_emb)  # [B,1,H,W]
            return out
