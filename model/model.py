import torch
import torch.nn as nn

from .encoder import RDN as encoder
from .decoder import RDN as decoder
from .get_angle_embedding import get_angle_embedding


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
        slices: 2, H, W
        angle: 1  (single relative angle from 0 to 1)
        """
        _, H, W = slices.shape

        assert angle.dim() == 1, "Angle should be a real number between 0 and 1"

        slices = slices.unsqueeze(1)  # [2, 1, H, W]

        feats = self.encoder(slices, angle)  # [2,C,H,W]

        angle_embedding = get_angle_embedding(angle, self.embd_dim).squeeze(0).expand(-1, H, W).unsqueeze(0)  # [1,C,H,W]

        out = self.decoder(feats, angle_embedding)

        return out