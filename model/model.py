import torch
import torch.nn as nn

from .registry import build_decoder
from .encoder import RDN

# Ensure decoder modules are registered on import
from . import rdn_decoder  # noqa: F401
from . import resunet_decoder  # noqa: F401


class ProjectI(nn.Module):
    def __init__(
        self,
        embd_dim,
        encoder_config,
        decoder_config,
        linres,
    ):
        super().__init__()
        self.embd_dim = embd_dim
        decoder_cfg = dict(decoder_config) if decoder_config is not None else {}

        if "name" in decoder_cfg:
            decoder_name = decoder_cfg.pop("name")
            params = decoder_cfg.pop("params", None)
            if params is not None:
                decoder_params = dict(params)
            else:
                decoder_params = decoder_cfg
        else:
            decoder_name = "rdn"
            decoder_params = decoder_cfg

        self.encoder = RDN(**encoder_config)
        self.decoder = build_decoder(decoder_name, **decoder_params)
        self.linres = linres

    def forward(self, slices, r, delta, m):
        """
        Forward pass with FiLM conditioning.
        Args:
            slices: [2,H,W] or [B,2,H,W] input slices (S is number of anchors, typically 2)
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
        B, _, H, W = slices.shape

        def _prep(t):
            if isinstance(t, (float, int)):
                t = torch.tensor([t], dtype=slices.dtype, device=slices.device)
            t = t if t.dim() > 1 else t.view(B, 1)
            return t
        
        r = _prep(r)
        delta = _prep(delta)
        m = _prep(m)

        inp_slices = slices.view(B*2, 1, H, W)  # [B,2*C,H,W]
        x = self.encoder(inp_slices)  # [B*2,embd_dim,H,W]
        inp_slices = x.view(B, 2*self.embd_dim, H, W)  # [B,2*embd_dim,H,W]
        x = self.decoder(inp_slices, r, delta, m)  # [B,1,H,W]
        out = x

        if self.linres:
            r4 = r.view(-1, 1, 1, 1)
            x_0 = torch.lerp(slices[:, 0:1], slices[:, 1:2], r4)
            out = x + x_0

        return out
