import torch
import torch.nn as nn

from .registry import build_encoder
from .utils import angle_aware_resample
from .decoder import MLP
from .pos_embed import AddativePositionalEmbedding
from .attention_resample import AttentionResample


class ProjectI(nn.Module):
    def __init__(
        self,
        embd_dim,
        encoder_type,
        is_attention_resample,
        encoder_config,
        decoder_config,
    ):
        super().__init__()
        self.embd_dim = embd_dim
        self.encoder = build_encoder(encoder_type, encoder_config)
        self.decoder = MLP(
            zero_one=decoder_config.get("zero_one", False),
            in_dim=embd_dim + 2,
            out_dim=1,
            depth=decoder_config["decoder_depth"],
            width=decoder_config["decoder_hidden_size"],
        )
        self.pos_embd = AddativePositionalEmbedding(num_harmonics=12, embd_dim=embd_dim, radians=True)
        if is_attention_resample:
            self.attn_resample = AttentionResample(embd_dim, num_heads=4)
            if encoder_type == "rdn":
                self.pos_embd_2 = AddativePositionalEmbedding(num_harmonics=12, embd_dim=embd_dim, radians=True)
        self.is_attention_resample = is_attention_resample
        self.encoder_type = encoder_type

    def forward(self, slices, angles, query_angles, is_train=False):
        """
        slices: T, H, W
        angles: T
        query_angles: T + R = Q
        """
        T, H, W = slices.shape
        Q = query_angles.shape[0]
        C = self.embd_dim  # Feature dimension from FiLM layer

        slices = slices.unsqueeze(1)  # [T, 1, H, W]

        feats = self.encoder(slices, angles)  # [T,C,H,W]

        """
        Since I'm not querying the model outside of the fixed grid, this is equivalent to trilinear
        but eventually I want to make true INR
        """

        if self.is_attention_resample:
            sampled = self.pos_embd(query_angles[None, :], torch.zeros((Q, C, H, W), device=query_angles.device)).squeeze(0)  # [Q,C,H,W]
            if self.encoder_type == "rdn":
                feats = self.pos_embd_2(angles[None, :], feats).squeeze(0)  # [T,C,H,W]

            sampled = self.attn_resample(feats, sampled)  # [Q,C,H,W]
        else:
            sampled = angle_aware_resample(
                angles=angles,
                slices=feats,
                query_angles=query_angles,
                radians=True,
            )  # [Q,C,H,W]
        
            sampled = self.pos_embd(query_angles[None, :], sampled).squeeze(0)  # [Q,C,H,W]
        """
        ######################################################
        """

        feature_vectors = sampled.permute(0, 2, 3, 1).reshape(
            Q * H * W, C
        )  # [Q*H*W, C]

        device = sampled.device
        dtype = sampled.dtype
        r_vals = (
            (torch.arange(H, device=device, dtype=dtype) + 0.5) / H
        ) * 2.0 - 1.0  # [H]
        x_vals = (
            (torch.arange(W, device=device, dtype=dtype) + 0.5) / W
        ) * 2.0 - 1.0  # [W]
        r_hw = r_vals[:, None].expand(H, W)  # [H,W]
        x_hw = x_vals[None, :].expand(H, W)  # [H,W]
        r_rhw = r_hw.unsqueeze(0).expand(Q, H, W)  # [Q,H,W]
        x_rhw = x_hw.unsqueeze(0).expand(Q, H, W)  # [Q,H,W]

        coords = torch.stack([r_rhw, x_rhw], dim=-1)  # [Q,H,W,2]
        coords_flat = coords.reshape(Q * H * W, 2)  # [Q*H*W, 2]

        decoder_in = torch.cat([feature_vectors, coords_flat], dim=-1)  # [Q*H*W, C+2]
        intensity_prediction = self.decoder(decoder_in).squeeze(-1)  # [Q*H*W]
        out = intensity_prediction.view(Q, H, W)  # [Q,H,W]

        if is_train:
            inp = out.detach().unsqueeze(1)  # [Q, 1, H, W]
            c_feats = self.encoder(inp, query_angles)  # [Q,C,H,W]

            if self.is_attention_resample:
                c_sampled = self.pos_embd(angles[None, :], torch.zeros((T, C, H, W), device=query_angles.device)).squeeze(0)  # [T,C,H,W]
                if self.encoder_type == "rdn":
                    c_feats = self.pos_embd_2(query_angles[None, :], c_feats).squeeze(0)  # [Q,C,H,W]

                c_sampled = self.attn_resample(c_feats, c_sampled)  # [T,C,H,W]
            else:
                c_sampled = angle_aware_resample(
                    angles=query_angles,
                    slices=c_feats,
                    query_angles=angles,
                    radians=True,
                )  # [T,C,H,W]

                c_sampled = self.pos_embd(angles[None, :], c_sampled).squeeze(0)  # [T,C,H,W]

            c_feature_vectors = c_sampled.permute(0, 2, 3, 1).reshape(
                T * H * W, C
            )  # [T*H*W, C]

            c_r_rhw = r_hw.unsqueeze(0).expand(T, H, W)  # [T,H,W]
            c_x_rhw = x_hw.unsqueeze(0).expand(T, H, W)  # [T,H,W]

            c_coords = torch.stack([c_r_rhw, c_x_rhw], dim=-1)  # [T,H,W,2]
            c_coords_flat = c_coords.reshape(T * H * W, 2)  # [T*H*W, 2]

            c_decoder_in = torch.cat(
                [c_feature_vectors, c_coords_flat], dim=-1
            )  # [T*H*W, C+2]
            c_intensity_prediction = self.decoder(c_decoder_in).squeeze(-1)  # [T*H*W]
            c_out = c_intensity_prediction.view(T, H, W)
        else:
            c_out = None

        return out, c_out
