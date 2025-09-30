"""FiLM-conditioned residual U-Net decoder."""
from __future__ import annotations

import math
from typing import List, Optional, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .angle_embedding import FiLM
from .registry import register_decoder


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        self.film = FiLM(cond_dim, out_channels)
        self.skip_proj = None
        if in_channels != out_channels:
            self.skip_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        residual = x if self.skip_proj is None else self.skip_proj(x)
        h = self.act(self.conv1(x))
        h = self.conv2(h)
        h = self.film(h, cond)
        return self.act(h + residual)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, cond_dim: int):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.res_block = ResidualBlock(out_channels + skip_channels, out_channels, cond_dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.res_block(x, cond)


@register_decoder("resunet")
class FiLMResUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        base_channels: int = 64,
        channel_multipliers: Sequence[int] = (1, 2, 4),
        num_res_blocks: int = 2,
        cond_dim: Optional[int] = None,
        delta_max: float = 2.0,
    ) -> None:
        super().__init__()

        if cond_dim is None:
            cond_dim = 6
        self.cond_dim = cond_dim
        self.delta_max = float(delta_max)

        self.stem = nn.Conv2d(in_channels, base_channels * channel_multipliers[0], kernel_size=3, padding=1)
        self.stem_film = FiLM(self.cond_dim, base_channels * channel_multipliers[0])
        self.act = nn.SiLU()

        # Encoder path
        self.down_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()

        in_ch = base_channels * channel_multipliers[0]
        for idx, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResidualBlock(in_ch, out_ch, self.cond_dim))
                in_ch = out_ch
            self.down_blocks.append(blocks)
            if idx < len(channel_multipliers) - 1:
                self.downsamplers.append(nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1))

        # Bottleneck blocks
        bottleneck_ch = base_channels * channel_multipliers[-1]
        self.mid_blocks = nn.ModuleList(
            [ResidualBlock(bottleneck_ch, bottleneck_ch, self.cond_dim) for _ in range(2)]
        )

        # Decoder path
        self.up_blocks = nn.ModuleList()
        rev_multipliers = list(channel_multipliers[:-1])[::-1]
        in_ch = bottleneck_ch
        for mult in rev_multipliers:
            skip_ch = base_channels * mult
            out_ch = base_channels * mult
            self.up_blocks.append(UpsampleBlock(in_ch, skip_ch, out_ch, self.cond_dim))
            in_ch = out_ch

        self.tail = ResidualBlock(in_ch, base_channels * channel_multipliers[0], self.cond_dim)
        self.head_film = FiLM(self.cond_dim, base_channels * channel_multipliers[0])
        self.output = nn.Conv2d(base_channels * channel_multipliers[0], out_channels, kernel_size=3, padding=1)

    def _build_condition(self, r: torch.Tensor, delta: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        delta_clamped = torch.clamp(delta, min=1e-6)
        delta_norm = torch.clamp(delta / self.delta_max, min=0.0, max=1.0)
        m_rad = m * (math.pi / 180.0)
        cond = torch.cat(
            [
                r,
                r**2,
                delta_norm,
                torch.log(delta_clamped),
                torch.sin(m_rad),
                torch.cos(m_rad),
            ],
            dim=1,
        )
        return cond

    def forward(self, x: torch.Tensor, r: torch.Tensor, delta: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        cond = self._build_condition(r, delta, m)

        x = self.stem(x)
        x = self.act(self.stem_film(x, cond))

        skips: List[torch.Tensor] = []
        downsamplers: List[Optional[nn.Module]] = list(self.downsamplers) + [None]
        for idx, (blocks, down) in enumerate(zip(self.down_blocks, downsamplers)):
            for block in blocks:
                x = block(x, cond)
            if idx < len(self.downsamplers):
                skips.append(x)
            if down is not None:
                x = down(x)

        for block in self.mid_blocks:
            x = block(x, cond)

        for up_block in self.up_blocks:
            skip = skips.pop()
            x = up_block(x, skip, cond)

        x = self.tail(x, cond)
        x = self.head_film(x, cond)
        return self.output(x)
