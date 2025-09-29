import math

import torch
from torch import nn

from .angle_embedding import FiLM


class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3 // 2)
        self.relu = nn.SiLU()

    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], 1)


class RDB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super().__init__()
        self.layers = nn.Sequential(
            *[
                DenseLayer(in_channels + growth_rate * i, growth_rate)
                for i in range(num_layers)
            ]
        )

        # local feature fusion
        self.lff = nn.Conv2d(
            in_channels + growth_rate * num_layers, growth_rate, kernel_size=1
        )

    def forward(self, x):
        return x + self.lff(self.layers(x))  # local residual learning


class RDN(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=128,
        num_features=64,
        growth_rate=64,
        num_blocks=8,
        num_layers=3,
        cond_dim=None,
        delta_max=2.0,
    ):
        super().__init__()

        self.G0 = num_features
        self.G = growth_rate
        self.D = num_blocks
        self.C = num_layers

        if cond_dim is None:
            cond_dim = 6
        self.cond_dim = cond_dim
        self.delta_max = float(delta_max)

        self.film_sfe = FiLM(self.cond_dim, num_features)

        # shallow feature extraction
        self.sfe1 = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=3 // 2)
        self.sfe2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=3 // 2)

        # residual dense blocks
        self.rdbs = nn.ModuleList([RDB(self.G0, self.G, self.C)])
        self.rdb_films = nn.ModuleList([FiLM(self.cond_dim, self.G)])
        for _ in range(self.D - 1):
            self.rdbs.append(RDB(self.G, self.G, self.C))
            self.rdb_films.append(FiLM(self.cond_dim, self.G))

        # global feature fusion
        self.gff = nn.Sequential(
            nn.Conv2d(self.G * self.D, self.G0, kernel_size=1),
            nn.Conv2d(self.G0, self.G0, kernel_size=3, padding=3 // 2),
        )

        self.film_gff = FiLM(self.cond_dim, self.G0)

        self.output = nn.Conv2d(self.G0, out_channels, kernel_size=3, padding=3 // 2)

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

    def forward(self, x, r, delta, m):
        cond = self._build_condition(r, delta, m)

        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)

        x = self.film_sfe(sfe2, cond)
        local_features = []
        for rdb, film in zip(self.rdbs, self.rdb_films):
            x = rdb(x)
            x = film(x, cond)
            local_features.append(x)

        x = self.gff(torch.cat(local_features, 1))
        x = self.film_gff(x, cond) + sfe1  # global residual learning
        x = self.output(x)
        return x
