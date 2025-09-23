# metaformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .pos_embed import AddativePositionalEmbedding


class LayerNorm2d(nn.Module):
    """LayerNorm for 2D features"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] or [T, C, H, W]
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[None, :, None, None] * x + self.bias[None, :, None, None]


class PooledSliceAttention(nn.Module):
    """Pooled attention across slices"""

    def __init__(self, dim, pool_size=8, num_heads=4):
        super().__init__()
        self.pool_size = pool_size
        self.norm = LayerNorm2d(dim)
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, x):
        # x: [T, C, H, W]
        T, C, H, W = x.shape
        if T == 1:
            return x

        identity = x
        x = self.norm(x)

        # Pool spatially
        h_pooled = max(1, H // self.pool_size)
        w_pooled = max(1, W // self.pool_size)
        x_pooled = F.adaptive_avg_pool2d(x, (h_pooled, w_pooled))

        # Attention across slices at each spatial position
        x_flat = x_pooled.flatten(2).permute(2, 0, 1)  # [HW_pooled, T, C]
        x_attended, _ = self.attention(x_flat, x_flat, x_flat)
        x_attended = x_attended.permute(1, 2, 0).reshape(T, C, h_pooled, w_pooled)

        # Upsample
        x_attended = F.interpolate(
            x_attended, size=(H, W), mode="bilinear", align_corners=False
        )

        return identity + x_attended


class ConvMLP(nn.Module):
    """1x1 Conv MLP block"""

    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.norm = LayerNorm2d(dim)
        self.fc1 = nn.Conv2d(dim, hidden_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        # x: [T, C, H, W]
        identity = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return identity + x


class MetaFormerBlock(nn.Module):
    """One MetaFormer block: Attention -> MLP"""

    def __init__(self, dim, pool_size=8, num_heads=4, mlp_ratio=4.0):
        super().__init__()
        self.attention = PooledSliceAttention(dim, pool_size, num_heads)
        self.mlp = ConvMLP(dim, mlp_ratio)

    def forward(self, x):
        x = self.attention(x)
        x = self.mlp(x)
        return x


class MetaFormerEncoder(nn.Module):
    """
    Clean MetaFormer architecture for slice encoding
    Input: [T, 1, H, W] - T slices
    Output: [T, out_channels, H, W]
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=128,
        dim=128,  # Internal dimension
        depth=4,  # Number of MetaFormer blocks
        pool_size=8,
        num_heads=4,
        mlp_ratio=4.0,
        stem_type="simple",  # "simple" or "heavy"
        patch_size=(64, 64),  # Input patch size (H, W)
    ):
        super().__init__()
        # Stem: 1 -> dim channels
        self.stem_type = stem_type
        self.dim = dim
        self.pos_embed = AddativePositionalEmbedding(
            num_harmonics=10, embd_dim=dim, radians=True
        )
        self.H, self.W = patch_size

        if stem_type == "simple":
            self.conv_stem1 = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1)
            self.norm_stem1 = LayerNorm2d(dim)
            self.gelu_stem1 = nn.GELU()
            self.conv_stem2 = None
            self.norm_stem2 = None
            self.gelu_stem2 = None
        else:
            self.conv_stem1 = nn.Conv2d(in_channels, dim // 2, kernel_size=3, padding=1)
            self.norm_stem1 = LayerNorm2d(dim // 2)
            self.gelu_stem1 = nn.GELU()
            self.conv_stem2 = nn.Conv2d(dim // 2, dim, kernel_size=3, padding=1)
            self.norm_stem2 = LayerNorm2d(dim)
            self.gelu_stem2 = nn.GELU()

        # Main blocks
        self.blocks = nn.ModuleList(
            [
                MetaFormerBlock(dim, pool_size, num_heads, mlp_ratio)
                for _ in range(depth)
            ]
        )

        # Output projection
        self.head = nn.Conv2d(dim, out_channels, kernel_size=1)

        self.in_slice_positional_embedding = nn.Embedding(
            num_embeddings=self.H * self.W, embedding_dim=self.dim
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, angles):
        # x: [T, 1, H, W]
        if self.stem_type == "simple":
            x = self.conv_stem1(x)
            x = self.pos_embed(angles.unsqueeze(0), x.unsqueeze(0)).squeeze(0)
            x = self.norm_stem1(x)
            x = self.gelu_stem1(x)
        else:
            x = self.conv_stem1(x)
            x = self.norm_stem1(x)
            x = self.gelu_stem1(x)
            x = self.conv_stem2(x)
            x = self.pos_embed(angles.unsqueeze(0), x.unsqueeze(0)).squeeze(0)
            x = self.norm_stem2(x)
            x = self.gelu_stem2(x)

        spatial_embedding = self.in_slice_positional_embedding(
            torch.arange(self.H*self.W, device=x.device)
        )  # [H*W, C]
        spatial_embedding = spatial_embedding.permute(1, 0)  # [C, H*W]
        spatial_embedding = spatial_embedding.view(self.dim, self.H, self.W).unsqueeze(
            0
        )  # [1, C, H, W]
        x = x + spatial_embedding  # [T, C, H, W]

        for block in self.blocks:
            x = block(x)

        x = self.head(x)
        return x
