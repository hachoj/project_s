import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, zero_one, in_dim=128 + 3, out_dim=1, depth=4, width=256):
        super(MLP, self).__init__()
        self.layer_1 = nn.Sequential(
            nn.Linear(in_dim, width),
            nn.GELU(),
        )
        for i in range(depth):
            setattr(self, f"block_{i+1}", nn.Sequential(nn.Linear(width, width), nn.GELU()))
        self.out = nn.Sequential(nn.Linear(width, out_dim), nn.Sigmoid() if zero_one else nn.Tanh())
        self.depth = depth

    def forward(self, x):
        x = self.layer_1(x)
        for i in range(1, self.depth + 1):
            x = x + getattr(self, f"block_{i}")(x)
        x = self.out(x)
        return x.squeeze(-1) if x.dim() == 2 else x
