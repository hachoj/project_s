import math
import torch
import torch.nn as nn


def _init_zero(module: nn.Module) -> None:
    with torch.no_grad():
        module.weight.zero_()
        module.bias.zero_()

def get_angle_embedding(angles, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    args:
        time: a 1-D tensor of angles representing degree offsets
        embedding_dim: the dimension of the output
    returns:
        angles x embedded_dim tensor
    """
    assert len(angles.shape) == 1

    angles = angles * 1000.0  # scale to [0, 1000]

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=time.device)
    emb = time.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

class FiLM(nn.Module):
    def __init__(self, cond_dim: int, feat_channels: int, hidden_dim: int = 64, scale: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feat_channels * 2),
        )
        _init_zero(self.mlp[-1])
        self.scale = scale

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Apply feature-wise linear modulation with precomputed conditioning.

        Args:
            x: [B,C,H,W]
            cond: [B, cond_dim]
        """
        gamma_beta = self.mlp(cond)  # [B,2C]
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        gamma = (1.0 + self.scale * gamma).unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)
        return x * gamma + beta
