import math
import torch
import torch.nn as nn

def get_angle_embedding(angles, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    args:
        angles: a 1-D tensor of angles representing degree offsets
        embedding_dim: the dimension of the output
    returns:
        angles x embedded_dim tensor
    """
    assert len(angles.shape) == 1

    angles = angles * 1000.0  # scale to [0, 1000]

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=angles.device)
    emb = angles.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

class AddativePositionalEmbedding(nn.Module):
    def __init__(self, num_harmonics, embd_dim, mlp_ratio=4, radians=True):
        super().__init__()
        self.num_harmonics = num_harmonics
        self.embd_dim = embd_dim
        self.radians = radians

        self.proj = nn.Sequential(
            nn.Linear(num_harmonics * 2, embd_dim),
            nn.GELU(),
            nn.Linear(embd_dim, embd_dim*mlp_ratio),
            nn.GELU(),
            nn.Linear(embd_dim*mlp_ratio, embd_dim),
        )

        freqs = torch.arange(1, 1 + num_harmonics, dtype=torch.float32)
        self.register_buffer("freqs", freqs)

    def forward(self, angles, image_tensor):
        angles = angles.to(device=image_tensor.device, dtype=image_tensor.dtype)
        freqs = self.freqs.to(device=image_tensor.device, dtype=image_tensor.dtype)

        if not self.radians:
            angles = angles * (torch.pi / 180.0)

        angles = angles[..., None] * freqs[None, None, :]  # [1, T, H]

        harmonics = torch.cat(
            [torch.sin(angles), torch.cos(angles)], dim=-1
        )  # [1, T, 2H]

        pos_embedding = self.proj(harmonics)  # [1, T, C]
        pos_embedding = pos_embedding[..., None, None]  # [1, T, C, 1, 1]
        return image_tensor + pos_embedding  # [1, T, C, H, W]
