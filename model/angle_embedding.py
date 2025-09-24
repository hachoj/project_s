import math
import torch

def get_angle_embedding(angles, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    args:
        angles: a 1-D tensor of angles normalized to [0, 1]
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
