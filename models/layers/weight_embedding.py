import torch
from torch import nn

__all__ = ['WeightEmbedding']

class WeightEmbedding(nn.Module):
    r"""Split weights into bins defined by boundaries and
    assign learnable (usual PyTorch) embedding for every bin
    """

    def __init__(self, emb_dim: int, boundaries: torch.Tensor, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.emb = nn.Embedding(len(boundaries), emb_dim)
        self.register_buffer('boundaries', boundaries)

    def forward(self, weights) -> torch.Tensor:
        idxs = torch.bucketize(weights, self.get_buffer('boundaries'))
        return self.emb(idxs)
