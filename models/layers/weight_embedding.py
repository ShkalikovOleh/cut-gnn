import torch
from torch import nn

__all__ = ['WeightEmbedding', 'min_max_norm_weights']

def min_max_norm_weights(weights: torch.Tensor):
    r"""Mapping weights value into interval [-1, 1]
    with use of min-max normalization
    """

    wmin, wmax = torch.min(weights), torch.max(weights)
    weights = 2 * (weights - wmin) / (wmax - wmin) - 1
    return weights

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
