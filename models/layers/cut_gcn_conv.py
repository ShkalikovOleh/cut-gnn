import torch
from torch_geometric.nn import MessagePassing, Linear
from torch_geometric.utils import scatter


class CutGCNConv(MessagePassing):
    r"""
    Graph Convolution Layer proposed in the paper

    Jung, S., & Keuper, M. (2022). Learning to solve Minimum Cost Multicuts efficiently using
    Edge-Weighted Graph Convolutional Neural Networks.
    """

    def __init__(self, in_channels: int, out_channels: int, normalize: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize

        self.lin = Linear(in_channels, out_channels, bias=False,
                          weight_initializer='glorot')

    def apply_norm(self, edge_index: torch.Tensor, edge_weight: torch.Tensor):
        row, col = edge_index[0], edge_index[1]
        idx = col if self.flow == 'source_to_target' else row

        deg = scatter(torch.abs(edge_weight), idx, dim=0, reduce='sum')
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(torch.isinf(deg_inv_sqrt), 0)

        norm_weight = edge_weight * deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return norm_weight

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor = None) -> torch.Tensor:
        x = self.lin(x)

        if self.normalize:
            w_norm = self.apply_norm(edge_index, edge_weight)
        else:
            w_norm = edge_weight

        out = self.propagate(edge_index, x=x, edge_weight=w_norm)
        out = out + x

        return out

    def message(self, x_j: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        return edge_weight.view(-1, 1) * x_j
