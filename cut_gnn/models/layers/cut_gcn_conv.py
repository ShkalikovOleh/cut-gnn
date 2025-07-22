import torch
from torch.nn import Parameter
from torch_geometric.nn import Linear, MessagePassing
from torch_geometric.utils import scatter

__all__ = ["CutGCNConv"]


class CutGCNConv(MessagePassing):
    r"""
    Graph Convolution Layer proposed in the paper

    Jung, S., & Keuper, M. (2022). Learning to solve Minimum Cost Multicuts efficiently using
    Edge-Weighted Graph Convolutional Neural Networks.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        apply_linear_before: bool = True,
        **kwargs,
    ):
        """Create a CutGCN layer.

        Args:
            in_channels (int): dim of input embeddings
            out_channels (int): dim of output embeddings
            apply_linear_before (bool, optional): apply linear layer before message passing or after. Defaults to True.
        """
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.apply_linear_before = apply_linear_before

        self.lin = Linear(
            in_channels,
            out_channels,
            bias=not apply_linear_before,
            weight_initializer="glorot",
        )
        if self.apply_linear_before:
            self.bias = Parameter(torch.empty(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        if self.apply_linear_before:
            self.bias.data.zero_()

    def apply_norm(
        self, edge_index: torch.Tensor, edge_weight: torch.Tensor, num_nodes: int
    ):
        row, col = edge_index
        idx = col if self.flow == "source_to_target" else row
        deg = scatter(
            torch.abs(edge_weight), idx, dim=0, reduce="sum", dim_size=num_nodes
        )

        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(torch.isinf(deg_inv_sqrt), 0)

        norm_weight = edge_weight * deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return norm_weight

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor = None,
    ) -> torch.Tensor:
        if self.apply_linear_before:
            x = self.lin(x)

        w_norm = self.apply_norm(edge_index, edge_weight, x.shape[0])

        h = self.propagate(edge_index, x=x, edge_weight=w_norm)
        h += x

        if self.apply_linear_before:
            out = h + self.bias
        else:
            out = self.lin(h)

        return out

    def message(self, x_j: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        return edge_weight.view(-1, 1) * x_j
