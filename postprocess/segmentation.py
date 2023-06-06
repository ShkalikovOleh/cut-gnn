import torch
from torch_geometric.nn.conv import MessagePassing

import typing

class Segmenter(MessagePassing):
    r"""Perform segmentation based on given predictions.
    Starting with totally separated clusters (1 cluster -> 1 node)
    merges all nodes connected by edge with predicted value less than threshold.
    The resulted cluster has a label equal to minimum of the source labels.

    Also, compute the edge 0,1-labeling for the segmentation, e.g. uncut edges
    which violates cycle consistency constraint
    """

    def __init__(self, threshold: float = 0.5, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs, aggr='min')
        self.threshhold = threshold

    def forward(self, edge_index: torch.Tensor, edge_pred: torch.Tensor,
                n_nodes: int, max_prop_iter: typing.Optional[int] = None) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        if max_prop_iter is not None and max_prop_iter < n_nodes:
            n_iter = max_prop_iter
        else:
            n_iter = n_nodes

        x = torch.arange(n_nodes, dtype=torch.long, device=edge_index.device).view(-1, 1)
        for _ in range(n_iter):
            x = self.propagate(edge_index, x=x, edge_pred=edge_pred)

        y = torch.where(x[edge_index[0]] == x[edge_index[1]], 0., 1.)

        return x.view(-1), y.view(-1)

    def message(self, x_j: torch.Tensor, x_i: torch.Tensor, edge_pred: torch.Tensor) -> torch.Tensor:
        return torch.where(edge_pred < self.threshhold, x_j[:, 0], x_i[:, 0]).view(-1, 1)
