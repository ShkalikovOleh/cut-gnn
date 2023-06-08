import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import Sequential, BatchNorm

from typing import Tuple, List

from .layers import CutGCNConv

__all__ = ['CutGCN']

class CutGCN(nn.Module):
    def __init__(self, n_layers: int = 12, hidden_dims: int = 256,
                 out_dims: int = 16,
                 apply_batch_norm: bool = True, dropout_rate: float = 0,
                 apply_relu: bool = True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        gnn_layers = []

        if apply_batch_norm:
            gnn_layers.append((BatchNorm(2), 'x -> x'))

        #first GNN layer block
        gnn_layers.extend(CutGCN.__make_gnn_layer(2, hidden_dims, apply_batch_norm,
                                                  apply_relu, dropout_rate))

        # hidden GNN layer blocks
        for _ in range(1, n_layers-1):
            gnn_layers.extend(CutGCN.__make_gnn_layer(hidden_dims, hidden_dims,
                                                      apply_batch_norm, apply_relu, dropout_rate))

        # last GNN layer block
        gnn_layers.extend(CutGCN.__make_gnn_layer(hidden_dims, out_dims,
                                                  apply_batch_norm, False, 0))

        self.gcn = Sequential('x, edge_index, edge_weight', gnn_layers)

    @staticmethod
    def __make_gnn_layer(in_dim: int, out_dim: int, apply_bn: bool,
                         apply_relu: bool, dropout_rate: float) -> List[Tuple[torch.nn.Module, str]]:
        layers = []

        layers.append((CutGCNConv(in_dim, out_dim), 'x, edge_index, edge_weight -> x'))

        if apply_bn:
            layers.append((BatchNorm(out_dim), 'x -> x'))

        if apply_relu:
            layers.append((torch.nn.ReLU(inplace=True), 'x -> x'))

        if dropout_rate > 0:
            layers.append((torch.nn.Dropout(p=dropout_rate), 'x -> x'))

        return layers

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor) -> torch.Tensor:
        node_feat = self.gcn(x=x, edge_index=edge_index, edge_weight=edge_weight)
        return node_feat