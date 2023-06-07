import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import Sequential, BatchNorm

from typing import Tuple, List

from .layers import CutGCNConv

__all__ = ['CutGCN']

class CutGCN(nn.Module):
    def __init__(self, n_gnn_layers: int = 12, n_gnn_hidden_dims: int = 256,
                 gnn_out_dims: int = 16, n_mlp_hidden_dims: int = 256,
                 apply_batch_norm: bool = True, dropout_rate: float = 0,
                 apply_relu: bool = True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        gnn_layers = []

        if apply_batch_norm:
            gnn_layers.append((BatchNorm(2), 'x -> x'))

        #first GNN layer block
        gnn_layers.extend(CutGCN.__make_gnn_layer(2, n_gnn_hidden_dims, apply_batch_norm,
                                                  apply_relu, dropout_rate))

        # hidden GNN layer blocks
        for _ in range(1, n_gnn_layers-1):
            gnn_layers.extend(CutGCN.__make_gnn_layer(n_gnn_hidden_dims, n_gnn_hidden_dims,
                                                      apply_batch_norm, apply_relu, dropout_rate))

        # last GNN layer block
        gnn_layers.extend(CutGCN.__make_gnn_layer(n_gnn_hidden_dims, gnn_out_dims,
                                                  apply_batch_norm, False, 0))

        self.gcn = Sequential('x, edge_index, edge_weight', gnn_layers)

        mlp_layers = [torch.nn.Linear(gnn_out_dims*2, n_mlp_hidden_dims)]
        if apply_batch_norm:
            mlp_layers.append(torch.nn.BatchNorm1d(n_mlp_hidden_dims))
        if apply_relu:
            mlp_layers.append(torch.nn.ReLU(inplace=True))
        mlp_layers.append(torch.nn.Linear(n_mlp_hidden_dims, 1))

        self.mlp = torch.nn.Sequential(*mlp_layers)

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

        row, col = edge_index
        edge_feat = torch.hstack([node_feat[row], node_feat[col]])

        edge_pred_1 = self.mlp(edge_feat).view(-1)
        edge_pred_2 = self.mlp(torch.roll(edge_feat, shifts=16, dims=1)).view(-1)

        return F.sigmoid((edge_pred_1 + edge_pred_2) / 2)