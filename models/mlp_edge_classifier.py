import torch
from torch import nn
import torch.nn.functional as F

__all__ = ['MLPEdgeClassifier']

class MLPEdgeClassifier(nn.Module):

    def __init__(self, node_dim: int, hidden_dim: int = 256,
                 apply_relu: bool = True, apply_bn: bool = True,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        mlp_layers = [torch.nn.Linear(node_dim*2, hidden_dim)]
        if apply_bn:
            mlp_layers.append(torch.nn.BatchNorm1d(hidden_dim))
        if apply_relu:
            mlp_layers.append(torch.nn.ReLU(inplace=True))
        mlp_layers.append(torch.nn.Linear(hidden_dim, 1))

        self.mlp = torch.nn.Sequential(*mlp_layers)

    def forward(self, node_feat: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        row, col = edge_index
        edge_feat = torch.hstack([node_feat[row], node_feat[col]])

        edge_pred_1 = self.mlp(edge_feat).view(-1)
        edge_pred_2 = self.mlp(torch.roll(edge_feat, shifts=16, dims=1)).view(-1)

        return F.sigmoid((edge_pred_1 + edge_pred_2) / 2)

