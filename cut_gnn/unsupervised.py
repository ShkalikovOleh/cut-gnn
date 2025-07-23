from math import exp
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from torch_geometric.data import Batch

from .losses import relaxed_ccl_loss
from .metrics import MultiCutObjectiveRatio
from .models import CutGCN, MLPEdgeClassifier
from .postprocess import Segmenter


class UnsupervisedMultiCut(LightningModule):
    def __init__(
        self,
        gnn_n_layers: int = 12,
        gnn_h_dim: int = 128,
        mlp_h_dim: int = 256,
        apply_batch_norm: bool = True,
        dropout_rate: float = 0,
        apply_relu: bool = True,
        linear_before_mp: bool = True,
        alpha_ccl: float = 2.0,
        beta_cost_1: float = 5.0,
        beta_cost_2: float = 300.0,
        lr: float = 3 * 10**-4,
        max_segm_step: Optional[int] = 1,
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        self.gnn = CutGCN(
            n_layers=gnn_n_layers,
            hidden_dims=gnn_h_dim,
            out_dims=gnn_h_dim,
            apply_batch_norm=apply_batch_norm,
            dropout_rate=dropout_rate,
            apply_relu=apply_relu,
            linear_before_mp=linear_before_mp,
        )
        self.edge_classifier = MLPEdgeClassifier(
            node_dim=gnn_h_dim, hidden_dim=mlp_h_dim
        )
        self.segmenter = Segmenter()

        self.test_obj_ratio_metric = MultiCutObjectiveRatio()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        params = list(self.gnn.parameters()) + list(self.edge_classifier.parameters())
        opt = torch.optim.Adam(
            params, lr=self.hparams.lr, betas=(0.9, 0.999), weight_decay=5 * 10**-4
        )
        return opt

    def forward(self, batch: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        node_emb = self.gnn(
            x=batch.x, edge_index=batch.edge_index, edge_weight=batch.edge_weight
        )
        edge_logits = self.edge_classifier(
            node_feat=node_emb, edge_index=batch.edge_index
        )
        edge_preds = F.sigmoid(edge_logits)
        return self.segmenter(
            batch.edge_index,
            edge_preds.detach(),
            batch.x.size(0),
            self.hparams.max_segm_step,
        )

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        node_emb = self.gnn(
            x=batch.x, edge_index=batch.edge_index, edge_weight=batch.edge_weight
        )
        edge_logits = self.edge_classifier(
            node_feat=node_emb, edge_index=batch.edge_index
        )
        edge_preds = F.sigmoid(edge_logits)
        _, edge_labels = self.segmenter(
            batch.edge_index,
            edge_preds.detach(),
            batch.x.size(0),
            self.hparams.max_segm_step,
        )

        # MultiCut cost Loss
        cost_loss = self.cost_loss(batch, edge_preds)
        self.log(
            "train/cost_loss",
            cost_loss,
            on_step=False,
            on_epoch=True,
            batch_size=batch.num_graphs,
        )

        # CCL Loss
        cycle_loss = relaxed_ccl_loss(edge_preds, edge_labels)
        self.log(
            "train/ccl_loss",
            cycle_loss,
            on_step=False,
            on_epoch=True,
            batch_size=batch.num_graphs,
        )

        # Total Loss
        loss = cost_loss + self.hparams.alpha_ccl * cycle_loss
        self.log(
            "train/total_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch.num_graphs,
        )

        # MultiCut Cost metric
        multicut_cost = batch.edge_weight @ edge_labels
        self.log(
            "train/multicut_cost",
            multicut_cost,
            on_step=False,
            on_epoch=True,
            batch_size=batch.num_graphs,
        )

        return loss

    def cost_loss(self, batch: Batch, edge_pred: torch.Tensor) -> torch.Tensor:
        neg_edges = batch.edge_weight < 0
        cost_lb = torch.sum(batch.edge_weight[neg_edges])
        cost = batch.edge_weight @ edge_pred

        b1, b2 = self.hparams.beta_cost_1, self.hparams.beta_cost_2
        cost_loss = (
            b1 * exp(-self.current_epoch / b2) * (cost - cost_lb) / edge_pred.size(0)
        )

        return cost_loss

    def validation_step(self, batch: Batch, batch_idx: int) -> None:
        node_emb = self.gnn(
            x=batch.x, edge_index=batch.edge_index, edge_weight=batch.edge_weight
        )
        edge_logits = self.edge_classifier(
            node_feat=node_emb, edge_index=batch.edge_index
        )
        edge_preds = F.sigmoid(edge_logits)
        _, edge_labels = self.segmenter(
            batch.edge_index, edge_preds, batch.x.size(0), self.hparams.max_segm_step
        )

        # MultiCut Cost metric
        multicut_cost = batch.edge_weight @ edge_labels
        self.log(
            "val/multicut_cost",
            multicut_cost,
            on_step=False,
            on_epoch=True,
            batch_size=batch.num_graphs,
        )

    def test_step(self, batch: Batch, batch_idx: int) -> None:
        node_emb = self.gnn(
            x=batch.x, edge_index=batch.edge_index, edge_weight=batch.edge_weight
        )
        edge_logits = self.edge_classifier(
            node_feat=node_emb, edge_index=batch.edge_index
        )
        edge_preds = F.sigmoid(edge_logits)
        _, edge_labels = self.segmenter(
            batch.edge_index, edge_preds, batch.x.size(0), self.hparams.max_segm_step
        )

        # MultiCut Cost metric
        multicut_cost = batch.edge_weight @ edge_labels
        self.log(
            "test/multicut_cost",
            multicut_cost,
            on_step=False,
            on_epoch=True,
            batch_size=batch.num_graphs,
        )

        # Rel Cost metric
        self.test_obj_ratio_metric(edge_labels, batch.y, batch.edge_weight)
        self.log(
            "test/obj_ratio",
            self.test_obj_ratio_metric,
            on_step=False,
            on_epoch=True,
            batch_size=batch.num_graphs,
        )
