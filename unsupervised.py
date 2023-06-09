import torch
from torch_geometric.data import Batch
from lightning.pytorch import LightningModule
from torchmetrics import MetricCollection
from torchmetrics.classification import (BinaryAccuracy, BinaryPrecision,
                                        BinaryRecall, BinaryAveragePrecision)

from math import exp
from typing import Tuple, Optional

from models import CutGCN, MLPEdgeClassifier
from losses import full_ccl_loss, relaxed_ccl_loss
from postprocess import Segmenter
from metrics import MultiCutRelativeCost


class UnsupervisedMultiCut(LightningModule):

    def __init__(self, gnn_h_dim: int = 128, mlp_h_dim: int = 256,
                 alpha_ccl: float = 2., model: str = 'CutGCN',
                 beta_cost_1: float = 5., beta_cost_2: float = 300.,
                 lr: float = 3*10**-4, max_segm_step: Optional[int] = 1) -> None:
        super().__init__()

        self.save_hyperparameters()

        self.gnn = CutGCN(hidden_dims=gnn_h_dim, out_dims=gnn_h_dim)
        self.edge_classifier = MLPEdgeClassifier(node_dim=gnn_h_dim, hidden_dim=mlp_h_dim)

        self.segmenter = Segmenter()

        self.train_rel_cost_metric = MultiCutRelativeCost()
        self.val_rel_cost_metric = MultiCutRelativeCost()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        params = list(self.gnn.parameters()) + list(self.edge_classifier.parameters())
        opt = torch.optim.Adam(params, lr=self.hparams.lr,
                               betas=(0.9, 0.999), weight_decay=5*10**-4)
        return opt

    def forward(self, batch: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        node_emb = self.gnn(x=batch.x, edge_index=batch.edge_index, edge_weight=batch.weight)
        edge_pred = self.edge_classifier(node_feat=node_emb, edge_index=batch.edge_index)
        return self.segmenter(batch.edge_index, edge_pred, batch.ptr[-1], self.hparams.max_segm_step)

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        node_emb = self.gnn(x=batch.x, edge_index=batch.edge_index, edge_weight=batch.weight)
        edge_pred = self.edge_classifier(node_feat=node_emb, edge_index=batch.edge_index)

        node_clusters, edge_labels = self.segmenter(batch.edge_index, edge_pred, batch.ptr[-1], self.hparams.max_segm_step)

        batch_size = batch.ptr.shape[0] - 1

        # MultiCut cost Loss
        b1, b2 = self.hparams.beta_cost_1, self.hparams.beta_cost_2
        neg_edges = batch.weight < 1
        cost_lb = torch.sum(batch.weight[neg_edges])
        cost = batch.weight @ edge_pred

        cost_loss = b1 * exp(-self.current_epoch / b2) * (cost - cost_lb) / len(edge_pred)
        self.log('train_cost_loss', cost_loss, on_step=False, on_epoch=True, batch_size=batch_size)

        # CCL Loss
        cycle_loss = relaxed_ccl_loss(edge_pred, edge_labels)
        self.log('train_ccl_loss', cycle_loss, on_step=False, on_epoch=True, batch_size=batch_size)

        # Total Loss
        loss = cost_loss + self.hparams.alpha_ccl * cycle_loss
        self.log('train_total_loss', loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size)

        # MultiCut Cost metric
        self.log('train_multicut_cost', cost, on_step=False, on_epoch=True, batch_size=batch_size)

        # Rel Cost metric
        self.train_rel_cost_metric(edge_labels, batch.gt, batch.weight)
        self.log('train_rel_cost', self.train_rel_cost_metric, on_step=False, on_epoch=True, batch_size=batch_size)

        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> None:
        node_emb = self.gnn(x=batch.x, edge_index=batch.edge_index, edge_weight=batch.weight)
        edge_pred = self.edge_classifier(node_feat=node_emb, edge_index=batch.edge_index)

        _, edge_labels = self.segmenter(batch.edge_index, edge_pred, batch.ptr[-1], self.hparams.max_segm_step)

        batch_size = batch.ptr.shape[0] - 1

        cost = batch.weight @ edge_pred

        # MultiCut Cost metric
        self.log('val_multicut_cost', cost, on_step=False, on_epoch=True, batch_size=batch_size)

        # Rel Cost metric
        self.val_rel_cost_metric(edge_labels, batch.gt, batch.weight)
        self.log('val_rel_cost', self.val_rel_cost_metric, on_step=False, on_epoch=True, batch_size=batch_size)

