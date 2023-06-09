import torch
from torch_geometric.data import Batch
from lightning.pytorch import LightningModule
from torchmetrics import MetricCollection
from torchmetrics.classification import (BinaryAccuracy, BinaryPrecision,
                                        BinaryRecall, BinaryAveragePrecision)

from typing import Tuple, Optional

from models import CutGCN, MLPEdgeClassifier
from losses import full_ccl_loss, relaxed_ccl_loss
from postprocess import Segmenter
from metrics import MultiCutRelativeCost


class SupervisedMultiCut(LightningModule):

    def __init__(self, gnn_h_dim: int = 128, mlp_h_dim: int = 256,
                 alpha_ccl: float = 10**-3, model: str = 'CutGCN',
                 ccl_type: str = 'relaxed', lr: float = 3*10**-4,
                 max_segm_step: Optional[int] = 1) -> None:
        super().__init__()

        self.save_hyperparameters()

        self.gnn = CutGCN(hidden_dims=gnn_h_dim, out_dims=gnn_h_dim)
        self.edge_classifier = MLPEdgeClassifier(node_dim=gnn_h_dim, hidden_dim=mlp_h_dim)

        self.segmenter = Segmenter()

        class_metrics = MetricCollection([BinaryAccuracy(), BinaryPrecision(),
                                    BinaryRecall(), BinaryAveragePrecision()])
        self.train_class_metrics = class_metrics.clone('train_')
        self.val_class_metrics = class_metrics.clone('val_')
        self.test_class_metrics = class_metrics.clone('test_')

        self.train_rel_cost_metric = MultiCutRelativeCost()
        self.val_rel_cost_metric = MultiCutRelativeCost()
        self.test_rel_cost_metric = MultiCutRelativeCost()

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

        # BCE Loss
        bce_loss = torch.nn.functional.binary_cross_entropy(edge_pred, batch.gt)
        self.log('train_bce_loss', bce_loss, on_step=False, on_epoch=True, batch_size=batch_size)

        # CCL Loss
        if self.hparams.ccl_type == 'relaxed':
            cycle_loss = relaxed_ccl_loss(edge_pred, edge_labels)
        else:
            cycle_loss = full_ccl_loss(edge_pred, batch.edge_index, batch.cycles, batch.ptr)
        self.log('train_ccl_loss', cycle_loss, on_step=False, on_epoch=True, batch_size=batch_size)

        # Total Loss
        loss = bce_loss + self.hparams.alpha_ccl * cycle_loss
        self.log('train_total_loss', loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size)

        # Binary Classification metrics
        self.train_class_metrics(edge_pred, batch.gt.to(torch.long))
        self.log_dict(self.train_class_metrics, on_step=False, on_epoch=True, batch_size=batch_size)

        # Rel Cost metric
        self.train_rel_cost_metric(edge_labels, batch.gt, batch.weight)
        self.log('train_rel_cost', self.train_rel_cost_metric, on_step=False, on_epoch=True, batch_size=batch_size)

        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> None:
        node_emb = self.gnn(x=batch.x, edge_index=batch.edge_index, edge_weight=batch.weight)
        edge_pred = self.edge_classifier(node_feat=node_emb, edge_index=batch.edge_index)

        _, edge_labels = self.segmenter(batch.edge_index, edge_pred, batch.ptr[-1], self.hparams.max_segm_step)

        batch_size = batch.ptr.shape[0] - 1

        # Binary Classification metrics
        self.val_class_metrics(edge_pred, batch.gt.to(torch.long))
        self.log_dict(self.val_class_metrics, on_step=False, on_epoch=True, batch_size=batch_size)

        # Rel Cost metric
        self.val_rel_cost_metric(edge_labels, batch.gt, batch.weight)
        self.log('val_rel_cost', self.val_rel_cost_metric, on_step=False, on_epoch=True, batch_size=batch_size)





