from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from torch_geometric.data import Batch
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAveragePrecision,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)

from .losses import full_ccl_loss, relaxed_ccl_loss
from .metrics import MultiCutObjectiveRatio
from .models import CutGCN, MLPEdgeClassifier
from .postprocess import Segmenter


class SupervisedMultiCut(LightningModule):
    def __init__(
        self,
        gnn_n_layers: int = 12,
        gnn_h_dim: int = 128,
        mlp_h_dim: int = 256,
        apply_batch_norm: bool = True,
        dropout_rate: float = 0,
        apply_relu: bool = True,
        linear_before_mp: bool = True,
        alpha_ccl: float = 10**-3,
        ccl_type: str = "relaxed",
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

        class_metrics = MetricCollection(
            [
                BinaryAccuracy(),
                BinaryPrecision(),
                BinaryRecall(),
                BinaryF1Score(),
                BinaryAveragePrecision(),
            ]
        )
        self.raw_train_class_metrics = class_metrics.clone("train/raw_")
        self.raw_val_class_metrics = class_metrics.clone("val/raw_")
        self.raw_test_class_metrics = class_metrics.clone("test/raw_")
        self.post_train_class_metrics = class_metrics.clone("train/postproc_")
        self.post_val_class_metrics = class_metrics.clone("val/postproc_")
        self.post_test_class_metrics = class_metrics.clone("test/postproc_")

        self.train_obj_ratio_metric = MultiCutObjectiveRatio()
        self.val_obj_ratio_metric = MultiCutObjectiveRatio()
        self.test_obj_ratio_metric = MultiCutObjectiveRatio()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        params = list(self.gnn.parameters()) + list(self.edge_classifier.parameters())
        opt = torch.optim.Adam(
            params, lr=self.hparams.lr, betas=(0.9, 0.999), weight_decay=5 * 10**-4
        )
        return opt

    def forward(self, batch: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        node_emb = self.gnn(
            x=batch.x, edge_index=batch.edge_index, edge_weight=batch.weight
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
            x=batch.x, edge_index=batch.edge_index, edge_weight=batch.weight
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

        # BCE Loss
        bce_loss = F.binary_cross_entropy_with_logits(edge_logits, batch.gt)
        self.log(
            "train/bce_loss",
            bce_loss,
            on_step=False,
            on_epoch=True,
            batch_size=batch.num_graphs,
        )

        # CCL Loss
        if self.hparams.ccl_type == "relaxed":
            cycle_loss = relaxed_ccl_loss(edge_preds, edge_labels)
        else:
            cycle_loss = full_ccl_loss(edge_preds, batch.edge_index, batch.cycles)
        self.log(
            "train/ccl_loss",
            cycle_loss,
            on_step=False,
            on_epoch=True,
            batch_size=batch.num_graphs,
        )

        # Total Loss
        loss = bce_loss + self.hparams.alpha_ccl * cycle_loss
        self.log(
            "train/total_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch.num_graphs,
        )

        # Binary Classification metrics
        self.raw_train_class_metrics(edge_preds, batch.gt.to(torch.long))
        self.log_dict(
            self.raw_train_class_metrics,
            on_step=False,
            on_epoch=True,
            batch_size=batch.num_graphs,
        )
        self.post_train_class_metrics(edge_labels, batch.gt.to(torch.long))
        self.log_dict(
            self.post_train_class_metrics,
            on_step=False,
            on_epoch=True,
            batch_size=batch.num_graphs,
        )

        # Rel Cost metric
        self.train_obj_ratio_metric(edge_labels, batch.gt, batch.weight)
        self.log(
            "train/obj_ratio",
            self.train_obj_ratio_metric,
            on_step=False,
            on_epoch=True,
            batch_size=batch.num_graphs,
        )

        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> None:
        node_emb = self.gnn(
            x=batch.x, edge_index=batch.edge_index, edge_weight=batch.weight
        )
        edge_logits = self.edge_classifier(
            node_feat=node_emb, edge_index=batch.edge_index
        )
        edge_preds = F.sigmoid(edge_logits)
        _, edge_labels = self.segmenter(
            batch.edge_index, edge_preds, batch.x.size(0), self.hparams.max_segm_step
        )

        # Binary Classification metrics
        self.raw_val_class_metrics(edge_preds, batch.gt.to(torch.long))
        self.log_dict(
            self.raw_val_class_metrics,
            on_step=False,
            on_epoch=True,
            batch_size=batch.num_graphs,
        )
        self.post_val_class_metrics(edge_labels, batch.gt.to(torch.long))
        self.log_dict(
            self.post_val_class_metrics,
            on_step=False,
            on_epoch=True,
            batch_size=batch.num_graphs,
        )

        # Rel Cost metric
        self.val_obj_ratio_metric(edge_labels, batch.gt, batch.weight)
        self.log(
            "val/obj_ratio",
            self.val_obj_ratio_metric,
            on_step=False,
            on_epoch=True,
            batch_size=batch.num_graphs,
        )

    def test_step(self, batch: Batch, batch_idx: int) -> None:
        node_emb = self.gnn(
            x=batch.x, edge_index=batch.edge_index, edge_weight=batch.weight
        )
        edge_logits = self.edge_classifier(
            node_feat=node_emb, edge_index=batch.edge_index
        )
        edge_preds = F.sigmoid(edge_logits)
        _, edge_labels = self.segmenter(
            batch.edge_index, edge_preds, batch.x.size(0), self.hparams.max_segm_step
        )

        # Binary Classification metrics
        self.raw_test_class_metrics(edge_preds, batch.gt.to(torch.long))
        self.log_dict(
            self.raw_test_class_metrics,
            on_step=False,
            on_epoch=True,
            batch_size=batch.num_graphs,
        )
        self.post_test_class_metrics(edge_labels, batch.gt.to(torch.long))
        self.log_dict(
            self.post_test_class_metrics,
            on_step=False,
            on_epoch=True,
            batch_size=batch.num_graphs,
        )

        # Rel Cost metric
        self.test_obj_ratio_metric(edge_labels, batch.gt, batch.weight)
        self.log(
            "test/obj_ratio",
            self.test_obj_ratio_metric,
            on_step=False,
            on_epoch=True,
            batch_size=batch.num_graphs,
        )
