from typing import Optional

import torch
import torchmetrics

__all__ = ["MultiCutRelativeCost"]


class MultiCutRelativeCost(torchmetrics.Metric):
    is_differentiable: Optional[bool] = False
    higher_is_better: Optional[bool] = True
    full_state_update: bool = False

    def __init__(self) -> None:
        super().__init__()
        self.add_state("cost", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self, pred_edges: torch.Tensor, gt: torch.Tensor, weights: torch.Tensor
    ) -> None:
        pred_cost = weights @ pred_edges
        true_cost = weights @ gt
        rel_cost = torch.clamp_min(pred_cost / true_cost, 0)

        self.cost += rel_cost
        self.num += 1

    def compute(self) -> torch.Tensor:
        return self.cost / self.num
