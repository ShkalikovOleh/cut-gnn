from typing import Sequence

import torch

__all__ = ["full_ccl_loss", "relaxed_ccl_loss"]


def full_ccl_loss(
    preds: torch.Tensor,
    edge_index: torch.Tensor,
    cycles: Sequence[Sequence[int]],
    threshold: float = 0.5,
) -> torch.Tensor:
    r"""Cycle consistency loss from the paper (normalized by number of cycles)

    Jung, S., & Keuper, M. (2022). Learning to solve Minimum Cost Multicuts efficiently using
    Edge-Weighted Graph Convolutional Neural Networks.
    """

    cycle_loss = 0.0

    for cycle in cycles:
        cut = preds[cycle]
        total_log_sum = torch.sum(torch.log(1 - cut))

        L = len(cycle)
        for i in range(L):
            cycle_loss += torch.sum(
                (cut > threshold) * cut * torch.exp(total_log_sum - (1 - cut))
            )

    return cycle_loss / len(cycles)


def relaxed_ccl_loss(preds: torch.Tensor, cc_preds: torch.Tensor):
    r"""Cycle consistency loss based on the postprocesed prediction with enforced cycle consistency.
    Instead of computing full CCL it returns only the sum of cutted edges which
    violate constraints, i.e. the product term from the full loss has only 2 values:
    0 when cut is feasible and 1 in the opposite case.
    """
    return torch.mean(preds[cc_preds < preds])
