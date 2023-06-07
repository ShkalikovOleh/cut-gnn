import torch

from typing import Iterable, Mapping

def full_ccl_loss(preds: torch.Tensor, edge_index: torch.Tensor,
                cycles: Iterable[Iterable[Mapping[int, torch.Tensor]]],
                ptr: torch.Tensor) -> torch.Tensor:
    r"""Cycle consistency loss from the paper

    Jung, S., & Keuper, M. (2022). Learning to solve Minimum Cost Multicuts efficiently using
    Edge-Weighted Graph Convolutional Neural Networks.
    """

    cycle_loss = torch.tensor(0., device=edge_index.device)

    for batch_idx, g_cycles in enumerate(cycles):
        shift = torch.argwhere(edge_index[0] >= ptr[batch_idx])[0]

        for l, cycle in g_cycles[0].items():
            idxs = cycle + shift

            cutted = preds[idxs]
            uncutted = 1 - cutted

            for i in range(l):
                y = cutted[:, i]
                cycle_loss += torch.sum(y * torch.prod(uncutted[:, 1:], dim=1))
                uncutted = torch.roll(uncutted, -1, 1)

    return cycle_loss

def relaxed_ccl_loss(preds: torch.Tensor, cc_preds: torch.Tensor):
    r"""Cycle consistency loss based on the preprocesed prediction with enforced cycle consistency.
    Instead of computing full CCL returns only the sum of cutted edges which
    violate constraints, e.g. the product term has only 2 values: 0 when there is no
    violation and 1 in the opposite case.
    """

    return torch.sum(preds[cc_preds < preds])
