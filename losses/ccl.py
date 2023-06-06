import torch

import typing

cycles_type = typing.Iterable[typing.Iterable[typing.Iterable[int]]]

def full_ccl_loss(preds: torch.Tensor, edge_index: torch.Tensor,
                cycles: cycles_type, ptr: torch.Tensor) -> torch.Tensor:
    r"""Cycle consistency loss from the paper

    Jung, S., & Keuper, M. (2022). Learning to solve Minimum Cost Multicuts efficiently using
    Edge-Weighted Graph Convolutional Neural Networks.
    """

    cycle_loss = torch.tensor(0., device=edge_index.device)

    for batch_idx, g_cycles in enumerate(cycles):
        shift = torch.argwhere(edge_index[0] >= ptr[batch_idx])[0]

        for cycle in g_cycles:
            idxs = torch.tensor(cycle) + shift

            cutted = preds[idxs]
            uncutted = 1 - cutted

            for i in range(len(cycle)):
                mask = torch.ones(len(cycle), dtype=torch.bool)
                mask[i] = False
                cycle_loss += cutted[i] * torch.cumprod(uncutted[mask][0], dim=-1)

    return cycle_loss

def relaxed_ccl_loss(preds: torch.Tensor, cc_preds: torch.Tensor):
    r"""Cycle consistency loss based on the preprocesed prediction with enforced cycle consistency.
    Instead of computing full CCL returns only the sum of cutted edges which
    violate constraints, e.g. the product term has only 2 values: 0 when there is no
    violation and 1 in the opposite case.
    """

    return torch.sum(preds[cc_preds < preds])
