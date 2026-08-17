# Hand-maintained solutions mirror — canonical, lives in `main` (see AGENTS.md).
"""Solutions for g2c.specdec.verify pedagogical scaffolds.

Patched onto the scaffold targets by g2c.solutions.apply().
"""
from __future__ import annotations

import torch

from g2c.specdec.verify import EPS


def greedy_verify(
    draft_ids: torch.Tensor, target_logits: torch.Tensor
) -> tuple[int, int]:
    if draft_ids.ndim != 1 or draft_ids.numel() == 0:
        raise ValueError(
            f"draft_ids must be a non-empty 1-D tensor; got shape "
            f"{tuple(draft_ids.shape)}"
        )
    k = draft_ids.shape[0]
    if target_logits.shape[0] != k + 1:
        raise ValueError(
            f"target_logits must have k + 1 = {k + 1} rows; got "
            f"{target_logits.shape[0]}"
        )
    targets = target_logits.argmax(dim=-1)
    n = 0
    while n < k and int(targets[n]) == int(draft_ids[n]):
        n += 1
    return n, int(targets[n])


def speculative_verify(
    draft_ids: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> tuple[int, int]:
    if draft_ids.ndim != 1 or draft_ids.numel() == 0:
        raise ValueError(
            f"draft_ids must be a non-empty 1-D tensor; got shape "
            f"{tuple(draft_ids.shape)}"
        )
    k = draft_ids.shape[0]
    if target_probs.shape[0] != k + 1:
        raise ValueError(
            f"target_probs must have k + 1 = {k + 1} rows; got "
            f"{target_probs.shape[0]}"
        )
    if draft_probs.shape[0] != k:
        raise ValueError(
            f"draft_probs must have k = {k} rows; got "
            f"{draft_probs.shape[0]}"
        )
    for i in range(k):
        x = int(draft_ids[i])
        ratio = target_probs[i, x] / draft_probs[i, x].clamp_min(EPS)
        u = torch.rand((), generator=generator)
        if u < min(1.0, float(ratio)):
            continue
        residual = (target_probs[i] - draft_probs[i]).clamp_min(0.0)
        total = residual.sum()
        if total <= 0:
            residual, total = target_probs[i], target_probs[i].sum()
        next_token = torch.multinomial(
            residual / total, 1, generator=generator
        )
        return i, int(next_token)
    next_token = torch.multinomial(target_probs[k], 1, generator=generator)
    return k, int(next_token)
