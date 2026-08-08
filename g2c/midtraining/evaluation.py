"""Per-domain loss evaluation for adaptation-versus-retention experiments."""
from __future__ import annotations

from collections.abc import Mapping

import torch

from g2c.nn import resolve_device  # noqa: F401 - used by scaffold
from g2c.pretraining import (  # noqa: F401 - helpers used by scaffold
    get_lm_batch,
    lm_cross_entropy,
)
from g2c.pretraining.data import SupportsLMBatch

TokenSource = torch.Tensor | SupportsLMBatch


def evaluate_domain_losses(
    model,
    domains: Mapping[str, TokenSource],
    *,
    batch_size: int,
    context_length: int,
    eval_iters: int = 20,
    device: str | torch.device | None = "auto",
    seed: int = 0,
) -> dict[str, float]:
    """Measure next-token loss independently on each named domain.

    The same seeded batch sequence is used for every checkpoint that calls
    this function with the same arguments. That makes base-versus-midtrained
    deltas attributable to weights rather than different random windows.

    Recipe:
        1. Validate non-empty `domains` and positive batch/eval dimensions.
        2. Resolve `device`, move `model` there, and enter `torch.no_grad()`.
        3. For each domain, create a FRESH generator seeded with `seed`. For
           `eval_iters` iterations, sample with `get_lm_batch`, move `x/y` to
           the device, run the model, and append `lm_cross_entropy(...).item()`.
        4. Return `{domain_name: mean_loss}` in input mapping order.

    A fresh generator per domain is intentional: adding another domain must
    not change which windows an existing domain evaluates on.
    """
    # TODO
    raise NotImplementedError
