# Hand-maintained solutions mirror — canonical, lives in `main` (see AGENTS.md).
"""Solutions for g2c.pretraining.loss pedagogical scaffolds.

Patched onto the scaffold targets by g2c.solutions.apply().
"""
from __future__ import annotations

import torch

from g2c.nn import CrossEntropyLoss

_loss_fn = CrossEntropyLoss()


def lm_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute the next-token cross-entropy loss across a (B, T) batch.

    Args:
        logits: (B, T, V) — one logit vector per (batch, position).
            From `TransformerLM.forward(token_ids)`.
        targets: (B, T) of integer token IDs in `[0, V)`. Each
            `targets[b, t]` is the token the model should predict at
            position `t` of batch element `b` — typically the input
            tokens shifted left by one (see `get_lm_batch`).

    Returns:
        Scalar tensor — mean cross-entropy across all `B × T` positions.

    Recipe:
        1. B, T, V = logits.shape
        2. flat_logits = logits.reshape(B * T, V)         # (B*T, V)
        3. flat_targets = targets.reshape(B * T)          # (B*T,)
        4. return CrossEntropyLoss()(flat_logits, flat_targets)

    A working sanity value: with random-init logits (no training), the
    loss should be approximately `log(V)` — the entropy of a uniform
    distribution over the vocabulary. If you see a much smaller value
    at step 0, double-check the reshape — a shape bug can make
    `flat_targets` line up with the wrong rows of `flat_logits`, which
    will silently train the wrong objective.

    Reuses your `CrossEntropyLoss` from Module 03 rather than calling
    `torch.nn.functional.cross_entropy` — same level of abstraction we
    use everywhere else in this course.
    """
    B, T, V = logits.shape
    flat_logits = logits.reshape(B * T, V)
    flat_targets = targets.reshape(B * T)
    return _loss_fn(flat_logits, flat_targets)
