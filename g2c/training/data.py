"""Batch sampling for transformer language-model pretraining.

Module 06 trained one prediction per window: a context of length `N`
predicted ONE next token. That worked when the model could only look
back `N` tokens — a bigram model, an MLP language model — but it
wasted compute. The transformer reads a (B, T) block and emits a
prediction at EVERY position simultaneously, so we want batches that
exercise that. The right batch shape is `(B, T)` for inputs AND
`(B, T)` for targets, with the targets being the inputs shifted left
by one.

    ids = [4, 7, 1, 3, 9, 2, 6, 5, 0, ...]

    Window starting at index 2, length T=4:
        x:  [1, 3, 9, 2]      tokens 2, 3, 4, 5
        y:  [3, 9, 2, 6]      tokens 3, 4, 5, 6 — each is the NEXT
                              token of the corresponding `x` position

    For position t in the window:
        x[t] is the model's input at position t
        y[t] is what the model should predict from x[:t+1]

    The cross-entropy loss is then "average CE across all (B*T)
    (input prefix, target token) pairs."

`get_lm_batch` is implemented for you — it's the natural extension of
`g2c.lm.get_batch` from Module 06 to multi-position targets. The shift-
by-one is the conceptual core; everything else is index arithmetic.
"""
from __future__ import annotations

import torch


def get_lm_batch(
    ids: torch.Tensor,
    batch_size: int,
    context_length: int,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch of (input, target) windows for next-token training.

    Args:
        ids: 1-D LongTensor of token IDs — the corpus.
        batch_size: number of windows to sample (`B`).
        context_length: tokens per window (`T`).
        generator: optional `torch.Generator` for reproducibility.

    Returns:
        x: (batch_size, context_length) — the input contexts.
        y: (batch_size, context_length) — `x` shifted left by one. For
            position `t` of window `b`, `y[b, t]` is the token IMMEDIATELY
            FOLLOWING `x[b, t]` in the corpus.

    The model's job: at every position `t` in `x`, predict `y[t]` from
    the prefix `x[:t+1]`. With a causal transformer this is computed in
    a single forward pass — every position's prediction in parallel.

    Implementation:
        n = ids.shape[0]
        # We need T+1 contiguous tokens per window: T for x, plus one
        # extra for the trailing target. So starts must satisfy
        # start + T < n, i.e. start in [0, n - T - 1].
        starts = torch.randint(0, n - T, (B,), generator=generator)
        x = stack of ids[s : s + T]
        y = stack of ids[s + 1 : s + T + 1]
    """
    n = ids.shape[0]
    if n <= context_length:
        raise ValueError(
            f"ids has length {n}; need at least context_length+1 = "
            f"{context_length + 1} for a single (x, y) pair."
        )
    starts = torch.randint(
        0, n - context_length, (batch_size,), generator=generator
    )
    x = torch.stack([ids[s : s + context_length] for s in starts])
    y = torch.stack([ids[s + 1 : s + context_length + 1] for s in starts])
    return x, y
