"""Top-k filtering — keep the K most-likely tokens, drop the rest.

The model's raw distribution at any step has nonzero probability on
*every* token in the vocabulary. With a vocabulary of 8k or 50k, the
long tail — tokens with vanishingly small but nonzero probability —
contributes a noticeable amount of mass that, once in a while, gets
sampled. The result is the "the elephant suddenly walks down the
hallway" failure mode: a low-quality token, unrelated to the prefix,
that the model should never have proposed seriously.

Top-k filtering is the simplest fix:

    Keep the K logits with the highest values.
    Set every other logit to -inf.

After the softmax, the dropped tokens have probability exactly 0 and
the surviving K tokens are renormalized to sum to 1. Sampling can no
longer land on a long-tail token, by construction.

Two things to internalize:

  * **Filter in logit space, not probability space.** Setting dropped
    logits to `-inf` is the cleanest way to express "this token has
    zero mass" — `softmax([..., -inf, ...])` gives `[..., 0, ...]`
    automatically. Doing it in probability space (zeroing out probs and
    renormalizing) gets the same answer but doesn't compose with the
    other warpers, which all work in logit space.

  * **`-inf` is real, not approximate.** Use `float('-inf')` (or
    `torch.tensor(float('-inf'))`). A "very large negative number"
    like `-1e9` works most of the time but introduces small but
    nonzero probability for the dropped tokens — and that small
    nonzero probability, summed across thousands of tokens, becomes a
    real bias.

Scaffolded — about five lines.
"""
from __future__ import annotations

import torch


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Mask out everything except the top-`k` logits per row.

    Args:
        logits: tensor of shape `(..., V)`. Operates on the last dim;
            leading dims are preserved (batch, position, etc.).
        k: number of top logits to keep per row. Must be `>= 1`. Values
            larger than `V` are clamped to `V` (filter is a no-op).

    Returns:
        Tensor of the same shape as `logits`, where every entry NOT in
        the top-`k` of its row is replaced by `-inf`. Top-`k` entries
        keep their original logit values.

    Raises:
        ValueError: if `k < 1`.

    Recipe:
        1. if k < 1:
               raise ValueError(...)
        2. k = min(k, logits.shape[-1])
           # Don't crash if the caller asks for more than V tokens —
           # just keep them all.
        3. # Find the k-th largest value in each row. Anything below
           # this threshold gets masked; anything at or above stays.
           kth_values = torch.topk(logits, k, dim=-1).values[..., -1, None]
           # `kth_values` has shape (..., 1) — the row-wise threshold.
        4. mask = logits < kth_values
        5. return logits.masked_fill(mask, float('-inf'))

    The use of `< kth_values` (strict less-than) means ties at the
    boundary all survive — if positions 5 and 6 have the same logit
    and both are at the k-th rank, both stay. This matches PyTorch's
    `topk` semantics on tied inputs.

    Why not loop over rows and use Python `sorted()`? Because the
    whole point of doing this in PyTorch is that `torch.topk` is one
    op on the GPU/MPS and a Python loop is hundreds of host-device
    round-trips.
    """
    if k < 1:
        raise ValueError(f'k must be >= 1, got {k}')
    k = min(k, logits.shape[-1])
    kth_values = torch.topk(logits, k, dim=-1).values[..., -1, None]
    mask = logits < kth_values
    return logits.masked_fill(mask, float('-inf'))
