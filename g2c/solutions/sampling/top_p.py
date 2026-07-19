# Hand-maintained solutions mirror — canonical, lives in `main` (see AGENTS.md).
"""Solutions for g2c.sampling.top_p pedagogical scaffolds.

Patched onto the scaffold targets by g2c.solutions.apply().
"""
from __future__ import annotations

import torch


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Mask out tail tokens whose probability mass isn't needed to reach `p`.

    Args:
        logits: tensor of shape `(..., V)`. Last dim is the vocabulary;
            leading dims are preserved.
        p: cumulative-probability threshold in `(0, 1]`. `1.0` is a
            no-op (every token kept). Smaller values aggressively prune
            the tail.

    Returns:
        Tensor of the same shape as `logits`, with tail entries
        replaced by `-inf`. The top tokens whose cumulative probability
        reaches `p` keep their original logits.

    Raises:
        ValueError: if `p` is not in `(0, 1]`.

    Recipe:
        1. if not (0 < p <= 1):
               raise ValueError(...)

        2. # Sort logits descending along the last dim. We need both
           # the sorted values AND the original indices, so we can
           # apply the surviving mask back to the un-sorted tensor.
           sorted_logits, sorted_indices = torch.sort(
               logits, dim=-1, descending=True
           )

        3. # Compute the cumulative probability under the descending
           # order. After softmax, sorted_probs starts at the highest
           # prob and decays.
           sorted_probs = torch.softmax(sorted_logits, dim=-1)
           cumulative = sorted_probs.cumsum(dim=-1)

        4. # Mark which sorted positions to DROP. The rule:
           #   drop position i if the prefix [0..i-1] already covers
           #   p of the mass — i.e. cumulative[i-1] >= p.
           # Equivalently in vectorized form:
           sorted_mask = cumulative > p
           # `cumulative > p` is True at the FIRST position that
           # crosses p AND every position thereafter. We want to KEEP
           # that first crossing (the smallest prefix that reaches p),
           # so we shift the mask right by one to spare it:
           sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
           sorted_mask[..., 0] = False
           # After the shift, the argmax is always kept (position 0 is
           # never masked), even if it alone has probability > p.

        5. # `sorted_mask` is in sorted order. Scatter it back to the
           # original (un-sorted) order so we can apply it to `logits`.
           mask = torch.zeros_like(sorted_mask)
           mask.scatter_(-1, sorted_indices, sorted_mask)

        6. return logits.masked_fill(mask, float('-inf'))

    Why softmax inside the warper? Because the cumulative-probability
    test is meaningful only against *probabilities*, not raw logits.
    The temperature-scaled logits coming in still carry the right
    relative ranking, but the absolute mass distribution can only be
    read off the softmax. (We softmax INSIDE the warper but return
    masked LOGITS, not probabilities — the caller will softmax again
    downstream.)
    """
    if not (0 < p <= 1):
        raise ValueError(f'p must be in (0, 1], got {p}')

    sorted_logits, sorted_indices = torch.sort(
        logits, dim=-1, descending=True)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = sorted_probs.cumsum(dim=-1)

    sorted_mask = cumulative > p
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False

    mask = torch.zeros_like(sorted_mask)
    mask.scatter_(-1, sorted_indices, sorted_mask)
    return logits.masked_fill(mask, float('-inf'))
