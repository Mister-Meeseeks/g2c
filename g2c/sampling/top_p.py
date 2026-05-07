"""Top-p (nucleus) filtering — keep the smallest set of tokens whose
combined probability reaches `p`.

Top-k's flaw: a fixed `k` is wrong for both ends of the distribution.
When the model is *confident* (one token has 95% of the mass), top-50
keeps 49 long-tail distractors. When the model is *uncertain* (a flat
distribution over a few hundred reasonable tokens), top-50 cuts off
plausible alternatives. The cutoff doesn't track the model's
confidence.

Top-p (Holtzman et al., "The Curious Case of Neural Text
Degeneration", 2020) replaces a fixed-count cutoff with a fixed-mass
cutoff:

    Sort tokens by descending probability.
    Cumulatively sum the probabilities.
    Keep the smallest prefix whose cumulative sum is >= p.
    Drop everything else.

When the model is confident, the prefix is one or two tokens — the
nucleus narrows. When the model is uncertain, the prefix expands to
include more candidates. The size of the surviving set adapts to
the entropy of the distribution.

Two design choices to internalize:

  * **The "first token to cross `p`" is INCLUDED.** A common
    off-by-one writes the rule as "drop everything once cumulative
    mass exceeds `p`" — but that can drop the top token outright when
    the very first token already has probability `> p`. The correct
    rule is: drop everything *after* the smallest prefix that reaches
    `p`. The prefix itself stays. The argmax is always kept,
    regardless of `p`.

  * **Mask in logit space.** Same as top-k: set the masked positions
    to `-inf` so they cleanly fall out of any subsequent softmax. The
    function returns transformed logits, not normalized probabilities.

Scaffolded.
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
