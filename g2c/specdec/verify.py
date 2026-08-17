"""Verification — the half of speculative decoding that makes it lossless.

Module 11's loop generates one token per target-model forward pass
because each token depends on the one before it. Speculative decoding
does not break that dependency; it *guesses* through it. A cheap
drafter proposes a block of `k` future tokens, and the expensive target
model checks the whole block in ONE forward pass — teacher forcing
makes logits for every position of the block available at once
(Module 06's parallel-prediction fact, finally cashed in at decode
time).

Verification answers: which prefix of the guess would the target have
produced anyway?

  * `greedy_verify` — the deterministic rule. Accept draft tokens
    while they equal the target's argmax; the target's own argmax at
    the first disagreement (or one past an accepted block) comes free.
    Speculative greedy decoding therefore emits EXACTLY the sequence
    plain greedy decoding would — same tokens, fewer target passes.

  * `speculative_verify` — the stochastic rule (Leviathan et al. 2022;
    Chen et al. 2023). Accept draft token `x` with probability
    `min(1, p_target(x) / p_draft(x))`; on rejection, resample from
    the normalized residual `(p_target − p_draft)⁺`. The emitted
    tokens are then distributed EXACTLY according to the target — the
    drafter's quality changes the speed, never the distribution.

Both functions are pure tensor-in, tensor-out — no model objects. The
`(k+1, V)` logits/probs convention: row `i` is the target's prediction
for draft position `i`; row `k` is the prediction for the token AFTER
the full draft block (the "bonus" position).

Both are scaffolded.
"""
from __future__ import annotations

import torch

# Floor for the draft probability in the acceptance ratio, so a draft
# token the drafter itself assigned ~zero probability to cannot produce
# an infinite ratio.
EPS = 1e-10


def greedy_verify(
    draft_ids: torch.Tensor, target_logits: torch.Tensor
) -> tuple[int, int]:
    """Deterministic verification against the target's argmax.

    Args:
        draft_ids: `(k,)` LongTensor — the drafter's proposed block.
        target_logits: `(k + 1, V)` — the target's logits, one row per
            draft position plus one row past the block. Row `i` is the
            target's prediction for the token at draft position `i`.

    Returns:
        `(n_accepted, next_token)` where `n_accepted ∈ [0, k]` is the
        length of the accepted prefix and `next_token` is the target's
        argmax at the first disagreement — the *correction* — or, if
        the whole block was accepted, at the bonus row. Either way one
        extra token beyond the accepted prefix comes out of the single
        target pass, so an iteration always advances by
        `n_accepted + 1` tokens.

    Recipe:
        1. if draft_ids.ndim != 1 or draft_ids.numel() == 0:
               raise ValueError(...)
           if target_logits.shape[0] != draft_ids.shape[0] + 1:
               raise ValueError(...)   # the (k+1, V) contract
        2. targets = target_logits.argmax(dim=-1)      # (k+1,)
        3. n = 0
           while n < k and targets[n] == draft_ids[n]:
               n += 1
        4. return n, int(targets[n])

    Step 4 is why this rule is lossless for greedy decoding: at every
    position up to and including `n`, the emitted token equals what
    the target's own argmax chain would have produced.
    """
    # TODO
    raise NotImplementedError


def speculative_verify(
    draft_ids: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> tuple[int, int]:
    """Stochastic verification that preserves the target distribution.

    Args:
        draft_ids: `(k,)` LongTensor — tokens the drafter SAMPLED from
            its own distributions.
        target_probs: `(k + 1, V)` — the target's probability rows
            (post-warping, post-softmax), one per draft position plus
            the bonus row.
        draft_probs: `(k, V)` — the drafter's probability rows that
            `draft_ids` were sampled from.
        generator: optional `torch.Generator` for reproducibility.

    Returns:
        `(n_accepted, next_token)` — as in `greedy_verify`, but
        `next_token` is now *sampled*: from the normalized residual
        `(p_target − p_draft)⁺` at the first rejected position, or
        from the target's bonus row if everything was accepted.

    The guarantee (the speculative sampling theorem): if the drafter
    samples from `p_draft` and this rule filters, each emitted token is
    distributed exactly as `p_target` — as if the drafter never
    existed. Acceptance rate is where drafter quality shows up; the
    output distribution is where it can't.

    Recipe:
        1. validate shapes as in greedy_verify (draft_probs is (k, V)).
        2. for i in range(k):
               x = int(draft_ids[i])
               ratio = target_probs[i, x] / draft_probs[i, x].clamp_min(EPS)
               u = torch.rand((), generator=generator)
               if u < min(1.0, ratio): continue      # accepted
               # Rejected: resample from the residual — the part of the
               # target's mass the drafter under-served.
               residual = (target_probs[i] - draft_probs[i]).clamp_min(0.0)
               total = residual.sum()
               if total <= 0:
                   residual, total = target_probs[i], target_probs[i].sum()
               next_token = torch.multinomial(
                   residual / total, 1, generator=generator
               )
               return i, int(next_token)
        3. # Whole block accepted — the bonus row is a free target
           # sample.
           next_token = torch.multinomial(
               target_probs[k], 1, generator=generator
           )
           return k, int(next_token)

    The `total <= 0` fallback fires only when `p_target ≤ p_draft`
    everywhere — i.e. the distributions are identical — in which case
    rejection was a measure-zero event and sampling from the target
    directly is exact.
    """
    # TODO
    raise NotImplementedError
