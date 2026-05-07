"""Repetition penalty — discourage tokens that already appeared.

Small language models loop. They emit "the the the the..." or "I
think I think I think...", or get into longer cycles where a fixed
phrase reappears every few sentences. The cause is structural: at
every step, the most-likely token is some function of the recent
context, and once the model is in a state where token `X` is the
most-likely continuation, emitting `X` doesn't change the state much,
so `X` stays most-likely.

Repetition penalty (Keskar et al., "CTRL", 2019) is the standard
defense:

    For every token id that appears in the prior context,
    rescale its logit BEFORE softmax — pushing its probability down.

The asymmetric formula handles the sign of the logit so the penalty
always *decreases* probability:

    if logit > 0:    new_logit = logit / penalty
    if logit < 0:    new_logit = logit * penalty

With `penalty = 1.0` both branches are identity → no penalty. With
`penalty > 1`, positive logits shrink toward zero and negative logits
grow more negative — both make the token less likely. With
`penalty < 1` the penalty becomes a *bonus* (rarely useful, but the
formula is symmetric).

Two design choices to internalize:

  * **Penalty applies to TOKEN IDs, not positions.** If token 7 has
    appeared at any earlier position, its logit gets penalized — even
    if it appeared only once at the start of a long context. Some
    variants weight by recency or count; the original CTRL formula
    does neither. We follow CTRL.

  * **The penalty is stateless.** It takes the prior token IDs as an
    input and returns the warped logits. There's no internal state to
    track across calls. In `generate`, the prior IDs are the prompt
    plus everything sampled so far; the warper just sees the tensor.

Scaffolded.
"""
from __future__ import annotations

import torch


def apply_repetition_penalty(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    """Penalize logits at token IDs that appear in `token_ids`.

    Args:
        logits: tensor of shape `(B, V)` (or any `(..., V)` shape;
            but in this course's `generate`, it's `(B, V)`).
        token_ids: integer tensor of prior tokens. Either:
              - shape `(B, T)` — per-row history, one row per logits
                row. The standard case in batched generation.
              - shape `(T,)` — a shared history broadcast across all
                rows. Convenient when `B = 1`.
            Tokens are deduplicated per row before applying the penalty
            (an id that appears 5 times is penalized exactly the same
            as one that appears once — this is CTRL's convention).
        penalty: the rescaling factor. `1.0` is no-op. Typical values
            are `1.05` to `1.5` — larger values risk killing the
            token's probability entirely.

    Returns:
        Tensor of shape matching `logits`, with the relevant entries
        rescaled. Other entries are unchanged.

    Raises:
        ValueError: if `penalty <= 0`.

    Recipe:
        1. if penalty <= 0:
               raise ValueError(...)
        2. if penalty == 1.0:
               return logits     # exact identity, no allocation
        3. # Broadcast token_ids to (B, T) if it came in as (T,).
           if token_ids.dim() == 1:
               token_ids = token_ids.unsqueeze(0).expand(logits.shape[0], -1)
        4. # Gather the logits at the prior-token positions:
           prev_logits = logits.gather(-1, token_ids)         # (B, T)
        5. # Compute the asymmetric penalty:
           penalized = torch.where(
               prev_logits > 0,
               prev_logits / penalty,
               prev_logits * penalty,
           )
        6. # Scatter back into a copy of `logits`:
           out = logits.clone()
           out.scatter_(-1, token_ids, penalized)
           return out

    Notes on the `scatter_` step:
        If a row in `token_ids` contains the SAME token id twice (the
        token was repeated in the prior context), `scatter_` will
        write twice at the same logit position — the second write
        wins. Either order produces the same value (`penalized` was
        computed via the same formula), so duplicates within a row
        give the same answer as deduplicating first. This matches
        CTRL's "penalize each id once" convention as a side effect of
        scatter's last-write-wins semantics.
    """
    if penalty <= 0:
        raise ValueError(f'Penalty must be > 0, got {penalty}')
    if penalty == 1.0:
        return logits
    
    if token_ids.dim() == 1:
        token_ids = token_ids.unsqueeze(0).expand(logits.shape[0], -1)
    
    prev_logits = logits.gather(-1, token_ids)
    
    penalized = torch.where(
        prev_logits > 0,
        prev_logits / penalty,
        prev_logits * penalty,
    )
    
    out = logits.clone()
    out.scatter_(-1, token_ids, penalized)
    return out
