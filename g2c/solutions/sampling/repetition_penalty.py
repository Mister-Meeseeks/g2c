# Hand-maintained solutions mirror — canonical, lives in `main` (see AGENTS.md).
"""Solutions for g2c.sampling.repetition_penalty pedagogical scaffolds.

Patched onto the scaffold targets by g2c.solutions.apply().
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
