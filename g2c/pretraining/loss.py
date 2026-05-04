"""The transformer language-modeling loss.

In Module 03 you built `CrossEntropyLoss` for `(batch, num_classes)` →
`(batch,)`-style classification. The transformer outputs `(B, T, V)`
logits — *every* position is a classification example, with target
the next token. The right loss is "average cross-entropy across all
`B × T` positions."

Mechanically there are two routes:

  1. Reshape `logits` to `(B*T, V)` and `targets` to `(B*T,)`, then
     call your existing per-row `CrossEntropyLoss`. Done.
  2. Compute per-position log-softmax + gather + mean by hand.

Route 1 is one line and is the right answer for this course. The
reason this gets its own module — instead of just inlining the
reshape — is pedagogical: this is THE loss function that pretrains
every modern LM. Naming it makes the recipe legible.

Two confusions worth pre-empting:

  * **Don't average per-sequence first, then across the batch.** The
    standard recipe averages over ALL `B × T` positions uniformly. If
    you average within a sequence first and then across the batch, you
    weight each sequence equally regardless of length — a different
    objective, and a meaningful drift from what you read about online.
    Here all sequences are the same length `T`, so the two definitions
    coincide; in production with packed/variable-length batches they
    don't.

  * **Don't sum.** Both `CrossEntropyLoss` (Module 03) and PyTorch's
    own `nn.functional.cross_entropy` default to `reduction='mean'`.
    Summing instead of averaging makes the effective learning rate
    scale with `B × T`, which couples your hyperparameter sweeps to
    your batch shape in a way that no one wants.

`lm_cross_entropy` is scaffolded — three lines.
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
