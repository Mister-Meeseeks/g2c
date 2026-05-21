"""The masked cross-entropy loss for SFT.

The pretraining loss (Module 10's `lm_cross_entropy`) averages CE
uniformly across all `B × T` positions. The SFT loss averages only
across positions where `loss_mask == 1` — the assistant tokens — and
literally ignores everything else.

Mathematically, the SFT loss is:

    L_sft  =  ( Σ_{(b,t) : mask[b,t]=1}  CE(logits[b,t], targets[b,t]) )
              ─────────────────────────────────────────────────────────
                                  Σ mask

The denominator is the *count of training-on positions*, not `B·T`.
Using `B·T` would be a bug: it makes the gradient magnitude shrink as
the prompt gets longer, which is not what we want — the model is
supposed to learn the response equally well regardless of how much
prompt preceded it.

Two implementation paths:

  1. Compute per-position CE with `reduction='none'`-style behavior,
     then mean over masked positions by hand. The path used here.
  2. Replace targets at mask=0 positions with a sentinel `ignore_index`
     and use a CE that supports `ignore_index`. The PyTorch route. We
     use route 1 because Module 03's `CrossEntropyLoss` doesn't take
     an `ignore_index`, and reusing it via reshape + manual masking
     stays consistent with the rest of the course.
"""
from __future__ import annotations

import torch

from g2c.nn import CrossEntropyLoss

_loss_fn = CrossEntropyLoss()


def masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute next-token cross-entropy averaged over masked positions.

    Args:
        logits: (B, T, V). One logit vector per (batch, position).
        targets: (B, T) integer LongTensor. Each `targets[b, t]` is the
            token the model should predict at position `t` of batch
            element `b`.
        mask: (B, T) integer or float tensor. `mask[b, t] != 0` means
            "this position counts toward the loss"; `0` means "ignore."
            Typically a 0/1 LongTensor from `pad_and_collate`.

    Returns:
        Scalar tensor — the masked mean cross-entropy. If the mask is
        all zeros (no positions count), returns a scalar `0.0` rather
        than NaN. (This guards against the corner case of a batch
        whose every example has no assistant tokens, which shouldn't
        happen with a well-built dataset but must not crash the
        trainer.)

    Recipe:
        1. B, T, V = logits.shape
        2. flat_logits  = logits.reshape(B * T, V)             # (B*T, V)
           flat_targets = targets.reshape(B * T)               # (B*T,)
           flat_mask    = mask.reshape(B * T).to(logits.dtype) # (B*T,)

           Cast mask to float here (or to logits.dtype) so the
           multiply-and-sum below stays in floating point. Integer
           mask × float per_pos_loss works in PyTorch but has been a
           silent dtype-promotion source of NaNs in some versions —
           cast explicitly.

        3. Compute per-position cross-entropy. We can't just call
           `CrossEntropyLoss()(flat_logits, flat_targets)` directly
           because it averages — we need per-row values. Two options:

             (a) Inline the formula:
                 m = flat_logits.max(dim=-1, keepdim=True).values
                 lse = m.squeeze(-1) + (flat_logits - m).exp().sum(-1).log()
                 correct = flat_logits.gather(
                     1, flat_targets.unsqueeze(1)
                 ).squeeze(1)
                 per_pos_loss = -correct + lse              # (B*T,)

             (b) Loop over positions calling _loss_fn one row at a time
                 — semantically clean but spectacularly slow.

           Use option (a). It's exactly Module 03's recipe with the
           `.mean()` removed.

        4. masked_total = (per_pos_loss * flat_mask).sum()
           masked_count = flat_mask.sum()

        5. Guard: if masked_count.item() == 0:
               return torch.zeros((), dtype=logits.dtype, device=logits.device)
           else:
               return masked_total / masked_count

    Sanity values:

        * With a uniform mask (every position counted), this returns
          the same value as Module 10's `lm_cross_entropy`.
        * With a mask of all 1s on one position and 0 elsewhere, this
          returns the per-row cross-entropy at that single position.
        * At step 0 with random-init weights and a properly-built
          mask, expect values around `log(V)` — the entropy of a
          uniform distribution over the vocab. Significantly smaller
          values mean the mask is malformed (e.g. masking only
          padding positions, where the random-init model accidentally
          places high probability on the pad id).
    """
    # TODO
    raise NotImplementedError
