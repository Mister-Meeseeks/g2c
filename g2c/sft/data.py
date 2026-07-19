"""SFT example container and the (pad + shift + mask) batch collator.

Pretraining (Module 10) sampled random fixed-length windows from one
long token stream — every batch was the same shape and every position
was a training target. SFT examples are different in two ways:

  1. Each example is a complete conversation with hard boundaries; we
     don't sample windows out of it.
  2. Examples are variable-length; a batch must pad to the longest
     example.

`pad_and_collate` is the analogue of `get_lm_batch` for SFT — it takes
a list of ready-to-use `SFTExample`s and produces a `(x, y, loss_mask)`
triple. The shift-by-one is the same as in pretraining; the new piece
is the loss mask, which travels with the targets.

`SFTExample` is a NamedTuple — boilerplate, no implementation. The
fields are simple lists of ints (not tensors) because raw Python lists
compose cleanly with the JSON-style datasets you'll author by hand;
the collator handles the conversion to tensors at batch time.
"""
from __future__ import annotations

from typing import NamedTuple

import torch


class SFTExample(NamedTuple):
    """One tokenized SFT example.

    Attributes:
        ids:  full token sequence including role markers and the
              trailing `<|end|>` of the last assistant turn. Length T.
        mask: same length as `ids`. `mask[t] == 1` iff `ids[t]` is part
              of an assistant turn's CONTENT (including the trailing
              `<|end|>`); `0` for every prompt token (user, role
              markers, the assistant role marker itself).

    The mask here is aligned with `ids`. The collator shifts it during
    batching so the resulting `loss_mask` aligns with the *targets*
    (`y = ids[1:]`).
    """

    ids: list[int]
    mask: list[int]


def pad_and_collate(
    examples: list[SFTExample],
    *,
    max_seq_len: int,
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad a list of SFT examples and assemble a (B, T-1) batch.

    Args:
        examples: list of `SFTExample`. Length B (the batch size).
        max_seq_len: total ids per padded example BEFORE the
            shift-by-one. Examples longer than this are truncated to
            this many tokens; shorter examples are padded with
            `pad_id`. (After the shift, both `x` and `y` have length
            `max_seq_len - 1`.)
        pad_id: token id to pad short examples with. Padded positions
            also get `loss_mask = 0` so they don't contribute to the
            loss.

    Returns:
        x:         LongTensor (B, max_seq_len - 1). Input contexts.
        y:         LongTensor (B, max_seq_len - 1). Targets — `x` shifted
                   left by one, padded as needed.
        loss_mask: LongTensor (B, max_seq_len - 1). `1` exactly at
                   positions where the loss should fire; `0` elsewhere.

    Why is shape `(B, max_seq_len - 1)` rather than `(B, max_seq_len)`?
    Because the LM training contract is "predict position t+1 from the
    prefix ending at position t" — so we form `(x, y) = (ids[:-1],
    ids[1:])` per example. The collator does this shift internally.

    Recipe:
        1. Truncate every example's `ids` and `mask` to at most
           `max_seq_len` tokens (head-truncation, drop excess from the
           end). If you want to keep the assistant turn at the cost of
           the user turn, that's a more sophisticated truncation
           policy — for now, simple end-truncation is fine.

        2. Pad every (truncated) `ids` to length exactly `max_seq_len`
           by appending `pad_id` as needed. Pad every `mask` to the
           same length by appending `0`.

        3. Stack into tensors:
              ids_b   = torch.tensor(...)              # (B, max_seq_len)
              mask_b  = torch.tensor(...)              # (B, max_seq_len)

        4. Shift to (input, target) form:
              x         = ids_b[:, :-1]                # (B, max_seq_len-1)
              y         = ids_b[:, 1:]                 # (B, max_seq_len-1)
              loss_mask = mask_b[:, 1:]                # (B, max_seq_len-1)

           IMPORTANT: the loss_mask is `mask_b` shifted by ONE — it
           must align with `y`, not with `ids_b`. The element
           `loss_mask[b, t]` says "should the model be supervised at
           position t of x[b], where the target is y[b, t]". That's an
           assistant-content question, which is what the un-shifted
           mask answered for `ids[b, t+1]` — which is exactly
           `y[b, t]`.

        5. Return (x, y, loss_mask). All three are LongTensors. The
           loss function casts `loss_mask` to float as needed.

    Example (T = max_seq_len = 6):

        examples = [
            SFTExample(ids=[10, 11, 12, 13, 14], mask=[0, 0, 1, 1, 1]),
        ]
        pad_id   = 0

        truncate / pad ids:   [10, 11, 12, 13, 14, 0]    (6,)
        truncate / pad mask:  [ 0,  0,  1,  1,  1, 0]    (6,)

        x         = [10, 11, 12, 13, 14]           (1, 5)
        y         = [11, 12, 13, 14,  0]           (1, 5)
        loss_mask = [ 0,  1,  1,  1,  0]           (1, 5)
                                       ▲
                       0 because the target at this position is the
                       padding token — we never want loss on padding.
    """
    # TODO
    raise NotImplementedError

