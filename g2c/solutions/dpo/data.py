# Hand-maintained solutions mirror — canonical, lives in `main` (see AGENTS.md).
"""Solutions for g2c.dpo.data pedagogical scaffolds.

Patched onto the scaffold targets by g2c.solutions.apply().
"""
from __future__ import annotations

from typing import NamedTuple

import torch


def pad_and_collate_pref(
    examples: list[PreferenceExample],
    *,
    max_seq_len: int,
    pad_id: int,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
]:
    """Pad a list of preference examples and assemble two parallel batches.

    Args:
        examples: list of `PreferenceExample`. Length B (the batch size).
        max_seq_len: total ids per padded example BEFORE the
            shift-by-one. Examples whose `prompt + chosen` (or
            `prompt + rejected`) is longer than this are truncated;
            shorter ones are padded with `pad_id`. (After the shift,
            both `x` and `y` have length `max_seq_len - 1`.)
        pad_id: token id to pad short examples with. Padded positions
            also get `loss_mask = 0` so they don't contribute to the
            sequence-level log-probability.

    Returns a six-tuple in this order:
        chosen_x, chosen_y, chosen_mask:        each (B, max_seq_len - 1)
        rejected_x, rejected_y, rejected_mask:  each (B, max_seq_len - 1)

    All `*_x` and `*_y` tensors are `torch.long`. The masks may be
    long or float; the loss function casts as needed.

    Recipe:
        For each example, build two SFT-style records:

            chosen_full   = prompt_ids + chosen_ids
            chosen_full_mask  = [0] * len(prompt_ids) + [1] * len(chosen_ids)

            rejected_full = prompt_ids + rejected_ids
            rejected_full_mask = [0] * len(prompt_ids) + [1] * len(rejected_ids)

        Then truncate each `*_full` (and matching `*_full_mask`) to
        `max_seq_len` tokens (head-truncation — keep the start, drop
        the tail; same policy as Module 13). Pad up to `max_seq_len`
        with `pad_id` for ids and `0` for mask.

        Stack into `(B, max_seq_len)` tensors, then shift:

            chosen_x    = chosen_ids_b[:, :-1]      # (B, T-1)
            chosen_y    = chosen_ids_b[:,  1:]      # (B, T-1)
            chosen_mask = chosen_mask_b[:, 1:]      # (B, T-1)

        and analogously for rejected.

        Return `(chosen_x, chosen_y, chosen_mask,
                 rejected_x, rejected_y, rejected_mask)`.

    Worked example (max_seq_len = 6, pad_id = 0):

        ex = PreferenceExample(
            prompt_ids=[10, 11, 12],   # e.g. "<|user|>\\nHi\\n<|assistant|>\\n" tokens
            chosen_ids=[20, 21],       # the chosen response
            rejected_ids=[30, 31, 32], # the rejected response
        )

        chosen_full       = [10, 11, 12, 20, 21]      mask = [0, 0, 0, 1, 1]
        chosen_full_pad   = [10, 11, 12, 20, 21, 0]   mask = [0, 0, 0, 1, 1, 0]
        chosen_x          = [10, 11, 12, 20, 21]      (5,)
        chosen_y          = [11, 12, 20, 21,  0]      (5,)
        chosen_mask       = [ 0,  0,  1,  1,  0]      (5,)

        rejected_full     = [10, 11, 12, 30, 31, 32]  mask = [0, 0, 0, 1, 1, 1]
        rejected_full_pad = [10, 11, 12, 30, 31, 32]  mask = [0, 0, 0, 1, 1, 1]
        rejected_x        = [10, 11, 12, 30, 31]      (5,)
        rejected_y        = [11, 12, 30, 31, 32]      (5,)
        rejected_mask     = [ 0,  0,  1,  1,  1]      (5,)

    Three correctness pins worth restating:

      * The mask is aligned with `y`, not `ids`. Same shift-by-one as
        Module 13. A mask aligned with `ids` (forgot to shift) trains
        the model to score the prompt token as if it were a response
        token — wrong objective.
      * Padded positions have `mask = 0` in BOTH chosen and rejected.
        Otherwise the implicit-reward signal depends on the pad-id's
        log-probability, which is meaningless.
      * The chosen and rejected batches are independent — their
        truncation and padding happen separately. The chosen sequence
        can have entirely different length than the rejected.
    """
    chosen_ids_rows: list[list[int]] = []
    chosen_mask_rows: list[list[int]] = []
    rejected_ids_rows: list[list[int]] = []
    rejected_mask_rows: list[list[int]] = []

    def fit(ids: list[int], mask: list[int]) -> tuple[list[int], list[int]]:
        ids = ids[:max_seq_len]
        mask = mask[:max_seq_len]
        pad_count = max_seq_len - len(ids)
        if pad_count > 0:
            ids = ids + [pad_id] * pad_count
            mask = mask + [0] * pad_count
        return ids, mask

    for ex in examples:
        chosen_full = ex.prompt_ids + ex.chosen_ids
        chosen_full_mask = [0] * len(ex.prompt_ids) + [1] * len(ex.chosen_ids)
        chosen_ids, chosen_mask = fit(chosen_full, chosen_full_mask)

        rejected_full = ex.prompt_ids + ex.rejected_ids
        rejected_full_mask = [0] * len(ex.prompt_ids) + [1] * len(ex.rejected_ids)
        rejected_ids, rejected_mask = fit(rejected_full, rejected_full_mask)

        chosen_ids_rows.append(chosen_ids)
        chosen_mask_rows.append(chosen_mask)
        rejected_ids_rows.append(rejected_ids)
        rejected_mask_rows.append(rejected_mask)

    chosen_ids_b = torch.tensor(chosen_ids_rows, dtype=torch.long)
    chosen_mask_b = torch.tensor(chosen_mask_rows, dtype=torch.long)
    rejected_ids_b = torch.tensor(rejected_ids_rows, dtype=torch.long)
    rejected_mask_b = torch.tensor(rejected_mask_rows, dtype=torch.long)

    chosen_x = chosen_ids_b[:, :-1]
    chosen_y = chosen_ids_b[:, 1:]
    chosen_mask = chosen_mask_b[:, 1:]

    rejected_x = rejected_ids_b[:, :-1]
    rejected_y = rejected_ids_b[:, 1:]
    rejected_mask = rejected_mask_b[:, 1:]

    return (
        chosen_x,
        chosen_y,
        chosen_mask,
        rejected_x,
        rejected_y,
        rejected_mask,
    )
