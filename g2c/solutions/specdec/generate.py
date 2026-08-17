# Hand-maintained solutions mirror — canonical, lives in `main` (see AGENTS.md).
"""Solutions for g2c.specdec.generate pedagogical scaffolds.

Patched onto the scaffold targets by g2c.solutions.apply().
"""
from __future__ import annotations

import torch

from g2c.specdec.generate import SpecStats, draft_greedy
from g2c.specdec.verify import greedy_verify


@torch.no_grad()
def speculative_generate(
    target,
    drafter,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    *,
    k: int = 4,
    eos_id: int | None = None,
) -> tuple[torch.Tensor, SpecStats]:
    if prompt_ids.dim() != 1 or prompt_ids.numel() == 0:
        raise ValueError(
            f"prompt_ids must be a non-empty 1-D tensor; got shape "
            f"{tuple(prompt_ids.shape)}"
        )
    if k < 1:
        raise ValueError(f"k must be at least 1; got {k}")

    full_ids = prompt_ids.detach().cpu().clone()
    stats = SpecStats()
    device = getattr(target, "device", torch.device("cpu"))

    while stats.generated < max_new_tokens:
        k_step = min(k, max_new_tokens - stats.generated)

        draft = draft_greedy(drafter, full_ids, k_step)

        ctx = full_ids[-(target.max_seq_len - k_step):]
        seq = torch.cat([ctx, draft]).to(device).unsqueeze(0)
        logits = target(seq)
        block = logits[0, -(k_step + 1):, :].cpu()

        n_acc, nxt = greedy_verify(draft, block)
        new_ids = torch.cat([draft[:n_acc], torch.tensor([nxt])])
        new_ids = new_ids[: max_new_tokens - stats.generated]

        if eos_id is not None and (new_ids == eos_id).any():
            cut = int((new_ids == eos_id).nonzero()[0]) + 1
            new_ids = new_ids[:cut]

        stats.record(
            drafted=k_step, accepted=n_acc, generated=int(new_ids.numel())
        )
        full_ids = torch.cat([full_ids, new_ids])

        if eos_id is not None and full_ids[-1].item() == eos_id:
            break

    return full_ids, stats
