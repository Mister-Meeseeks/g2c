# Hand-maintained solutions mirror — canonical, lives in `main` (see AGENTS.md).
"""Solutions for g2c.rl.grpo pedagogical scaffolds.

Patched onto the scaffold targets by g2c.solutions.apply().
"""
from __future__ import annotations

import torch

from g2c.rl.grpo import DEGENERATE_STD


def group_advantages(rewards: torch.Tensor) -> torch.Tensor:
    mean = rewards.mean()
    std = rewards.std(unbiased=False)
    if std < DEGENERATE_STD:
        return torch.zeros_like(rewards)
    return (rewards - mean) / std


def completion_log_prob(
    model,
    ids: torch.Tensor,
    prompt_len: int,
) -> torch.Tensor:
    _, T = ids.shape
    if not 1 <= prompt_len < T:
        raise ValueError(
            f"prompt_len must be in [1, T={T}), got {prompt_len}"
        )
    logits = model(ids[:, :-1])
    logprobs = torch.log_softmax(logits, dim=-1)
    targets = ids[:, 1:]
    token_lp = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    mask = torch.zeros_like(token_lp)
    mask[:, prompt_len - 1 :] = 1.0
    return (token_lp * mask).sum(dim=-1)


def grpo_loss(
    logp: torch.Tensor,
    ref_logp: torch.Tensor,
    advantages: torch.Tensor,
    kl_coef: float,
) -> torch.Tensor:
    pg = -(advantages.detach() * logp).mean()
    d = ref_logp.detach() - logp
    kl = (d.exp() - d - 1.0).mean()
    return pg + kl_coef * kl
