# Hand-maintained solutions mirror — canonical, lives in `main` (see AGENTS.md).
"""Solutions for g2c.sampling.best_of_n pedagogical scaffolds.

Patched onto the scaffold targets by g2c.solutions.apply().
"""
from __future__ import annotations

import torch

from g2c.sampling.generate import generate


def sequence_log_prob(
    model,
    token_ids: torch.Tensor,
    *,
    prompt_len: int = 0,
) -> float:
    """Score an existing sequence: total log-probability under `model`.

    See the scaffold docstring in g2c/sampling/best_of_n.py for the
    full contract.
    """
    if token_ids.dim() != 1:
        raise ValueError(f'token_ids must be 1-D, got shape {tuple(token_ids.shape)}')
    if token_ids.numel() < 2:
        raise ValueError('token_ids must have at least 2 tokens to score')
    if prompt_len < 0:
        raise ValueError(f'prompt_len must be >= 0, got {prompt_len}')
    if prompt_len >= token_ids.numel():
        raise ValueError(
            f'prompt_len {prompt_len} leaves nothing to score in a '
            f'{token_ids.numel()}-token sequence'
        )

    with torch.no_grad():
        logits = model(token_ids.unsqueeze(0))
    log_probs = torch.log_softmax(logits[0], dim=-1)

    start = max(prompt_len - 1, 0)
    pred = log_probs[start:-1]
    targets = token_ids[start + 1:]
    chosen = pred.gather(1, targets.unsqueeze(1)).squeeze(1)
    return float(chosen.sum())


def best_of_n(
    model,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    *,
    n: int = 8,
    length_normalize: bool = False,
    generator: torch.Generator | None = None,
    **generate_kwargs,
) -> tuple[torch.Tensor, list[tuple[torch.Tensor, float]]]:
    """Sample `n` continuations and return the highest-scoring one.

    See the scaffold docstring in g2c/sampling/best_of_n.py for the
    full contract.
    """
    if n < 1:
        raise ValueError(f'n must be >= 1, got {n}')

    prompt_len = prompt_ids.numel()
    scored: list[tuple[torch.Tensor, float]] = []
    for _ in range(n):
        ids = generate(
            model,
            prompt_ids,
            max_new_tokens,
            generator=generator,
            **generate_kwargs,
        )
        total = sequence_log_prob(model, ids, prompt_len=prompt_len)
        if length_normalize:
            new_tokens = max(ids.numel() - prompt_len, 1)
            total = total / new_tokens
        scored.append((ids, total))

    best_ids = max(scored, key=lambda pair: pair[1])[0]
    return best_ids, scored
