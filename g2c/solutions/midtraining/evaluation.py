# Hand-maintained solutions mirror — canonical, lives in `main` (see AGENTS.md).
from __future__ import annotations

import torch

from g2c.nn import resolve_device
from g2c.pretraining import get_lm_batch, lm_cross_entropy


def evaluate_domain_losses(
    model,
    domains,
    *,
    batch_size: int,
    context_length: int,
    eval_iters: int = 20,
    device="auto",
    seed: int = 0,
):
    if not domains:
        raise ValueError("domains must be non-empty")
    if batch_size <= 0 or context_length <= 0 or eval_iters <= 0:
        raise ValueError("batch_size, context_length, and eval_iters must be positive")

    resolved = resolve_device(device)
    model.to(resolved)
    results: dict[str, float] = {}
    with torch.no_grad():
        for name, source in domains.items():
            generator = torch.Generator().manual_seed(seed)
            losses: list[float] = []
            for _ in range(eval_iters):
                x, y = get_lm_batch(
                    source,
                    batch_size,
                    context_length,
                    generator=generator,
                )
                logits = model(x.to(resolved))
                losses.append(float(lm_cross_entropy(logits, y.to(resolved))))
            results[name] = sum(losses) / len(losses)
    return results
