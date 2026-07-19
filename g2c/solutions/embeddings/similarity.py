# Hand-maintained solutions mirror — canonical, lives in `main` (see AGENTS.md).
"""Solutions for g2c.embeddings.similarity pedagogical scaffolds.

Patched onto the scaffold targets by g2c.solutions.apply().
"""
from __future__ import annotations

from pathlib import Path

import torch


def nearest_by_cosine(
    query: torch.Tensor,
    vectors: dict[str, torch.Tensor],
    *,
    exclude: set[str] | None = None,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Return nearest words to ``query`` by cosine similarity."""
    exclude = exclude or set()
    q = normalized(query)

    # SOLUTION
    scores: list[tuple[str, float]] = []
    for word, vector in vectors.items():
        if word in exclude:
            continue
        score = float((q * normalized(vector)).sum().item())
        scores.append((word, score))
    scores.sort(key=lambda item: item[1], reverse=True)
    return scores[:top_k]
