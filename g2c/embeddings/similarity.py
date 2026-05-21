"""Vector-similarity helpers for embedding inspection."""
from __future__ import annotations

from pathlib import Path

import torch


def load_glove_subset(path: Path, words: set[str]) -> dict[str, torch.Tensor]:
    """Load only selected word vectors from a whitespace-separated GloVe file."""
    vectors: dict[str, torch.Tensor] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            pieces = line.rstrip().split(" ")
            word = pieces[0]
            if word in words:
                values = [float(x) for x in pieces[1:]]
                vectors[word] = torch.tensor(values, dtype=torch.float32)
    return vectors


def normalized(v: torch.Tensor) -> torch.Tensor:
    """Return a unit-norm copy of ``v`` with a zero-safe denominator."""
    return v / v.norm().clamp_min(1e-12)


def nearest_by_cosine(
    query: torch.Tensor,
    vectors: dict[str, torch.Tensor],
    *,
    exclude: set[str] | None = None,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Return nearest words to ``query`` by cosine similarity."""
    # TODO
    raise NotImplementedError


def analogy(
    a: str,
    b: str,
    c: str,
    vectors: dict[str, torch.Tensor],
    *,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Return nearest words to ``a - b + c``."""
    query = vectors[a] - vectors[b] + vectors[c]
    return nearest_by_cosine(query, vectors, exclude={a, b, c}, top_k=top_k)
