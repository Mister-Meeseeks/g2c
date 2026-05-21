"""Small skip-gram helpers for Module 05 embedding experiments."""
from __future__ import annotations

import torch

from g2c.nn import CrossEntropyLoss, Linear, SGD

from .token import TokenEmbedding


def make_skipgram_pairs(
    ids: list[int],
    window: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return center IDs and context target IDs for a skip-gram objective."""
    # TODO
    raise NotImplementedError


class SkipGramEmbeddingModel:
    """Minimal skip-gram model: center token -> embedding -> context logits."""

    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        self.embedding = TokenEmbedding(vocab_size, embedding_dim)
        self.output = Linear(embedding_dim, vocab_size)

    def parameters(self) -> list[torch.Tensor]:
        return list(self.embedding.parameters()) + list(self.output.parameters())

    def __call__(self, center_ids: torch.Tensor) -> torch.Tensor:
        center_vectors = self.embedding(center_ids)
        return self.output(center_vectors)


def train_skipgram(
    model: SkipGramEmbeddingModel,
    center_ids: torch.Tensor,
    context_ids: torch.Tensor,
    *,
    steps: int = 500,
    batch_size: int = 128,
    lr: float = 0.2,
    generator: torch.Generator | None = None,
) -> list[float]:
    """Train on random center/context batches and return the loss curve."""
    # TODO
    raise NotImplementedError
