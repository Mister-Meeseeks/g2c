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
    if window < 1:
        raise ValueError("window must be at least 1")
    centers: list[int] = []
    contexts: list[int] = []

    # SOLUTION
    for i, center in enumerate(ids):
        left = max(0, i - window)
        right = min(len(ids), i + window + 1)
        for j in range(left, right):
            if j == i:
                continue
            centers.append(center)
            contexts.append(ids[j])

    return torch.tensor(centers, dtype=torch.long), torch.tensor(contexts, dtype=torch.long)


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
    if center_ids.numel() != context_ids.numel():
        raise ValueError("center_ids and context_ids must have the same length")
    if center_ids.numel() == 0:
        raise ValueError("no skip-gram pairs to train on")

    loss_fn = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr)
    losses: list[float] = []

    # SOLUTION
    for _ in range(steps):
        batch_indices = torch.randint(
            0,
            center_ids.numel(),
            (batch_size,),
            generator=generator,
        )
        batch_centers = center_ids[batch_indices]
        batch_contexts = context_ids[batch_indices]

        optimizer.zero_grad()
        logits = model(batch_centers)
        loss = loss_fn(logits, batch_contexts)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    return losses
