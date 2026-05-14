"""Notebook-only helpers for Module 09 transformer-block experiments.

Module 09 teaches LayerNorm, the FFN, residuals, and the shape of
``TransformerLM``. The pre-norm/post-norm and residual/no-norm comparisons
in the notebook are the lesson; the surrounding training loop, batch
sampling, loss reshape, and "swap a different ``Block`` subclass into the
LM" plumbing are not. Those helpers live here so the notebook cells that
remain inline are the ones a student should read.
"""

from __future__ import annotations

import torch

from g2c.nn import CrossEntropyLoss, SGD, resolve_device
from g2c.transformer import Block, TransformerLM

__all__ = [
    "get_sequence_batch",
    "lm_loss",
    "make_lm_with_block",
    "train_tiny_transformer",
]


_loss_fn = CrossEntropyLoss()


def lm_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    B, T, V = logits.shape
    return _loss_fn(logits.reshape(B * T, V), targets.reshape(B * T))


def get_sequence_batch(
    ids: torch.Tensor,
    *,
    batch_size: int,
    context_length: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(0, len(ids) - context_length, (batch_size,), generator=generator)
    x = torch.stack([ids[s : s + context_length] for s in starts])
    y = torch.stack([ids[s + 1 : s + context_length + 1] for s in starts])
    return x, y


def make_lm_with_block(
    block_cls: type[Block],
    *,
    vocab_size: int = 50,
    embedding_dim: int = 64,
    num_layers: int = 6,
    num_heads: int = 4,
    max_seq_len: int = 32,
    hidden_dim: int | None = None,
) -> TransformerLM:
    """Build a ``TransformerLM`` whose blocks are instances of ``block_cls``.

    Used to plug experimental ``Block`` subclasses (post-norm, residual-free,
    no-LayerNorm) into the standard LM without modifying ``TransformerLM``.
    """
    model = TransformerLM(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        hidden_dim=hidden_dim,
    )
    model.blocks = [
        block_cls(
            embedding_dim,
            num_heads,
            hidden_dim=model.hidden_dim,
            causal=True,
        )
        for _ in range(num_layers)
    ]
    return model


def train_tiny_transformer(
    model: TransformerLM,
    ids: torch.Tensor,
    *,
    steps: int = 300,
    lr: float = 1e-3,
    batch_size: int = 32,
    context_length: int = 32,
    log_every: int = 25,
    seed: int = 0,
    device: str | torch.device = "auto",
) -> list[tuple[int, float]]:
    """Train ``model`` on a flat token stream and return the (step, loss) curve."""
    device = resolve_device(device)
    model.to(device)
    optimizer = SGD(model.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)
    curve: list[tuple[int, float]] = []

    for step in range(steps):
        xb, yb = get_sequence_batch(
            ids,
            batch_size=batch_size,
            context_length=context_length,
            generator=generator,
        )
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = lm_loss(logits, yb)
        loss.backward()
        optimizer.step()

        if step % log_every == 0 or step == steps - 1:
            curve.append((step, float(loss.item())))

    return curve
