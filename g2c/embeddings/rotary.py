"""Rotary positional embeddings (RoPE).

A different way of injecting position into a transformer. Instead of ADDING
a position vector to token embeddings, RoPE ROTATES query and key vectors
inside attention by an angle proportional to their position. The rotation
is structured so that the dot product of two RoPE'd vectors depends only on
their RELATIVE position (m − n), not absolute position. That property turns
out to be remarkably useful for attention.

Most modern LLMs (Llama, Mistral, Qwen, GPT-NeoX descendants) use RoPE.

We implement the "split-halves" variant used by Llama and friends: split the
last dim in half, treat dims `i` and `d/2 + i` as a 2-vector, rotate each
pair by an angle proportional to position. This is mathematically equivalent
to (and computationally faster than) the original interleaved formulation
in Su et al.'s paper.

This class isn't wired into any larger model in Module 05 — that happens
in Module 07 (attention). Here we just implement and unit-test it in
isolation.

Boilerplate is implemented. The cos/sin table construction (in `__init__`)
and the rotation itself (in `forward`) are scaffolded.
"""
from __future__ import annotations

from collections.abc import Iterable

import torch

from g2c.nn import Module


class RotaryEmbedding(Module):
    """Rotary positional embedding (RoPE), split-halves variant.

    Args:
        max_seq_len: longest sequence length we'll ever need.
        embedding_dim: dimension of the vectors being rotated. Must be even.
    """

    max_seq_len: int
    embedding_dim: int
    cos: torch.Tensor   # (max_seq_len, embedding_dim)
    sin: torch.Tensor   # (max_seq_len, embedding_dim)

    def __init__(self, max_seq_len: int, embedding_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if embedding_dim % 2 != 0:
            raise ValueError("embedding_dim must be even for RoPE")
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim

        # TODO
        raise NotImplementedError

    def parameters(self) -> Iterable[torch.Tensor]:
        return []

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Helper for the split-halves rotation.

        Splits the last dim in half: x = (x1, x2). Returns (-x2, x1).

        Why this odd-looking operation? The 2D rotation
            (a, b) → (a cos θ − b sin θ,  a sin θ + b cos θ)
        can be re-expressed as
            (a, b) → (a, b) ⊙ cos θ + (−b, a) ⊙ sin θ
        and `_rotate_half` produces exactly the (−b, a) part, applied across
        all paired dimensions in one go.
        """
        d = x.shape[-1]
        half = d // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RoPE to `x`.

        Args:
            x: tensor with shape (..., seq_len, embedding_dim). The
               second-to-last dim is treated as the position axis;
               position 0 is at index 0, etc.

        Returns:
            Same shape as `x`, with each position rotated.

        Recipe:
            seq_len = x.shape[-2]
            cos = self.cos[:seq_len]                # (seq_len, dim)
            sin = self.sin[:seq_len]                # (seq_len, dim)
            return x * cos + self._rotate_half(x) * sin

        At position 0, cos is all 1s and sin is all 0s, so the rotation is
        the identity — the test suite verifies this.
        """
        # TODO
        raise NotImplementedError
