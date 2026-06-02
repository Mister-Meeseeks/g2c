# Hand-maintained solutions mirror — canonical, lives in `main` (see AGENTS.md).
"""Solutions for g2c.embeddings.rotary pedagogical scaffolds.

Patched onto the scaffold targets by g2c.solutions.apply().
"""
from __future__ import annotations

from collections.abc import Iterable
import torch
from g2c.nn import Module

from g2c.embeddings.rotary import RotaryEmbedding


class _RotaryEmbeddingImpl:  # patched onto RotaryEmbedding by apply()
    def __init__(self, max_seq_len: int, embedding_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if embedding_dim % 2 != 0:
            raise ValueError("embedding_dim must be even for RoPE")
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim

        # TODO: precompute self.cos and self.sin, each of shape
        # (max_seq_len, embedding_dim). Neither requires grad.
        #
        # Recipe (split-halves):
        #   1. inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2) / dim))
        #                                                 shape (dim/2,)
        #   2. positions = torch.arange(max_seq_len)      shape (max_seq_len,)
        #   3. freqs = torch.outer(positions, inv_freq)   shape (max_seq_len, dim/2)
        #   4. emb = torch.cat([freqs, freqs], dim=-1)    shape (max_seq_len, dim)
        #   5. self.cos = emb.cos()
        #      self.sin = emb.sin()
        inv_freq = 1.0 / (base ** (torch.arange(0, embedding_dim, 2) / embedding_dim))
        positions = torch.arange(max_seq_len)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.cos = emb.cos()
        self.sin = emb.sin()
        self.cos.requires_grad_(False)
        self.sin.requires_grad_(False)

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
        seq_lqen = x.shape[-2]
        cos = self.cos[:seq_lqen]
        sin = self.sin[:seq_lqen]
        return x * cos + self._rotate_half(x) * sin

