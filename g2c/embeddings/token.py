"""Learned token embeddings.

A `TokenEmbedding` is a lookup table that maps each token ID to a learnable
vector. Internally it's just a `(vocab_size, embedding_dim)` matrix; the
forward pass is integer indexing.

This is the bridge between Module 04's discrete token IDs and the continuous
vectors the rest of the network operates on. Every modern LLM has one of these
as its very first layer.

Boilerplate (init with small random values, parameter list) is implemented.
The forward — the actual lookup — is scaffolded.
"""
from __future__ import annotations

import math
from collections.abc import Iterable

import torch

from g2c.nn import Module


class TokenEmbedding(Module):
    """A learned lookup table from token IDs to vectors.

    Args:
        vocab_size: number of distinct tokens the table can represent.
        embedding_dim: dimensionality of each token's vector.
    """

    vocab_size: int
    embedding_dim: int
    weight: torch.Tensor

    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        bound = 1.0 / math.sqrt(embedding_dim)
        W = torch.empty(vocab_size, embedding_dim).uniform_(-bound, bound)
        W.requires_grad_(True)
        self.weight = W

    def parameters(self) -> Iterable[torch.Tensor]:
        return [self.weight]

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Look up embeddings for each token ID.

        Args:
            ids: integer tensor of any shape. Typical: (batch, seq_len).

        Returns:
            Tensor with shape `ids.shape + (embedding_dim,)`.

        The whole point of this method is to internalize that an "embedding
        lookup" is just integer indexing into a learnable table. Hint: it's
        a one-liner. The autograd will correctly route gradients back to the
        rows of `self.weight` that were touched.
        """
        # TODO
        raise NotImplementedError
