"""Positional encodings (additive variants).

Without positional information, a transformer treats its input as a bag of
tokens — `dog bites man` and `man bites dog` look identical. Two additive
schemes for fixing this:

  - `LearnedPositionalEmbedding`: another lookup table, this one indexed by
    position rather than token ID. Same shape and machinery as `TokenEmbedding`.
  - `SinusoidalPositionalEmbedding` (Vaswani et al. 2017): a fixed table of
    sines and cosines at multiple frequencies. No learnable parameters; can
    be evaluated at positions beyond what was seen in training.

Both produce a `(seq_len, embedding_dim)` tensor that's typically ADDED to
token embeddings before the first attention block.

Rotary positional embeddings (RoPE) are a different beast — they multiply
inside attention rather than add to embeddings. They live in `rotary.py`.

Boilerplate is implemented. The forwards (and the sinusoidal table
construction, which is the real lesson here) are scaffolded.
"""
from __future__ import annotations

import math
from collections.abc import Iterable

import torch

from g2c.nn import Module


class LearnedPositionalEmbedding(Module):
    """A learned lookup table from positions to vectors.

    Mechanically identical to `TokenEmbedding`, just indexed by position
    instead of token ID.

    Limitation: the table has a fixed maximum length (`max_seq_len`); the
    model can't process sequences longer than this without re-training.
    """

    max_seq_len: int
    embedding_dim: int
    weight: torch.Tensor

    def __init__(self, max_seq_len: int, embedding_dim: int) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim

        bound = 1.0 / math.sqrt(embedding_dim)
        W = torch.empty(max_seq_len, embedding_dim).uniform_(-bound, bound)
        W.requires_grad_(True)
        self.weight = W

    def parameters(self) -> Iterable[torch.Tensor]:
        return [self.weight]

    def forward(self, seq_len: int) -> torch.Tensor:
        """Return positional embeddings for positions 0..seq_len-1.

        Args:
            seq_len: length of the sequence (must be ≤ max_seq_len).

        Returns:
            (seq_len, embedding_dim) tensor — the first `seq_len` rows of
            the learned table. This will broadcast correctly when added to
            a (batch, seq_len, embedding_dim) tensor of token embeddings.

        Hint: a one-liner — slice the weight table.
        """
        # TODO
        raise NotImplementedError


class SinusoidalPositionalEmbedding(Module):
    """Fixed sinusoidal positional encoding (Vaswani et al. 2017).

    Each position gets a vector built from sines and cosines at exponentially
    decaying frequencies:

        PE[pos, 2i]     = sin(pos / 10000^(2i/d))
        PE[pos, 2i+1]   = cos(pos / 10000^(2i/d))

    where `d = embedding_dim` and `i = 0, 1, ..., d/2 - 1`.

    Properties:
      - No learnable parameters. The whole table is determined by the formula.
      - Each dimension oscillates at a different frequency: low `i` → fast
        oscillation, high `i` → slow. This gives the model information at
        multiple scales of "how far apart are these tokens?"
      - Can be evaluated at positions beyond `max_seq_len` (just compute the
        formula at the new position) — though our implementation only stores
        a fixed-size table for simplicity.

    Implementation requires `embedding_dim` to be even (so we can pair the
    dimensions into sin/cos halves cleanly).
    """

    max_seq_len: int
    embedding_dim: int
    weight: torch.Tensor

    def __init__(self, max_seq_len: int, embedding_dim: int) -> None:
        super().__init__()
        if embedding_dim % 2 != 0:
            raise ValueError("embedding_dim must be even for sinusoidal encoding")
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim

        # TODO: build the (max_seq_len, embedding_dim) sinusoidal table and
        # assign it to self.weight. The table should NOT require grad.
        #
        # Recipe:
        #   1. positions = arange(max_seq_len)                shape (max_seq_len,)
        #   2. i = arange(0, embedding_dim, 2)                shape (embedding_dim/2,)
        #      div_term = 1.0 / (10000 ** (i / embedding_dim))
        #   3. angles = positions[:, None] * div_term[None, :]   shape (max_seq_len, embedding_dim/2)
        #   4. weight = empty(max_seq_len, embedding_dim)
        #      weight[:, 0::2] = sin(angles)
        #      weight[:, 1::2] = cos(angles)
        #   5. self.weight = weight   (no requires_grad — fixed table)
        raise NotImplementedError

    def parameters(self) -> Iterable[torch.Tensor]:
        return []

    def forward(self, seq_len: int) -> torch.Tensor:
        """Return positional encodings for positions 0..seq_len-1.

        Args:
            seq_len: length of the sequence (must be ≤ max_seq_len).

        Returns:
            (seq_len, embedding_dim) tensor.

        Hint: a one-liner — slice `self.weight`.
        """
        # TODO
        raise NotImplementedError
