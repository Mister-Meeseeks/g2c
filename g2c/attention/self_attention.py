"""Single-head causal self-attention.

The smallest piece of the transformer. Given a sequence of token vectors
`x` of shape `(batch, T, embedding_dim)`, attention produces a new
sequence of the same shape where each output position is a learned
weighted average of all preceding (and current) positions:

    out[t] = sum_{s <= t}  attention_weight(t, s) * value(s)

The attention weight `attention_weight(t, s)` is the softmax-normalized
similarity between a *query* vector for position `t` and a *key* vector
for position `s`. Queries, keys, and values are all linear projections of
the same input — that's the "self" in self-attention.

Shapes throughout:

    x          (B, T, D)
    Q, K, V    (B, T, D)             — three projections of x
    scores     (B, T, T)              — Q @ K^T / sqrt(D)
    weights    (B, T, T)              — softmax(scores) row-wise
    mixed      (B, T, D)              — weights @ V
    output     (B, T, D)              — out_proj(mixed)

The `causal=True` flag installs an upper-triangular `-inf` mask on
`scores` BEFORE the softmax, so position `t` can only attend to
positions `0..t`. This is the property that makes attention safe to use
in a language model: at training time, position `t`'s prediction is not
allowed to peek at the answer (positions `> t`).

Boilerplate (`__init__`, `parameters`, `causal_mask`) is implemented.
The `forward` and `attention_weights` methods are scaffolded — they
contain the real lesson.
"""
from __future__ import annotations

import math
from collections.abc import Iterable

import torch

from g2c.nn import Linear, Module


class SelfAttention(Module):
    """Single-head self-attention with optional causal masking.

    Construction creates four `Linear` layers — three for the Q/K/V
    projections and one for the output projection. All four are
    `(embedding_dim, embedding_dim)` for single-head attention.

    Parameters:
        embedding_dim:  the dimensionality of token vectors. Q, K, V, and
            the output all live in this dimension. (Module 08 generalizes
            to multi-head, where Q/K/V live in `head_dim < embedding_dim`.)
        causal:         if True, install a `-inf` upper-triangular mask
            on the attention scores so position `t` only attends to
            positions `<= t`. Default True; set False for the strip-mask
            experiment in exercise 4.

    No `torch.nn.MultiheadAttention`, no `torch.nn.functional.scaled_dot_product_attention`
    — the entire point of this module is to build the mechanism by hand.
    """

    embedding_dim: int
    causal: bool
    q_proj: Linear
    k_proj: Linear
    v_proj: Linear
    out_proj: Linear

    def __init__(self, embedding_dim: int, *, causal: bool = True) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.causal = causal
        self.q_proj = Linear(embedding_dim, embedding_dim)
        self.k_proj = Linear(embedding_dim, embedding_dim)
        self.v_proj = Linear(embedding_dim, embedding_dim)
        self.out_proj = Linear(embedding_dim, embedding_dim)

    def parameters(self) -> Iterable[torch.Tensor]:
        return [
            *self.q_proj.parameters(),
            *self.k_proj.parameters(),
            *self.v_proj.parameters(),
            *self.out_proj.parameters(),
        ]

    @staticmethod
    def causal_mask(seq_len: int, device: torch.device | None = None) -> torch.Tensor:
        """Boolean upper-triangular mask of shape `(seq_len, seq_len)`.

        Entry `[i, j]` is True when `j > i` — i.e. position `i` should
        NOT be allowed to attend to position `j`. Pass this to
        `scores.masked_fill(mask, -inf)` to apply it.

        Why True-means-mask? `masked_fill(mask, value)` overwrites entries
        where the mask is True, which matches the convention that "masked"
        means "blocked." (PyTorch's own attention modules use the same
        polarity.)
        """
        return torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
            diagonal=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention. Returns `(B, T, D)` — same shape as input.

        Args:
            x: tensor of shape `(batch, T, embedding_dim)`. The sequence
                of token vectors to attend over.

        Returns:
            Tensor of shape `(batch, T, embedding_dim)`. For each position
            `t` in the output, the value is a learned weighted mixture of
            value vectors from positions `0..t` (or all positions, if
            `causal=False`).

        Recipe:
            1. Project to queries, keys, values:
                   q = self.q_proj(x)            # (B, T, D)
                   k = self.k_proj(x)            # (B, T, D)
                   v = self.v_proj(x)            # (B, T, D)

            2. Compute scaled dot-product attention scores:
                   scores = q @ k.transpose(-2, -1) / sqrt(D)
                                                  # (B, T, T)

               The transpose is on the LAST TWO dims so the batch dim is
               preserved. The `1/sqrt(D)` scaling keeps the variance of
               the dot product in a regime where softmax has useful
               gradients — without it, softmax saturates as `D` grows
               and gradients vanish.

            3. Apply the causal mask if `self.causal`:
                   T = x.shape[-2]
                   mask = self.causal_mask(T, device=scores.device)
                   scores = scores.masked_fill(mask, float("-inf"))

               The mask must be applied BEFORE the softmax — masking
               post-softmax destroys the row-sums-to-one property.

            4. Convert scores to attention weights:
                   weights = scores.softmax(dim=-1)   # (B, T, T)

               Softmax is over the LAST dim — for each query position,
               the attention weights over all key positions sum to 1.

            5. Mix the value vectors:
                   mixed = weights @ v            # (B, T, D)

            6. Output projection:
                   return self.out_proj(mixed)    # (B, T, D)

        Implementation note: `attention_weights` below recomputes steps
        1-4. Feel free to factor the shared work into a helper if you
        like — the lesson is in the math, not the code structure.
        """
        # TODO
        raise NotImplementedError

    def attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Return the attention weight matrix for visualization.

        Args:
            x: tensor of shape `(batch, T, embedding_dim)`.

        Returns:
            Tensor of shape `(batch, T, T)`. Entry `[b, i, j]` is the
            attention weight from query position `i` to key position `j`
            in batch element `b`. Each row sums to 1 (it's a softmax).

        This is exactly steps 1-4 of `forward`, without the value
        mixing or output projection. It exists so exercise 2 — visualizing
        the attention pattern on the canonical "the animal didn't cross
        the street because it was too tired" sentence — has a clean
        method to call.

        Recipe:
            1. q = self.q_proj(x)
            2. k = self.k_proj(x)
            3. scores = q @ k.transpose(-2, -1) / sqrt(D)
            4. apply causal mask if self.causal
            5. return scores.softmax(dim=-1)
        """
        # TODO
        raise NotImplementedError
