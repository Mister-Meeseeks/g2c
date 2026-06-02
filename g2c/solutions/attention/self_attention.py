# Hand-maintained solutions mirror — canonical, lives in `main` (see AGENTS.md).
"""Solutions for g2c.attention.self_attention pedagogical scaffolds.

Patched onto the scaffold targets by g2c.solutions.apply().
"""
from __future__ import annotations

import math
from collections.abc import Iterable
import torch
from g2c.nn import Linear, Module

from g2c.attention.self_attention import SelfAttention


class _SelfAttentionImpl:  # patched onto SelfAttention by apply()
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
        q = self.q_proj(x)  # (B, T, D)
        k = self.k_proj(x)  # (B, T, D)
        v = self.v_proj(x)  # (B, T, D)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.embedding_dim )  # (B, T, T)
        if self.causal:
            T = x.shape[-2]
            mask = self.causal_mask(T, device=scores.device)
            scores = scores.masked_fill(mask, float("-inf"))

        weights = scores.softmax(dim=-1)  # (B, T, T)
        mixed = weights @ v  # (B, T, D)
        return self.out_proj(mixed)  # (B, T, D)

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
        q = self.q_proj(x)
        k = self.k_proj(x)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.embedding_dim)
        if self.causal:
            T = x.shape[-2]
            mask = self.causal_mask(T, device=scores.device)
            scores = scores.masked_fill(mask, float("-inf"))
        return scores.softmax(dim=-1)
