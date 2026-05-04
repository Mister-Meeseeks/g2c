"""Multi-head causal self-attention.

Multi-head generalizes Module 07's single-head SelfAttention by splitting
the embedding dimension `D` into `H` parallel sub-channels of size
`head_dim = D / H`. Each head computes its own Q/K/V/scores/softmax/mix
INDEPENDENTLY in its own subspace, and the H per-head outputs are
concatenated and passed through a final output projection.

The crucial implementation choice — the one the syllabus singles out —
is that this is done via *reshaping*, not by instantiating H independent
Linear layers. We use ONE `(D, D)` projection for queries (and likewise
for keys and values), then `view` the result as `(B, T, H, head_dim)`
to expose the per-head structure. This is mathematically identical to
H independent `(D, head_dim)` projections, but it's a single matmul
on the GPU/MPS instead of H separate ones — and it's the standard
implementation idiom that you'll see in every real transformer
codebase.

Shapes throughout:

    x          (B, T, D)
    Q, K, V    (B, T, D) → reshape → (B, H, T, head_dim)
    scores     (B, H, T, T)              — Q @ K^T / sqrt(head_dim)
    weights    (B, H, T, T)              — softmax(scores) row-wise
    mixed      (B, H, T, head_dim)       — weights @ V
    concat     (B, T, D)                 — transpose + reshape
    output     (B, T, D)                 — out_proj(concat)

Two important details to internalize:

  * The √D scaling from Module 07 becomes √head_dim — each dot product
    is over a head_dim-dimensional subspace, not the full embedding.
    Using √D here is the single most common multi-head bug.
  * The output projection is now load-bearing. In single-head it was
    almost ceremonial, but here it must MIX the H per-head outputs back
    into a coherent representation. Without it, you'd have H independent
    attention layers stacked on top of each other with no way to combine
    their information.

Boilerplate (`__init__`, `parameters`, `causal_mask`) is implemented.
The `forward` and `attention_weights` methods are scaffolded — the math
is the same as Module 07 with reshape/transpose calls bolted on.
"""
from __future__ import annotations

import math
from collections.abc import Iterable

import torch

from g2c.nn import Linear, Module


class MultiHeadAttention(Module):
    """Multi-head self-attention with optional causal masking.

    Construction creates four `Linear(D, D)` layers — three for the
    Q/K/V projections and one for the output projection. The Q/K/V
    outputs are reshaped from `(B, T, D)` to `(B, H, T, head_dim)` so
    all H heads can be computed in parallel via batched matmul.

    Parameters:
        embedding_dim:  total dimensionality of token vectors. Must be
            divisible by `num_heads`. The Q, K, V projections all map
            `D → D`; the per-head subspace has `head_dim = D / H`.
        num_heads:      number of parallel attention heads. Must divide
            `embedding_dim` evenly.
        causal:         if True, install a `-inf` upper-triangular mask
            on the attention scores so position `t` only attends to
            positions `<= t`. Default True.

    No `torch.nn.MultiheadAttention`, no `torch.nn.functional.scaled_dot_product_attention`,
    no shortcuts. The point is to build the mechanism by hand and feel
    why the reshape works.
    """

    embedding_dim: int
    num_heads: int
    head_dim: int
    causal: bool
    q_proj: Linear
    k_proj: Linear
    v_proj: Linear
    out_proj: Linear

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        *,
        causal: bool = True,
    ) -> None:
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
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

        Same as `SelfAttention.causal_mask` from Module 07. Repeated
        here so the multi-head class is self-contained — students can
        read this file end-to-end without flipping back to Module 07.

        The mask broadcasts naturally over the head dimension: a
        `(T, T)` mask applied to `(B, H, T, T)` scores via
        `masked_fill` blocks the same positions in every head.
        """
        return torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
            diagonal=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-head self-attention. Returns `(B, T, D)`.

        Args:
            x: tensor of shape `(batch, T, embedding_dim)`.

        Returns:
            Tensor of shape `(batch, T, embedding_dim)`. For each
            position `t`, the value is the concatenation of `H`
            independent weighted mixtures of value vectors — each
            mixture computed in its own `head_dim`-dimensional subspace
            — followed by a final output projection.

        Recipe:
            1. Project to Q, K, V:
                   q = self.q_proj(x)            # (B, T, D)
                   k = self.k_proj(x)            # (B, T, D)
                   v = self.v_proj(x)            # (B, T, D)

            2. Reshape into per-head form:
                   B, T, _ = x.shape
                   H, d_h = self.num_heads, self.head_dim
                   q = q.view(B, T, H, d_h).transpose(1, 2)  # (B, H, T, d_h)
                   k = k.view(B, T, H, d_h).transpose(1, 2)
                   v = v.view(B, T, H, d_h).transpose(1, 2)

               The view splits the last dim D into H slots of size d_h.
               The transpose makes H the leading batch-like dim so the
               next matmul applies independently to each head.

            3. Scaled dot-product scores (per head, in parallel):
                   scores = q @ k.transpose(-2, -1) / sqrt(d_h)
                                                    # (B, H, T, T)

               Note: scaling is by sqrt(d_h), NOT sqrt(D). Each dot
               product is over a d_h-dimensional subspace. Using sqrt(D)
               here over-scales the scores by sqrt(H), saturating the
               softmax in the wrong direction — a frequent multi-head
               bug.

            4. Apply the causal mask if self.causal:
                   mask = self.causal_mask(T, device=scores.device)
                   scores = scores.masked_fill(mask, float("-inf"))

               The (T, T) mask broadcasts over the (B, H, ., .) dims —
               the same positions are blocked in every head.

            5. Convert scores to attention weights:
                   weights = scores.softmax(dim=-1)   # (B, H, T, T)

            6. Mix the value vectors PER HEAD:
                   mixed = weights @ v                # (B, H, T, d_h)

            7. Concatenate heads back into a single (B, T, D) tensor:
                   mixed = mixed.transpose(1, 2).contiguous().view(B, T, D)

               Transpose moves H back next to d_h so the view can
               flatten the last two dims into D = H * d_h. The
               `.contiguous()` is required because `transpose` returns
               a non-contiguous view that `.view()` cannot reshape
               directly.

            8. Output projection:
                   return self.out_proj(mixed)        # (B, T, D)
        """
        q = self.q_proj(x)  # (B, T, D)
        k = self.k_proj(x)  # (B, T, D)
        v = self.v_proj(x)  # (B, T, D)

        B, T, _ = x.shape
        H, d_h = self.num_heads, self.head_dim

        q = q.view(B, T, H, d_h).transpose(1, 2)  # (B, H, T, d_h)
        k = k.view(B, T, H, d_h).transpose(1, 2)
        v = v.view(B, T, H, d_h).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(d_h)  # (B, H, T, T)

        if self.causal:
            mask = self.causal_mask(T, device=scores.device)
            scores = scores.masked_fill(mask, float("-inf"))

        weights = scores.softmax(dim=-1)  # (B, H, T, T)
        mixed = weights @ v  # (B, H, T, d_h)
        mixed = mixed.transpose(1, 2).contiguous().view(B, T, self.embedding_dim)  # (B, T, D)

        return self.out_proj(mixed)  # (B, T, D)
                                           

    def attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Return the per-head attention weight matrices for visualization.

        Args:
            x: tensor of shape `(batch, T, embedding_dim)`.

        Returns:
            Tensor of shape `(batch, H, T, T)`. Entry `[b, h, i, j]` is
            the attention weight from query position `i` to key position
            `j` in batch element `b`, head `h`. Each `[b, h, i, :]` row
            sums to 1.

        This is steps 1-5 of `forward` — Q/K projection, reshape, scaled
        dot product, mask, softmax — without value mixing or the output
        projection. It exists so exercise 4 (per-head visualization) has
        a clean public method to call.

        Recipe:
            1. q = self.q_proj(x)
            2. k = self.k_proj(x)
            3. reshape both into (B, H, T, d_h)
            4. scores = q @ k.transpose(-2, -1) / sqrt(d_h)
            5. apply causal mask if self.causal
            6. return scores.softmax(dim=-1)
        """
        q = self.q_proj(x)
        k = self.k_proj(x)

        B, T, _ = x.shape
        H, d_h = self.num_heads, self.head_dim

        q = q.view(B, T, H, d_h).transpose(1, 2)  # (B, H, T, d_h)
        k = k.view(B, T, H, d_h).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(d_h)  # (B, H, T, T)

        if self.causal:
            mask = self.causal_mask(T, device=scores.device)
            scores = scores.masked_fill(mask, float("-inf"))

        return scores.softmax(dim=-1)  # (B, H, T, T)
