"""The transformer block — pre-norm, MHA, residual, FFN, residual.

This is the unit that gets stacked `N` times to build a transformer.
Every modern decoder-only transformer (GPT-2, GPT-3, Llama, Mistral,
...) is structurally `embed → N × Block → unembed`. Once you have this
class, scaling up is "use a larger D, T, and N."

The two sublayers — multi-head attention and the position-wise FFN —
are wrapped in two architectural patterns that don't appear elsewhere
in the course but are decisive for training stability:

  * **Layer normalization BEFORE each sublayer** (pre-norm). Each
    sublayer reads a normalized input. The original transformer paper
    (Vaswani et al. 2017) put LN AFTER each sublayer (post-norm); the
    field has since converged on pre-norm because it trains more stably
    at depth without needing a learning-rate warmup hack. See "The
    pre-norm vs post-norm choice" in the Module 09 lesson page.

  * **Residual connections AROUND each sublayer.** The output of the
    sublayer is ADDED to its input rather than replacing it. This means
    every sublayer's job is to *adjust* the residual stream, not to
    overwrite it. Residual connections are what makes deep transformers
    trainable — without them, gradients vanish a few layers down. See
    "Why residual connections matter" in the lesson page.

The full forward pass:

    x = x + attn(ln1(x))         # residual + attention sublayer
    x = x + ffn(ln2(x))          # residual + FFN sublayer

Two LN's, two residuals, in that exact order. The student's job is to
write those two lines. Boilerplate (`__init__`, `parameters`) is
implemented; the `forward` method is scaffolded.

Note on dependencies: this module composes `LayerNorm` (Module 09),
`MultiHeadAttention` (Module 08), and `FeedForward` (Module 09). The
Block tests will only pass once all three are implemented — the suite
won't tell you "implement attention first, then layer norm, then FFN."
The recommended order is below in the test docstring.
"""
from __future__ import annotations

from collections.abc import Iterable

import torch

from g2c.attention import MultiHeadAttention
from g2c.nn import Module

from .ffn import FeedForward
from .layer_norm import LayerNorm


class Block(Module):
    """A single decoder-only transformer block (pre-norm).

    Structure:

        x ──► LayerNorm ──► MultiHeadAttention ──► + ──► LayerNorm ──► FeedForward ──► +
        │                                          ▲                                   ▲
        └──────────────── residual ────────────────┘                                   │
        └────────────────────────────────── residual ──────────────────────────────────┘

    Args:
        embedding_dim:  channel dim (`D`). Forwarded to MHA, FFN, and
                        both LNs.
        num_heads:      number of attention heads. Forwarded to MHA.
                        Must divide `embedding_dim` evenly.
        hidden_dim:     FFN inner dim. Defaults to `4 * embedding_dim`.
        causal:         passed through to MHA. Default True (decoder-only).
    """

    embedding_dim: int
    num_heads: int
    hidden_dim: int
    causal: bool

    ln1: LayerNorm
    attn: MultiHeadAttention
    ln2: LayerNorm
    ffn: FeedForward

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        *,
        hidden_dim: int | None = None,
        causal: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * embedding_dim
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.causal = causal
        self.ln1 = LayerNorm(embedding_dim)
        self.attn = MultiHeadAttention(embedding_dim, num_heads, causal=causal)
        self.ln2 = LayerNorm(embedding_dim)
        self.ffn = FeedForward(embedding_dim, hidden_dim=hidden_dim)

    def parameters(self) -> Iterable[torch.Tensor]:
        return [
            *self.ln1.parameters(),
            *self.attn.parameters(),
            *self.ln2.parameters(),
            *self.ffn.parameters(),
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply one transformer block to `x`. Returns `(B, T, D)`.

        Args:
            x: tensor of shape `(batch, T, embedding_dim)`. The output of
                the previous block — or, for the first block, of the
                token + position embedding sum.

        Returns:
            Tensor of the same shape as `x`. The residual stream after
            both sublayers have written into it.

        Recipe (two lines, in this exact order):

            1. x = x + self.attn(self.ln1(x))
            2. x = x + self.ffn(self.ln2(x))
            3. return x

        Read the structure carefully:

          * `ln1(x)` normalizes the input BEFORE attention sees it.
            Attention writes its update into a normalized space.
          * `attn(ln1(x))` is the attention sublayer's contribution. It
            has the same shape as `x`.
          * `x + attn(ln1(x))` adds that contribution back into the
            residual stream. The unnormalized `x` keeps flowing forward,
            with the attention update added.
          * `ln2(x)` (where `x` is now the post-attn residual) normalizes
            again before the FFN sublayer.
          * `ffn(ln2(x))` is the FFN sublayer's contribution.
          * Final `x = x + ffn(...)` adds it back.

        Common confusions to avoid:

          * The residuals are NOT normalized. Only the sublayer INPUT is
            normalized; the residual `x` is added unmodified. This is
            what makes the residual stream a "highway" that gradients
            flow through cleanly.
          * The TWO LayerNorms are independent — `ln1` and `ln2` each
            have their own gamma/beta. They're not the same module
            applied twice.
          * The order is "norm, then sublayer, then add." Post-norm
            (the order in the original 2017 paper) is "sublayer, then
            add, then norm" and trains differently. See the lesson page
            for why pre-norm won.
        """
        # TODO
        raise NotImplementedError

    def forward_cached(self, x: torch.Tensor, cache):
        """Apply one block to a single token using a layer KV cache.

        ``x`` is the residual stream for the newest token only, shape
        ``(B, 1, D)``. Attention reads cached past K/V rows; the FFN remains
        per-token and needs no cache.
        """
        attn_out, cache = self.attn.forward_cached(self.ln1(x), cache)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, cache
