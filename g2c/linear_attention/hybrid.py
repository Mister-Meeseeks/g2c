"""HybridBlock and HybridTransformerLM — a minimal hybrid configuration.

Linear attention compresses history; full attention everywhere pays the
full KV cache. Many practical architectures mix recurrent and full
layers. This file is provided plumbing — the pedagogy lives in
`LinearAttention.forward` / `step`; here we just stack blocks according
to a `layer_pattern` like:

    ["linear", "linear", "linear", "full"] * 2      # a 3:1 hybrid

The pattern vocabulary is exactly {"linear", "full"}.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch

from g2c.attention import MultiHeadAttention
from g2c.embeddings import LearnedPositionalEmbedding, TokenEmbedding
from g2c.nn import Module
from g2c.transformer.ffn import FeedForward
from g2c.transformer.layer_norm import LayerNorm

from .attention import LinearAttention

_KINDS = ("linear", "full")


class HybridBlock(Module):
    """Module 09's pre-norm block with a selectable attention sublayer."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        kind: str,
        *,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if kind not in _KINDS:
            raise ValueError(f"kind must be one of {_KINDS}, got {kind!r}")
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.kind = kind
        self.ln1 = LayerNorm(embedding_dim)
        if kind == "full":
            self.attn: Module = MultiHeadAttention(
                embedding_dim, num_heads, causal=True
            )
        else:
            self.attn = LinearAttention(embedding_dim, num_heads)
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
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class HybridTransformerLM(Module):
    """A decoder-only LM whose per-layer attention kind follows a pattern.

    Args:
        vocab_size, embedding_dim, num_heads, max_seq_len, hidden_dim:
            as in Module 09's `TransformerLM`.
        layer_pattern: sequence of `"linear"` / `"full"`, one entry per
            layer, bottom to top. `["full"] * N` reproduces the
            Module 09 architecture; `["linear", "linear", "linear",
            "full"] * k` is the 3:1 hybrid from the lesson page.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        layer_pattern: Sequence[str],
        num_heads: int,
        max_seq_len: int,
        *,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if not layer_pattern:
            raise ValueError("layer_pattern must have at least one entry")
        for kind in layer_pattern:
            if kind not in _KINDS:
                raise ValueError(
                    f"layer_pattern entries must be one of {_KINDS}, "
                    f"got {kind!r}"
                )
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.layer_pattern = list(layer_pattern)
        self.num_layers = len(self.layer_pattern)
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        self.token_embed = TokenEmbedding(vocab_size, embedding_dim)
        self.pos_embed = LearnedPositionalEmbedding(max_seq_len, embedding_dim)
        self.blocks = [
            HybridBlock(embedding_dim, num_heads, kind, hidden_dim=hidden_dim)
            for kind in self.layer_pattern
        ]
        self.ln_final = LayerNorm(embedding_dim)
        head_bias = torch.zeros(vocab_size)
        head_bias.requires_grad_(True)
        self.head_bias = head_bias

    def parameters(self) -> Iterable[torch.Tensor]:
        params: list[torch.Tensor] = []
        params.extend(self.token_embed.parameters())
        params.extend(self.pos_embed.parameters())
        for block in self.blocks:
            params.extend(block.parameters())
        params.extend(self.ln_final.parameters())
        params.append(self.head_bias)
        return params

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        _, T = token_ids.shape
        if T > self.max_seq_len:
            raise ValueError(
                f"sequence length {T} exceeds max_seq_len {self.max_seq_len}"
            )
        x = self.token_embed(token_ids) + self.pos_embed(T)
        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)
        return x @ self.token_embed.weight.T + self.head_bias
