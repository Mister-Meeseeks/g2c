"""MoEBlock and MoETransformerLM — Module 09's stack with a routed FFN slot.

Both classes here are provided plumbing, not scaffolds: the pedagogy of
this Beyond module lives in `Router.forward`, `MoEFeedForward.forward`,
and `load_balancing_loss`. Everything in this file is the Module 09
architecture with the FFN swapped — which is itself the lesson's
headline: MoE changes ONE slot in the block and nothing else.

`MoETransformerLM` also carries the parameter-accounting helpers
(`total_parameter_count`, `active_parameter_count`) used by the
notebook's model-card arithmetic.
"""
from __future__ import annotations

from collections.abc import Iterable

import torch

from g2c.attention import MultiHeadAttention
from g2c.embeddings import LearnedPositionalEmbedding, TokenEmbedding
from g2c.nn import Module
from g2c.transformer.layer_norm import LayerNorm

from .layer import MoEFeedForward


class MoEBlock(Module):
    """Module 09's pre-norm block with `MoEFeedForward` in the FFN slot."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        num_experts: int,
        top_k: int,
        *,
        hidden_dim: int | None = None,
        causal: bool = True,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.ln1 = LayerNorm(embedding_dim)
        self.attn = MultiHeadAttention(embedding_dim, num_heads, causal=causal)
        self.ln2 = LayerNorm(embedding_dim)
        self.moe_ffn = MoEFeedForward(
            embedding_dim, num_experts, top_k, hidden_dim=hidden_dim
        )

    def parameters(self) -> Iterable[torch.Tensor]:
        return [
            *self.ln1.parameters(),
            *self.attn.parameters(),
            *self.ln2.parameters(),
            *self.moe_ffn.parameters(),
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.moe_ffn(self.ln2(x))
        return x


class MoETransformerLM(Module):
    """A decoder-only transformer LM over `MoEBlock`s.

    Same composition as Module 09's `TransformerLM` — embed, blocks,
    final LayerNorm, tied unembedding — with two MoE additions:
    `load_balancing_loss()` averages the per-block auxiliary losses,
    and the parameter-count helpers report the total/active split.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_layers: int,
        num_heads: int,
        max_seq_len: int,
        num_experts: int,
        top_k: int,
        *,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.num_experts = num_experts
        self.top_k = top_k

        self.token_embed = TokenEmbedding(vocab_size, embedding_dim)
        self.pos_embed = LearnedPositionalEmbedding(max_seq_len, embedding_dim)
        self.blocks = [
            MoEBlock(
                embedding_dim,
                num_heads,
                num_experts,
                top_k,
                hidden_dim=hidden_dim,
                causal=True,
            )
            for _ in range(num_layers)
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

    def load_balancing_loss(self) -> torch.Tensor:
        """Mean of the per-block auxiliary losses from the last forward."""
        losses = [block.moe_ffn.load_balancing_loss() for block in self.blocks]
        return torch.stack(losses).mean()

    def total_parameter_count(self) -> int:
        """Every parameter — the model-card 'total' number."""
        return sum(p.numel() for p in self.parameters())

    def active_parameter_count(self) -> int:
        """Parameters one token's forward pass touches — the 'active' number.

        Total minus the `(E - k)` experts per block a token skips.
        """
        inactive = 0
        for block in self.blocks:
            per_expert = block.moe_ffn.expert_parameter_count()
            inactive += (self.num_experts - self.top_k) * per_expert
        return self.total_parameter_count() - inactive
