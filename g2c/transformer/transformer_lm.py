"""The full decoder-only transformer language model.

A `TransformerLM` is the smallest object that can be trained on
next-token prediction with the architecture you'll use for everything
else in the course. It composes:

    token_ids  ──► TokenEmbedding   ──┐
                                       ├─► +  ──► N × Block ──► LayerNorm ──► unembedding ──► logits
    positions  ──► PositionalEmbed  ──┘

Concretely:

    1. Look up token embeddings.                       (B, T, D)
    2. Look up positional embeddings.                  (T, D)  → broadcast to (B, T, D)
    3. Add them.                                       (B, T, D)
    4. Pass through `num_layers` transformer blocks.   (B, T, D)
    5. Apply a final layer norm.                       (B, T, D)
    6. Project to vocab logits via the unembedding.    (B, T, vocab_size)

The output is `(B, T, vocab_size)` — one logit vector per position. At
training time, position `t`'s logit predicts the token at position `t+1`,
so the standard training objective is cross-entropy between the logits
and the input shifted by one. (Module 09B / Module 10 wire that up.)

Two design choices to internalize:

  * **The final LayerNorm.** Modern transformers (GPT-2 onward) put one
    extra LN after the last block, before the unembedding. Without it,
    the residual stream's scale is unconstrained at the output and the
    unembedding's logits can drift. It's a small, cheap addition that
    materially improves training stability.

  * **The unembedding is the input embedding, transposed (tied
    embeddings).** Instead of a separate `Linear(D, V)` head, we reuse
    `TokenEmbedding.weight` (shape `(V, D)`) and compute logits as
    `x @ token_embed.weight.T + head_bias`. One matrix lives at both
    ends of the model. The geometric story is direct: scoring token
    `v` reduces to "how aligned is the residual stream with the
    embedding row that *put* token `v` in." Tying saves one full
    `(V, D)` matrix of parameters at no quality cost (Press & Wolf
    2017) and is standard in GPT-2, T5, Gemma, and most modern LMs.

Boilerplate (`__init__`, `parameters`) is implemented. The `forward`
method — the embed/blocks/norm/unembed pipeline — is scaffolded.
"""
from __future__ import annotations

from collections.abc import Iterable

import torch

from g2c.embeddings import LearnedPositionalEmbedding, TokenEmbedding
from g2c.nn import Module

from .block import Block
from .layer_norm import LayerNorm


class TransformerLM(Module):
    """A small decoder-only transformer language model.

    Args:
        vocab_size:    size of the token vocabulary (`V`).
        embedding_dim: channel dim for the residual stream (`D`).
        num_layers:    number of transformer blocks stacked (`N`).
        num_heads:     attention heads per block. Must divide `embedding_dim`.
        max_seq_len:   maximum sequence length (sets the size of the
                       learned positional embedding table). Inputs longer
                       than this fail with a `ValueError` in `forward`.
        hidden_dim:    FFN inner dim. Defaults to `4 * embedding_dim`.
    """

    vocab_size: int
    embedding_dim: int
    num_layers: int
    num_heads: int
    max_seq_len: int
    hidden_dim: int

    token_embed: TokenEmbedding
    pos_embed: LearnedPositionalEmbedding
    blocks: list[Block]
    ln_final: LayerNorm
    head_bias: torch.Tensor

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_layers: int,
        num_heads: int,
        max_seq_len: int,
        *,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * embedding_dim
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim

        self.token_embed = TokenEmbedding(vocab_size, embedding_dim)
        self.pos_embed = LearnedPositionalEmbedding(max_seq_len, embedding_dim)
        self.blocks = [
            Block(
                embedding_dim,
                num_heads,
                hidden_dim=hidden_dim,
                causal=True,
            )
            for _ in range(num_layers)
        ]
        self.ln_final = LayerNorm(embedding_dim)
        # The unembedding reuses `token_embed.weight` (transposed) — no
        # separate `(D, V)` weight matrix lives here. We keep a learned
        # per-token bias of shape `(V,)` on the output side.
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
        """Compute next-token logits for every position.

        Args:
            token_ids: integer tensor of shape `(batch, T)`. Each entry
                is a token ID in `[0, vocab_size)`. `T` must be ≤
                `max_seq_len` — sequences longer than the positional
                table will raise.

        Returns:
            Tensor of shape `(batch, T, vocab_size)`. Entry `[b, t, v]`
            is the unnormalized logit for token `v` at position `t` of
            batch element `b`. Standard training reads "predict
            `token_ids[:, t+1]` from `logits[:, t]`" — so the logit at
            position `t` is the model's prediction for what comes NEXT.

        Recipe:
            1. B, T = token_ids.shape
               if T > self.max_seq_len:
                   raise ValueError(...)
                   (the bound is enforced here, not in __init__,
                   because the constructor doesn't see the input length.)

            2. tok = self.token_embed(token_ids)        # (B, T, D)
            3. pos = self.pos_embed(T)                  # (T, D)
            4. x = tok + pos                            # broadcasts to (B, T, D)

               The position embedding adds the same vector at each batch
               element — that's the whole point. Position `t`'s contribution
               depends only on `t`, not on what token sits there.

            5. for block in self.blocks:
                   x = block(x)                         # (B, T, D)

               Each block reads and writes the residual stream. After
               `N` blocks, `x` carries information from up to `N` rounds
               of attention + FFN refinement.

            6. x = self.ln_final(x)                     # (B, T, D)

               One last normalization before the unembedding. This is
               the GPT-2 / modern-transformer convention; the original
               2017 paper didn't have it.

            7. return x @ self.token_embed.weight.T + self.head_bias  # (B, T, V)

               The unembedding reuses the input embedding matrix —
               `token_embed.weight` is `(V, D)`, so `weight.T` is
               `(D, V)` and projects the residual stream to vocab
               logits. The per-token `head_bias` adds a learned offset
               for each vocabulary item. Autograd routes gradient back
               into `token_embed.weight` from BOTH the input lookup
               AND this matmul on every step.

               The student-facing softmax / cross-entropy happens
               OUTSIDE this method, in the training loop.
        """
        B, T = token_ids.shape
        if T > self.max_seq_len:
            raise ValueError(
                f"Sequence length {T} exceeds model's max_seq_len {self.max_seq_len}"
            )
        x = self.token_embed(token_ids) + self.pos_embed(T)
        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)
        return x @ self.token_embed.weight.T + self.head_bias
