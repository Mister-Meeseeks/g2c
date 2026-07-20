# Hand-maintained solutions mirror — canonical, lives in `main` (see AGENTS.md).
"""Solutions for g2c.transformer.transformer_lm pedagogical scaffolds.

Patched onto the scaffold targets by g2c.solutions.apply().
"""
from __future__ import annotations

from collections.abc import Iterable

import torch

from g2c.embeddings import LearnedPositionalEmbedding, TokenEmbedding
from g2c.nn import Module
from g2c.transformer.block import Block
from g2c.transformer.kv_cache import KVCache
from g2c.transformer.layer_norm import LayerNorm
from g2c.transformer.transformer_lm import TransformerLM


class _TransformerLMImpl:  # patched onto TransformerLM by apply()
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


    def forward_cached(
        self,
        token_ids: torch.Tensor,
        cache: KVCache | None = None,
    ) -> tuple[torch.Tensor, KVCache]:
        """Compute next-token logits for one new token and update cache.

        Args:
            token_ids: integer tensor of shape ``(batch, 1)``. Cached decoding
                processes exactly one new position per call.
            cache: existing cache, or ``None`` to start from an empty cache.

        Returns:
            ``(logits, cache)`` where logits has shape ``(batch, 1, vocab_size)``
            and cache now includes this token's key/value rows in every layer.

        This is the inference-only sibling of ``forward``. It should match the
        final-position logits from ``forward(token_ids_so_far)`` when called
        step by step on the same token sequence.
        """
        if token_ids.dim() != 2 or token_ids.shape[1] != 1:
            raise ValueError(
                "forward_cached expects token_ids with shape (batch, 1); "
                f"got {tuple(token_ids.shape)}"
            )
        if cache is None:
            cache = self.empty_kv_cache()
        if len(cache) != self.num_layers:
            raise ValueError(
                f"cache has {len(cache)} layers, but model has {self.num_layers}"
            )

        position = cache.length
        if position >= self.max_seq_len:
            raise ValueError(
                f"KV cache length {position} has reached max_seq_len {self.max_seq_len}"
            )

        tok = self.token_embed(token_ids)
        # Position `position`, not 0 — the cache length is where we are.
        pos = self.pos_embed.weight[position : position + 1].to(tok.device)
        x = tok + pos
        for idx, block in enumerate(self.blocks):
            x, cache.layers[idx] = block.forward_cached(x, cache.layers[idx])
        x = self.ln_final(x)
        return x @ self.token_embed.weight.T + self.head_bias, cache
