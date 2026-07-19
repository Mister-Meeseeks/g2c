# Hand-maintained solutions mirror — canonical, lives in `main` (see AGENTS.md).
"""Solutions for g2c.lm.mlp pedagogical scaffolds.

Patched onto the scaffold targets by g2c.solutions.apply().
"""
from __future__ import annotations

from collections.abc import Iterable

import torch

from g2c.embeddings import TokenEmbedding
from g2c.lm.mlp import MLPLanguageModel
from g2c.nn import Linear, Module


class _MLPLanguageModelImpl:  # patched onto MLPLanguageModel by apply()
    def forward(self, ctx_ids: torch.Tensor) -> torch.Tensor:
        """Compute next-token logits from a context window.

        Args:
            ctx_ids: integer tensor of shape `(batch, context_length)`. Each
                row is a window of the most recent `context_length` tokens.

        Returns:
            (batch, vocab_size) tensor of unnormalized logits.

        Recipe:
            1. e = self.embed(ctx_ids)                       # (B, N, D)
            2. flat = e.view(e.shape[0], -1)                 # (B, N*D)
                  (Equivalently: e.reshape(e.shape[0], -1).
                   `view` is faster but requires contiguous memory; the
                   tensor coming out of `embed` is contiguous, so view is fine.)
            3. h = torch.tanh(self.hidden(flat))             # (B, H)
            4. return self.output(h)                         # (B, V)

        Why `tanh` and not `relu`? Bengio's original paper used tanh, and it
        works fine for this small model. Modern transformers use GELU or
        SwiGLU. The choice mostly stops mattering once you're stacking many
        layers; here it doesn't matter much.

        Why concatenation rather than averaging or summing the embeddings?
        Concatenation preserves position information — `(a, b)` is a
        different vector from `(b, a)` after concat, but the same after
        averaging. A trigram model that couldn't tell `a b` from `b a` would
        be useless. Modules 05 and 07 introduce more elegant ways of
        injecting position; concatenation is the brute-force-but-correct
        version that fits in one MLP.
        """
        e = self.embed(ctx_ids)  # (B, N, D)
        flat = e.view(e.shape[0], -1)  # (B, N*D)
        h = torch.tanh(self.hidden(flat))  # (B, H)
        return self.output(h)  # (B, V)

