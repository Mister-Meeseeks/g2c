# Hand-maintained solutions mirror — canonical, lives in `main` (see AGENTS.md).
"""Solutions for g2c.lm.bigram pedagogical scaffolds.

Patched onto the scaffold targets by g2c.solutions.apply().
"""
from __future__ import annotations

from collections.abc import Iterable

import torch

from g2c.embeddings import TokenEmbedding
from g2c.lm.bigram import CountsBigramLM, NeuralBigramLM
from g2c.nn import Linear, Module


class _CountsBigramLMImpl:  # patched onto CountsBigramLM by apply()
    def fit(self, ids: torch.Tensor) -> None:
        """Tally bigram counts from a 1-D sequence of token IDs.

        Args:
            ids: 1-D LongTensor of token IDs, length N. The corpus.

        Effect:
            Increments `self.counts[a, b]` for every adjacent pair `(a, b)`
            in `ids`. Calling `.fit()` repeatedly accumulates — useful if
            you want to fit on multiple corpora.

        Recipe:
            1. Form pairs: `prev = ids[:-1]`, `nxt = ids[1:]`.
            2. For each (a, b), do `self.counts[a, b] += 1`. The vectorized
               way to do this without a Python loop is:
                   self.counts.index_put_((prev, nxt), torch.ones_like(prev), accumulate=True)
               A plain `for` loop is also fine — the corpus is small.

        N adjacent pairs come from a length-N sequence: there are `N-1` of them.
        """
        for i in range(len(ids) - 1):
            prev = ids[i]
            next = ids[i + 1]
            self.counts[prev, next] += 1

    def logits(self, ctx_ids: torch.Tensor) -> torch.Tensor:
        """Return log-probabilities of next token, given the previous token.

        Args:
            ctx_ids: integer tensor of shape `(batch,)` or `(batch, 1)`. The
                last (and only) token of context for each example.

        Returns:
            (batch, vocab_size) tensor of LOG-PROBABILITIES (not unnormalized
            logits — they really are log p, summing to 1 in probability space).

        Recipe:
            1. If `ctx_ids` has shape `(batch, 1)`, squeeze the trailing dim
               (so `self.counts[ctx_ids]` returns `(batch, V)` not `(batch, 1, V)`).
            2. Look up the relevant rows: `row_counts = self.counts[ctx_ids]`,
               shape `(batch, vocab_size)`.
            3. Add smoothing: `smoothed = row_counts.float() + self.smoothing`.
            4. Normalize each row: divide by row sum (kept-dim).
            5. Return `log(probs)`.

        Why log probabilities and not raw probabilities? Cross-entropy and
        perplexity both want logs. And returning logs (vs. raw probs) keeps
        the interface consistent with `NeuralBigramLM.forward`, which returns
        unnormalized logits — both can be fed to a softmax/log-softmax for
        sampling and to `gather` for evaluation.
        """
        if ctx_ids.dim() == 2:
            ctx_ids = ctx_ids.squeeze(-1)
        row_counts = self.counts[ctx_ids]  # (batch, vocab_size)
        smoothed = row_counts.float() + self.smoothing
        probs = smoothed / smoothed.sum(dim=-1, keepdim=True)
        return torch.log(probs)



class _NeuralBigramLMImpl:  # patched onto NeuralBigramLM by apply()
    def forward(self, ctx_ids: torch.Tensor) -> torch.Tensor:
        """Compute next-token logits.

        Args:
            ctx_ids: integer tensor of shape `(batch,)` or `(batch, 1)` —
                the previous token for each example.

        Returns:
            (batch, vocab_size) tensor of unnormalized logits.

        Recipe:
            1. If `ctx_ids` has shape `(batch, 1)`, squeeze the trailing dim
               (so `embed` returns `(batch, D)` not `(batch, 1, D)`):
                   if ctx_ids.dim() == 2:
                       ctx_ids = ctx_ids.squeeze(-1)
            2. `e = self.embed(ctx_ids)`        # (batch, embedding_dim)
            3. Return `self.proj(e)`            # (batch, vocab_size)
        """
        if ctx_ids.dim() == 2:
            ctx_ids = ctx_ids.squeeze(-1)
        e = self.embed(ctx_ids)  # (batch, embedding_dim)
        return self.proj(e)     # (batch, vocab_size)

