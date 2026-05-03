"""Bigram language models — counts-based and neural.

A bigram model predicts the next token from the immediately preceding one:

    P(x_t | x_{t-1})

Two ways to build that distribution:

  - `CountsBigramLM`: walk the corpus once, tally how often each ordered
    pair (a, b) appears, normalize each row to a probability distribution.
    No gradient descent. No parameters in the deep-learning sense — just
    counts.

  - `NeuralBigramLM`: an embedding lookup followed by a linear projection
    to vocab logits. Trained with cross-entropy and SGD. Same expressive
    power as the counts model (in fact, it CAN learn the same distribution
    given enough data and capacity), but parameterized so it can be the
    starting point of a stack.

Both implement `.logits(ctx_ids) → (batch, vocab_size)`. That's the unified
interface used by `perplexity`, `sample`, and the training loop in
`train.py`. Both also set `.context_length = 1` so model-agnostic helpers
know how much history to feed.

Boilerplate is implemented. The two scaffolded methods on `CountsBigramLM`
(`fit`, `logits`) and the one on `NeuralBigramLM` (`forward`) are the lesson.
"""
from __future__ import annotations

from collections.abc import Iterable

import torch

from g2c.embeddings import TokenEmbedding
from g2c.nn import Linear, Module


class CountsBigramLM:
    """A bigram language model built from raw co-occurrence counts.

    The model is an integer table `counts` of shape `(vocab_size, vocab_size)`,
    where `counts[a, b]` is the number of times token `b` followed token `a`
    in the training corpus.

    Prediction:
        P(b | a)  =  (counts[a, b] + smoothing) / sum_b' (counts[a, b'] + smoothing)

    `smoothing` defaults to 1.0 (add-one / Laplace smoothing). Without it,
    any (a, b) pair never seen during training would have probability 0,
    which makes log-prob `-inf` and breaks perplexity.

    This class is NOT a `Module` — there's nothing to differentiate. It's
    a plain Python class with `.fit()` (one pass over the corpus) and
    `.logits()` (table lookup + log).
    """

    vocab_size: int
    smoothing: float
    counts: torch.Tensor    # (vocab_size, vocab_size), integer dtype
    context_length: int

    def __init__(self, vocab_size: int, smoothing: float = 1.0) -> None:
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.counts = torch.zeros(vocab_size, vocab_size, dtype=torch.long)
        self.context_length = 1

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

    def __call__(self, ctx_ids: torch.Tensor) -> torch.Tensor:
        return self.logits(ctx_ids)


class NeuralBigramLM(Module):
    """A neural bigram language model.

    Architecture: TokenEmbedding → Linear → vocab logits.

        ctx_ids ──→ embed ──→ proj ──→ logits
        (B,)        (B, D)    (B, V)

    Conceptually: the model learns one D-dimensional vector per token (the
    embedding), and one D-dimensional vector per OUTPUT class (a column of
    `proj.W`). The score that token `b` follows token `a` is the dot product
    of `embed(a)` with `proj.W[:, b]`. Softmax over the row of dot products
    gives a distribution.

    With a large enough D, this can perfectly represent the same conditional
    distribution that `CountsBigramLM` learns by counting. With smaller D
    it's a low-rank factorization — and that turns out to be a feature, not
    a bug, because it forces the model to share statistical strength across
    similar tokens.

    Trained with `train_lm()` from `train.py` against `CrossEntropyLoss`.
    """

    vocab_size: int
    embedding_dim: int
    embed: TokenEmbedding
    proj: Linear
    context_length: int

    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embed = TokenEmbedding(vocab_size, embedding_dim)
        self.proj = Linear(embedding_dim, vocab_size)
        self.context_length = 1

    def parameters(self) -> Iterable[torch.Tensor]:
        return [*self.embed.parameters(), *self.proj.parameters()]

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

    def logits(self, ctx_ids: torch.Tensor) -> torch.Tensor:
        """Alias for `forward` — provides the same interface as `CountsBigramLM`."""
        return self.forward(ctx_ids)
