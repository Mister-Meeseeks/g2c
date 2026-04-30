"""Bengio-style neural probabilistic language model.

A bigram model only looks at the previous token. That ceiling shows up
quickly: most language patterns span multiple tokens, and a model that
only remembers one token of history will plateau at a perplexity floor
set by the bigram statistics of the corpus.

Bengio et al. (2003) showed how to extend this: take the previous N tokens,
look up an embedding for each, concatenate them into one big vector, pass
through a hidden layer (with `tanh`), then project to vocab logits.

    ctx_ids   ─→  embed   ─→  flatten  ─→  hidden  ─→  tanh  ─→  output  ─→  logits
    (B, N)        (B,N,D)     (B, N·D)     (B, H)       (B, H)     (B, V)

That's it. The whole architecture fits on a postcard. With `context_length=2`
this is a "trigram" model — it predicts the third token from the previous two.

It's also the direct conceptual ancestor of the transformer: a fixed-context
neural language model. The transformer replaces the fixed-window concatenate-
and-MLP with attention over a variable window — but the prediction objective
and the basic recipe (embed → process → project → softmax over vocab) are
identical.

Boilerplate is implemented. The forward pass — where the architecture above
becomes code — is scaffolded.
"""
from __future__ import annotations

from collections.abc import Iterable

import torch

from g2c.embeddings import TokenEmbedding
from g2c.nn import Linear, Module


class MLPLanguageModel(Module):
    """Bengio-style fixed-context neural language model.

    Args:
        vocab_size: size of the token vocabulary.
        context_length: number of previous tokens used as input. With
            `context_length=2`, this is a trigram model (predict the 3rd
            from the previous 2).
        embedding_dim: dimensionality of the per-token embedding.
        hidden_dim: dimensionality of the hidden layer.

    Parameter count:
        embedding table:  vocab_size * embedding_dim
        hidden weight:    (context_length * embedding_dim) * hidden_dim   + hidden_dim bias
        output weight:    hidden_dim * vocab_size                          + vocab_size bias
    """

    vocab_size: int
    context_length: int
    embedding_dim: int
    hidden_dim: int

    embed: TokenEmbedding
    hidden: Linear
    output: Linear

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        embedding_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.embed = TokenEmbedding(vocab_size, embedding_dim)
        self.hidden = Linear(context_length * embedding_dim, hidden_dim)
        self.output = Linear(hidden_dim, vocab_size)

    def parameters(self) -> Iterable[torch.Tensor]:
        return [
            *self.embed.parameters(),
            *self.hidden.parameters(),
            *self.output.parameters(),
        ]

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
        # TODO
        raise NotImplementedError

    def logits(self, ctx_ids: torch.Tensor) -> torch.Tensor:
        """Alias for `forward` — matches the `CountsBigramLM` interface."""
        return self.forward(ctx_ids)
