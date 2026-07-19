"""Position-wise feed-forward network.

The transformer block has two sublayers: attention (which mixes
information ACROSS positions) and a feed-forward network (which mixes
information ACROSS channels at each position independently). The FFN is
a two-layer MLP applied to every position in parallel:

    y_t = W_2 · GELU(W_1 · x_t + b_1) + b_2     for every position t

The hidden dimension is conventionally `4 * embedding_dim` (set by
Vaswani et al. and inherited by every major transformer since). That
4× expansion is where most of a modern transformer's parameters live —
roughly two-thirds of the parameter count of a standard block.

`GELU` (Gaussian Error Linear Unit) is the activation. It's a smooth
relative of ReLU defined as `x * Phi(x)` where `Phi` is the standard
normal CDF. It outperforms ReLU empirically in transformer training and
is the activation used by GPT-2, GPT-3, and most modern transformers.
We use `torch.nn.functional.gelu` for the computation — same level of
abstraction as `torch.softmax` in attention.

Two things to internalize:

  * **The FFN is per-position.** It does not communicate across
    positions; that's attention's job. Every position passes through
    the SAME `W_1, W_2` weights independently — this is why the
    transformer is sometimes called "a stack of attention layers
    interleaved with per-token MLPs."
  * **Most parameters live here.** With `hidden_dim = 4 * D`, the FFN
    has `2 * 4 * D^2 = 8 * D^2` weight parameters versus the
    attention block's `4 * D^2`. In a standard transformer, roughly
    2/3 of the parameter count is FFN, 1/3 is attention.

Boilerplate (`__init__`, `parameters`) is implemented. The `forward`
method — three lines of Linear → GELU → Linear — is scaffolded.
"""
from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F  # noqa: F401 (for the student implementation)

from g2c.nn import Linear, Module


class FeedForward(Module):
    """Two-layer position-wise MLP with GELU activation.

    Args:
        embedding_dim: input/output channel dim (`D`).
        hidden_dim:    inner hidden dim. Defaults to `4 * embedding_dim`,
                       matching the standard transformer.
    """

    embedding_dim: int
    hidden_dim: int
    fc1: Linear
    fc2: Linear

    def __init__(
        self,
        embedding_dim: int,
        *,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * embedding_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.fc1 = Linear(embedding_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, embedding_dim)

    def parameters(self) -> Iterable[torch.Tensor]:
        return [
            *self.fc1.parameters(),
            *self.fc2.parameters(),
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Linear → GELU → Linear to each position independently.

        Args:
            x: tensor of shape `(..., embedding_dim)`. Typical:
                `(B, T, D)`. The function is point-wise across all
                leading dims — including position — so every position
                is processed independently with the same weights.

        Returns:
            Tensor of the same shape as `x`.

        Recipe:
            1. h = self.fc1(x)                  # (..., hidden_dim)
            2. h = F.gelu(h)                    # (..., hidden_dim)
            3. return self.fc2(h)               # (..., embedding_dim)

        That's it — the FFN is structurally simple. The pedagogical
        point is "the position dim flows through unchanged, and the
        channel dim expands to `hidden_dim` and contracts back to
        `embedding_dim`."

        Why GELU and not ReLU? GELU is smoother (differentiable
        everywhere, no kink at zero), which empirically gives slightly
        better gradient flow in transformer-shaped networks. It's the
        modern default. ReLU works fine too — Module 03's MLP used it.
        The difference is a few percent in convergence speed at the
        scales we'll train at.
        """
        # TODO
        raise NotImplementedError
