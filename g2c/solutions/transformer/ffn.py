# Hand-maintained solutions mirror — canonical, lives in `main` (see AGENTS.md).
"""Solutions for g2c.transformer.ffn pedagogical scaffolds.

Patched onto the scaffold targets by g2c.solutions.apply().
"""
from __future__ import annotations

from collections.abc import Iterable
import torch
import torch.nn.functional as F
from g2c.nn import Linear, Module

from g2c.transformer.ffn import FeedForward


class _FeedForwardImpl:  # patched onto FeedForward by apply()
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
        h = self.fc1(x)
        h = F.gelu(h)
        return self.fc2(h)

