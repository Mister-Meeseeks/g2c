# Hand-maintained solutions mirror — canonical, lives in `main` (see AGENTS.md).
"""Solutions for g2c.embeddings.token pedagogical scaffolds.

Patched onto the scaffold targets by g2c.solutions.apply().
"""
from __future__ import annotations

import math
from collections.abc import Iterable
import torch
from g2c.nn import Module

from g2c.embeddings.token import TokenEmbedding


class _TokenEmbeddingImpl:  # patched onto TokenEmbedding by apply()
    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Look up embeddings for each token ID.

        Args:
            ids: integer tensor of any shape. Typical: (batch, seq_len).

        Returns:
            Tensor with shape `ids.shape + (embedding_dim,)`.

        The whole point of this method is to internalize that an "embedding
        lookup" is just integer indexing into a learnable table. Hint: it's
        a one-liner. The autograd will correctly route gradients back to the
        rows of `self.weight` that were touched.
        """
        return self.weight[ids]

