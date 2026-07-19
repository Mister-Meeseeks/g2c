# Hand-maintained solutions mirror — canonical, lives in `main` (see AGENTS.md).
"""Solutions for g2c.transformer.layer_norm pedagogical scaffolds.

Patched onto the scaffold targets by g2c.solutions.apply().
"""
from __future__ import annotations

from collections.abc import Iterable

import torch

from g2c.nn import Module
from g2c.transformer.layer_norm import LayerNorm


class _LayerNormImpl:  # patched onto LayerNorm by apply()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize `x` over the last dim, then apply learned affine.

        Args:
            x: tensor of shape `(..., embedding_dim)`. Typical shapes are
                `(B, T, D)` (transformer activations) or `(B, D)` (a
                single per-example vector). The leading dims are treated
                as batch dims — every leading-dim slice is normalized
                independently.

        Returns:
            Tensor of the same shape as `x`.

        Recipe:
            1. mean = x.mean(dim=-1, keepdim=True)                   # (..., 1)
            2. var  = x.var(dim=-1, unbiased=False, keepdim=True)    # (..., 1)

               `unbiased=False` is the population variance (divide by N,
               not N-1). PyTorch's `nn.LayerNorm` uses the population
               variance; using `unbiased=True` here is a subtle bug that
               causes a small mismatch with the standard implementation
               and shows up only at small `D`.

            3. x_hat = (x - mean) / sqrt(var + self.eps)             # (..., D)

               Both `mean` and `var` broadcast against `x` because
               `keepdim=True` preserved the trailing 1 in their shapes.

            4. return self.gamma * x_hat + self.beta                 # (..., D)

               `gamma` and `beta` are 1-D `(D,)` tensors; they broadcast
               against `(..., D)` so each channel gets its own scale and
               shift.
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta

