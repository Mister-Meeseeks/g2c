# Hand-maintained solutions mirror — canonical, lives in `main` (see AGENTS.md).
"""Solutions for g2c.nn.modules pedagogical scaffolds.

Patched onto the scaffold targets by g2c.solutions.apply().
"""
from __future__ import annotations

import math
from collections.abc import Iterable
import torch
from g2c.nn.device import resolve_device

from g2c.nn.modules import Linear, ReLU, Sequential, Sigmoid, Tanh


class _LinearImpl:  # patched onto Linear by apply()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute y = x @ W + b. The bias broadcasts across the batch dim.

        Hint: this is a one-liner. The real lesson is that this is one line.
        """
        return torch.matmul(x, self.W) + self.b



class _ReLUImpl:  # patched onto ReLU by apply()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return max(0, x), element-wise.

        Hint: `x.clamp(min=0)` or `torch.relu(x)` (the functional form, not
        `torch.nn.ReLU` — that's the high-level abstraction we're avoiding).
        """
        return torch.relu(x)



class _TanhImpl:  # patched onto Tanh by apply()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return tanh(x), element-wise. Use `torch.tanh(x)`."""
        return torch.tanh(x)



class _SigmoidImpl:  # patched onto Sigmoid by apply()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return sigmoid(x), element-wise. Use `torch.sigmoid(x)`."""
        return torch.sigmoid(x)



class _SequentialImpl:  # patched onto Sequential by apply()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run x through each layer in order. Output of layer i is input of layer i+1.

        Hint: a 3-line implementation with a for-loop.
        """
        for layer in self.layers:
            x = layer(x)
        return x

