# Hand-maintained solutions mirror — canonical, lives in `main` (see AGENTS.md).
"""Solutions for g2c.nn.loss pedagogical scaffolds.

Patched onto the scaffold targets by g2c.solutions.apply().
"""
from __future__ import annotations

import torch

from g2c.nn.loss import CrossEntropyLoss
from g2c.nn.modules import Module


class _CrossEntropyLossImpl:  # patched onto CrossEntropyLoss by apply()
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Return the mean cross-entropy loss between logits and targets.

        See the docstring for the mathematical definition and numerical stability
        trick.

        Hint: follow the implementation outline in the docstring, step by step.
        """
        m = logits.max(dim=-1, keepdim=True).values
        logsumexp = m.squeeze(-1) + torch.log(torch.exp(logits - m).sum(dim=-1))
        correct_class_logits = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
        per_sample_loss = -correct_class_logits + logsumexp
        return per_sample_loss.mean()

