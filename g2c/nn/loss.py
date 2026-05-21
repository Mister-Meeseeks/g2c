"""Loss functions.

A loss function turns model output and ground truth into a single scalar that
gets smaller as the model gets better. Calling `.backward()` on that scalar
populates `.grad` on every parameter that contributed to it, which the
optimizer then uses to take a step.

`MSELoss` is implemented for you (it's a one-liner; no learning value in
re-deriving it). `CrossEntropyLoss` is scaffolded — it has a real numerical
stability trick that's worth working through by hand.
"""
from __future__ import annotations

import torch

from .modules import Module


class MSELoss(Module):
    """Mean squared error: mean((predictions - targets) ** 2).

    Standard loss for regression problems. Implemented for you.
    """

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return ((predictions - targets) ** 2).mean()


class CrossEntropyLoss(Module):
    """Cross-entropy loss for multi-class classification.

    Combines (numerically stable) log-softmax with negative log likelihood
    in one step.

    Inputs:
        logits:  (batch, num_classes)   raw, unnormalized scores
        targets: (batch,)               integer class indices in [0, num_classes)

    Output:
        scalar — mean over the batch of:
            loss_i = -log( softmax(logits_i)[ targets_i ] )
                   = -logits_i[ targets_i ] + log(sum_j exp(logits_i[j]))

    The numerical stability trick (essential — without it, naive softmax
    overflows on real-world logits):

        m_i = max_j logits_i[j]          # per-row max, kept-shape
        log(sum_j exp(logits_i[j])) = m_i + log(sum_j exp(logits_i[j] - m_i))

    The argument to `exp` is now ≤ 0, so it stays in float range. This is
    exactly the same shift-invariance trick as in stable softmax (Module 02)
    — same derivation, applied to the log-sum-exp.

    Implementation outline:
        1. m = logits.max(dim=-1, keepdim=True).values      # (batch, 1)
        2. logsumexp = m.squeeze(-1) + torch.log(torch.exp(logits - m).sum(dim=-1))
        3. correct_class_logits = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
        4. per_sample_loss = -correct_class_logits + logsumexp
        5. return per_sample_loss.mean()
    """

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Return the mean cross-entropy loss between logits and targets.

        See the docstring for the mathematical definition and numerical stability
        trick.

        Hint: follow the implementation outline in the docstring, step by step.
        """
        # TODO
        raise NotImplementedError
