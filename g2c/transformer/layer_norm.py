"""Layer normalization.

Layer normalization stabilizes training by re-normalizing each token's
activation vector to have zero mean and unit variance, then applying a
learned per-channel scale and shift. It's applied along the LAST
(channel) dimension of the input — independently for every batch
element and every sequence position.

    given  x of shape (..., D):
        mean = x.mean(-1, keepdim=True)              # (..., 1)
        var  = x.var(-1, unbiased=False, keepdim=True)
        x_hat = (x - mean) / sqrt(var + eps)         # zero mean, unit var
        y    = gamma * x_hat + beta                  # learned affine

`gamma` (init 1) and `beta` (init 0) are both shape `(D,)` and are the
only learnable parameters. They give the model a way to *un-normalize*
back to whatever scale/shift is useful — without them, every layer's
output would be forced into a unit-variance distribution that the rest
of the network can't change.

The two non-obvious things to internalize:

  * **It normalizes over the LAST dim, not the batch dim.** That's the
    structural difference from BatchNorm — LayerNorm doesn't pool
    statistics across batch elements, so it works identically at
    inference time and training time, and works fine with batch_size = 1.

  * **The eps is load-bearing.** Without it, a near-constant input
    vector (e.g. all-zeros) gives a near-zero variance, and the divide
    by `sqrt(var)` blows up. `eps = 1e-5` keeps the divisor away from
    zero with negligible effect on normalized values.

Boilerplate (`__init__`, `parameters`) is implemented. The `forward`
method — the four-line normalization recipe — is scaffolded.
"""
from __future__ import annotations

from collections.abc import Iterable

import torch

from g2c.nn import Module


class LayerNorm(Module):
    """Layer normalization with learned per-channel affine.

    Args:
        embedding_dim: size of the channel dim being normalized (`D`).
        eps:           small constant added to the variance for numerical
                       stability. `1e-5` is the standard choice and what
                       PyTorch's own `nn.LayerNorm` uses by default.
    """

    embedding_dim: int
    eps: float
    gamma: torch.Tensor
    beta: torch.Tensor

    def __init__(self, embedding_dim: int, *, eps: float = 1e-5) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.eps = eps
        # gamma initialized to ones, beta to zeros — at init, LayerNorm
        # is exactly "subtract mean, divide by std" with no affine.
        gamma = torch.ones(embedding_dim)
        gamma.requires_grad_(True)
        self.gamma = gamma
        beta = torch.zeros(embedding_dim)
        beta.requires_grad_(True)
        self.beta = beta

    def parameters(self) -> Iterable[torch.Tensor]:
        return [self.gamma, self.beta]

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
