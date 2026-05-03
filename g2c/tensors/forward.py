"""A linear layer and a numerically stable softmax.

Together these form the forward pass of a single-layer classifier — the
simplest model that maps an input vector to a probability distribution
over classes. Both functions use raw `torch.Tensor` ops; we don't track
gradients in this module.

The point of the exercise is shape discipline (always know the shape of
every tensor) and the numerical stability trick that makes softmax safe
on real-world (large-magnitude) logits.

Search for `# TODO` to find the spots that need work.
"""
from __future__ import annotations

import torch


def linear(x: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Apply a linear layer: y = x @ W + b.

    Shapes:
        x: (batch, in_features)
        W: (in_features, out_features)
        b: (out_features,)              — broadcasts across the batch dim

    Returns:
        y: (batch, out_features)

    The bias `b` is added to every row via broadcasting. (NB: `torch.nn.Linear`
    stores W as (out, in) and computes `x @ W.T + b`. Our convention here is
    (in, out) which makes the matmul shape-rule more obvious.)
    """
    return torch.matmul(x, W) + b


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically stable softmax along `dim`.

    Mathematically:
        softmax(x_i) = exp(x_i) / sum_j exp(x_j)

    Naively this overflows when any x_i is large (exp(89.0) > float32 max).
    The fix: subtract the max along `dim` before exponentiating. softmax is
    shift-invariant — softmax(x) == softmax(x - c) for any constant c —
    so the result is mathematically identical but stays in float range.

    Steps:
        m = x.max(dim=dim, keepdim=True).values     # max along dim, kept-shape
        e = torch.exp(x - m)                         # safe: argument <= 0
        return e / e.sum(dim=dim, keepdim=True)

    Args:
        x: any shape.
        dim: the axis along which to apply softmax. Default -1 (last axis).

    Returns:
        Same shape as x. Values along `dim` sum to 1.
    """
    max = x.max(dim=dim, keepdim=True).values
    e = torch.exp(x - max)
    return e / e.sum(dim=dim, keepdim=True)
