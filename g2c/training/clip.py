"""Gradient clipping by global norm.

Pretraining a transformer occasionally produces a single batch with
a wildly large gradient — a numerical edge case (a near-zero softmax
denominator, an unusually peaked attention pattern, an outlier token
sequence). One bad step with a normal-sized lr can ruin a run that
was otherwise on track. The standard defense is to clip the gradient
norm: compute the global L2 norm of all gradients, and if it exceeds
a threshold `max_norm`, scale every gradient down so the norm equals
`max_norm` exactly.

    g_total = sqrt( ∑_p ||p.grad||² )    # over every parameter

    if g_total > max_norm:
        scale = max_norm / g_total
        for p in params: p.grad.mul_(scale)

Two things to internalize:

  * **The norm is GLOBAL, not per-parameter.** A model with one
    enormous gradient and many small ones gets the *whole vector*
    scaled down by the same factor. The relative magnitudes between
    parameter gradients are preserved — clipping rescales the
    direction without changing it. This is the property that makes
    clipping safe: it never reverses the sign of a gradient or
    makes the loss go up; it just shortens the step.

  * **No-op when below the threshold.** If `g_total ≤ max_norm`,
    nothing happens. Clipping should be invisible on the vast
    majority of well-behaved steps and only kick in on the
    pathological ones. A typical pretraining run clips on maybe
    1–10% of steps depending on `max_norm`.

The function follows PyTorch's `torch.nn.utils.clip_grad_norm_`
convention: in-place mutation of `.grad`, returning the *pre-clip*
total norm so the caller can log it.

Scaffolded — about five lines.
"""
from __future__ import annotations

from collections.abc import Iterable

import torch


def clip_grad_norm_(
    params: Iterable[torch.Tensor],
    max_norm: float,
) -> float:
    """Rescale gradients so their global L2 norm is at most `max_norm`.

    Args:
        params: iterable of parameter tensors. Tensors with `.grad is
            None` are skipped (they didn't participate in this step).
        max_norm: the threshold. If the global gradient norm is above
            this, every gradient is multiplied by `max_norm / total_norm`.

    Returns:
        The PRE-clip total norm as a Python float. Useful for logging:
        you can plot it over training and see how often clipping
        actually fires.

    Recipe:
        1. params = [p for p in params if p.grad is not None]
           # Materialize the iterable once — we'll loop twice.
           # If `params` is empty, return 0.0.

        2. # Compute the global L2 norm: sqrt( sum of ||p.grad||² ).
           total_norm_sq = sum( (p.grad ** 2).sum() for p in params )
           total_norm = sqrt(total_norm_sq).item()

        3. # If we're under the threshold, nothing to do.
           if total_norm > max_norm:
               scale = max_norm / (total_norm + 1e-6)
               # The +1e-6 is a defensive divisor floor — it never
               # matters when total_norm > max_norm since we're
               # already nonzero, but it removes a class of NaN bugs
               # in degenerate edge cases.
               for p in params:
                   p.grad.mul_(scale)

        4. return total_norm

    Implementation notes:
      - Use `torch.no_grad()` (or operate on `.grad` directly — it's
        already a leaf tensor) so the rescaling itself isn't tracked
        by autograd. The simplest version uses `.mul_` on `.grad`
        which doesn't go through autograd.
      - `(p.grad ** 2).sum()` is fine; `p.grad.norm()` is also fine.
        Either is two ops.
    """
    # TODO
    raise NotImplementedError
