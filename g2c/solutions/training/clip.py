# Hand-maintained solutions mirror — canonical, lives in `main` (see AGENTS.md).
"""Solutions for g2c.training.clip pedagogical scaffolds.

Patched onto the scaffold targets by g2c.solutions.apply().
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
    params = [p for p in params if p.grad is not None]
    if not params:
        return 0.0
    
    with torch.no_grad():
        total_norm_sq = sum((p.grad ** 2).sum() for p in params)
        total_norm = torch.sqrt(total_norm_sq).item()
        if total_norm > max_norm:
            scale = max_norm / (total_norm + 1e-6)
            for p in params:
                p.grad.mul_(scale)
                
    return total_norm
