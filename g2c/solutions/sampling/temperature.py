# Hand-maintained solutions mirror — canonical, lives in `main` (see AGENTS.md).
"""Solutions for g2c.sampling.temperature pedagogical scaffolds.

Patched onto the scaffold targets by g2c.solutions.apply().
"""
from __future__ import annotations

import torch


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Divide logits by `temperature`.

    Args:
        logits: tensor of shape `(..., V)`. Operates on the last dim;
            any leading batch / position dims are preserved.
        temperature: a strictly positive float. `1.0` is identity;
            smaller values sharpen, larger values flatten.

    Returns:
        Tensor of the same shape as `logits`, equal to `logits / temperature`.

    Raises:
        ValueError: if `temperature <= 0`. Use `argmax` directly for
            greedy decoding rather than passing `temperature=0` here.

    Recipe:
        1. if temperature <= 0:
               raise ValueError(...)
        2. return logits / temperature

    Why is this its own function instead of an inline divide? Because
    naming the operation matters when you compose four warpers in a
    row, and because the `temperature <= 0` guard centralizes a class
    of bugs that otherwise turn up as `nan`s much later in the
    pipeline.
    """
    
    if temperature <= 0:
        raise ValueError(f'Temperature must be > 0, got {temperature}')
    
    return logits / temperature
