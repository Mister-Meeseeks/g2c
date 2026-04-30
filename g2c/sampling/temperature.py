"""Temperature scaling — the most basic logit warper.

Temperature is a single scalar that rescales logits *before* the softmax:

    softmax(logits / T)

The behavior at the limits is decisive:

    T → 0⁺   the distribution collapses onto its argmax (one-hot)
    T  = 1   identity — the model's "native" distribution
    T → ∞    the distribution flattens toward uniform

Three things to internalize:

  * **Temperature scales logits, not probabilities.** Dividing the logits
    by `T` and then softmaxing is mathematically distinct from softmaxing
    first and then sharpening / flattening the probabilities. Logit-space
    is the right place because the softmax's exponential turns linear
    rescaling into multiplicative reshaping of the distribution — exactly
    what we want.

  * **Temperature never reorders tokens.** Whatever logit was largest
    stays largest; whatever was smallest stays smallest. Temperature is
    a monotone transformation of probabilities. Top-k / top-p change
    *which* tokens are eligible; temperature only changes *how peaked*
    the eligible distribution is.

  * **`T = 0` is undefined here.** Division by zero. The "greedy" decode
    path lives in `generate.py`, not here — when the caller wants
    argmax, the generator skips temperature and multinomial entirely
    and takes `argmax`. Forcing the warper to be a clean, total
    function (`T > 0` always) keeps the composition pipeline simple.

Scaffolded — the docstring's recipe is two lines.
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
    # TODO
    raise NotImplementedError
