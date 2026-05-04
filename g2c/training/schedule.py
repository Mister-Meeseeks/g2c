"""Learning-rate schedule: linear warmup then cosine decay.

Pretraining a transformer with a constant learning rate works at toy
scale but breaks down once the model is big enough to matter. The
canonical schedule, used by GPT-2/3, Llama, and essentially every
modern open LM, has two phases:

  1. **Warmup.** For the first few hundred to few thousand steps,
     ramp `lr` linearly from 0 up to `max_lr`. Without warmup, the
     very first updates — taken with random-init weights and large
     gradients — can knock the model into a region that's hard to
     recover from. Warmup gives the model a chance to find a sane
     local geometry before you push it hard.

  2. **Cosine decay.** From the end of warmup to the end of training,
     decay `lr` along the right half of a cosine curve from `max_lr`
     down to `min_lr`. Cosine starts off slow, accelerates the decay,
     then slows down again at the bottom — empirically a better
     balance of "make fast progress" and "polish the final loss" than
     either linear decay or step decay.

```
  lr
   ▲
   │              ╮
   │           ╱   ╰─╮
   │         ╱        ╰─╮
   │       ╱             ╰─╮
   │     ╱                  ╰─╮
   │   ╱                      ╰─╮___
   │ ╱
   └─────────────┬────────────────────► step
   0    warmup_steps               max_steps
```

Two design choices to internalize:

  * **The warmup starts near 0, not from `min_lr`.** Some
    implementations make step 0 exactly 0. This course uses the
    common `(step + 1) / warmup_steps` convention: the first optimizer
    update uses `max_lr / warmup_steps`, and the last warmup step
    reaches exactly `max_lr`.

  * **`step > max_steps` keeps `lr = min_lr`.** Don't crash, don't
    grow — if training continues past the schedule horizon, the lr
    just stays at the floor. In practice you'd stop at `max_steps`,
    but the function should be safe to call past it.

The whole schedule is pure arithmetic on `step` — no internal state,
no PyTorch dependence. Three to five lines. Scaffolded.
"""
from __future__ import annotations

import math


def cosine_with_warmup(
    step: int,
    *,
    warmup_steps: int,
    max_steps: int,
    max_lr: float,
    min_lr: float = 0.0,
) -> float:
    """Compute the learning rate at the given training step.

    Args:
        step: current training step (0-indexed).
        warmup_steps: number of linear-warmup steps. `0` disables warmup.
        max_steps: total training steps (the end of cosine decay).
        max_lr: the peak learning rate (reached at the end of warmup).
        min_lr: the floor learning rate (reached at `step = max_steps`
            and held thereafter).

    Returns:
        The learning rate as a float.

    Recipe:
        1. If `step < warmup_steps`:
               # Linear ramp from 0 to max_lr over warmup_steps.
               return max_lr * (step + 1) / warmup_steps
               # The `+1` is conventional: at step `warmup_steps - 1`
               # (the last warmup step) the lr is exactly `max_lr`,
               # not `max_lr * (warmup_steps - 1)/warmup_steps`. Either
               # convention is defensible; this one matches nanoGPT.

        2. If `step >= max_steps`:
               return min_lr

        3. Otherwise (cosine decay phase):
               progress = (step - warmup_steps) / (max_steps - warmup_steps)
               # progress ∈ [0, 1]; 0 at end of warmup, 1 at max_steps.
               coeff = 0.5 * (1.0 + cos(pi * progress))
               # coeff ∈ [0, 1]; 1 at progress=0, 0 at progress=1.
               return min_lr + coeff * (max_lr - min_lr)

    Sanity values to check by hand:
        warmup_steps=10, max_steps=110, max_lr=1e-3, min_lr=0.0
        - step=0:    1e-4  (first warmup step; 1/10 of max)
        - step=9:    1e-3  (last warmup step; equals max_lr)
        - step=10:   1e-3  (first cosine step; cos(0) = 1, so still max_lr)
        - step=60:   5e-4  (halfway through cosine; cos(π/2) = 0 → 0.5)
        - step=110:  0.0   (end of cosine; cos(π) = -1, so coeff = 0)
        - step=999:  0.0   (past max_steps, held at min_lr)
    """
    if warmup_steps > 0 and step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    elif step >= max_steps:
        return min_lr
    else:
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + coeff * (max_lr - min_lr)
