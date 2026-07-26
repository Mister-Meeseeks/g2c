# Hand-maintained solutions mirror — canonical, lives in `main` (see AGENTS.md).
"""Solutions for g2c.lora.inject pedagogical scaffolds.

Patched onto the scaffold targets by g2c.solutions.apply().
"""
from __future__ import annotations

import torch

from g2c.lora.inject import count_parameters  # noqa: F401 (sibling context)
from g2c.lora.layer import LoRALinear  # noqa: F401 (used by the impl)


def mark_only_lora_trainable(model: torch.nn.Module) -> tuple[int, int]:
    """Freeze the whole model except the LoRA `A`/`B` matrices.

    After this call, exactly the `lora_A` and `lora_B` parameters of
    every `LoRALinear` in the tree have `requires_grad=True`; every
    other parameter — including the wrapped `base` layers *inside* the
    `LoRALinear`s — has `requires_grad=False`.

    Returns:
        `(trainable, total)` from `count_parameters(model)`, so the
        caller sees the ratio the moment the freeze lands.

    Recipe:

        1. Freeze everything:

               for p in model.parameters():
                   p.requires_grad_(False)

           A parameter with requires_grad=False is invisible to
           autograd: no gradient is computed for it, `.grad` stays
           None, and an optimizer that skips grad-less parameters
           never touches it. This one flag is where ALL of LoRA's
           optimizer-memory savings come from.

        2. Re-enable only the adapters:

               for m in model.modules():
                   if isinstance(m, LoRALinear):
                       m.lora_A.requires_grad_(True)
                       m.lora_B.requires_grad_(True)

           Freeze-then-unfreeze (rather than deciding per-parameter in
           one pass) is deliberate: `lora_A` and `lora_B` are the
           allowlist, and everything not on it — however the host model
           names its weights — ends up frozen.

        3. return count_parameters(model)
    """
    for p in model.parameters():
        p.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.lora_A.requires_grad_(True)
            m.lora_B.requires_grad_(True)
    return count_parameters(model)
