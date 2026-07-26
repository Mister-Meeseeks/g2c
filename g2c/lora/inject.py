"""Injection and freezing — putting LoRA layers into a real model tree.

`LoRALinear` adapts one layer. A real model has hundreds of linear
layers, addressed by name (`model.layers.17.self_attn.q_proj`), and LoRA
practice is to adapt a *named subset* — classically the attention
projections `q_proj` and `v_proj` — and freeze everything else.

Two functions, two separate decisions:

  * `inject_lora` — the tree surgery. Walks the model, replaces every
    matching `torch.nn.Linear` with a `LoRALinear` wrapping it. This is
    plumbing, so it is implemented for you.
  * `mark_only_lora_trainable` — the freeze. Turns off `requires_grad`
    everywhere except the adapter matrices. This is scaffolded, because
    it IS the lesson: `requires_grad` is the valve on the gradient flow
    you built in Module 01, and the memory economics of LoRA follow
    entirely from which parameters the optimizer has to track.

Injecting without freezing trains the whole model with a LoRA bolted
on — the loss goes down either way, and only the parameter counts tell
the truth. That is the module's headline pitfall, and it is why the two
steps are not one function.
"""
from __future__ import annotations

from collections.abc import Iterable

import torch

from .layer import LoRALinear


def inject_lora(
    model: torch.nn.Module,
    target_names: Iterable[str],
    *,
    rank: int,
    alpha: float | None = None,
) -> list[str]:
    """Replace every matching child `torch.nn.Linear` with a `LoRALinear`.

    Implemented for you — module-tree surgery is plumbing, not the
    concept under study.

    Args:
        model: the model to modify in place. For BaseLM, pass the loaded
            adapter (`artifact.model`); the walk descends into it.
        target_names: attribute names to adapt wherever they appear,
            e.g. `{"q_proj", "v_proj"}`. Matching is by the child's
            attribute name, exactly as in the LoRA paper's practice of
            adapting a named subset of projections.
        rank: forwarded to `LoRALinear`.
        alpha: forwarded to `LoRALinear`.

    Returns:
        The dotted names of every replaced layer, in tree order — print
        it; seeing 64 replacements across 32 layers makes the surgery
        concrete.

    Raises:
        ValueError: if nothing matched. A typo in a projection name
            ("qproj") would otherwise silently train nothing.
    """
    targets = set(target_names)
    # Collect first, then mutate: replacing children while named_modules()
    # is mid-walk would iterate a tree that no longer exists.
    to_replace: list[tuple[torch.nn.Module, str, str, torch.nn.Linear]] = []
    for parent_name, parent in model.named_modules():
        for child_name, child in parent.named_children():
            if child_name not in targets:
                continue
            if not isinstance(child, torch.nn.Linear) or isinstance(
                child, LoRALinear
            ):
                continue
            dotted = f"{parent_name}.{child_name}" if parent_name else child_name
            to_replace.append((parent, child_name, dotted, child))
    if not to_replace:
        raise ValueError(
            f"inject_lora matched no torch.nn.Linear children named "
            f"{sorted(targets)}. Check the names against "
            "[n for n, _ in model.named_modules()]."
        )
    replaced: list[str] = []
    for parent, child_name, dotted, child in to_replace:
        setattr(parent, child_name, LoRALinear(child, rank=rank, alpha=alpha))
        replaced.append(dotted)
    return replaced


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Return `(trainable, total)` parameter counts.

    Implemented for you. `trainable` counts parameters with
    `requires_grad=True`; `total` counts everything.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


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
    # TODO
    raise NotImplementedError
