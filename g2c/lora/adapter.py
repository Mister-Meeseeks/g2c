"""What you carry away from a LoRA run — and how the trainer sees it.

Implemented for you end to end; nothing here is scaffolded. Two small
pieces:

  * `lora_state_dict` / `load_lora_state_dict` — the adapter file. A
    LoRA run's durable output is the A/B matrices alone, a few
    megabytes riding on a base model of gigabytes. Saving anything more
    would defeat the point; the exercise notebook makes you look at the
    two file sizes side by side.

  * `LoRAModel` — the trainer-facing view. `SFTTrainer` builds its
    optimizer from `model.parameters()`, and the course `AdamW`
    allocates `m`/`v` state eagerly for every tensor it is handed
    (`g2c/training/optim.py`). Hand it the raw BaseLM tree and it will
    dutifully allocate two full copies of a frozen model — the exact
    memory LoRA exists to save. The course `g2c.nn.Module` contract
    reads "parameters() returns all *trainable* parameters", and this
    wrapper is that contract applied to a torch tree: the trainer sees
    only what training can move.
"""
from __future__ import annotations

import torch

from g2c.nn import Module, resolve_device

from .layer import LoRALinear  # noqa: F401  (re-exported context for notebooks)

_LORA_SUFFIXES = (".lora_A", ".lora_B", "lora_A", "lora_B")


def _is_lora_param_name(name: str) -> bool:
    return name.endswith("lora_A") or name.endswith("lora_B")


def lora_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Extract only the adapter weights, detached and on CPU.

    The keys are the model's own dotted parameter names, so an adapter
    saved from one BaseLM loads into any other BaseLM injected with the
    same `target_names` and `rank`.
    """
    return {
        name: p.detach().cpu().clone()
        for name, p in model.named_parameters()
        if _is_lora_param_name(name)
    }


def load_lora_state_dict(
    model: torch.nn.Module,
    state: dict[str, torch.Tensor],
) -> None:
    """Copy saved adapter weights into a LoRA-injected model, in place.

    Strict by design: the injected model's adapter parameters and the
    saved state must name exactly the same set. A missing key means the
    model was injected with different targets or rank than the adapter
    was trained with — silently loading the intersection would "work"
    and behave wrong.
    """
    own = {
        name: p
        for name, p in model.named_parameters()
        if _is_lora_param_name(name)
    }
    missing = sorted(set(own) - set(state))
    unexpected = sorted(set(state) - set(own))
    if missing or unexpected:
        raise ValueError(
            "adapter/state mismatch — was the model injected with the same "
            f"target_names and rank? missing={missing[:4]}, "
            f"unexpected={unexpected[:4]}"
        )
    with torch.no_grad():
        for name, p in own.items():
            source = state[name]
            if source.shape != p.shape:
                raise ValueError(
                    f"adapter tensor {name} has shape {tuple(source.shape)}, "
                    f"model expects {tuple(p.shape)} — rank mismatch?"
                )
            p.copy_(source.to(device=p.device, dtype=p.dtype))


class LoRAModel(Module):
    """A course-`Module` view of a LoRA-injected torch model.

    Wrap the injected (and frozen) model in this before handing it to
    `SFTTrainer`. Forward passes go straight through; the only thing
    this class changes is what `parameters()` exposes — the trainable
    parameters, per the course contract — so the optimizer's state is
    sized to the adapter, not the base model.

    Generation does not need the wrapper: after `inject_lora`, the
    original model object already computes with the LoRA delta live
    inside it, so sample from it exactly as in Module 13.

    Attribute access falls through to the wrapped model, so
    `model.vocab_size`, `model.max_seq_len`, etc. keep working.
    """

    def __init__(self, base: torch.nn.Module) -> None:
        self.base = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x)

    def parameters(self) -> list[torch.Tensor]:
        return [p for p in self.base.parameters() if p.requires_grad]

    def to(self, device: str | torch.device | None = "auto") -> LoRAModel:
        self.base.to(resolve_device(device))
        return self

    def train(self, mode: bool = True) -> LoRAModel:
        self.base.train(mode)
        self.training = bool(mode)
        return self

    def __getattr__(self, name: str):
        return getattr(self.base, name)
