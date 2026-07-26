"""LoRA — a trainable low-rank delta on a frozen linear layer (Module 13B).

Full fine-tuning updates every weight in the model, which means the
optimizer must hold state for every weight in the model. AdamW keeps two
extra tensors per parameter (`m` and `v`), so fine-tuning a model costs
roughly 3× its weight memory before a single activation is allocated.
That is fine at 360M parameters. It is the wall at 1–3B on a laptop.

LoRA (Hu et al., 2021) sidesteps the wall by freezing the pretrained
weight `W` and training a low-rank correction beside it:

    y = x @ W.T + b          (frozen — exactly the pretrained layer)
      + (x @ A) @ B * (alpha / r)     (trainable — the LoRA delta)

    A: (in_features, r)      r is the rank, typically 4–16
    B: (r, out_features)     alpha is a constant scale, typically = r

`A @ B` is a rank-`r` matrix the same shape as `W` (transposed), but it
costs `r * (in_features + out_features)` parameters instead of
`in_features * out_features`. At rank 8 on BaseLM's attention
projections, that is a few tenths of a percent of the model.

The initialization is asymmetric on purpose: `A` starts random (the same
`Uniform(-1/sqrt(in), 1/sqrt(in))` scheme as `g2c.nn.Linear` — see
Module 03, "Where weights start"), `B` starts at zero. The product
`A @ B` is therefore exactly zero, and the wrapped layer computes
*bit-identically* the same function as the frozen original until
training moves `B`. The exercise notebook asks why the zero must live in
`B` and not in both matrices — the answer is in Module 01's chain rule.

This class subclasses `torch.nn.Module`, not `g2c.nn.Module`. That is a
deliberate exception to the course house style: the model being adapted
is BaseLM, a Hugging Face `torch.nn.Module` tree, and the wrapper must
live inside that tree — moved by its `.to()`, found by its
`named_parameters()`. The adapter has to speak the host's dialect.

Boilerplate (`__init__`, initialization, `extra_repr`) is implemented.
`forward`, `merge`, and `unmerge` — the actual content of LoRA — are
scaffolded. Search for `# TODO`.
"""
from __future__ import annotations

import math

import torch


class LoRALinear(torch.nn.Module):
    """Wrap a `torch.nn.Linear` with a trainable low-rank delta.

    The wrapped layer (`self.base`) is left untouched — same object, same
    weights, same bias. The delta path adds `(x @ A) @ B * scaling` on
    top of its output. Freezing the base is *not* done here; that is
    `mark_only_lora_trainable`'s job (see `g2c.lora.inject`), and keeping
    the two steps separate is the point — wrapping and freezing are
    different decisions, and forgetting the second one is the module's
    headline pitfall.

    Args:
        base: the `torch.nn.Linear` to adapt. Stored as-is.
        rank: width of the low-rank bottleneck. The paper's `r`.
        alpha: numerator of the `alpha / rank` scale on the delta.
            Defaults to `rank`, making `scaling == 1.0` — the least
            surprising choice at course scale.

    Attributes:
        base: the wrapped linear layer, untouched.
        lora_A: `(in_features, rank)` — random at init.
        lora_B: `(rank, out_features)` — zero at init, so the delta
            starts as an exact no-op.
        scaling: `alpha / rank`, applied to the delta.
        merged: whether the delta is currently folded into
            `base.weight`. Toggled by `merge()` / `unmerge()`.
    """

    base: torch.nn.Linear
    lora_A: torch.nn.Parameter
    lora_B: torch.nn.Parameter
    rank: int
    alpha: float
    scaling: float
    merged: bool

    def __init__(
        self,
        base: torch.nn.Linear,
        *,
        rank: int,
        alpha: float | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(base, torch.nn.Linear):
            raise TypeError(
                f"LoRALinear wraps torch.nn.Linear, got {type(base).__name__}. "
                "(The course TransformerLM's g2c.nn.Linear is a different "
                "class — see the Module 13B lesson's scope notes.)"
            )
        if rank < 1:
            raise ValueError(f"rank must be >= 1, got {rank}.")
        self.base = base
        self.rank = rank
        self.alpha = float(alpha) if alpha is not None else float(rank)
        self.scaling = self.alpha / rank
        self.merged = False

        in_features = base.in_features
        out_features = base.out_features
        device = base.weight.device
        dtype = base.weight.dtype

        # A random, B zero: the product starts at exactly zero, but the
        # gradient path through B stays alive. Same init family as
        # g2c.nn.Linear (Module 03, "Where weights start").
        bound = 1.0 / math.sqrt(in_features)
        lora_A = torch.empty(
            in_features, rank, device=device, dtype=dtype
        ).uniform_(-bound, bound)
        self.lora_A = torch.nn.Parameter(lora_A)
        self.lora_B = torch.nn.Parameter(
            torch.zeros(rank, out_features, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the base layer's output plus the scaled low-rank delta.

        Args:
            x: `(..., in_features)` — any number of leading dims. The
                attention projections this wraps see `(B, T, in_features)`.

        Returns:
            `(..., out_features)`, equal to `base(x)` plus the delta —
            unless `self.merged` is True, in which case the delta already
            lives inside `base.weight` and must NOT be added again.

        Recipe:

            1. y = self.base(x)

            2. If self.merged, return y as-is. (After merge(), the delta
               is folded into base.weight; adding it again would double
               it.)

            3. delta = (x @ self.lora_A) @ self.lora_B

               Compute (x @ A) FIRST. Two skinny matmuls cost
               O(T * r * (in + out)); materializing A @ B first would
               build a full (in, out) matrix on every forward — the
               exact cost LoRA exists to avoid paying per-step.

            4. return y + delta * self.scaling
        """
        # TODO
        raise NotImplementedError

    def merge(self) -> None:
        """Fold the delta into `base.weight` in place, so `forward`
        costs exactly one linear layer — the deployment trick.

        Idempotent: calling `merge()` on an already-merged layer is a
        no-op, never a double-add.

        Recipe:

            1. If self.merged, return.

            2. with torch.no_grad():
                   self.base.weight += (self.lora_A @ self.lora_B).T * self.scaling

               Orientation: torch.nn.Linear stores weight as
               (out_features, in_features); A @ B is
               (in_features, out_features) — hence the transpose.
               The no_grad matters: base.weight may still require grad,
               and an in-place edit on a live leaf is an autograd error.

            3. self.merged = True
        """
        # TODO
        raise NotImplementedError

    def unmerge(self) -> None:
        """Subtract the delta back out of `base.weight`, restoring the
        separated form — the "eject the adapter" operation.

        Idempotent: calling `unmerge()` on an unmerged layer is a no-op.

        Recipe: the exact mirror of `merge()` — same guard, same
        `torch.no_grad()`, same transpose, `-=` instead of `+=`, and
        `self.merged = False` at the end.
        """
        # TODO
        raise NotImplementedError

    def extra_repr(self) -> str:
        return (
            f"in_features={self.base.in_features}, "
            f"out_features={self.base.out_features}, "
            f"rank={self.rank}, alpha={self.alpha:g}, merged={self.merged}"
        )
