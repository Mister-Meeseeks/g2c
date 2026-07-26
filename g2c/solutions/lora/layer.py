# Hand-maintained solutions mirror — canonical, lives in `main` (see AGENTS.md).
"""Solutions for g2c.lora.layer pedagogical scaffolds.

Patched onto the scaffold targets by g2c.solutions.apply().
"""
from __future__ import annotations

import torch

from g2c.lora.layer import LoRALinear  # noqa: F401 (target class context)


class _LoRALinearImpl:  # patched onto LoRALinear by apply()
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
        y = self.base(x)
        if self.merged:
            return y
        delta = (x @ self.lora_A) @ self.lora_B
        return y + delta * self.scaling

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
        if self.merged:
            return
        with torch.no_grad():
            self.base.weight += (self.lora_A @ self.lora_B).T * self.scaling
        self.merged = True

    def unmerge(self) -> None:
        """Subtract the delta back out of `base.weight`, restoring the
        separated form — the "eject the adapter" operation.

        Idempotent: calling `unmerge()` on an unmerged layer is a no-op.

        Recipe: the exact mirror of `merge()` — same guard, same
        `torch.no_grad()`, same transpose, `-=` instead of `+=`, and
        `self.merged = False` at the end.
        """
        if not self.merged:
            return
        with torch.no_grad():
            self.base.weight -= (self.lora_A @ self.lora_B).T * self.scaling
        self.merged = False
