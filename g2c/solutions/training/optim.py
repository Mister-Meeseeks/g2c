# Hand-maintained solutions mirror — canonical, lives in `main` (see AGENTS.md).
"""Solutions for g2c.training.optim pedagogical scaffolds.

Patched onto the scaffold targets by g2c.solutions.apply().
"""
from __future__ import annotations

from collections.abc import Iterable
import torch

from g2c.training.optim import AdamW


class _AdamWImpl:  # patched onto AdamW by apply()
    def step(self) -> None:
        """Apply one AdamW update.

        Update rule for each parameter with a populated gradient:

            step_count += 1

            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad.pow(2)

            m_hat = m / (1 - beta1 ** step_count)
            v_hat = v / (1 - beta2 ** step_count)

            param *= (1 - lr * weight_decay)
            param -= lr * m_hat / (sqrt(v_hat) + eps)

        Important details:
          - Increment `step_count` once per optimizer step, not once per
            parameter.
          - Skip parameters whose `.grad is None`, but keep their `m` and `v`
            state aligned in the lists.
          - Apply weight decay directly to `param`, not by adding
            `weight_decay * param` to the gradient. That direct shrinkage is
            the "W" in AdamW.
          - Wrap the whole update in `torch.no_grad()` so the optimizer state
            and parameter update are not tracked by autograd.
        """
        self.step_count += 1

        with torch.no_grad():
            for i, p in enumerate(self.params):
                if p.grad is None:
                    continue

                grad = p.grad
                self.m[i].mul_(self.beta1).add_(grad, alpha=1.0 - self.beta1)
                self.v[i].mul_(self.beta2).addcmul_(
                    grad,
                    grad,
                    value=1.0 - self.beta2,
                )

                m_hat = self.m[i] / (1.0 - self.beta1 ** self.step_count)
                v_hat = self.v[i] / (1.0 - self.beta2 ** self.step_count)

                if self.weight_decay != 0.0:
                    p.mul_(1.0 - self.lr * self.weight_decay)
                p.addcdiv_(m_hat, v_hat.sqrt().add(self.eps), value=-self.lr)

