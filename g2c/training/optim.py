"""AdamW optimizer for Module 03B.

`SGD` lives in `g2c.nn` because Module 03 introduces the basic neural
network training loop. `AdamW` lives here because Module 03B is about
training dynamics: adaptive step scales, decoupled weight decay, clipping,
learning-rate schedules, and reading curves.

`AdamW` is the modern adaptive optimizer used for most transformer training.
It keeps two pieces of state per parameter:

    m <- beta1 * m + (1 - beta1) * grad       # momentum / first moment
    v <- beta2 * v + (1 - beta2) * grad^2     # scale / second moment

The update uses bias-corrected `m_hat` and `v_hat`, then applies decoupled
weight decay. The constructor and state setup are implemented; `step` is
scaffolded.
"""
from __future__ import annotations

from collections.abc import Iterable

import torch


class AdamW:
    """Adam with decoupled weight decay.

    Args:
        params: iterable of parameter tensors.
        lr: learning rate. AdamW usually uses a much smaller nominal lr than
            SGD; `1e-3` is common for MLPs, `3e-4` is a common transformer
            starting point.
        betas: exponential-decay rates for the first and second moments.
            Defaults match PyTorch and common transformer recipes.
        eps: small denominator floor for numerical stability.
        weight_decay: decoupled shrinkage applied directly to the parameter
            before the adaptive gradient update. `0.0` disables it.

    Public state:
        params: parameter tensors.
        m: first-moment tensors, one per parameter, same shapes as params.
        v: second-moment tensors, one per parameter, same shapes as params.
        step_count: number of `step()` calls applied so far.

    The external interface intentionally matches `SGD`: Module 10's trainer
    only needs `zero_grad()`, mutable `lr`, and `step()`.
    """

    params: list[torch.Tensor]
    lr: float
    beta1: float
    beta2: float
    eps: float
    weight_decay: float
    m: list[torch.Tensor]
    v: list[torch.Tensor]
    step_count: int

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float,
        *,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.step_count = 0

    def zero_grad(self) -> None:
        """Set every parameter's gradient to zero in place."""
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

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
