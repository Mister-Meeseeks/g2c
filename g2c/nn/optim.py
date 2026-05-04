"""Optimizers.

The optimizer's job is to apply the update rule using the gradients PyTorch
populated via `loss.backward()`.

`SGD` is the simplest one:

    param ← param − learning_rate · gradient

With weight decay (L2 regularization), the update becomes:

    param ← param − learning_rate · (gradient + weight_decay · param)

This is equivalent to adding `(weight_decay / 2) · ||param||²` to the loss
and differentiating, but it's cheaper to apply at update time.

`__init__` and `zero_grad` are implemented for you. `step` is scaffolded —
the lesson is the update rule, not the bookkeeping.

`AdamW` is the modern adaptive optimizer used for most transformer training.
It keeps two pieces of state per parameter:

    m ← beta1 * m + (1 - beta1) * grad       # momentum / first moment
    v ← beta2 * v + (1 - beta2) * grad²      # scale / second moment

The update uses bias-corrected `m_hat` and `v_hat`, then applies decoupled
weight decay. Its constructor and state setup are implemented; `step` is
scaffolded.
"""
from __future__ import annotations

from collections.abc import Iterable

import torch


class SGD:
    """Vanilla stochastic gradient descent with optional weight decay.

    Args:
        params: iterable of parameter tensors (each with `requires_grad=True`).
        lr: learning rate.
        weight_decay: L2 regularization coefficient. 0.0 disables it.

    Usage:
        optimizer = SGD(model.parameters(), lr=0.01)
        for x, y in batches:
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
    """

    params: list[torch.Tensor]
    lr: float
    weight_decay: float

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float,
        weight_decay: float = 0.0,
    ) -> None:
        self.params = list(params)
        self.lr = lr
        self.weight_decay = weight_decay

    def zero_grad(self) -> None:
        """Set every parameter's gradient to zero in place.

        Must be called between iterations because PyTorch's `.backward()`
        accumulates into `.grad` rather than overwriting (the same accumulation
        rule we built ourselves in Module 01).
        """
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self) -> None:
        """Apply the SGD update to every parameter.

        Update rule (per parameter, using `.data` to bypass autograd tracking):
            delta = grad + weight_decay * param            # data, not graph
            param.data -= lr * delta

        Implementation hints:
            - Wrap the body in `with torch.no_grad():` so the update itself
              is not tracked by autograd. (Equivalent: operate on `.data`
              directly; cleaner: use `no_grad`.)
            - Skip parameters whose `.grad` is None (they didn't participate).
            - When weight_decay is 0, the term drops out — no need to special-case.
        """
        # TODO
        raise NotImplementedError


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
        # TODO
        raise NotImplementedError
