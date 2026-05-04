"""Optimizers.

The optimizer's job is to apply the update rule using the gradients PyTorch
populated via `loss.backward()`.

`SGD` is the simplest one:

    param ← param − learning_rate · gradient

With weight decay (L2 regularization), the update becomes:

    param ← param − learning_rate · (gradient + weight_decay · param)

This is equivalent to adding `(weight_decay / 2) · ||param||²` to the loss
and differentiating, but it's cheaper to apply at update time.

`__init__` and `zero_grad` are implemented for you. `step` is scaffolded;
the lesson is the update rule, not the bookkeeping. AdamW is introduced
later in Module 03B and lives in `g2c.training`.
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
        with torch.no_grad():
            for p in self.params:
                if p.grad is not None:
                    delta = p.grad + self.weight_decay * p
                    p -= self.lr * delta
