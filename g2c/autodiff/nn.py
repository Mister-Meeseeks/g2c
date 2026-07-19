"""Tiny neural-network helpers built on scalar ``Value`` objects.

Module 01's core deliverable is the autodiff engine itself. These helpers are
small enough to understand directly, but they are real editor-backed code
rather than notebook-local functions. The notebook uses them to build the XOR
exercise once the scalar operations and ``backward`` are working.
"""
from __future__ import annotations

import random
from collections.abc import Sequence

from .value import Value

XOR_DATA: tuple[tuple[tuple[float, float], float], ...] = (
    ((0.0, 0.0), -1.0),
    ((0.0, 1.0), 1.0),
    ((1.0, 0.0), 1.0),
    ((1.0, 1.0), -1.0),
)


def single_neuron_forward(
    x1: float,
    x2: float,
    w1: Value,
    w2: Value,
    b: Value,
) -> Value:
    """Return ``tanh(w1*x1 + w2*x2 + b)`` using only ``Value`` operations."""
    # TODO
    raise NotImplementedError


class ScalarNeuron:
    """One tanh neuron backed by scalar ``Value`` parameters."""

    def __init__(self, n_inputs: int, *, rng: random.Random) -> None:
        if n_inputs <= 0:
            raise ValueError("n_inputs must be positive")
        self.w = [Value(rng.uniform(-1.0, 1.0)) for _ in range(n_inputs)]
        self.b = Value(rng.uniform(-1.0, 1.0))

    def __call__(self, x: Sequence[float | Value]) -> Value:
        """Return ``tanh(w dot x + b)``."""
        # TODO
        raise NotImplementedError

    def parameters(self) -> list[Value]:
        return self.w + [self.b]


class ScalarMLP:
    """A fixed 2-2-1 MLP for the XOR exercise."""

    def __init__(self, *, rng: random.Random) -> None:
        self.hidden = [ScalarNeuron(2, rng=rng), ScalarNeuron(2, rng=rng)]
        self.out = ScalarNeuron(2, rng=rng)

    def __call__(self, x: tuple[float, float]) -> Value:
        """Run the 2-2-1 MLP on one XOR input."""
        # TODO
        raise NotImplementedError

    def parameters(self) -> list[Value]:
        params: list[Value] = []
        for neuron in self.hidden:
            params.extend(neuron.parameters())
        params.extend(self.out.parameters())
        return params


def zero_grad(params: Sequence[Value]) -> None:
    """Reset gradients on scalar parameters in place."""
    for param in params:
        param.grad = 0.0


def xor_loss(
    model: ScalarMLP,
    data: Sequence[tuple[tuple[float, float], float]] = XOR_DATA,
) -> Value:
    """Return average squared error across the XOR truth table."""
    # TODO
    raise NotImplementedError


def train_xor_step(model: ScalarMLP, lr: float) -> float:
    """Run one full-batch gradient-descent step and return pre-update loss."""
    # TODO
    raise NotImplementedError
