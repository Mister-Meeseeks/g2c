# Hand-maintained solutions mirror — canonical, lives in `main` (see AGENTS.md).
"""Solutions for g2c.autodiff.nn pedagogical scaffolds.

Patched onto the scaffold targets by g2c.solutions.apply().
"""
from __future__ import annotations

import random
from collections.abc import Sequence

from g2c.autodiff.value import Value

XOR_DATA: tuple[tuple[tuple[float, float], float], ...] = (
    ((0.0, 0.0), -1.0),
    ((0.0, 1.0), 1.0),
    ((1.0, 0.0), 1.0),
    ((1.0, 1.0), -1.0),
)

from g2c.autodiff.nn import ScalarMLP, ScalarNeuron


class _ScalarNeuronImpl:  # patched onto ScalarNeuron by apply()
    def __call__(self, x: Sequence[float | Value]) -> Value:
        """Return ``tanh(w dot x + b)``."""
        if len(x) != len(self.w):
            raise ValueError(f"expected {len(self.w)} inputs, got {len(x)}")
        # SOLUTION
        out = self.b
        for weight, value in zip(self.w, x, strict=True):
            out = out + weight * value
        return out.tanh()



class _ScalarMLPImpl:  # patched onto ScalarMLP by apply()
    def __call__(self, x: tuple[float, float]) -> Value:
        """Run the 2-2-1 MLP on one XOR input."""
        # SOLUTION
        hidden_values = [neuron(x) for neuron in self.hidden]
        return self.out(hidden_values)



def single_neuron_forward(
    x1: float,
    x2: float,
    w1: Value,
    w2: Value,
    b: Value,
) -> Value:
    """Return ``tanh(w1*x1 + w2*x2 + b)`` using only ``Value`` operations."""
    # SOLUTION
    return (w1 * x1 + w2 * x2 + b).tanh()


def xor_loss(
    model: ScalarMLP,
    data: Sequence[tuple[tuple[float, float], float]] = XOR_DATA,
) -> Value:
    """Return average squared error across the XOR truth table."""
    if not data:
        raise ValueError("data must not be empty")
    # SOLUTION
    total = Value(0.0)
    for x, target in data:
        pred = model(x)
        total = total + (pred - target) ** 2
    return total / len(data)


def train_xor_step(model: ScalarMLP, lr: float) -> float:
    """Run one full-batch gradient-descent step and return pre-update loss."""
    params = model.parameters()
    zero_grad(params)
    loss = xor_loss(model)
    loss.backward()

    # SOLUTION
    for param in params:
        param.data -= lr * param.grad
    return loss.data
