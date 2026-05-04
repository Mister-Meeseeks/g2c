"""Neural network building blocks.

A `Module` is the base class for everything that has learnable parameters
or processes a tensor. `Linear` is a learned affine map. `ReLU`, `Tanh`,
`Sigmoid` are element-wise nonlinearities. `Sequential` composes modules
in order. Together these are the LEGO bricks for everything from a tiny
MLP to a transformer block.

The substrate is `torch.Tensor` with autograd. We do NOT use `torch.nn.Linear`
or any other `torch.nn` module — the point of this module is to build them.
We DO use PyTorch tensors and rely on `loss.backward()` to populate gradients;
that's the substrate, not the lesson.

Boilerplate (the `Module` base class, all `__init__` methods, `parameters()`
methods) is implemented for you. The forward passes — the actual mathematical
content of each layer — are scaffolded. Search for `# TODO`.
"""
from __future__ import annotations

import math
from collections.abc import Iterable

import torch

from .device import resolve_device


class Module:
    """Base class for all neural-network modules.

    Subclasses implement `forward()`. Calling a module instance (`module(x)`)
    dispatches to `forward()`. Modules with learnable parameters override
    `parameters()` to expose them; default is no parameters.

    This is a minimal teaching version of `torch.nn.Module`. We don't bother
    with hooks, training/eval mode, state_dict, etc. — those are real-PyTorch
    concerns, not pedagogy concerns.
    """

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement forward()")

    def parameters(self) -> Iterable[torch.Tensor]:
        """Return all trainable parameters of this module. Override in subclasses."""
        return []

    def to(self, device: str | torch.device | None = "auto") -> "Module":
        """Move all trainable parameters to `device` and return self.

        This mirrors the everyday `torch.nn.Module.to(...)` ergonomics,
        but keeps the implementation visible: the course's modules expose
        raw tensor parameters, so moving a model is just moving those
        tensors and any existing gradients.
        """
        resolved = resolve_device(device)
        for p in self.parameters():
            p.data = p.data.to(resolved)
            if p.grad is not None:
                p.grad = p.grad.to(resolved)
        return self

    @property
    def device(self) -> torch.device:
        """Device of the first parameter, or CPU for parameter-free modules."""
        for p in self.parameters():
            return p.device
        return torch.device("cpu")


# ----------------------------------------------------------------------
# Linear layer: y = x @ W + b
# ----------------------------------------------------------------------

class Linear(Module):
    """A learned affine transformation:  y = x @ W + b.

    Shapes:
        x: (batch, in_features)
        W: (in_features, out_features)
        b: (out_features,)
        y: (batch, out_features)

    Initialization:
        W ~ Uniform(-1/sqrt(in), 1/sqrt(in))   (a simple variant of Kaiming init)
        b = 0
    """

    in_features: int
    out_features: int
    W: torch.Tensor
    b: torch.Tensor

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        bound = 1.0 / math.sqrt(in_features)
        W = torch.empty(in_features, out_features).uniform_(-bound, bound)
        W.requires_grad_(True)
        self.W = W

        b = torch.zeros(out_features)
        b.requires_grad_(True)
        self.b = b

    def parameters(self) -> Iterable[torch.Tensor]:
        return [self.W, self.b]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute y = x @ W + b. The bias broadcasts across the batch dim.

        Hint: this is a one-liner. The real lesson is that this is one line.
        """
        # TODO
        raise NotImplementedError


# ----------------------------------------------------------------------
# Activation functions
# ----------------------------------------------------------------------

class ReLU(Module):
    """Element-wise rectified linear unit: max(0, x).

    Why this matters: stacking linear layers without a nonlinearity in between
    is mathematically equivalent to a single linear layer (linear ∘ linear =
    linear). The nonlinearity is what gives a deep network its expressive
    power. ReLU is the cheapest, simplest nonlinearity that works well in
    practice.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return max(0, x), element-wise.

        Hint: `x.clamp(min=0)` or `torch.relu(x)` (the functional form, not
        `torch.nn.ReLU` — that's the high-level abstraction we're avoiding).
        """
        # TODO
        raise NotImplementedError


class Tanh(Module):
    """Element-wise hyperbolic tangent: (e^x - e^-x) / (e^x + e^-x).

    Squashes inputs to (-1, 1). Smooth, zero-centered. Was the dominant
    activation before ReLU took over for deep networks.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return tanh(x), element-wise. Use `torch.tanh(x)`."""
        # TODO
        raise NotImplementedError


class Sigmoid(Module):
    """Element-wise logistic sigmoid: 1 / (1 + e^-x).

    Squashes inputs to (0, 1). Used historically as the canonical activation;
    today mostly seen as the output activation for binary classification.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return sigmoid(x), element-wise. Use `torch.sigmoid(x)`."""
        # TODO
        raise NotImplementedError


# ----------------------------------------------------------------------
# Sequential — composes modules in order
# ----------------------------------------------------------------------

class Sequential(Module):
    """Apply a sequence of modules in order.

    Example:
        model = Sequential(
            Linear(784, 128),
            ReLU(),
            Linear(128, 10),
        )

        logits = model(x)   # shape (batch, 10)
    """

    layers: list[Module]

    def __init__(self, *layers: Module) -> None:
        super().__init__()
        self.layers = list(layers)

    def parameters(self) -> Iterable[torch.Tensor]:
        params: list[torch.Tensor] = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run x through each layer in order. Output of layer i is input of layer i+1.

        Hint: a 3-line implementation with a for-loop.
        """
        # TODO
        raise NotImplementedError
