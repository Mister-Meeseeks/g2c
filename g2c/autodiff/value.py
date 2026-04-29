"""Scalar autodiff engine — Module 01.

A `Value` is a node in a computation graph. Each forward operation between
Values returns a new Value that records its parents and a local gradient rule
in `_backward`. Calling `.backward()` on the root populates `.grad` on every
node in the graph via reverse-mode autodifferentiation.

Boilerplate (`__init__`, `__repr__`, the unary/right-hand-side convenience
operators) is implemented for you. The actual autodiff logic — the forward
math and the local gradient rule for each primitive operation, plus the
topological-sort `backward()` driver — is left to you to implement.

Search for `# TODO` to find the spots that need work.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Union

Number = Union[int, float]


class Value:
    """A scalar value in the autodiff graph."""

    data: float
    grad: float
    _prev: set["Value"]
    _op: str
    _backward: Callable[[], None]

    def __init__(
        self,
        data: Number,
        _children: Iterable["Value"] = (),
        _op: str = "",
    ) -> None:
        self.data = float(data)
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None  # default no-op; overridden by ops below

    def __repr__(self) -> str:
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    # ------------------------------------------------------------------
    # Core primitive operations — STUDENT IMPLEMENTS
    #
    # Each method should:
    #   1. If `other` is a Number, wrap it in a Value.
    #   2. Compute the forward value.
    #   3. Construct `out = Value(<forward_value>, (<parents>,), "<op_name>")`.
    #   4. Define a `_backward` closure that ACCUMULATES (`+=`) gradient
    #      contributions from `out.grad` into each parent's `.grad`.
    #   5. Assign `out._backward = _backward`.
    #   6. Return `out`.
    # ------------------------------------------------------------------

    def __add__(self, other: Union["Value", Number]) -> "Value":
        """Forward: self + other.

        Local rule:
            d(out)/d(self)  = 1
            d(out)/d(other) = 1
        """
        # TODO
        raise NotImplementedError

    def __mul__(self, other: Union["Value", Number]) -> "Value":
        """Forward: self * other.

        Local rule:
            d(out)/d(self)  = other.data
            d(out)/d(other) = self.data
        """
        # TODO
        raise NotImplementedError

    def __pow__(self, exponent: Number) -> "Value":
        """Forward: self ** exponent. `exponent` is a numeric constant.

        Local rule:
            d(out)/d(self) = exponent * self.data ** (exponent - 1)
        """
        # TODO
        raise NotImplementedError

    def exp(self) -> "Value":
        """Forward: e ** self.

        Local rule:
            d(out)/d(self) = e ** self.data    (which equals out.data)
        """
        # TODO
        raise NotImplementedError

    def log(self) -> "Value":
        """Forward: ln(self). Requires self.data > 0.

        Local rule:
            d(out)/d(self) = 1 / self.data
        """
        # TODO
        raise NotImplementedError

    def tanh(self) -> "Value":
        """Forward: tanh(self).

        Local rule:
            d(out)/d(self) = 1 - tanh(self.data) ** 2     (= 1 - out.data ** 2)
        """
        # TODO
        raise NotImplementedError

    def relu(self) -> "Value":
        """Forward: max(0, self).

        Local rule:
            d(out)/d(self) = 1 if self.data > 0 else 0
        """
        # TODO
        raise NotImplementedError

    # ------------------------------------------------------------------
    # The reverse pass — STUDENT IMPLEMENTS
    # ------------------------------------------------------------------

    def backward(self) -> None:
        """Run reverse-mode autodiff from this Value.

        Steps:
          1. Build the topological order of the computation graph rooted at `self`
             (each node appears after all its parents).
          2. Set `self.grad = 1.0` (the seed: d(self) / d(self) = 1).
          3. Walk the topological order in REVERSE, calling each node's
             `_backward()` to accumulate gradients into its parents' `.grad`.
        """
        # TODO
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Convenience operators — already implemented in terms of the primitives.
    # No student work needed here; once the primitives above are implemented,
    # these auto-work.
    # ------------------------------------------------------------------

    def __neg__(self) -> "Value":
        return self * -1

    def __sub__(self, other: Union["Value", Number]) -> "Value":
        return self + (-other)

    def __truediv__(self, other: Union["Value", Number]) -> "Value":
        return self * (other ** -1)

    def __radd__(self, other: Number) -> "Value":
        return self + other

    def __rsub__(self, other: Number) -> "Value":
        return (-self) + other

    def __rmul__(self, other: Number) -> "Value":
        return self * other

    def __rtruediv__(self, other: Number) -> "Value":
        return (self ** -1) * other
