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

        Hint: Gradient is the partial derivatives of y = x + z
        """
        # TODO
        raise NotImplementedError

    def __mul__(self, other: Union["Value", Number]) -> "Value":
        """Forward: self * other.

        Hint: Gradient is the partial derivatives of y = x * z
        """
        # TODO
        raise NotImplementedError

    def __pow__(self, exponent: Number) -> "Value":
        """Forward: self ** exponent. `exponent` is a numeric constant.

        Hint: Gradient is the derivative of y = x^c
        """
        # TODO
        raise NotImplementedError

    def exp(self) -> "Value":
        """Forward: e ** self.

        Hint: Gradient is the derivative of y = e^x
        """
        # TODO
        raise NotImplementedError

    def log(self) -> "Value":
        """Forward: ln(self). Requires self.data > 0.

        Hint: Gradient is the derivative of y = ln(x)
        """
        # TODO
        raise NotImplementedError

    def tanh(self) -> "Value":
        """Forward: tanh(self).

        Hint: Gradient is the derivative of y = tanh(x)
        """
        # TODO
        raise NotImplementedError

    def relu(self) -> "Value":
        """Forward: max(0, self).

        Hint: Gradient is the derivative of y = max(0, x)
        """
        # TODO
        raise NotImplementedError

    # ------------------------------------------------------------------
    # The reverse pass — STUDENT IMPLEMENTS
    # ------------------------------------------------------------------

    def backward(self) -> None:
        """Run reverse-mode autodiff from this Value.

        Calls _backward on every node in the graph in reverse topological order

        Hints:
            Make sure to initialize self.grad at 1.0.
        """
        # TODO
        raise NotImplementedError
    
    # ------------------------------------------------------------------
    # Helper for backward() — STUDENT IMPLEMENTS
    # ------------------------------------------------------------------
    def topological_sort(self) -> list["Value"]:
        """ Helper for topological sorting of the child nodes.

        Returns:
            A list of Values in topological order (i.e., each node only appears after all its parents).

        Hint: 
            You will have to use recursion. Don't add the same node twice.
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
