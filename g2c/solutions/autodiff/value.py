# Hand-maintained solutions mirror — canonical, lives in `main` (see AGENTS.md).
"""Solutions for g2c.autodiff.value pedagogical scaffolds.

Patched onto the scaffold targets by g2c.solutions.apply().
"""
from __future__ import annotations

import math
from collections.abc import Callable, Iterable

Number = int | float

from g2c.autodiff.value import Value


class _ValueImpl:  # patched onto Value by apply()
    def __add__(self, other: Value | Number) -> Value:
        """Forward: self + other.

        Hint: Gradient is the partial derivatives of y = x + z
        """
        if not isinstance(other, Value):
            other = Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            self.grad += out.grad * 1.0
            other.grad += out.grad * 1.0
        out._backward = _backward
        return out

    def __mul__(self, other: Value | Number) -> Value:
        """Forward: self * other.

        Hint: Gradient is the partial derivatives of y = x * z
        """
        if not isinstance(other, Value):
            other = Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backward = _backward
        return out

    def __pow__(self, exponent: Number) -> Value:
        """Forward: self ** exponent. `exponent` is a numeric constant.

        Hint: Gradient is the derivative of y = x^c
        """
        out = Value(self.data ** exponent, (self,), f"**{exponent}")

        def _backward() -> None:
            self.grad += out.grad * exponent * self.data ** (exponent - 1)
        out._backward = _backward
        return out

    def exp(self) -> Value:
        """Forward: e ** self.

        Hint: Gradient is the derivative of y = e^x
        """
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward() -> None:
            self.grad += out.grad * out.data
        out._backward = _backward
        return out

    def log(self) -> Value:
        """Forward: ln(self). Requires self.data > 0.

        Hint: Gradient is the derivative of y = ln(x)
        """
        out = Value(math.log(self.data), (self,), "log")

        def _backward() -> None:
            self.grad += out.grad * (1 / self.data)
        out._backward = _backward
        return out

    def tanh(self) -> Value:
        """Forward: tanh(self).

        Hint: Gradient is the derivative of y = tanh(x)
        """
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward() -> None:
            self.grad += out.grad * (1 - t ** 2)
        out._backward = _backward
        return out

    def relu(self) -> Value:
        """Forward: max(0, self).

        Hint: Gradient is the derivative of y = max(0, x)
        """
        out = Value(self.data if self.data > 0 else 0, (self,), "ReLU")

        def _backward() -> None:
            self.grad += out.grad * (1.0 if self.data > 0 else 0.0)
        out._backward = _backward
        return out

    def backward(self) -> None:
        """Run reverse-mode autodiff from this Value.

        Calls _backward on every node in the graph in reverse topological order

        Hints:
            Make sure to initialize self.grad at 1.0.
        """
        topo_nodes = self.topological_sort()
        topo_nodes.reverse()

        self.grad = 1.0
        for node in topo_nodes:
            node._backward()

    def topological_sort(self) -> list[Value]:
        """ Helper for topological sorting of the child nodes.

        Returns:
            A list of Values in topological order (i.e., each node only
            appears after all its parents).

        Hint:
            Recursive depth-first search with a `visited` set: skip nodes
            already seen, recurse into `_prev` first, then append the node.
        """
        topo_nodes: list[Value] = []
        visited: set[int] = set()

        def visit(node: Value) -> None:
            if id(node) in visited:
                return
            visited.add(id(node))
            for child in node._prev:
                visit(child)
            topo_nodes.append(node)

        visit(self)
        return topo_nodes

