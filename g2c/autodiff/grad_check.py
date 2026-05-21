"""Numerical gradient checking — the canonical sanity check for autodiff.

The idea: estimate df/dx by finite differences and compare against the
analytic gradient produced by `Value.backward()`. They should agree to
several digits for any correct autodiff engine.
"""
from __future__ import annotations

from collections.abc import Callable

from .value import Value


def numerical_grad(f: Callable[[Value], Value], x: Value, h: float = 1e-5) -> float:
    """Estimate df/dx at `x` via central finite differences.

    Args:
        f: A single-argument function from Value to Value (the expression
           whose gradient at `x` we want to estimate).
        x: The Value at which to evaluate the gradient. Only `x.data` is used.
        h: Step size. 1e-5 is a reasonable default — small enough to be
           accurate, large enough to avoid float precision blowup.

    Returns:
        The estimated derivative df/dx at x.data, as a Python float.

    Implementation hint:
        df/dx = f(x+h) - f(x-h) / 2h
    """
    # TODO
    raise NotImplementedError
