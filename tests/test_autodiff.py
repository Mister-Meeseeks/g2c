"""Tests for Module 01: scalar autodiff.

These tests pin down the contract of `g2c.autodiff.Value` and `numerical_grad`.
Run with `pytest`. As you implement each primitive in `g2c/autodiff/value.py`,
the corresponding tests will start to pass.

Suggested order to implement & turn green:
  1. __add__  → test_add_*, test_radd*
  2. __mul__  → test_mul_*, test_rmul*, test_neg, test_sub_*, test_rsub
  3. __pow__  → test_pow_*, test_truediv_*, test_rtruediv
  4. exp      → test_exp_*
  5. log      → test_log_*
  6. tanh     → test_tanh_*
  7. relu     → test_relu_*
  8. backward → all backward / composition / accumulation tests
  9. numerical_grad → test_numerical_grad_*
"""
from __future__ import annotations

import math

import pytest

from g2c.autodiff import Value, numerical_grad


# ----------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------

def test_value_construction():
    v = Value(3.14)
    assert v.data == pytest.approx(3.14)
    assert v.grad == 0.0


def test_value_repr_runs():
    repr(Value(1.0))  # should not raise


# ----------------------------------------------------------------------
# Forward — primitive operations
# ----------------------------------------------------------------------

def test_add_forward():
    assert (Value(2.0) + Value(3.0)).data == pytest.approx(5.0)


def test_add_with_constant():
    assert (Value(2.0) + 3).data == pytest.approx(5.0)


def test_radd():
    assert (3 + Value(2.0)).data == pytest.approx(5.0)


def test_mul_forward():
    assert (Value(2.0) * Value(3.0)).data == pytest.approx(6.0)


def test_mul_with_constant():
    assert (Value(2.0) * 3).data == pytest.approx(6.0)


def test_rmul():
    assert (3 * Value(2.0)).data == pytest.approx(6.0)


def test_neg():
    assert (-Value(3.0)).data == pytest.approx(-3.0)


def test_sub_forward():
    assert (Value(5.0) - Value(2.0)).data == pytest.approx(3.0)


def test_sub_with_constant():
    assert (Value(5.0) - 2).data == pytest.approx(3.0)


def test_rsub():
    assert (10 - Value(3.0)).data == pytest.approx(7.0)


def test_pow_forward():
    assert (Value(2.0) ** 3).data == pytest.approx(8.0)


def test_pow_negative_exponent():
    assert (Value(2.0) ** -1).data == pytest.approx(0.5)


def test_truediv_forward():
    assert (Value(6.0) / Value(2.0)).data == pytest.approx(3.0)


def test_truediv_with_constant():
    assert (Value(6.0) / 2).data == pytest.approx(3.0)


def test_rtruediv():
    assert (10 / Value(2.0)).data == pytest.approx(5.0)


def test_exp_forward():
    assert Value(0.0).exp().data == pytest.approx(1.0)
    assert Value(1.0).exp().data == pytest.approx(math.e)


def test_log_forward():
    assert Value(1.0).log().data == pytest.approx(0.0)
    assert Value(math.e).log().data == pytest.approx(1.0)


def test_tanh_forward():
    assert Value(0.0).tanh().data == pytest.approx(0.0)
    assert Value(1.0).tanh().data == pytest.approx(math.tanh(1.0))


def test_relu_forward_positive():
    assert Value(2.5).relu().data == pytest.approx(2.5)


def test_relu_forward_negative():
    assert Value(-1.5).relu().data == pytest.approx(0.0)


def test_relu_forward_zero():
    assert Value(0.0).relu().data == pytest.approx(0.0)


# ----------------------------------------------------------------------
# Backward — primitive operations (one parent or two)
# ----------------------------------------------------------------------

def test_add_backward():
    a, b = Value(2.0), Value(3.0)
    c = a + b
    c.backward()
    assert a.grad == pytest.approx(1.0)
    assert b.grad == pytest.approx(1.0)


def test_mul_backward():
    a, b = Value(2.0), Value(3.0)
    c = a * b
    c.backward()
    assert a.grad == pytest.approx(3.0)  # b.data
    assert b.grad == pytest.approx(2.0)  # a.data


def test_pow_backward():
    a = Value(3.0)
    c = a ** 2
    c.backward()
    assert a.grad == pytest.approx(6.0)  # 2 * 3^1


def test_pow_backward_higher():
    a = Value(2.0)
    c = a ** 4
    c.backward()
    assert a.grad == pytest.approx(32.0)  # 4 * 2^3


def test_exp_backward():
    a = Value(1.0)
    c = a.exp()
    c.backward()
    assert a.grad == pytest.approx(math.e)


def test_log_backward():
    a = Value(2.0)
    c = a.log()
    c.backward()
    assert a.grad == pytest.approx(0.5)  # 1 / 2


def test_tanh_backward():
    a = Value(0.5)
    c = a.tanh()
    c.backward()
    expected = 1 - math.tanh(0.5) ** 2
    assert a.grad == pytest.approx(expected)


def test_relu_backward_positive():
    a = Value(2.0)
    c = a.relu()
    c.backward()
    assert a.grad == pytest.approx(1.0)


def test_relu_backward_negative():
    a = Value(-2.0)
    c = a.relu()
    c.backward()
    assert a.grad == pytest.approx(0.0)


def test_neg_backward():
    a = Value(3.0)
    c = -a
    c.backward()
    assert a.grad == pytest.approx(-1.0)


def test_sub_backward():
    a, b = Value(5.0), Value(2.0)
    c = a - b
    c.backward()
    assert a.grad == pytest.approx(1.0)
    assert b.grad == pytest.approx(-1.0)


def test_truediv_backward():
    a, b = Value(6.0), Value(2.0)
    c = a / b
    c.backward()
    # d(a/b)/da = 1/b = 0.5
    # d(a/b)/db = -a/b^2 = -6/4 = -1.5
    assert a.grad == pytest.approx(0.5)
    assert b.grad == pytest.approx(-1.5)


# ----------------------------------------------------------------------
# Composition — multi-op expressions
# ----------------------------------------------------------------------

def test_compose_mul_add():
    a, b = Value(2.0), Value(3.0)
    c = a * b + a  # value = 8; dc/da = b + 1 = 4; dc/db = a = 2
    c.backward()
    assert c.data == pytest.approx(8.0)
    assert a.grad == pytest.approx(4.0)
    assert b.grad == pytest.approx(2.0)


def test_long_chain():
    a, b, c = Value(1.0), Value(2.0), Value(0.5)
    f = (a * b + b ** 2) * c.tanh()
    f.backward()

    th = math.tanh(0.5)
    sech2 = 1 - th ** 2

    assert f.data == pytest.approx(6 * th)              # (1*2 + 4) * tanh(0.5)
    assert a.grad == pytest.approx(2 * th)              # b * tanh(c)
    assert b.grad == pytest.approx(5 * th)              # (a + 2b) * tanh(c)
    assert c.grad == pytest.approx(6 * sech2)           # (ab + b^2) * sech^2(c)


# ----------------------------------------------------------------------
# Gradient accumulation — Module 01 Exercise 5 ("topology stress test")
# ----------------------------------------------------------------------

def test_shared_node_simple():
    """f = a * a + a; df/da = 2a + 1. If gradient is not accumulated, this fails."""
    a = Value(3.0)
    f = a * a + a
    f.backward()
    assert a.grad == pytest.approx(7.0)


def test_shared_node_cube():
    """c = a^3 via repeated multiplication; dc/da = 3a^2."""
    a = Value(2.0)
    c = a * a * a
    c.backward()
    assert a.grad == pytest.approx(12.0)


def test_shared_node_diamond():
    """Diamond graph: f = (a+b) * (a-b); df/da = 2a, df/db = -2b."""
    a, b = Value(3.0), Value(2.0)
    f = (a + b) * (a - b)
    f.backward()
    assert f.data == pytest.approx(5.0)  # 9 - 4
    assert a.grad == pytest.approx(6.0)
    assert b.grad == pytest.approx(-4.0)


# ----------------------------------------------------------------------
# Numerical gradient checking — Module 01 Exercise 2
# ----------------------------------------------------------------------

def test_numerical_grad_quadratic():
    g = numerical_grad(lambda x: x ** 2, Value(3.0))
    assert g == pytest.approx(6.0, rel=1e-3)


def test_numerical_grad_cubic():
    g = numerical_grad(lambda x: x ** 3, Value(2.0))
    assert g == pytest.approx(12.0, rel=1e-3)


def test_numerical_grad_matches_analytic_simple():
    f = lambda x: x ** 3 + 2 * x  # noqa: E731

    x = Value(1.5)
    f(x).backward()
    analytic = x.grad

    numeric = numerical_grad(f, Value(1.5))
    assert analytic == pytest.approx(numeric, rel=1e-3)


def test_numerical_grad_matches_analytic_compound():
    f = lambda x: (x ** 3 + 2 * x).tanh()  # noqa: E731

    x = Value(0.7)
    f(x).backward()
    analytic = x.grad

    numeric = numerical_grad(f, Value(0.7))
    assert analytic == pytest.approx(numeric, rel=1e-3)
