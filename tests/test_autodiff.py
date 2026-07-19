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
import random

import pytest

from g2c.autodiff import (
    ScalarMLP,
    ScalarNeuron,
    Value,
    numerical_grad,
    single_neuron_forward,
    train_xor_step,
    xor_loss,
    zero_grad,
)

# ----------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------

def test_value_construction():
    v = Value(3.14)
    assert v.data == pytest.approx(3.14)
    assert v.grad == 0.0


def test_value_repr_runs():
    repr(Value(1.0))  # should not raise


def test_scalar_neuron_construction_parameters():
    neuron = ScalarNeuron(3, rng=random.Random(0))
    params = neuron.parameters()
    assert len(params) == 4
    assert all(isinstance(param, Value) for param in params)


def test_scalar_neuron_rejects_nonpositive_inputs():
    with pytest.raises(ValueError):
        ScalarNeuron(0, rng=random.Random(0))


def test_scalar_mlp_construction_parameters():
    model = ScalarMLP(rng=random.Random(0))
    assert len(model.parameters()) == 9


def test_zero_grad_resets_scalar_parameters():
    params = [Value(1.0), Value(2.0)]
    params[0].grad = 0.5
    params[1].grad = -1.5
    zero_grad(params)
    assert [param.grad for param in params] == [0.0, 0.0]


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


# ----------------------------------------------------------------------
# Module 01 notebook helpers — editor-backed scalar XOR pieces
# ----------------------------------------------------------------------

def test_single_neuron_forward_matches_formula():
    w1 = Value(0.2)
    w2 = Value(-0.3)
    b = Value(0.1)
    out = single_neuron_forward(0.75, 0.25, w1, w2, b)
    expected = math.tanh(0.2 * 0.75 - 0.3 * 0.25 + 0.1)
    assert out.data == pytest.approx(expected)


def test_scalar_neuron_forward_known_weights():
    neuron = ScalarNeuron(2, rng=random.Random(0))
    neuron.w[0].data = 0.5
    neuron.w[1].data = -0.25
    neuron.b.data = 0.1
    out = neuron((2.0, 4.0))
    assert out.data == pytest.approx(math.tanh(0.5 * 2.0 - 0.25 * 4.0 + 0.1))


def test_scalar_neuron_rejects_wrong_input_width():
    neuron = ScalarNeuron(2, rng=random.Random(0))
    with pytest.raises(ValueError):
        neuron((1.0,))


def test_scalar_mlp_forward_returns_value():
    model = ScalarMLP(rng=random.Random(0))
    assert isinstance(model((0.0, 1.0)), Value)


def test_xor_loss_returns_scalar_value():
    model = ScalarMLP(rng=random.Random(0))
    loss = xor_loss(model)
    assert isinstance(loss, Value)
    assert loss.data >= 0.0


def test_train_xor_step_reduces_loss_over_many_steps():
    model = ScalarMLP(rng=random.Random(0))
    start = xor_loss(model).data
    for _ in range(300):
        train_xor_step(model, lr=0.05)
    end = xor_loss(model).data
    assert end < start
