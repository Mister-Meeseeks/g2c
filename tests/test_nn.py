"""Tests for Module 03: a first neural network.

These tests pin down the contract of `g2c.nn`. Run with `pytest`.

Suggested order to implement & turn green:

  1. Linear.forward            → test_linear_*
  2. ReLU.forward              → test_relu_*
  3. Tanh.forward              → test_tanh_*
  4. Sigmoid.forward           → test_sigmoid_*
  5. Sequential.forward        → test_sequential_*
  6. CrossEntropyLoss.forward  → test_cross_entropy_*
  7. SGD.step                  → test_sgd_step_*
  8. End-to-end                → test_train_linear_regression

Construction tests, parameter-counting tests, and `MSELoss` (which is
implemented for you) pass from the start.
"""
from __future__ import annotations

import math

import pytest
import torch

from g2c.nn import (
    SGD,
    CrossEntropyLoss,
    Linear,
    MSELoss,
    Module,
    ReLU,
    Sequential,
    Sigmoid,
    Tanh,
    resolve_device,
)


# ----------------------------------------------------------------------
# Module base class (boilerplate — passes from the start)
# ----------------------------------------------------------------------

def test_module_default_no_parameters():
    class Empty(Module):
        def forward(self, x):
            return x

    assert list(Empty().parameters()) == []


def test_module_call_dispatches_to_forward():
    class Identity(Module):
        def forward(self, x):
            return x * 2

    m = Identity()
    x = torch.tensor([1.0, 2.0, 3.0])
    assert torch.allclose(m(x), torch.tensor([2.0, 4.0, 6.0]))


def test_resolve_device_auto_is_real_device():
    expected = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    assert resolve_device("auto") == expected


def test_module_to_cpu_moves_parameters():
    layer = Linear(5, 3)
    assert layer.to("cpu") is layer
    assert layer.device == torch.device("cpu")
    for p in layer.parameters():
        assert p.device.type == "cpu"


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS is only available on supported Apple Silicon machines",
)
def test_module_to_mps_moves_parameters():
    layer = Linear(5, 3)
    layer.to("mps")
    assert layer.device == torch.device("mps")
    for p in layer.parameters():
        assert p.device.type == "mps"


# ----------------------------------------------------------------------
# Linear — construction (boilerplate)
# ----------------------------------------------------------------------

def test_linear_construction_shapes():
    layer = Linear(5, 3)
    assert layer.W.shape == (5, 3)
    assert layer.b.shape == (3,)


def test_linear_parameters_requires_grad():
    layer = Linear(5, 3)
    params = list(layer.parameters())
    assert len(params) == 2
    assert all(p.requires_grad for p in params)


def test_linear_bias_initialized_to_zero():
    layer = Linear(5, 3)
    assert torch.allclose(layer.b, torch.zeros(3))


# ----------------------------------------------------------------------
# Linear — forward (the lesson)
# ----------------------------------------------------------------------

def test_linear_forward_shape():
    layer = Linear(8, 4)
    x = torch.randn(16, 8)
    y = layer(x)
    assert y.shape == (16, 4)


def test_linear_forward_known():
    """y = x @ W + b with hand-picked weights."""
    layer = Linear(2, 2)
    layer.W.data = torch.tensor([[3.0, 4.0], [5.0, 6.0]])
    layer.b.data = torch.tensor([0.5, -0.5])
    x = torch.tensor([[1.0, 2.0]])
    # y = [[1*3 + 2*5, 1*4 + 2*6]] + [[0.5, -0.5]] = [[13.5, 15.5]]
    assert torch.allclose(layer(x), torch.tensor([[13.5, 15.5]]))


def test_linear_bias_broadcasts_across_batch():
    layer = Linear(3, 2)
    layer.W.data = torch.zeros(3, 2)
    layer.b.data = torch.tensor([1.0, 2.0])
    x = torch.zeros(5, 3)
    y = layer(x)
    expected = torch.tensor([1.0, 2.0]).unsqueeze(0).expand(5, 2)
    assert torch.allclose(y, expected)


def test_linear_backward_populates_grads():
    layer = Linear(3, 2)
    x = torch.randn(5, 3)
    y = layer(x)
    y.sum().backward()
    assert layer.W.grad is not None
    assert layer.b.grad is not None
    assert layer.W.grad.shape == layer.W.shape
    assert layer.b.grad.shape == layer.b.shape


# ----------------------------------------------------------------------
# Activations
# ----------------------------------------------------------------------

def test_relu_forward():
    relu = ReLU()
    x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    expected = torch.tensor([0.0, 0.0, 0.0, 0.5, 2.0])
    assert torch.allclose(relu(x), expected)


def test_relu_no_parameters():
    assert list(ReLU().parameters()) == []


def test_relu_backward():
    relu = ReLU()
    x = torch.tensor([-1.0, 1.0], requires_grad=True)
    relu(x).sum().backward()
    # gradient is 1 for x>0, 0 for x<=0
    assert torch.allclose(x.grad, torch.tensor([0.0, 1.0]))


def test_tanh_forward():
    tanh = Tanh()
    x = torch.tensor([-1.0, 0.0, 1.0])
    assert torch.allclose(tanh(x), torch.tanh(x))


def test_tanh_squashes_to_pm_one():
    tanh = Tanh()
    x = torch.tensor([-1000.0, 1000.0])
    y = tanh(x)
    assert torch.allclose(y, torch.tensor([-1.0, 1.0]), atol=1e-6)


def test_sigmoid_forward():
    sig = Sigmoid()
    x = torch.tensor([0.0])
    assert torch.allclose(sig(x), torch.tensor([0.5]))


def test_sigmoid_squashes_to_zero_one():
    sig = Sigmoid()
    x = torch.tensor([-1000.0, 1000.0])
    y = sig(x)
    assert torch.allclose(y, torch.tensor([0.0, 1.0]), atol=1e-6)


# ----------------------------------------------------------------------
# Sequential
# ----------------------------------------------------------------------

def test_sequential_forward_shape():
    model = Sequential(
        Linear(3, 4),
        ReLU(),
        Linear(4, 2),
    )
    x = torch.randn(8, 3)
    assert model(x).shape == (8, 2)


def test_sequential_parameters_count():
    model = Sequential(
        Linear(3, 4),
        ReLU(),         # no params
        Linear(4, 2),
    )
    assert len(list(model.parameters())) == 4   # W1, b1, W2, b2


def test_sequential_executes_in_order():
    """A purely linear stack of Linear layers (no nonlinearity) should
    behave exactly like one combined linear map."""
    torch.manual_seed(0)
    a = Linear(3, 5)
    b = Linear(5, 2)
    model = Sequential(a, b)

    x = torch.randn(7, 3)
    actual = model(x)
    expected = (x @ a.W + a.b) @ b.W + b.b
    assert torch.allclose(actual, expected, atol=1e-6)


# ----------------------------------------------------------------------
# MSELoss (boilerplate — passes from the start)
# ----------------------------------------------------------------------

def test_mse_known():
    loss = MSELoss()
    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.0, 2.0, 4.0])
    # MSE = (0 + 0 + 1) / 3 = 1/3
    assert loss(pred, target).item() == pytest.approx(1 / 3)


# ----------------------------------------------------------------------
# CrossEntropyLoss
# ----------------------------------------------------------------------

def test_cross_entropy_matches_torch():
    """Sanity check against torch.nn.functional.cross_entropy."""
    torch.manual_seed(0)
    ce = CrossEntropyLoss()
    logits = torch.randn(8, 5)
    targets = torch.randint(0, 5, (8,))
    ours = ce(logits, targets)
    theirs = torch.nn.functional.cross_entropy(logits, targets)
    assert torch.allclose(ours, theirs, atol=1e-6)


def test_cross_entropy_known():
    """Equal logits → uniform softmax → loss == log(num_classes)."""
    ce = CrossEntropyLoss()
    logits = torch.zeros(1, 4)
    targets = torch.tensor([0])
    # softmax = [0.25, 0.25, 0.25, 0.25]; -log(0.25) = log(4)
    assert ce(logits, targets).item() == pytest.approx(math.log(4))


def test_cross_entropy_handles_large_logits():
    """The numerical stability trick must keep large logits in range."""
    ce = CrossEntropyLoss()
    logits = torch.tensor([[1000.0, 1001.0, 1002.0]])
    targets = torch.tensor([2])
    loss = ce(logits, targets)
    assert torch.isfinite(loss).all()


def test_cross_entropy_perfect_prediction():
    """Logits strongly biased to the correct class → near-zero loss."""
    ce = CrossEntropyLoss()
    logits = torch.tensor([[100.0, 0.0, 0.0]])
    targets = torch.tensor([0])
    assert ce(logits, targets).item() == pytest.approx(0.0, abs=1e-4)


# ----------------------------------------------------------------------
# SGD
# ----------------------------------------------------------------------

def test_sgd_step_basic():
    """Simple SGD update: param -= lr * grad."""
    p = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    p.grad = torch.tensor([0.1, 0.2, 0.3])
    SGD([p], lr=0.5).step()
    assert torch.allclose(p, torch.tensor([0.95, 1.90, 2.85]))


def test_sgd_zero_grad():
    """zero_grad clears all parameter gradients in place."""
    p = torch.tensor([1.0], requires_grad=True)
    p.grad = torch.tensor([0.5])
    SGD([p], lr=0.1).zero_grad()
    assert torch.all(p.grad == 0)


def test_sgd_step_with_weight_decay():
    """With weight_decay=wd, the update becomes lr * (grad + wd * param)."""
    p = torch.tensor([1.0, 2.0], requires_grad=True)
    p.grad = torch.tensor([0.1, 0.1])
    SGD([p], lr=1.0, weight_decay=0.1).step()
    # delta = (0.1 + 0.1*1.0, 0.1 + 0.1*2.0) = (0.2, 0.3)
    # new p = (1 - 0.2, 2 - 0.3) = (0.8, 1.7)
    assert torch.allclose(p, torch.tensor([0.8, 1.7]))


def test_sgd_step_skips_no_grad_params():
    """A parameter without a populated grad should be left alone."""
    p1 = torch.tensor([1.0], requires_grad=True)
    p2 = torch.tensor([5.0], requires_grad=True)
    p1.grad = torch.tensor([0.5])
    # p2.grad is None
    SGD([p1, p2], lr=0.1).step()
    assert torch.allclose(p1, torch.tensor([0.95]))
    assert torch.allclose(p2, torch.tensor([5.0]))


def test_sgd_step_does_not_track_in_autograd():
    """The update itself must not be recorded by autograd
    (otherwise a subsequent .backward() would build through stale state)."""
    p = torch.tensor([1.0], requires_grad=True)
    p.grad = torch.tensor([0.1])
    SGD([p], lr=0.1).step()
    assert p.is_leaf
    assert p.grad_fn is None


# ----------------------------------------------------------------------
# End-to-end: train a one-layer linear regression from scratch
# ----------------------------------------------------------------------

def test_train_linear_regression():
    """Fit y = 3x + 2 using Linear + MSELoss + SGD. The trained layer
    should recover W ≈ 3 and b ≈ 2."""
    torch.manual_seed(0)
    x = torch.linspace(-1.0, 1.0, 100).unsqueeze(1)         # (100, 1)
    y = 3.0 * x + 2.0 + 0.01 * torch.randn_like(x)

    model = Linear(1, 1)
    optimizer = SGD(model.parameters(), lr=0.1)
    loss_fn = MSELoss()

    for _ in range(500):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()

    assert model.W.item() == pytest.approx(3.0, abs=0.05)
    assert model.b.item() == pytest.approx(2.0, abs=0.05)
