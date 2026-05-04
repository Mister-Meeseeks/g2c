"""Tests for Module 03B: training neural networks well.

Suggested order to implement & turn green:

  1. AdamW.step         -> test_adamw_step_*

Construction and zero_grad tests pass from the start. The step tests fail
with `NotImplementedError` until you implement the AdamW update rule in
`g2c.nn.optim`.
"""
from __future__ import annotations

import math

import torch

from g2c.nn import AdamW


def test_adamw_construction_initializes_state():
    p1 = torch.tensor([1.0, 2.0], requires_grad=True)
    p2 = torch.tensor([[3.0], [4.0]], requires_grad=True)
    opt = AdamW([p1, p2], lr=3e-4, betas=(0.8, 0.95), eps=1e-6, weight_decay=0.01)

    assert opt.params[0] is p1
    assert opt.params[1] is p2
    assert opt.lr == 3e-4
    assert opt.beta1 == 0.8
    assert opt.beta2 == 0.95
    assert opt.eps == 1e-6
    assert opt.weight_decay == 0.01
    assert opt.step_count == 0
    assert len(opt.m) == 2
    assert len(opt.v) == 2
    assert torch.equal(opt.m[0], torch.zeros_like(p1))
    assert torch.equal(opt.v[1], torch.zeros_like(p2))


def test_adamw_zero_grad():
    p = torch.tensor([1.0], requires_grad=True)
    p.grad = torch.tensor([0.5])
    AdamW([p], lr=0.1).zero_grad()
    assert torch.all(p.grad == 0)


def test_adamw_step_basic_bias_corrected():
    """First AdamW step with no weight decay should match the hand formula."""
    p = torch.tensor([1.0], requires_grad=True)
    p.grad = torch.tensor([0.1])
    opt = AdamW([p], lr=0.001, weight_decay=0.0)

    opt.step()

    # m = 0.01, v = 0.00001, but bias correction gives
    # m_hat = 0.1, v_hat = 0.01, so the update is about lr * 1.
    expected = torch.tensor([1.0 - 0.001 * 0.1 / (math.sqrt(0.01) + 1e-8)])
    assert torch.allclose(p, expected)
    assert opt.step_count == 1
    assert torch.allclose(opt.m[0], torch.tensor([0.01]))
    assert torch.allclose(opt.v[0], torch.tensor([0.00001]))


def test_adamw_step_accumulates_moments_across_steps():
    p = torch.tensor([1.0], requires_grad=True)
    opt = AdamW([p], lr=0.001, weight_decay=0.0)

    p.grad = torch.tensor([0.1])
    opt.step()
    first_m = opt.m[0].clone()
    first_v = opt.v[0].clone()

    p.grad = torch.tensor([0.2])
    opt.step()

    expected_m = 0.9 * first_m + 0.1 * torch.tensor([0.2])
    expected_v = 0.999 * first_v + 0.001 * torch.tensor([0.2]).pow(2)
    assert opt.step_count == 2
    assert torch.allclose(opt.m[0], expected_m)
    assert torch.allclose(opt.v[0], expected_v)


def test_adamw_weight_decay_is_decoupled_from_gradient():
    """With zero grad, AdamW still shrinks params through decoupled decay."""
    p = torch.tensor([2.0], requires_grad=True)
    p.grad = torch.tensor([0.0])
    AdamW([p], lr=0.1, weight_decay=0.01).step()
    assert torch.allclose(p, torch.tensor([2.0 * (1.0 - 0.1 * 0.01)]))


def test_adamw_step_skips_no_grad_params():
    p1 = torch.tensor([1.0], requires_grad=True)
    p2 = torch.tensor([5.0], requires_grad=True)
    p1.grad = torch.tensor([0.5])
    opt = AdamW([p1, p2], lr=0.1)

    opt.step()

    assert not torch.allclose(p1, torch.tensor([1.0]))
    assert torch.allclose(p2, torch.tensor([5.0]))
    assert torch.equal(opt.m[1], torch.zeros_like(p2))
    assert torch.equal(opt.v[1], torch.zeros_like(p2))


def test_adamw_step_does_not_track_in_autograd():
    p = torch.tensor([1.0], requires_grad=True)
    p.grad = torch.tensor([0.1])
    AdamW([p], lr=0.1).step()
    assert p.is_leaf
    assert p.grad_fn is None
