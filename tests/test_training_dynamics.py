"""Tests for Module 03B: training neural networks well.

Suggested order to implement & turn green:

  1. AdamW.step                  -> tests/test_optim.py
  2. clip_grad_norm_             -> test_clip_grad_norm_*
  3. cosine_with_warmup          -> test_cosine_with_warmup_*

These tests own the training-dynamics primitives that Module 10's
pretraining loop consumes. Keeping them here makes the curriculum
boundary explicit: Module 03B teaches the knobs; Module 10 wires them
into a tiny language-model trainer.
"""
from __future__ import annotations

import pytest
import torch

from g2c.training import clip_grad_norm_, cosine_with_warmup


# ----------------------------------------------------------------------
# clip_grad_norm_ -- scaffolded
# ----------------------------------------------------------------------

def _make_param_with_grad(grad_values: list[float]) -> torch.Tensor:
    p = torch.zeros(len(grad_values), requires_grad=True)
    p.grad = torch.tensor(grad_values)
    return p


def test_clip_grad_norm_below_threshold_unchanged():
    """If global grad norm is below `max_norm`, gradients are untouched."""
    p = _make_param_with_grad([0.3, 0.4])  # norm = 0.5
    grad_before = p.grad.clone()
    norm = clip_grad_norm_([p], max_norm=10.0)
    assert torch.equal(p.grad, grad_before)
    assert norm == pytest.approx(0.5, abs=1e-6)


def test_clip_grad_norm_above_threshold_scaled():
    """Above the threshold: global norm rescaled to about max_norm."""
    p = _make_param_with_grad([3.0, 4.0])  # norm = 5.0
    clip_grad_norm_([p], max_norm=1.0)
    new_norm = p.grad.pow(2).sum().sqrt().item()
    assert new_norm == pytest.approx(1.0, abs=1e-3)


def test_clip_grad_norm_returns_pre_clip_norm():
    """The returned float is the pre-clip total norm."""
    p = _make_param_with_grad([3.0, 4.0])  # norm = 5.0
    norm = clip_grad_norm_([p], max_norm=1.0)
    assert norm == pytest.approx(5.0, abs=1e-3)


def test_clip_grad_norm_global_across_params():
    """Norm pools across all params, not each parameter separately."""
    p1 = _make_param_with_grad([3.0])  # per-param norm 3
    p2 = _make_param_with_grad([4.0])  # per-param norm 4
    norm = clip_grad_norm_([p1, p2], max_norm=1.0)
    assert norm == pytest.approx(5.0, abs=1e-3)
    # Each grad is scaled by 1/5 -- the same scale.
    assert p1.grad.item() == pytest.approx(3.0 / 5.0, abs=1e-3)
    assert p2.grad.item() == pytest.approx(4.0 / 5.0, abs=1e-3)


def test_clip_grad_norm_skips_none_grads():
    """Params with `grad=None` are skipped."""
    p_real = _make_param_with_grad([3.0, 4.0])  # norm = 5
    p_none = torch.zeros(3, requires_grad=True)  # no .grad
    norm = clip_grad_norm_([p_real, p_none], max_norm=10.0)
    assert norm == pytest.approx(5.0, abs=1e-3)
    assert p_none.grad is None


def test_clip_grad_norm_empty_returns_zero():
    """No params, or all-None grads, gives total norm 0 and no error."""
    p_none = torch.zeros(3, requires_grad=True)
    norm = clip_grad_norm_([p_none], max_norm=1.0)
    assert norm == pytest.approx(0.0, abs=1e-6)


# ----------------------------------------------------------------------
# cosine_with_warmup -- scaffolded
# ----------------------------------------------------------------------

def test_cosine_with_warmup_during_warmup_is_linear():
    """During warmup (step in [0, warmup_steps)), the lr ramps linearly
    from `max_lr / warmup_steps` at step 0 up to `max_lr` at step
    `warmup_steps - 1`.

    Linear ramping during warmup is what lets the model find a sane
    geometry before being pushed hard. A constant-lr warmup or a
    slower-than-linear warmup misses the point.
    """
    lr_step0 = cosine_with_warmup(
        0, warmup_steps=10, max_steps=100, max_lr=1e-3
    )
    lr_step5 = cosine_with_warmup(
        5, warmup_steps=10, max_steps=100, max_lr=1e-3
    )
    lr_step9 = cosine_with_warmup(
        9, warmup_steps=10, max_steps=100, max_lr=1e-3
    )
    assert lr_step0 == pytest.approx(1e-3 * 1 / 10)
    assert lr_step5 == pytest.approx(1e-3 * 6 / 10)
    assert lr_step9 == pytest.approx(1e-3)  # last warmup step -> max_lr


def test_cosine_with_warmup_at_first_cosine_step():
    """At step == warmup_steps, we're at the start of the cosine phase.
    progress=0, cos(0)=1, so lr = max_lr.
    """
    lr = cosine_with_warmup(
        10, warmup_steps=10, max_steps=110, max_lr=1e-3, min_lr=1e-5
    )
    assert lr == pytest.approx(1e-3)


def test_cosine_with_warmup_halfway_through_decay():
    """Halfway through the cosine phase, cos(pi/2) = 0 so the coefficient
    is 0.5 -- the lr should be the midpoint between max_lr and min_lr.
    """
    lr = cosine_with_warmup(
        60,  # warmup_steps + (max_steps - warmup_steps) / 2
        warmup_steps=10,
        max_steps=110,
        max_lr=1e-3,
        min_lr=1e-5,
    )
    assert lr == pytest.approx(0.5 * (1e-3 + 1e-5), abs=1e-9)


def test_cosine_with_warmup_at_max_steps():
    """At step == max_steps, cos(pi) = -1, coeff = 0, so lr = min_lr."""
    lr = cosine_with_warmup(
        110, warmup_steps=10, max_steps=110, max_lr=1e-3, min_lr=1e-5
    )
    assert lr == pytest.approx(1e-5)


def test_cosine_with_warmup_after_max_steps_held_at_min_lr():
    """Past max_steps, the function clamps at min_lr."""
    for step in (111, 200, 5000):
        lr = cosine_with_warmup(
            step, warmup_steps=10, max_steps=110, max_lr=1e-3, min_lr=1e-5
        )
        assert lr == pytest.approx(1e-5)


def test_cosine_with_warmup_no_warmup_step_zero():
    """warmup_steps=0 is supported: at step 0 we're already in cosine
    phase. cos(0) = 1 -> max_lr.
    """
    lr = cosine_with_warmup(
        0, warmup_steps=0, max_steps=100, max_lr=1e-3
    )
    assert lr == pytest.approx(1e-3)


def test_cosine_with_warmup_decay_is_monotonic():
    """During the cosine phase, the lr only ever decreases."""
    lrs = [
        cosine_with_warmup(
            s, warmup_steps=5, max_steps=100, max_lr=1.0, min_lr=0.0
        )
        for s in range(5, 101)
    ]
    for a, b in zip(lrs, lrs[1:]):
        assert b <= a + 1e-12


def test_cosine_with_warmup_min_lr_default_is_zero():
    """Default min_lr is 0 -- lr decays all the way to zero by max_steps."""
    lr = cosine_with_warmup(
        100, warmup_steps=10, max_steps=100, max_lr=1.0
    )
    assert lr == pytest.approx(0.0, abs=1e-9)
