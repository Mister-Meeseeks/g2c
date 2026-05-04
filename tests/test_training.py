"""Tests for Module 03B: training neural networks well.

Suggested order to implement & turn green:

  1. AdamW.step                  -> test_adamw_step_*
  2. clip_grad_norm_             -> test_clip_grad_norm_*
  3. cosine_with_warmup          -> test_cosine_with_warmup_*

Construction and zero_grad tests pass from the start. The rest fail with
`NotImplementedError` until you implement the Module 03B training dynamics
in `g2c.training`.
"""
from __future__ import annotations

import math

import pytest
import torch

from g2c.training import AdamW, clip_grad_norm_, cosine_with_warmup


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
