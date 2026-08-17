"""Tests for g2c/muon — Beyond module: orthogonalized updates.

Suggested order to implement & turn green:

1. `zeropower_via_newtonschulz` — unblocks the `test_newton_schulz_*`
   tests. The three sanity anchors (orthogonal input is a fixed point,
   scale invariance, transpose consistency) catch nearly every bug:
   a forgotten Frobenius normalization diverges, a wrong coefficient
   order breaks the fixed point, a missing transpose-back breaks
   shapes.
2. `Muon.step` — unblocks the `test_muon_step_*` and smoke tests. The
   gradient-scale-invariance test is the signature check: Muon's
   update direction must not depend on the gradient's magnitude.

Construction, validation, and `zero_grad` are provided, so those tests
pass from the start. The AdamW-side test drives the internal
`g2c.training.AdamW`, so it additionally needs Module 03B's optimizer
(yours, or `G2C_APPLY_SOLUTIONS=03b`).
"""
from __future__ import annotations

import pytest
import torch

from g2c.muon import Muon, zeropower_via_newtonschulz


def _leaf(shape, seed=None, scale=1.0):
    if seed is not None:
        torch.manual_seed(seed)
    p = (torch.randn(*shape) * scale).requires_grad_(True)
    return p


# ---------------------------------------------------------------------------
# Provided plumbing — green from the start.
# ---------------------------------------------------------------------------


def test_construction_and_state():
    W = _leaf((8, 4), seed=0)
    b = _leaf((4,), seed=1)
    opt = Muon([W], [b], lr=0.02)
    assert opt.muon_params == [W]
    assert opt.adamw_params == [b]
    assert len(opt.momentum_buffers) == 1
    assert opt.momentum_buffers[0].shape == (8, 4)
    assert opt.momentum_buffers[0].abs().sum() == 0
    assert opt.step_count == 0
    assert opt.params == [W, b]


def test_construction_rejects_non_matrix_muon_params():
    with pytest.raises(ValueError):
        Muon([_leaf((4,), seed=2)])
    with pytest.raises(ValueError):
        Muon([_leaf((2, 3, 4), seed=3)])


def test_zero_grad_clears_both_groups():
    W, b = _leaf((4, 4), seed=4), _leaf((4,), seed=5)
    W.grad = torch.ones_like(W)
    b.grad = torch.ones_like(b)
    Muon([W], [b]).zero_grad()
    assert W.grad.abs().sum() == 0
    assert b.grad.abs().sum() == 0


# ---------------------------------------------------------------------------
# Newton–Schulz orthogonalization
# ---------------------------------------------------------------------------


def test_newton_schulz_rejects_non_matrices():
    with pytest.raises(ValueError):
        zeropower_via_newtonschulz(torch.randn(4))


def test_newton_schulz_preserves_shape():
    torch.manual_seed(10)
    for shape in [(16, 8), (8, 16), (8, 8)]:
        G = torch.randn(*shape)
        assert zeropower_via_newtonschulz(G).shape == shape


def test_newton_schulz_pushes_singular_values_toward_one():
    torch.manual_seed(11)
    G = torch.randn(16, 8)
    s_before = torch.linalg.svdvals(G / G.norm())
    s_after = torch.linalg.svdvals(zeropower_via_newtonschulz(G))
    # The normalized input's spectrum is spread out and small (sum of
    # squares is 1); the output's sits in a loose band around 1.
    assert s_before.max() < 1.0
    assert ((s_after > 0.4) & (s_after < 1.6)).all(), s_after
    assert 0.7 < s_after.mean() < 1.3


def test_newton_schulz_preserves_orthogonal_directions():
    torch.manual_seed(12)
    Q, _ = torch.linalg.qr(torch.randn(8, 8))
    # Q's singular values are all equal (exactly 1), so the polynomial
    # acts uniformly on them: the output must be a scalar multiple of Q
    # — same singular vectors, spectrum moved as one. (It is not Q
    # itself: the Frobenius pre-normalization first rescales the
    # spectrum to 1/sqrt(8).)
    R = zeropower_via_newtonschulz(Q)
    c = (R * Q).sum() / (Q * Q).sum()
    assert torch.allclose(R, c * Q, atol=1e-4)
    assert 0.7 < c.item() < 1.3


def test_newton_schulz_is_scale_invariant():
    torch.manual_seed(13)
    G = torch.randn(8, 8)
    assert torch.allclose(
        zeropower_via_newtonschulz(G),
        zeropower_via_newtonschulz(7.0 * G),
        atol=1e-4,
    )


def test_newton_schulz_transpose_consistency():
    torch.manual_seed(14)
    G = torch.randn(12, 4)
    assert torch.allclose(
        zeropower_via_newtonschulz(G.T),
        zeropower_via_newtonschulz(G).T,
        atol=1e-5,
    )


# ---------------------------------------------------------------------------
# The Muon step
# ---------------------------------------------------------------------------


def test_muon_step_moves_matrices_and_counts():
    W = _leaf((8, 4), seed=20)
    opt = Muon([W], lr=0.01)
    W.grad = torch.randn(8, 4)
    before = W.detach().clone()
    opt.step()
    assert not torch.allclose(W.detach(), before)
    assert opt.momentum_buffers[0].abs().sum() > 0
    assert opt.step_count == 1


def test_muon_step_skips_none_grads():
    W1, W2 = _leaf((4, 4), seed=21), _leaf((4, 4), seed=22)
    opt = Muon([W1, W2], lr=0.01)
    W2.grad = torch.randn(4, 4)
    before = W1.detach().clone()
    opt.step()
    assert torch.allclose(W1.detach(), before)


def test_muon_update_is_gradient_scale_invariant():
    """Muon's signature property: the update ignores gradient magnitude.

    The momentum buffer is normalized by its Frobenius norm inside
    Newton–Schulz, so a 100x larger gradient produces the same step.
    """
    torch.manual_seed(23)
    init = torch.randn(8, 4)
    grad = torch.randn(8, 4)

    results = []
    for factor in (1.0, 100.0):
        W = init.clone().requires_grad_(True)
        opt = Muon([W], lr=0.01)
        W.grad = grad * factor
        opt.step()
        results.append(W.detach().clone())
    assert torch.allclose(results[0], results[1], atol=1e-4)


def test_muon_update_is_near_orthogonal():
    torch.manual_seed(24)
    W = _leaf((8, 8), seed=24)
    before = W.detach().clone()
    opt = Muon([W], lr=0.01)
    W.grad = torch.randn(8, 8)
    opt.step()
    # Square matrix: scale factor is 1, so the applied update is
    # lr * O with O approximately orthogonal.
    update = (before - W.detach()) / 0.01
    s = torch.linalg.svdvals(update)
    assert ((s > 0.4) & (s < 1.6)).all(), s


def test_muon_rectangular_scale_factor():
    """A tall (8, 2) matrix takes a sqrt(8/2) = 2x larger step than its
    transposed twin — the rectangular compensation from the recipe."""
    torch.manual_seed(25)
    grad = torch.randn(8, 2)

    tall = torch.zeros(8, 2, requires_grad=True)
    wide = torch.zeros(2, 8, requires_grad=True)
    opt_t, opt_w = Muon([tall], lr=0.01), Muon([wide], lr=0.01)
    tall.grad = grad
    wide.grad = grad.T
    opt_t.step()
    opt_w.step()

    ratio = tall.detach().norm() / wide.detach().norm()
    assert torch.allclose(ratio, torch.tensor(2.0), atol=1e-3)


def test_muon_weight_decay_shrinks_before_update():
    torch.manual_seed(26)
    W = _leaf((4, 4), seed=26)
    opt = Muon([W], lr=0.1, weight_decay=0.5)
    W.grad = torch.zeros(4, 4)
    before = W.detach().clone()
    opt.step()
    # Zero gradient: the momentum buffer stays zero and Newton–Schulz
    # of a zero matrix is zero, so only the decay term acts.
    assert torch.allclose(W.detach(), before * (1 - 0.1 * 0.5), atol=1e-6)


def test_muon_adamw_side_follows_adamw():
    """The non-matrix group is driven by g2c.training.AdamW itself."""
    from g2c.training import AdamW

    torch.manual_seed(27)
    init = torch.randn(6)
    grad = torch.randn(6)

    b_muon = init.clone().requires_grad_(True)
    b_ref = init.clone().requires_grad_(True)
    opt = Muon([], [b_muon], adamw_lr=1e-3)
    ref = AdamW([b_ref], lr=1e-3)
    b_muon.grad = grad.clone()
    b_ref.grad = grad.clone()
    opt.step()
    ref.step()
    assert torch.allclose(b_muon.detach(), b_ref.detach(), atol=1e-8)


def test_muon_training_smoke():
    """Muon alone can drive a least-squares problem to lower loss."""
    torch.manual_seed(28)
    X = torch.randn(4, 32)
    W_true = torch.randn(4, 4)
    Y = W_true @ X

    W = torch.zeros(4, 4, requires_grad=True)
    opt = Muon([W], lr=0.05, momentum=0.9)

    def loss_value():
        return ((W @ X - Y) ** 2).mean()

    initial = loss_value().item()
    for _ in range(60):
        opt.zero_grad()
        loss = loss_value()
        loss.backward()
        opt.step()
    assert loss_value().item() < 0.2 * initial
