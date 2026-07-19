"""Tests for Module 07: single-head causal self-attention.

Suggested order to implement & turn green:

  1. SelfAttention.forward             → test_forward_*
  2. SelfAttention.attention_weights   → test_attention_weights_*

The construction tests and the `causal_mask` static-method tests pass
from the start because that part of the class is fully implemented.

The two scaffolded methods share most of their logic (Q/K projection,
score computation, mask, softmax). Implementing one makes the other
trivial — but the suite tests them independently so a regression in one
is caught even if the other is correct.

Most tests use very small dimensions (`embedding_dim=4`, `T=3..6`) so
the suite runs in well under a second.
"""
from __future__ import annotations

import math

import pytest
import torch

from g2c.attention import SelfAttention

# ----------------------------------------------------------------------
# Construction (boilerplate)
# ----------------------------------------------------------------------

def test_construction():
    SelfAttention(embedding_dim=4)


def test_construction_stores_args():
    a = SelfAttention(embedding_dim=8)
    assert a.embedding_dim == 8
    assert a.causal is True


def test_construction_non_causal():
    a = SelfAttention(embedding_dim=4, causal=False)
    assert a.causal is False


def test_construction_projection_shapes():
    a = SelfAttention(embedding_dim=8)
    assert a.q_proj.W.shape == (8, 8)
    assert a.k_proj.W.shape == (8, 8)
    assert a.v_proj.W.shape == (8, 8)
    assert a.out_proj.W.shape == (8, 8)


def test_parameters_count():
    """4 Linear layers × (W, b) = 8 parameter tensors."""
    a = SelfAttention(embedding_dim=4)
    params = list(a.parameters())
    assert len(params) == 8


# ----------------------------------------------------------------------
# causal_mask (boilerplate)
# ----------------------------------------------------------------------

def test_causal_mask_shape():
    mask = SelfAttention.causal_mask(5)
    assert mask.shape == (5, 5)


def test_causal_mask_dtype():
    mask = SelfAttention.causal_mask(5)
    assert mask.dtype == torch.bool


def test_causal_mask_upper_triangular():
    """Above-diagonal entries are True; diagonal and below are False."""
    mask = SelfAttention.causal_mask(4)
    expected = torch.tensor([
        [False, True,  True,  True],
        [False, False, True,  True],
        [False, False, False, True],
        [False, False, False, False],
    ])
    assert torch.equal(mask, expected)


def test_causal_mask_diagonal_is_unmasked():
    """A position must always be allowed to attend to itself."""
    mask = SelfAttention.causal_mask(6)
    diagonal = torch.diag(mask)
    assert not diagonal.any()


# ----------------------------------------------------------------------
# forward — the main pipeline
# ----------------------------------------------------------------------

def test_forward_shape():
    a = SelfAttention(embedding_dim=4)
    x = torch.randn(2, 5, 4)
    out = a(x)
    assert out.shape == (2, 5, 4)


def test_forward_shape_single_batch():
    a = SelfAttention(embedding_dim=8)
    x = torch.randn(1, 3, 8)
    out = a(x)
    assert out.shape == (1, 3, 8)


def test_forward_shape_single_token():
    """A length-1 sequence should still work (degenerate case, no mixing)."""
    a = SelfAttention(embedding_dim=4)
    x = torch.randn(2, 1, 4)
    out = a(x)
    assert out.shape == (2, 1, 4)


def test_forward_routes_gradients():
    """A backward pass should populate gradients on every parameter."""
    a = SelfAttention(embedding_dim=4)
    x = torch.randn(2, 3, 4)
    out = a(x)
    out.sum().backward()
    for p in a.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()


def test_forward_causality():
    """Headline test: with causal=True, changing x at position t must NOT
    change the output at positions < t.

    This is the property that makes self-attention safe to use in a
    language model — without it, the model trivially cheats by peeking at
    the answer during training.
    """
    torch.manual_seed(0)
    a = SelfAttention(embedding_dim=4, causal=True)
    x1 = torch.randn(1, 5, 4)
    x2 = x1.clone()
    x2[0, 3, :] = torch.randn(4)        # mutate position 3 only
    out1 = a(x1)
    out2 = a(x2)
    # Positions 0, 1, 2 must be unchanged — they cannot see position 3.
    assert torch.allclose(out1[0, :3], out2[0, :3], atol=1e-6)
    # Positions 3 and 4 SHOULD change — they can see position 3.
    assert not torch.allclose(out1[0, 3:], out2[0, 3:], atol=1e-4)


def test_forward_no_causal_sees_future():
    """With causal=False, every position attends to every other position,
    so changing x at any position should change the output at all positions.
    """
    torch.manual_seed(0)
    a = SelfAttention(embedding_dim=4, causal=False)
    x1 = torch.randn(1, 5, 4)
    x2 = x1.clone()
    x2[0, 3, :] = torch.randn(4)
    out1 = a(x1)
    out2 = a(x2)
    # Even position 0 — which the causal mask would have prevented from
    # seeing position 3 — should now reflect the change.
    assert not torch.allclose(out1[0, 0], out2[0, 0], atol=1e-5)


# ----------------------------------------------------------------------
# attention_weights — visualization-friendly weight extraction
# ----------------------------------------------------------------------

def test_attention_weights_shape():
    a = SelfAttention(embedding_dim=4)
    x = torch.randn(2, 5, 4)
    w = a.attention_weights(x)
    assert w.shape == (2, 5, 5)


def test_attention_weights_sum_to_one():
    """Every row of the attention matrix is a probability distribution."""
    a = SelfAttention(embedding_dim=4)
    x = torch.randn(2, 5, 4)
    w = a.attention_weights(x)
    row_sums = w.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_attention_weights_nonnegative():
    """Softmax output is always nonnegative."""
    a = SelfAttention(embedding_dim=4)
    x = torch.randn(2, 5, 4)
    w = a.attention_weights(x)
    assert (w >= 0).all()


def test_attention_weights_causal_zero_above_diagonal():
    """With causal=True, attention from position i to position j > i is 0
    (the -inf in scores becomes 0 after softmax).
    """
    a = SelfAttention(embedding_dim=4, causal=True)
    x = torch.randn(1, 6, 4)
    w = a.attention_weights(x)         # (1, 6, 6)
    # Upper triangle (excluding diagonal) should be all zeros.
    upper = torch.triu(w[0], diagonal=1)
    assert torch.allclose(upper, torch.zeros_like(upper), atol=1e-6)


def test_attention_weights_causal_first_row_is_one_hot():
    """Row 0 can only attend to position 0, so weight[0, 0] must be 1
    and weight[0, j>0] must be 0.
    """
    a = SelfAttention(embedding_dim=4, causal=True)
    x = torch.randn(1, 5, 4)
    w = a.attention_weights(x)
    assert w[0, 0, 0].item() == pytest.approx(1.0, abs=1e-6)
    assert torch.allclose(w[0, 0, 1:], torch.zeros(4), atol=1e-6)


def test_attention_weights_non_causal_full_matrix():
    """With causal=False, every weight should be strictly positive — no
    -inf masking suppresses anything.
    """
    a = SelfAttention(embedding_dim=4, causal=False)
    x = torch.randn(1, 5, 4)
    w = a.attention_weights(x)
    assert (w > 0).all()


def test_attention_weights_consistent_with_forward():
    """`forward(x)` must equal `out_proj(attention_weights(x) @ v_proj(x))`.

    This pins down that the two scaffolded methods agree about what the
    attention weights are. Catches the common bug where the student
    implements `forward` and `attention_weights` slightly differently
    (e.g., one applies the mask, the other doesn't).
    """
    torch.manual_seed(0)
    a = SelfAttention(embedding_dim=4, causal=True)
    x = torch.randn(2, 5, 4)
    w = a.attention_weights(x)
    v = a.v_proj(x)
    expected = a.out_proj(w @ v)
    actual = a(x)
    assert torch.allclose(actual, expected, atol=1e-5)


# ----------------------------------------------------------------------
# Mathematical sanity — the sqrt(D) scaling
# ----------------------------------------------------------------------

def test_attention_weights_use_sqrt_d_scaling():
    """The attention scores should be scaled by 1/sqrt(D).

    We construct an attention layer whose Q and K projections are the
    identity, so `scores = x @ x^T / sqrt(D)`. We then verify the actual
    attention weights against the same softmax computed by hand with
    1/sqrt(D) scaling. If the student forgot to divide by sqrt(D), the
    weights will be wrong by a softmax-temperature factor.
    """
    torch.manual_seed(0)
    D = 4
    a = SelfAttention(embedding_dim=D, causal=False)
    # Force Q and K projections to the identity (W = I, b = 0).
    with torch.no_grad():
        a.q_proj.W.copy_(torch.eye(D))
        a.q_proj.b.zero_()
        a.k_proj.W.copy_(torch.eye(D))
        a.k_proj.b.zero_()
    x = torch.randn(1, 3, D)
    w = a.attention_weights(x)             # (1, 3, 3)
    # Expected: softmax(x @ x^T / sqrt(D), dim=-1).
    expected_scores = (x @ x.transpose(-2, -1)) / math.sqrt(D)
    expected_w = expected_scores.softmax(dim=-1)
    assert torch.allclose(w, expected_w, atol=1e-5)
