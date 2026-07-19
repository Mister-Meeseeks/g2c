"""Tests for Module 08: multi-head causal self-attention.

Suggested order to implement & turn green:

  1. MultiHeadAttention.forward             → test_forward_*
  2. MultiHeadAttention.attention_weights   → test_attention_weights_*
  3. Optional cached inference path          → test_forward_cached_*

Construction tests, the `causal_mask` static-method tests, and the
parameter-count tests pass from the start — that part of the class is
fully implemented.

The two scaffolded methods share most of their logic (Q/K projection,
reshape, score computation, mask, softmax). Implementing one makes the
other essentially mechanical — but the suite tests them independently
so a regression in one is caught even if the other is correct.

Most tests use very small dimensions (`embedding_dim=8`, `num_heads=2..4`,
`T=3..6`) so the suite runs in well under a second.
"""
from __future__ import annotations

import math

import pytest
import torch

from g2c.attention import MultiHeadAttention
from g2c.transformer import LayerKVCache

# ----------------------------------------------------------------------
# Construction (boilerplate)
# ----------------------------------------------------------------------

def test_construction():
    MultiHeadAttention(embedding_dim=8, num_heads=2)


def test_construction_stores_args():
    a = MultiHeadAttention(embedding_dim=12, num_heads=3)
    assert a.embedding_dim == 12
    assert a.num_heads == 3
    assert a.head_dim == 4
    assert a.causal is True


def test_construction_non_causal():
    a = MultiHeadAttention(embedding_dim=8, num_heads=2, causal=False)
    assert a.causal is False


def test_construction_indivisible_raises():
    """`embedding_dim` must be divisible by `num_heads`."""
    with pytest.raises(ValueError):
        MultiHeadAttention(embedding_dim=8, num_heads=3)


def test_construction_projection_shapes():
    """All four projections are (D, D), regardless of head count.

    The whole point of multi-head-via-reshape is that we DON'T need
    separate (D, head_dim) projections per head — one (D, D) projection
    followed by a view is mathematically equivalent.
    """
    a = MultiHeadAttention(embedding_dim=8, num_heads=4)
    assert a.q_proj.W.shape == (8, 8)
    assert a.k_proj.W.shape == (8, 8)
    assert a.v_proj.W.shape == (8, 8)
    assert a.out_proj.W.shape == (8, 8)


def test_parameters_count():
    """4 Linear layers × (W, b) = 8 parameter tensors."""
    a = MultiHeadAttention(embedding_dim=8, num_heads=2)
    params = list(a.parameters())
    assert len(params) == 8


def test_parameters_independent_of_head_count():
    """Total parameter count is the same whether D is split into 1 head
    or 8 heads. The decomposition is structural, not parametric.
    """
    a1 = MultiHeadAttention(embedding_dim=64, num_heads=1)
    a8 = MultiHeadAttention(embedding_dim=64, num_heads=8)
    n1 = sum(p.numel() for p in a1.parameters())
    n8 = sum(p.numel() for p in a8.parameters())
    assert n1 == n8


# ----------------------------------------------------------------------
# causal_mask (boilerplate)
# ----------------------------------------------------------------------

def test_causal_mask_shape():
    mask = MultiHeadAttention.causal_mask(5)
    assert mask.shape == (5, 5)


def test_causal_mask_dtype():
    mask = MultiHeadAttention.causal_mask(5)
    assert mask.dtype == torch.bool


def test_causal_mask_upper_triangular():
    """Above-diagonal entries are True; diagonal and below are False."""
    mask = MultiHeadAttention.causal_mask(4)
    expected = torch.tensor([
        [False, True,  True,  True],
        [False, False, True,  True],
        [False, False, False, True],
        [False, False, False, False],
    ])
    assert torch.equal(mask, expected)


def test_causal_mask_diagonal_is_unmasked():
    """A position must always be allowed to attend to itself."""
    mask = MultiHeadAttention.causal_mask(6)
    diagonal = torch.diag(mask)
    assert not diagonal.any()


# ----------------------------------------------------------------------
# forward — the main pipeline
# ----------------------------------------------------------------------

def test_forward_shape():
    a = MultiHeadAttention(embedding_dim=8, num_heads=2)
    x = torch.randn(2, 5, 8)
    out = a(x)
    assert out.shape == (2, 5, 8)


def test_forward_shape_single_batch():
    a = MultiHeadAttention(embedding_dim=12, num_heads=3)
    x = torch.randn(1, 4, 12)
    out = a(x)
    assert out.shape == (1, 4, 12)


def test_forward_shape_single_token():
    """A length-1 sequence should still work (degenerate case, no mixing)."""
    a = MultiHeadAttention(embedding_dim=8, num_heads=4)
    x = torch.randn(2, 1, 8)
    out = a(x)
    assert out.shape == (2, 1, 8)


def test_forward_shape_one_head():
    """H=1 is a valid degenerate case; reshape is a no-op."""
    a = MultiHeadAttention(embedding_dim=8, num_heads=1)
    x = torch.randn(2, 4, 8)
    out = a(x)
    assert out.shape == (2, 4, 8)


def test_forward_routes_gradients():
    """A backward pass should populate gradients on every parameter."""
    a = MultiHeadAttention(embedding_dim=8, num_heads=2)
    x = torch.randn(2, 3, 8)
    out = a(x)
    out.sum().backward()
    for p in a.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()


def test_forward_causality():
    """Headline test: with causal=True, changing x at position t must NOT
    change the output at positions < t — across ALL heads.

    The same property as Module 07's single-head version. With multiple
    heads, every head must respect the causal mask independently — so
    the test is even stricter: any one head leaking the future would
    contaminate the concatenated output and fail this test.
    """
    torch.manual_seed(0)
    a = MultiHeadAttention(embedding_dim=8, num_heads=4, causal=True)
    x1 = torch.randn(1, 5, 8)
    x2 = x1.clone()
    x2[0, 3, :] = torch.randn(8)        # mutate position 3 only
    out1 = a(x1)
    out2 = a(x2)
    # Positions 0, 1, 2 must be unchanged — no head can see position 3.
    assert torch.allclose(out1[0, :3], out2[0, :3], atol=1e-6)
    # Positions 3 and 4 SHOULD change — they can see position 3.
    assert not torch.allclose(out1[0, 3:], out2[0, 3:], atol=1e-4)


def test_forward_no_causal_sees_future():
    """With causal=False, every position attends to every other across
    every head, so changing x at any position should change the output
    at all positions.
    """
    torch.manual_seed(0)
    a = MultiHeadAttention(embedding_dim=8, num_heads=2, causal=False)
    x1 = torch.randn(1, 5, 8)
    x2 = x1.clone()
    x2[0, 3, :] = torch.randn(8)
    out1 = a(x1)
    out2 = a(x2)
    assert not torch.allclose(out1[0, 0], out2[0, 0], atol=1e-5)


def test_forward_cached_matches_full_forward():
    """The cached single-token path should match the regular causal path.

    This pins the core KV-cache idea: after token t is processed, its K/V
    projections can be reused while producing token t+1, without changing the
    math.
    """
    torch.manual_seed(0)
    a = MultiHeadAttention(embedding_dim=8, num_heads=2, causal=True)
    x = torch.randn(2, 5, 8)
    full = a(x)

    cache = LayerKVCache()
    pieces = []
    for t in range(x.shape[1]):
        out_t, cache = a.forward_cached(x[:, t:t + 1], cache)
        pieces.append(out_t)

    cached = torch.cat(pieces, dim=1)
    assert cached.shape == full.shape
    assert cache.length == x.shape[1]
    assert torch.allclose(cached, full, atol=1e-5)


def test_forward_cached_requires_single_new_token():
    a = MultiHeadAttention(embedding_dim=8, num_heads=2, causal=True)
    x = torch.randn(1, 2, 8)
    with pytest.raises(ValueError):
        a.forward_cached(x, LayerKVCache())


def test_forward_cached_requires_causal_attention():
    a = MultiHeadAttention(embedding_dim=8, num_heads=2, causal=False)
    x = torch.randn(1, 1, 8)
    with pytest.raises(ValueError):
        a.forward_cached(x, LayerKVCache())


# ----------------------------------------------------------------------
# attention_weights — visualization-friendly weight extraction
# ----------------------------------------------------------------------

def test_attention_weights_shape():
    """(B, H, T, T) — one (T, T) attention matrix per head."""
    a = MultiHeadAttention(embedding_dim=8, num_heads=4)
    x = torch.randn(2, 5, 8)
    w = a.attention_weights(x)
    assert w.shape == (2, 4, 5, 5)


def test_attention_weights_sum_to_one():
    """Every (B, H, i, :) row is a probability distribution."""
    a = MultiHeadAttention(embedding_dim=8, num_heads=2)
    x = torch.randn(2, 5, 8)
    w = a.attention_weights(x)
    row_sums = w.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_attention_weights_nonnegative():
    a = MultiHeadAttention(embedding_dim=8, num_heads=2)
    x = torch.randn(2, 5, 8)
    w = a.attention_weights(x)
    assert (w >= 0).all()


def test_attention_weights_causal_zero_above_diagonal():
    """In every head, weights above the diagonal are exactly zero."""
    a = MultiHeadAttention(embedding_dim=8, num_heads=4, causal=True)
    x = torch.randn(1, 6, 8)
    w = a.attention_weights(x)         # (1, 4, 6, 6)
    for h in range(4):
        upper = torch.triu(w[0, h], diagonal=1)
        assert torch.allclose(upper, torch.zeros_like(upper), atol=1e-6)


def test_attention_weights_causal_first_row_is_one_hot():
    """In every head, position 0 can only attend to position 0."""
    a = MultiHeadAttention(embedding_dim=8, num_heads=4, causal=True)
    x = torch.randn(1, 5, 8)
    w = a.attention_weights(x)
    for h in range(4):
        assert w[0, h, 0, 0].item() == pytest.approx(1.0, abs=1e-6)
        assert torch.allclose(w[0, h, 0, 1:], torch.zeros(4), atol=1e-6)


def test_attention_weights_non_causal_full_matrix():
    """With causal=False, every weight should be strictly positive."""
    a = MultiHeadAttention(embedding_dim=8, num_heads=2, causal=False)
    x = torch.randn(1, 5, 8)
    w = a.attention_weights(x)
    assert (w > 0).all()


def test_attention_weights_heads_are_distinct():
    """Different heads should produce different attention patterns
    (with random init they almost certainly will).

    Pins down that the implementation actually splits D into H
    independent subspaces — a wrong implementation that uses the same
    q/k for every head would produce H IDENTICAL weight matrices.
    """
    torch.manual_seed(0)
    a = MultiHeadAttention(embedding_dim=8, num_heads=4, causal=False)
    x = torch.randn(1, 5, 8)
    w = a.attention_weights(x)             # (1, 4, 5, 5)
    for i in range(4):
        for j in range(i + 1, 4):
            assert not torch.allclose(w[0, i], w[0, j], atol=1e-4)


def test_attention_weights_consistent_with_forward():
    """`forward(x)` must equal applying the same projections, reshape,
    weights, value-mix, concat, and out_proj as `attention_weights(x)`
    plus the value path.

    Pins down that the two scaffolded methods agree about projections,
    reshape, scaling, and mask. Catches the common bug where one
    applies the mask but the other doesn't, or one uses sqrt(d_h) and
    the other uses sqrt(D).
    """
    torch.manual_seed(0)
    D, H = 8, 2
    a = MultiHeadAttention(embedding_dim=D, num_heads=H, causal=True)
    x = torch.randn(2, 5, D)
    B, T, _ = x.shape
    d_h = D // H
    w = a.attention_weights(x)                                 # (B, H, T, T)
    v = a.v_proj(x).view(B, T, H, d_h).transpose(1, 2)         # (B, H, T, d_h)
    mixed = w @ v                                              # (B, H, T, d_h)
    concat = mixed.transpose(1, 2).contiguous().view(B, T, D)  # (B, T, D)
    expected = a.out_proj(concat)
    actual = a(x)
    assert torch.allclose(actual, expected, atol=1e-5)


# ----------------------------------------------------------------------
# Mathematical sanity — the sqrt(head_dim) scaling
# ----------------------------------------------------------------------

def test_attention_weights_use_sqrt_head_dim_scaling():
    """The scores must be scaled by 1/sqrt(head_dim), NOT 1/sqrt(D).

    The single most common multi-head bug is keeping the single-head
    scaling factor sqrt(D) when the per-head dot products are now over
    a d_h-dimensional subspace. We pin down the correct scaling by
    forcing q_proj and k_proj to the identity, then comparing against
    an explicit per-head softmax(x_h @ x_h.T / sqrt(d_h)).
    """
    torch.manual_seed(0)
    D, H = 8, 4
    d_h = D // H
    a = MultiHeadAttention(embedding_dim=D, num_heads=H, causal=False)
    with torch.no_grad():
        a.q_proj.W.copy_(torch.eye(D))
        a.q_proj.b.zero_()
        a.k_proj.W.copy_(torch.eye(D))
        a.k_proj.b.zero_()
    x = torch.randn(1, 3, D)
    w = a.attention_weights(x)             # (1, H, T, T)
    # Build expected per-head: split x into H slices of d_h, do
    # softmax(x_h @ x_h.T / sqrt(d_h)) for each.
    x_per_head = x.view(1, 3, H, d_h).transpose(1, 2)   # (1, H, 3, d_h)
    expected_scores = (x_per_head @ x_per_head.transpose(-2, -1)) / math.sqrt(d_h)
    expected_w = expected_scores.softmax(dim=-1)
    assert torch.allclose(w, expected_w, atol=1e-5)
