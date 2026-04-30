"""Tests for Module 05: embeddings and positions.

Suggested order to implement & turn green:

  1. TokenEmbedding.forward                       → test_token_*
  2. LearnedPositionalEmbedding.forward           → test_learned_pos_*
  3. SinusoidalPositionalEmbedding.__init__       → test_sinusoidal_construction_*
                                                   (and the rest)
  4. SinusoidalPositionalEmbedding.forward        → test_sinusoidal_forward_*
  5. RotaryEmbedding.__init__ (cos/sin tables)    → test_rotary_construction_*
                                                   (and the rest)
  6. RotaryEmbedding.forward                      → test_rotary_*

Construction tests for `TokenEmbedding` and `LearnedPositionalEmbedding`
pass from the start because their `__init__`s are fully implemented (the
weight table is just a uniform random init).

Construction for the sinusoidal and rotary classes only succeeds once
their `__init__` table-building is implemented — that's where the math is.
"""
from __future__ import annotations

import math

import pytest
import torch

from g2c.embeddings import (
    LearnedPositionalEmbedding,
    RotaryEmbedding,
    SinusoidalPositionalEmbedding,
    TokenEmbedding,
)


# ----------------------------------------------------------------------
# TokenEmbedding — construction (boilerplate)
# ----------------------------------------------------------------------

def test_token_emb_weight_shape():
    emb = TokenEmbedding(vocab_size=100, embedding_dim=16)
    assert emb.weight.shape == (100, 16)


def test_token_emb_requires_grad():
    emb = TokenEmbedding(vocab_size=100, embedding_dim=16)
    assert emb.weight.requires_grad


def test_token_emb_parameters_yields_weight():
    emb = TokenEmbedding(vocab_size=100, embedding_dim=16)
    params = list(emb.parameters())
    assert len(params) == 1
    assert params[0] is emb.weight


# ----------------------------------------------------------------------
# TokenEmbedding — forward
# ----------------------------------------------------------------------

def test_token_emb_forward_shape_2d_input():
    emb = TokenEmbedding(vocab_size=100, embedding_dim=16)
    ids = torch.randint(0, 100, (4, 8))
    assert emb(ids).shape == (4, 8, 16)


def test_token_emb_forward_shape_1d_input():
    emb = TokenEmbedding(vocab_size=100, embedding_dim=16)
    ids = torch.tensor([1, 2, 3])
    assert emb(ids).shape == (3, 16)


def test_token_emb_forward_lookup_rows():
    """Each output row should literally equal the corresponding row of weight."""
    emb = TokenEmbedding(vocab_size=10, embedding_dim=4)
    ids = torch.tensor([3, 5, 7])
    out = emb(ids)
    assert torch.allclose(out[0], emb.weight[3])
    assert torch.allclose(out[1], emb.weight[5])
    assert torch.allclose(out[2], emb.weight[7])


def test_token_emb_backward_routes_to_weight():
    emb = TokenEmbedding(vocab_size=10, embedding_dim=4)
    ids = torch.tensor([1, 2, 3])
    emb(ids).sum().backward()
    assert emb.weight.grad is not None
    assert emb.weight.grad.shape == emb.weight.shape


# ----------------------------------------------------------------------
# LearnedPositionalEmbedding — construction (boilerplate)
# ----------------------------------------------------------------------

def test_learned_pos_weight_shape():
    pe = LearnedPositionalEmbedding(max_seq_len=128, embedding_dim=16)
    assert pe.weight.shape == (128, 16)


def test_learned_pos_requires_grad():
    pe = LearnedPositionalEmbedding(max_seq_len=128, embedding_dim=16)
    assert pe.weight.requires_grad


# ----------------------------------------------------------------------
# LearnedPositionalEmbedding — forward
# ----------------------------------------------------------------------

def test_learned_pos_forward_shape():
    pe = LearnedPositionalEmbedding(max_seq_len=128, embedding_dim=16)
    assert pe(seq_len=10).shape == (10, 16)


def test_learned_pos_forward_returns_prefix():
    pe = LearnedPositionalEmbedding(max_seq_len=128, embedding_dim=16)
    out = pe(seq_len=10)
    assert torch.allclose(out, pe.weight[:10])


def test_learned_pos_forward_seq_len_one():
    pe = LearnedPositionalEmbedding(max_seq_len=128, embedding_dim=16)
    assert pe(seq_len=1).shape == (1, 16)


# ----------------------------------------------------------------------
# SinusoidalPositionalEmbedding — construction (table is the lesson)
# ----------------------------------------------------------------------

def test_sinusoidal_weight_shape():
    pe = SinusoidalPositionalEmbedding(max_seq_len=128, embedding_dim=16)
    assert pe.weight.shape == (128, 16)


def test_sinusoidal_no_parameters():
    pe = SinusoidalPositionalEmbedding(max_seq_len=128, embedding_dim=16)
    assert list(pe.parameters()) == []


def test_sinusoidal_no_grad_on_table():
    """The sinusoidal table is fixed — should not require gradient."""
    pe = SinusoidalPositionalEmbedding(max_seq_len=128, embedding_dim=16)
    assert not pe.weight.requires_grad


def test_sinusoidal_position_zero_values():
    """At pos=0: sin(0)=0 in even slots, cos(0)=1 in odd slots."""
    pe = SinusoidalPositionalEmbedding(max_seq_len=10, embedding_dim=8)
    pos_0 = pe.weight[0]
    assert torch.allclose(pos_0[0::2], torch.zeros(4), atol=1e-6)   # sin(0) = 0
    assert torch.allclose(pos_0[1::2], torch.ones(4), atol=1e-6)    # cos(0) = 1


def test_sinusoidal_value_range():
    """All values are sines and cosines: must be in [-1, 1]."""
    pe = SinusoidalPositionalEmbedding(max_seq_len=128, embedding_dim=16)
    assert pe.weight.min().item() >= -1.0 - 1e-6
    assert pe.weight.max().item() <= 1.0 + 1e-6


def test_sinusoidal_known_value_at_pos_one_dim_zero():
    """PE[1, 0] = sin(1 / 10000^(0/d)) = sin(1)."""
    pe = SinusoidalPositionalEmbedding(max_seq_len=10, embedding_dim=4)
    assert pe.weight[1, 0].item() == pytest.approx(math.sin(1.0), abs=1e-5)


def test_sinusoidal_known_value_at_pos_one_dim_one():
    """PE[1, 1] = cos(1 / 10000^(0/d)) = cos(1)."""
    pe = SinusoidalPositionalEmbedding(max_seq_len=10, embedding_dim=4)
    assert pe.weight[1, 1].item() == pytest.approx(math.cos(1.0), abs=1e-5)


def test_sinusoidal_odd_dim_raises():
    """embedding_dim must be even."""
    with pytest.raises(ValueError):
        SinusoidalPositionalEmbedding(max_seq_len=10, embedding_dim=7)


def test_sinusoidal_forward_returns_prefix():
    pe = SinusoidalPositionalEmbedding(max_seq_len=128, embedding_dim=16)
    out = pe(seq_len=10)
    assert out.shape == (10, 16)
    assert torch.allclose(out, pe.weight[:10])


# ----------------------------------------------------------------------
# RotaryEmbedding — construction
# ----------------------------------------------------------------------

def test_rotary_construction_runs():
    RotaryEmbedding(max_seq_len=128, embedding_dim=16)


def test_rotary_no_parameters():
    rope = RotaryEmbedding(max_seq_len=128, embedding_dim=16)
    assert list(rope.parameters()) == []


def test_rotary_cos_sin_shapes():
    rope = RotaryEmbedding(max_seq_len=128, embedding_dim=16)
    assert rope.cos.shape == (128, 16)
    assert rope.sin.shape == (128, 16)


def test_rotary_cos_at_position_zero_is_one():
    """At pos=0: cos = 1 everywhere, sin = 0 everywhere."""
    rope = RotaryEmbedding(max_seq_len=10, embedding_dim=8)
    assert torch.allclose(rope.cos[0], torch.ones(8), atol=1e-6)
    assert torch.allclose(rope.sin[0], torch.zeros(8), atol=1e-6)


def test_rotary_odd_dim_raises():
    with pytest.raises(ValueError):
        RotaryEmbedding(max_seq_len=10, embedding_dim=7)


# ----------------------------------------------------------------------
# RotaryEmbedding — forward
# ----------------------------------------------------------------------

def test_rotary_forward_preserves_shape():
    rope = RotaryEmbedding(max_seq_len=128, embedding_dim=16)
    x = torch.randn(2, 8, 16)
    assert rope(x).shape == x.shape


def test_rotary_forward_at_position_zero_is_identity():
    """A sequence of length 1 means we only apply pos=0, which is identity."""
    rope = RotaryEmbedding(max_seq_len=128, embedding_dim=16)
    x = torch.randn(2, 1, 16)
    assert torch.allclose(rope(x), x, atol=1e-6)


def test_rotary_forward_preserves_norm():
    """A rotation preserves the L2 norm of each rotated vector."""
    rope = RotaryEmbedding(max_seq_len=128, embedding_dim=16)
    x = torch.randn(1, 8, 16)
    y = rope(x)
    x_norms = (x ** 2).sum(dim=-1)
    y_norms = (y ** 2).sum(dim=-1)
    assert torch.allclose(x_norms, y_norms, atol=1e-5)


def test_rotary_forward_distinct_at_different_positions():
    """Same vector, applied at different positions, should produce different
    outputs (otherwise the rotation isn't doing anything)."""
    rope = RotaryEmbedding(max_seq_len=10, embedding_dim=16)
    x = torch.randn(16)
    seq = torch.zeros(1, 5, 16)
    seq[0, 0] = x
    seq[0, 4] = x
    out = rope(seq)
    # Position 0 → identity; position 4 → rotated. They should differ.
    assert not torch.allclose(out[0, 0], out[0, 4], atol=1e-3)


def test_rotary_relative_position_property():
    """The headline property of RoPE: dot(R(q, m), R(k, n)) depends only on
    the relative offset (n - m), not on absolute m or n.

    This is the reason RoPE works so well in attention — token-pair scores
    naturally encode relative position rather than absolute position."""
    rope = RotaryEmbedding(max_seq_len=20, embedding_dim=16)

    torch.manual_seed(0)
    q = torch.randn(16)
    k = torch.randn(16)

    def rotated_dot(m: int, n: int) -> float:
        L = max(m, n) + 1
        seq_q = torch.zeros(1, L, 16)
        seq_k = torch.zeros(1, L, 16)
        seq_q[0, m] = q
        seq_k[0, n] = k
        return (rope(seq_q)[0, m] * rope(seq_k)[0, n]).sum().item()

    # Same relative offset (= 2), three different absolute placements
    a = rotated_dot(0, 2)
    b = rotated_dot(3, 5)
    c = rotated_dot(7, 9)
    assert a == pytest.approx(b, abs=1e-4)
    assert a == pytest.approx(c, abs=1e-4)
