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
    SkipGramEmbeddingModel,
    TokenEmbedding,
    analogy,
    load_glove_subset,
    make_skipgram_pairs,
    nearest_by_cosine,
    normalized,
    train_skipgram,
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


def test_skipgram_model_construction_parameters():
    model = SkipGramEmbeddingModel(vocab_size=25, embedding_dim=8)
    params = model.parameters()
    assert len(params) == 3
    assert params[0].shape == (25, 8)
    assert params[1].shape == (8, 25)
    assert params[2].shape == (25,)


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


# ----------------------------------------------------------------------
# Module 05 notebook helpers — skip-gram and vector similarity
# ----------------------------------------------------------------------

def test_make_skipgram_pairs_window_one():
    centers, contexts = make_skipgram_pairs([10, 20, 30], window=1)
    assert torch.equal(centers, torch.tensor([10, 20, 20, 30]))
    assert torch.equal(contexts, torch.tensor([20, 10, 30, 20]))


def test_make_skipgram_pairs_rejects_bad_window():
    with pytest.raises(ValueError):
        make_skipgram_pairs([1, 2, 3], window=0)


def test_skipgram_model_forward_shape():
    model = SkipGramEmbeddingModel(vocab_size=10, embedding_dim=4)
    logits = model(torch.tensor([1, 2, 3]))
    assert logits.shape == (3, 10)


def test_train_skipgram_reduces_loss():
    torch.manual_seed(0)
    centers, contexts = make_skipgram_pairs([0, 1, 0, 1, 0, 1, 0, 1], window=1)
    model = SkipGramEmbeddingModel(vocab_size=2, embedding_dim=4)

    losses = train_skipgram(
        model,
        centers,
        contexts,
        steps=80,
        batch_size=8,
        lr=0.2,
        generator=torch.Generator().manual_seed(0),
    )

    assert losses[-1] < losses[0]


def test_normalized_unit_norm_and_zero_safe():
    v = torch.tensor([3.0, 4.0])
    assert torch.allclose(normalized(v), torch.tensor([0.6, 0.8]))
    assert torch.allclose(normalized(torch.zeros(2)), torch.zeros(2))


def test_load_glove_subset(tmp_path):
    path = tmp_path / "mini_glove.txt"
    path.write_text(
        "king 1.0 0.0\n"
        "queen 0.8 0.2\n"
        "cat 0.0 1.0\n",
        encoding="utf-8",
    )

    vectors = load_glove_subset(path, {"king", "cat"})

    assert set(vectors) == {"king", "cat"}
    assert torch.allclose(vectors["king"], torch.tensor([1.0, 0.0]))


def test_nearest_by_cosine_excludes_and_sorts():
    vectors = {
        "east": torch.tensor([1.0, 0.0]),
        "northeast": torch.tensor([1.0, 1.0]),
        "north": torch.tensor([0.0, 1.0]),
        "west": torch.tensor([-1.0, 0.0]),
    }
    rows = nearest_by_cosine(
        torch.tensor([1.0, 0.0]),
        vectors,
        exclude={"east"},
        top_k=2,
    )
    assert [word for word, _ in rows] == ["northeast", "north"]
    assert rows[0][1] > rows[1][1]


def test_analogy_uses_vector_arithmetic():
    vectors = {
        "king": torch.tensor([1.0, 1.0]),
        "man": torch.tensor([1.0, 0.0]),
        "woman": torch.tensor([0.0, 1.0]),
        "queen": torch.tensor([0.0, 2.0]),
        "prince": torch.tensor([0.5, 1.0]),
    }
    rows = analogy("king", "man", "woman", vectors, top_k=1)
    assert rows[0][0] == "queen"
