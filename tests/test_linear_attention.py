"""Tests for g2c/linear_attention — Beyond module: efficient sequence models.

Suggested order to implement & turn green:

1. `LinearAttention.forward` (the parallel form) — unblocks
   `test_forward_*` and the causality tests.
2. `LinearAttention.step` (the recurrent form) — unblocks the
   equivalence tests, which are the module's anchor: nearly every bug
   (dropped normalizer, wrong update order, decay applied in one form
   only) fails `test_parallel_matches_recurrent` with a large error.

`feature_map`, construction, `init_state`, and the hybrid plumbing are
provided, so those tests pass from the start.
"""
from __future__ import annotations

import pytest
import torch

from g2c.linear_attention import (
    HybridBlock,
    HybridTransformerLM,
    LinearAttention,
    feature_map,
)
from g2c.linear_attention.attention import EPS

D, H = 16, 4


# ---------------------------------------------------------------------------
# Provided plumbing — green from the start.
# ---------------------------------------------------------------------------


def test_feature_map_is_positive():
    x = torch.linspace(-10, 10, 101)
    phi = feature_map(x)
    assert (phi > 0).all()
    # monotone: bigger input, bigger feature
    assert (phi[1:] >= phi[:-1]).all()


def test_construction_and_decay_range():
    la = LinearAttention(D, H)
    assert la.head_dim == D // H
    decay = la.decay
    assert decay.shape == (H,)
    assert ((decay > 0) & (decay < 1)).all()


def test_construction_rejects_indivisible_heads():
    with pytest.raises(ValueError):
        LinearAttention(D, 3)


def test_init_state_shapes():
    la = LinearAttention(D, H)
    S, z = la.init_state(2)
    assert S.shape == (2, H, D // H, D // H)
    assert z.shape == (2, H, D // H)
    assert S.abs().sum() == 0 and z.abs().sum() == 0


def test_hybrid_pattern_validation():
    with pytest.raises(ValueError):
        HybridBlock(D, H, "sparse")
    with pytest.raises(ValueError):
        HybridTransformerLM(32, D, [], H, 16)
    with pytest.raises(ValueError):
        HybridTransformerLM(32, D, ["linear", "nope"], H, 16)


def test_hybrid_block_kinds():
    from g2c.attention import MultiHeadAttention

    assert isinstance(HybridBlock(D, H, "full").attn, MultiHeadAttention)
    assert isinstance(HybridBlock(D, H, "linear").attn, LinearAttention)


def test_story_patterns_are_parameter_matched_except_for_decay_scalars():
    """Linear layers add only one learned decay scalar per attention head."""
    common = dict(
        vocab_size=32,
        embedding_dim=D,
        num_heads=H,
        max_seq_len=16,
    )
    patterns = {
        "full": ["full"] * 4,
        "linear": ["linear"] * 4,
        "hybrid": ["linear", "linear", "linear", "full"],
    }
    counts = {
        name: sum(p.numel() for p in HybridTransformerLM(
            layer_pattern=pattern, **common
        ).parameters())
        for name, pattern in patterns.items()
    }
    assert counts["linear"] == counts["full"] + 4 * H
    assert counts["hybrid"] == counts["full"] + 3 * H


# ---------------------------------------------------------------------------
# The parallel form
# ---------------------------------------------------------------------------


def test_forward_shape():
    torch.manual_seed(0)
    la = LinearAttention(D, H)
    x = torch.randn(2, 7, D)
    assert la(x).shape == (2, 7, D)


def test_forward_is_causal():
    """Changing the future must not change the past."""
    torch.manual_seed(1)
    la = LinearAttention(D, H)
    x1 = torch.randn(1, 8, D)
    x2 = x1.clone()
    x2[:, 5:] += torch.randn(1, 3, D)
    out1, out2 = la(x1), la(x2)
    assert torch.allclose(out1[:, :5], out2[:, :5], atol=1e-5)
    assert not torch.allclose(out1[:, 5:], out2[:, 5:], atol=1e-3)


def test_forward_matches_undecayed_reference_at_gamma_one():
    """With γ → 1 the decay mask degenerates to the plain causal mask."""
    torch.manual_seed(2)
    la = LinearAttention(D, H)
    la.gamma_logit.data.fill_(50.0)  # sigmoid → 1.0 exactly in float32
    x = torch.randn(1, 6, D)

    q = feature_map(la._split_heads(la.q_proj(x)))
    k = feature_map(la._split_heads(la.k_proj(x)))
    v = la._split_heads(la.v_proj(x))
    scores = q @ k.transpose(-2, -1)
    causal = torch.tril(torch.ones(6, 6, dtype=torch.bool))
    scores = scores * causal
    expected = la.out_proj(
        la._merge_heads(scores @ v / (scores.sum(-1, keepdim=True) + EPS))
    )

    assert torch.allclose(la(x), expected, atol=1e-5)


# ---------------------------------------------------------------------------
# The recurrent form and the equivalence — the module's anchor.
# ---------------------------------------------------------------------------


def test_step_shapes_and_state_is_fixed_size():
    torch.manual_seed(3)
    la = LinearAttention(D, H)
    state = None
    for t in range(20):
        out, state = la.step(torch.randn(2, D), state)
        assert out.shape == (2, D)
    S, z = state
    # After 20 tokens the "cache" is still one matrix per head.
    assert S.shape == (2, H, D // H, D // H)
    assert z.shape == (2, H, D // H)


def test_parallel_matches_recurrent():
    torch.manual_seed(4)
    la = LinearAttention(D, H)
    x = torch.randn(2, 12, D)
    parallel = la(x)

    state = None
    outs = []
    for t in range(12):
        out_t, state = la.step(x[:, t], state)
        outs.append(out_t)
    recurrent = torch.stack(outs, dim=1)

    assert torch.allclose(parallel, recurrent, atol=1e-4), (
        (parallel - recurrent).abs().max()
    )


def test_parallel_matches_recurrent_with_strong_decay():
    """The equivalence must hold for γ well below 1, not just the init."""
    torch.manual_seed(5)
    la = LinearAttention(D, H)
    la.gamma_logit.data.fill_(0.0)  # sigmoid → 0.5: aggressive forgetting
    x = torch.randn(1, 10, D)
    parallel = la(x)
    state = None
    outs = []
    for t in range(10):
        out_t, state = la.step(x[:, t], state)
        outs.append(out_t)
    assert torch.allclose(parallel, torch.stack(outs, dim=1), atol=1e-4)


# ---------------------------------------------------------------------------
# The hybrid model end to end
# ---------------------------------------------------------------------------


def test_hybrid_lm_forward_shape():
    torch.manual_seed(6)
    model = HybridTransformerLM(
        vocab_size=32,
        embedding_dim=D,
        layer_pattern=["linear", "linear", "linear", "full"],
        num_heads=H,
        max_seq_len=16,
    )
    ids = torch.randint(0, 32, (2, 9))
    assert model(ids).shape == (2, 9, 32)
    assert model.num_layers == 4


def test_hybrid_lm_rejects_overlong_input():
    model = HybridTransformerLM(32, D, ["linear"], H, max_seq_len=8)
    with pytest.raises(ValueError):
        model(torch.randint(0, 32, (1, 9)))
