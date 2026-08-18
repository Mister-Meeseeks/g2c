"""Tests for g2c/moe — Beyond module: mixture of experts.

Suggested order to implement & turn green:

1. `Router.forward` — unblocks `test_router_*`.
2. `MoEFeedForward.forward` — unblocks `test_moe_forward_*`,
   `test_moe_e1_k1_matches_dense`, and the gradient-behavior tests.
   (Requires Module 09's `FeedForward.forward` — Beyond modules assume
   the numbered prerequisites are done.)
3. `MoEFeedForward.load_balancing_loss` — unblocks `test_balance_*`.

Construction, validation, and parameter-count tests pass from the
start, as a sanity check on the scaffold itself.
"""
from __future__ import annotations

import pytest
import torch

from g2c.moe import MoEBlock, MoEFeedForward, MoETransformerLM, Router
from g2c.transformer import TransformerLM

D = 16


# ---------------------------------------------------------------------------
# Construction and accounting — green from the start.
# ---------------------------------------------------------------------------


def test_router_construction():
    router = Router(D, num_experts=4, top_k=2)
    assert router.num_experts == 4
    assert router.top_k == 2
    assert router.last_probs is None


@pytest.mark.parametrize("bad_k", [0, 5, -1])
def test_router_rejects_bad_top_k(bad_k):
    with pytest.raises(ValueError):
        Router(D, num_experts=4, top_k=bad_k)


def test_expert_parameter_count_matches_ffn_formula():
    """One expert is one Module 09 FFN: 8D² weights + 5D biases."""
    moe = MoEFeedForward(D, num_experts=3, top_k=1)
    assert moe.expert_parameter_count() == 8 * D * D + 5 * D


def test_moe_layer_parameter_count():
    moe = MoEFeedForward(D, num_experts=4, top_k=2)
    expected = 4 * (8 * D * D + 5 * D) + (D * 4 + 4)  # experts + gate
    assert sum(p.numel() for p in moe.parameters()) == expected


def test_total_vs_active_parameter_split():
    model = MoETransformerLM(
        vocab_size=32,
        embedding_dim=D,
        num_layers=2,
        num_heads=2,
        max_seq_len=16,
        num_experts=4,
        top_k=2,
    )
    total = model.total_parameter_count()
    active = model.active_parameter_count()
    per_expert = model.blocks[0].moe_ffn.expert_parameter_count()
    assert total - active == 2 * (4 - 2) * per_expert
    assert active < total


def test_k1_active_count_matches_dense_plus_routers():
    """The notebook's E=8, k=1 comparison differs only by its routers."""
    common = dict(
        vocab_size=32,
        embedding_dim=D,
        num_layers=2,
        num_heads=2,
        max_seq_len=16,
    )
    dense = TransformerLM(**common)
    moe = MoETransformerLM(**common, num_experts=8, top_k=1)
    dense_count = sum(p.numel() for p in dense.parameters())
    router_count = common["num_layers"] * (D * 8 + 8)
    assert moe.active_parameter_count() == dense_count + router_count


def test_balance_loss_before_forward_raises():
    moe = MoEFeedForward(D, num_experts=4, top_k=2)
    with pytest.raises((RuntimeError, NotImplementedError)):
        moe.load_balancing_loss()


# ---------------------------------------------------------------------------
# Router.forward
# ---------------------------------------------------------------------------


def test_router_output_shapes_and_ranges():
    torch.manual_seed(0)
    router = Router(D, num_experts=6, top_k=3)
    x = torch.randn(10, D)
    weights, indices = router(x)
    assert weights.shape == (10, 3)
    assert indices.shape == (10, 3)
    assert indices.min() >= 0 and indices.max() < 6
    # top-k of a softmax never repeats an expert within a row
    for row in indices:
        assert len(set(row.tolist())) == 3


def test_router_weights_renormalized():
    torch.manual_seed(1)
    router = Router(D, num_experts=6, top_k=2)
    weights, _ = router(torch.randn(7, D))
    assert torch.allclose(weights.sum(dim=-1), torch.ones(7), atol=1e-5)


def test_router_stores_full_distribution_attached():
    torch.manual_seed(2)
    router = Router(D, num_experts=4, top_k=2)
    router(torch.randn(5, D))
    assert router.last_probs is not None
    assert router.last_probs.shape == (5, 4)
    assert router.last_probs.requires_grad
    assert torch.allclose(router.last_probs.sum(dim=-1), torch.ones(5), atol=1e-5)


# ---------------------------------------------------------------------------
# MoEFeedForward.forward
# ---------------------------------------------------------------------------


def test_moe_forward_shape():
    torch.manual_seed(3)
    moe = MoEFeedForward(D, num_experts=4, top_k=2)
    x = torch.randn(2, 5, D)
    out = moe(x)
    assert out.shape == x.shape
    assert moe.last_indices is not None
    assert moe.last_indices.shape == (10, 2)


def test_moe_e1_k1_matches_dense():
    """With one expert and k=1, MoE IS Module 09's FFN — exactly."""
    torch.manual_seed(4)
    moe = MoEFeedForward(D, num_experts=1, top_k=1)
    ffn = moe.experts[0]
    x = torch.randn(2, 5, D)
    assert torch.allclose(moe(x), ffn(x), atol=1e-5)


def test_moe_gradients_reach_only_selected_experts():
    torch.manual_seed(5)
    moe = MoEFeedForward(D, num_experts=2, top_k=1)
    # Rig the gate so expert 0 always wins.
    moe.router.gate.W.data.zero_()
    moe.router.gate.b.data = torch.tensor([5.0, -5.0])
    out = moe(torch.randn(3, 4, D))
    out.sum().backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in moe.experts[0].parameters()
    )
    assert all(
        p.grad is None or p.grad.abs().sum() == 0
        for p in moe.experts[1].parameters()
    )


def test_renormalized_k1_task_gradient_to_router_effectively_vanishes():
    """A single surviving weight is 1, independent of router confidence."""
    torch.manual_seed(10)
    moe = MoEFeedForward(D, num_experts=4, top_k=1)
    moe(torch.randn(3, 5, D)).square().mean().backward()
    assert moe.router.gate.W.grad is not None
    assert moe.router.gate.W.grad.abs().sum() < 1e-6


def test_k2_task_gradient_reaches_router_through_relative_weights():
    """With two survivors, their renormalized relative weights can learn."""
    torch.manual_seed(10)
    moe = MoEFeedForward(D, num_experts=4, top_k=2)
    moe(torch.randn(3, 5, D)).square().mean().backward()
    assert moe.router.gate.W.grad is not None
    assert moe.router.gate.W.grad.abs().sum() > 1e-5


# ---------------------------------------------------------------------------
# load_balancing_loss
# ---------------------------------------------------------------------------


def _rig_routing(moe: MoEFeedForward, probs: torch.Tensor, indices: torch.Tensor):
    moe.router.last_probs = probs
    moe.last_indices = indices


def test_balance_loss_uniform_routing_is_one():
    moe = MoEFeedForward(D, num_experts=4, top_k=1)
    probs = torch.full((12, 4), 0.25)
    indices = (torch.arange(12) % 4).reshape(12, 1)
    _rig_routing(moe, probs, indices)
    assert torch.allclose(moe.load_balancing_loss(), torch.tensor(1.0), atol=1e-5)


def test_balance_loss_collapsed_routing_is_num_experts():
    moe = MoEFeedForward(D, num_experts=4, top_k=1)
    probs = torch.zeros(12, 4)
    probs[:, 0] = 1.0
    indices = torch.zeros(12, 1, dtype=torch.long)
    _rig_routing(moe, probs, indices)
    assert torch.allclose(moe.load_balancing_loss(), torch.tensor(4.0), atol=1e-5)


def test_balance_loss_gradient_reaches_the_gate():
    torch.manual_seed(6)
    moe = MoEFeedForward(D, num_experts=4, top_k=2)
    weights, indices = moe.router(torch.randn(9, D))
    moe.last_indices = indices
    loss = moe.load_balancing_loss()
    loss.backward()
    assert moe.router.gate.W.grad is not None
    assert moe.router.gate.W.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# The full model
# ---------------------------------------------------------------------------


def test_moe_transformer_lm_forward_and_balance():
    torch.manual_seed(7)
    model = MoETransformerLM(
        vocab_size=32,
        embedding_dim=D,
        num_layers=2,
        num_heads=2,
        max_seq_len=16,
        num_experts=4,
        top_k=2,
    )
    ids = torch.randint(0, 32, (2, 10))
    logits = model(ids)
    assert logits.shape == (2, 10, 32)
    balance = model.load_balancing_loss()
    assert balance.dim() == 0
    assert torch.isfinite(balance)
    assert balance.detach().item() > 0


def test_moe_block_forward_shape():
    """The block wires MoE into the same residual structure as Module 09."""
    torch.manual_seed(8)
    block = MoEBlock(D, num_heads=2, num_experts=2, top_k=1)
    x = torch.randn(1, 3, D)
    out = block(x)
    assert out.shape == x.shape
