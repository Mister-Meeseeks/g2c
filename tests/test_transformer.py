"""Tests for Module 09: the transformer block.

Suggested order to implement & turn green:

  1. LayerNorm.forward             → test_layer_norm_*
  2. FeedForward.forward           → test_ffn_*
  3. Block.forward                 → test_block_*
  4. TransformerLM.forward         → test_transformer_lm_*
  5. LayerKVCache.append           → test_layer_kv_cache_append_*
  6. TransformerLM.forward_cached  → test_*_cached_*

Steps 5 and 6 are the Module 16 cached-inference path. They are the only
part of this file you can leave for later without blocking anything else;
the rest of Modules 09-11 never touch the cache. Do step 5 before step 6,
and before `MultiHeadAttention.forward_cached` in Module 08's file —
both of those call `append`.

Construction tests, parameter-count tests, the LN init tests
(`gamma`/`beta` start as ones/zeros), and the FFN default-hidden-dim
test all pass from the start because that part of the code is fully
implemented.

Block depends on both LayerNorm AND `MultiHeadAttention` from Module 08
AND FeedForward. The Block tests will only pass once steps 1, 2, AND
Module 08's `MultiHeadAttention.forward` are all implemented. If you
haven't finished Module 08 yet, do that first — these tests aren't
trying to teach attention, they're trying to teach the block structure
WRAPPING attention.

TransformerLM additionally depends on `TokenEmbedding.forward` (Module
05) and `LearnedPositionalEmbedding.forward` (Module 05). If those are
still scaffolded, fill them in (one-liners each) before tackling step 4.

Most tests use very small dimensions (`embedding_dim=8`, `num_heads=2`,
`T=3..6`, `vocab_size=12`, `num_layers=2`) so the suite runs in well
under a second.
"""
from __future__ import annotations

import pytest
import torch

from g2c.nn import CrossEntropyLoss
from g2c.transformer import (
    Block,
    FeedForward,
    KVCache,
    LayerKVCache,
    LayerNorm,
    TransformerLM,
)

# ----------------------------------------------------------------------
# LayerNorm — construction (boilerplate)
# ----------------------------------------------------------------------

def test_layer_norm_construction():
    LayerNorm(embedding_dim=8)


def test_layer_norm_construction_stores_args():
    ln = LayerNorm(embedding_dim=16, eps=1e-4)
    assert ln.embedding_dim == 16
    assert ln.eps == 1e-4


def test_layer_norm_default_eps():
    ln = LayerNorm(embedding_dim=8)
    assert ln.eps == 1e-5


def test_layer_norm_parameters_count():
    """Two learnable params: gamma (scale) and beta (shift)."""
    ln = LayerNorm(embedding_dim=8)
    params = list(ln.parameters())
    assert len(params) == 2


def test_layer_norm_gamma_init_ones():
    """gamma starts as all-ones — at init, LN is just standardize+identity."""
    ln = LayerNorm(embedding_dim=8)
    assert torch.equal(ln.gamma, torch.ones(8))


def test_layer_norm_beta_init_zeros():
    """beta starts as all-zeros — at init, LN doesn't shift."""
    ln = LayerNorm(embedding_dim=8)
    assert torch.equal(ln.beta, torch.zeros(8))


def test_layer_norm_parameters_require_grad():
    ln = LayerNorm(embedding_dim=4)
    for p in ln.parameters():
        assert p.requires_grad


# ----------------------------------------------------------------------
# LayerNorm — forward
# ----------------------------------------------------------------------

def test_layer_norm_forward_shape():
    ln = LayerNorm(embedding_dim=8)
    x = torch.randn(2, 5, 8)
    out = ln(x)
    assert out.shape == (2, 5, 8)


def test_layer_norm_forward_handles_2d():
    """LN accepts any (..., D) input — including (B, D)."""
    ln = LayerNorm(embedding_dim=4)
    x = torch.randn(3, 4)
    out = ln(x)
    assert out.shape == (3, 4)


def test_layer_norm_normalizes_last_dim():
    """Headline test: with default gamma=1, beta=0, output has zero mean
    and unit variance along the last dim — for every (batch, position).
    """
    ln = LayerNorm(embedding_dim=8)
    # Arbitrary mean and scale to make sure normalization actually does work.
    x = torch.randn(2, 5, 8) * 3.0 + 7.0
    out = ln(x)
    means = out.mean(dim=-1)
    variances = out.var(dim=-1, unbiased=False)
    assert torch.allclose(means, torch.zeros(2, 5), atol=1e-5)
    assert torch.allclose(variances, torch.ones(2, 5), atol=1e-4)


def test_layer_norm_affine_applied():
    """Setting gamma=2, beta=1 should scale output variance to 4 and
    shift mean to 1. Pins down that the affine is applied AFTER the
    normalization, not the other way around.
    """
    ln = LayerNorm(embedding_dim=8)
    with torch.no_grad():
        ln.gamma.fill_(2.0)
        ln.beta.fill_(1.0)
    x = torch.randn(2, 5, 8)
    out = ln(x)
    means = out.mean(dim=-1)
    variances = out.var(dim=-1, unbiased=False)
    assert torch.allclose(means, torch.ones(2, 5), atol=1e-4)
    assert torch.allclose(variances, torch.ones(2, 5) * 4.0, atol=1e-3)


def test_layer_norm_normalizes_each_position_independently():
    """LayerNorm pools statistics ALONG the last dim, not across the
    batch or sequence dim. Mutating one position must NOT affect the
    normalization of any other position.

    This is the structural difference from BatchNorm — and the property
    that makes LN work identically at train and inference time.
    """
    ln = LayerNorm(embedding_dim=4)
    x1 = torch.randn(2, 3, 4)
    x2 = x1.clone()
    x2[0, 1, :] += 100.0   # shift one position by a big constant
    out1 = ln(x1)
    out2 = ln(x2)
    # The mutated position itself sees a normalized version (zero-mean
    # within the slice), so its OUTPUT is the same as if the shift hadn't
    # happened — that's the point of LN. The other positions are also
    # unchanged.
    assert torch.allclose(out1[0, 0], out2[0, 0])
    assert torch.allclose(out1[0, 2], out2[0, 2])
    assert torch.allclose(out1[1], out2[1])
    # And the mutated position itself is also unchanged — LN cancels the
    # constant shift inside the slice.
    assert torch.allclose(out1[0, 1], out2[0, 1], atol=1e-5)


def test_layer_norm_routes_gradients():
    ln = LayerNorm(embedding_dim=4)
    x = torch.randn(2, 3, 4)
    out = ln(x)
    out.sum().backward()
    for p in ln.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()


def test_layer_norm_handles_constant_input():
    """A constant-along-channel input has zero variance. The eps inside
    the sqrt prevents a divide-by-zero — output should be finite.
    """
    ln = LayerNorm(embedding_dim=4)
    x = torch.zeros(1, 1, 4)         # constant input, var = 0
    out = ln(x)
    assert torch.isfinite(out).all()


# ----------------------------------------------------------------------
# FeedForward — construction (boilerplate)
# ----------------------------------------------------------------------

def test_ffn_construction():
    FeedForward(embedding_dim=8)


def test_ffn_default_hidden_dim_is_4x():
    """Standard transformer convention: hidden_dim = 4 * embedding_dim."""
    ffn = FeedForward(embedding_dim=8)
    assert ffn.hidden_dim == 32
    assert ffn.fc1.W.shape == (8, 32)
    assert ffn.fc2.W.shape == (32, 8)


def test_ffn_custom_hidden_dim():
    ffn = FeedForward(embedding_dim=8, hidden_dim=16)
    assert ffn.hidden_dim == 16
    assert ffn.fc1.W.shape == (8, 16)
    assert ffn.fc2.W.shape == (16, 8)


def test_ffn_parameters_count():
    """Two Linear layers × (W, b) = 4 parameter tensors."""
    ffn = FeedForward(embedding_dim=8)
    params = list(ffn.parameters())
    assert len(params) == 4


# ----------------------------------------------------------------------
# FeedForward — forward
# ----------------------------------------------------------------------

def test_ffn_forward_shape():
    ffn = FeedForward(embedding_dim=8, hidden_dim=16)
    x = torch.randn(2, 5, 8)
    out = ffn(x)
    assert out.shape == (2, 5, 8)


def test_ffn_forward_per_position():
    """The FFN is per-position. Mutating x at one position must NOT
    affect the output at any other position. (Compare to attention,
    which DOES mix across positions.)
    """
    ffn = FeedForward(embedding_dim=8, hidden_dim=16)
    x1 = torch.randn(1, 4, 8)
    x2 = x1.clone()
    x2[0, 2, :] = torch.randn(8)
    out1 = ffn(x1)
    out2 = ffn(x2)
    # All positions OTHER than the mutated one must be unchanged.
    assert torch.allclose(out1[0, 0], out2[0, 0], atol=1e-6)
    assert torch.allclose(out1[0, 1], out2[0, 1], atol=1e-6)
    assert torch.allclose(out1[0, 3], out2[0, 3], atol=1e-6)
    # The mutated position itself should change.
    assert not torch.allclose(out1[0, 2], out2[0, 2], atol=1e-4)


def test_ffn_is_nonlinear():
    """The FFN includes a GELU between the two Linears, so it is NOT a
    linear function of its input. f(2x) != 2 f(x) for nonzero outputs.

    Pins down that the activation is actually applied — implementing
    `fc2(fc1(x))` (no nonlinearity) would silently pass shape tests but
    collapse the FFN to a single linear layer.
    """
    torch.manual_seed(0)
    ffn = FeedForward(embedding_dim=8, hidden_dim=16)
    # Zero out the fc1 bias so f(0) = fc2(gelu(0) + b_fc2) = fc2(b_fc2)
    # is a constant, and the linearity check below is meaningful.
    with torch.no_grad():
        ffn.fc1.b.zero_()
    x = torch.randn(1, 3, 8)
    f_x = ffn(x) - ffn(torch.zeros_like(x))
    f_2x = ffn(2.0 * x) - ffn(torch.zeros_like(x))
    # If f were linear, f(2x) - f(0) would equal 2*(f(x) - f(0)).
    assert not torch.allclose(f_2x, 2.0 * f_x, atol=1e-3)


def test_ffn_routes_gradients():
    ffn = FeedForward(embedding_dim=8, hidden_dim=16)
    x = torch.randn(2, 3, 8)
    out = ffn(x)
    out.sum().backward()
    for p in ffn.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()


# ----------------------------------------------------------------------
# Block — construction (boilerplate)
# ----------------------------------------------------------------------

def test_block_construction():
    Block(embedding_dim=8, num_heads=2)


def test_block_construction_stores_args():
    b = Block(embedding_dim=12, num_heads=3, hidden_dim=24, causal=False)
    assert b.embedding_dim == 12
    assert b.num_heads == 3
    assert b.hidden_dim == 24
    assert b.causal is False


def test_block_default_hidden_dim():
    """Block inherits the FFN's 4× default hidden dim."""
    b = Block(embedding_dim=8, num_heads=2)
    assert b.hidden_dim == 32
    assert b.ffn.hidden_dim == 32


def test_block_default_causal():
    b = Block(embedding_dim=8, num_heads=2)
    assert b.causal is True
    assert b.attn.causal is True


def test_block_parameters_count():
    """2 LN × 2 + MHA × 8 + FFN × 4 = 16 parameter tensors."""
    b = Block(embedding_dim=8, num_heads=2)
    params = list(b.parameters())
    assert len(params) == 16


def test_block_submodules():
    """Block exposes its sublayers as `ln1`, `attn`, `ln2`, `ffn`."""
    b = Block(embedding_dim=8, num_heads=2)
    assert isinstance(b.ln1, LayerNorm)
    assert isinstance(b.ln2, LayerNorm)
    assert isinstance(b.ffn, FeedForward)
    # MHA is from Module 08 — just verify the attribute exists.
    assert hasattr(b, "attn")
    assert b.attn.embedding_dim == 8
    assert b.attn.num_heads == 2


# ----------------------------------------------------------------------
# Block — forward
# ----------------------------------------------------------------------

def test_block_forward_shape():
    b = Block(embedding_dim=8, num_heads=2)
    x = torch.randn(2, 5, 8)
    out = b(x)
    assert out.shape == (2, 5, 8)


def test_block_forward_routes_gradients():
    b = Block(embedding_dim=8, num_heads=2)
    x = torch.randn(2, 3, 8)
    out = b(x)
    out.sum().backward()
    for p in b.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()


def test_block_residual_identity_when_sublayers_zeroed():
    """Headline test: when both sublayers' final projections are zeroed,
    the block reduces to the identity.

    The residual stream is the model's "default behavior": each sublayer
    ADDS its contribution to `x`. If the sublayer outputs are zero,
    `x + 0 = x` and the block passes input through unchanged. Zeroing
    `attn.out_proj` and `ffn.fc2` forces both sublayer outputs to zero
    (their weights and biases are the last linear step).

    This pins down the residual structure. Without residuals, this test
    fails because the block would output `attn(...) + ffn(...)`, not
    `x + attn(...) + ffn(...)`.
    """
    torch.manual_seed(0)
    b = Block(embedding_dim=8, num_heads=2, hidden_dim=16, causal=True)
    with torch.no_grad():
        b.attn.out_proj.W.zero_()
        b.attn.out_proj.b.zero_()
        b.ffn.fc2.W.zero_()
        b.ffn.fc2.b.zero_()
    x = torch.randn(2, 4, 8)
    out = b(x)
    assert torch.allclose(out, x, atol=1e-6)


def test_block_pre_norm_structure():
    """Headline test: Block uses pre-norm:
        x = x + attn(ln1(x))
        x = x + ffn(ln2(x))

    Pins down the exact composition by recomputing it independently.
    Distinguishes pre-norm (correct) from post-norm (`ln(x + sublayer(x))`)
    and from no-residual (`ln(sublayer(x))`).
    """
    torch.manual_seed(0)
    b = Block(embedding_dim=8, num_heads=2, hidden_dim=16, causal=True)
    x = torch.randn(1, 3, 8)
    expected = x + b.attn(b.ln1(x))
    expected = expected + b.ffn(b.ln2(expected))
    actual = b(x)
    assert torch.allclose(actual, expected, atol=1e-5)


def test_block_causality():
    """Headline test: a causal block must not let position t leak into
    the output at any position < t.

    The block contains a causal MHA, two LNs, and an FFN. LN is
    per-position and FFN is per-position, so the only mechanism that
    *could* leak future info is attention — which the causal mask
    prevents. This test confirms the wiring preserves that property.
    """
    torch.manual_seed(0)
    b = Block(embedding_dim=8, num_heads=2, hidden_dim=16, causal=True)
    x1 = torch.randn(1, 5, 8)
    x2 = x1.clone()
    x2[0, 3, :] = torch.randn(8)
    out1 = b(x1)
    out2 = b(x2)
    # Positions 0, 1, 2 must be unchanged.
    assert torch.allclose(out1[0, :3], out2[0, :3], atol=1e-6)
    # Positions 3 and 4 SHOULD change.
    assert not torch.allclose(out1[0, 3:], out2[0, 3:], atol=1e-4)


def test_block_non_causal():
    """With causal=False, the block has no temporal restriction — every
    position can leak into every other.
    """
    torch.manual_seed(0)
    b = Block(embedding_dim=8, num_heads=2, causal=False)
    x1 = torch.randn(1, 4, 8)
    x2 = x1.clone()
    x2[0, 2, :] = torch.randn(8)
    out1 = b(x1)
    out2 = b(x2)
    # Position 0 should now be affected — attn(causal=False) sees pos 2.
    assert not torch.allclose(out1[0, 0], out2[0, 0], atol=1e-5)


# ----------------------------------------------------------------------
# TransformerLM — construction (boilerplate)
# ----------------------------------------------------------------------

def test_transformer_lm_construction():
    TransformerLM(
        vocab_size=12,
        embedding_dim=8,
        num_layers=2,
        num_heads=2,
        max_seq_len=16,
    )


def test_transformer_lm_construction_stores_args():
    m = TransformerLM(
        vocab_size=12,
        embedding_dim=8,
        num_layers=2,
        num_heads=2,
        max_seq_len=16,
        hidden_dim=20,
    )
    assert m.vocab_size == 12
    assert m.embedding_dim == 8
    assert m.num_layers == 2
    assert m.num_heads == 2
    assert m.max_seq_len == 16
    assert m.hidden_dim == 20


def test_transformer_lm_blocks_count():
    """num_layers controls how many Block instances are stacked."""
    m = TransformerLM(
        vocab_size=12,
        embedding_dim=8,
        num_layers=3,
        num_heads=2,
        max_seq_len=16,
    )
    assert len(m.blocks) == 3
    for blk in m.blocks:
        assert isinstance(blk, Block)


def test_transformer_lm_parameter_chain_is_complete():
    """parameters() chains through token embed, pos embed, every block,
    final LN, and the unembedding's per-token bias. The unembedding's
    weight matrix is the SAME tensor as `token_embed.weight` (tied), so
    it appears once via `token_embed`, not twice.

    Counts:
        token_embed: 1 tensor
        pos_embed:   1 tensor
        blocks:      num_layers × 16 tensors
        ln_final:    2 tensors
        head_bias:   1 tensor
    Total for num_layers=2: 1 + 1 + 32 + 2 + 1 = 37
    """
    m = TransformerLM(
        vocab_size=12,
        embedding_dim=8,
        num_layers=2,
        num_heads=2,
        max_seq_len=16,
    )
    params = list(m.parameters())
    assert len(params) == 37


def test_transformer_lm_unembedding_is_tied_to_token_embedding():
    """Headline test: row `v` of `token_embed.weight` controls logit `v`
    of the unembedding. We pick a token id that does NOT appear in the
    input, mutate its embedding row, and verify ONLY the matching
    output column changes. That isolates the unembedding path (the
    input-lookup never touches the mutated row) and pins down that
    the unembedding really reuses the embedding matrix.
    """
    torch.manual_seed(0)
    V = 12
    m = TransformerLM(
        vocab_size=V,
        embedding_dim=8,
        num_layers=1,
        num_heads=2,
        max_seq_len=16,
    )
    ids = torch.tensor([[0, 1, 2, 3]])   # token 7 deliberately absent
    target_v = 7
    assert target_v not in ids.unique().tolist()

    out_before = m(ids).detach().clone()
    with torch.no_grad():
        m.token_embed.weight[target_v] = (
            torch.randn_like(m.token_embed.weight[target_v]) * 5.0
        )
    out_after = m(ids)

    # Logit column `target_v` should change.
    assert not torch.allclose(
        out_before[..., target_v], out_after[..., target_v], atol=1e-4
    )
    # Every other logit column should be unchanged — the input-lookup
    # path never touched any other row of the embedding, and the
    # unembedding only writes column `v` from row `v`.
    other_cols = [v for v in range(V) if v != target_v]
    assert torch.allclose(
        out_before[..., other_cols], out_after[..., other_cols], atol=1e-5
    )


def test_transformer_lm_head_bias_routes_gradient():
    """The per-token output bias is a learnable parameter that gets
    gradient on backward. Prevents a regression where `head_bias` is
    omitted from `parameters()` or detached from the graph.
    """
    torch.manual_seed(0)
    m = TransformerLM(
        vocab_size=12,
        embedding_dim=8,
        num_layers=1,
        num_heads=2,
        max_seq_len=16,
    )
    ids = torch.randint(0, 12, (2, 4))
    m(ids).sum().backward()
    assert m.head_bias.grad is not None
    assert torch.isfinite(m.head_bias.grad).all()


def test_transformer_lm_more_layers_more_params():
    """Stacking more blocks linearly grows the parameter count."""
    m1 = TransformerLM(
        vocab_size=12,
        embedding_dim=8,
        num_layers=1,
        num_heads=2,
        max_seq_len=16,
    )
    m4 = TransformerLM(
        vocab_size=12,
        embedding_dim=8,
        num_layers=4,
        num_heads=2,
        max_seq_len=16,
    )
    n1 = sum(p.numel() for p in m1.parameters())
    n4 = sum(p.numel() for p in m4.parameters())
    # The non-block params (embeds, final LN, head) are constant across
    # both models. The block params scale linearly with depth.
    block_param_count = sum(p.numel() for p in m1.blocks[0].parameters())
    assert n4 - n1 == 3 * block_param_count


def test_kv_cache_empty_has_one_layer_cache_per_block():
    cache = KVCache.empty(num_layers=3)
    assert len(cache) == 3
    assert cache.length == 0
    assert [layer.length for layer in cache.layers] == [0, 0, 0]


# ----------------------------------------------------------------------
# LayerKVCache.append
# ----------------------------------------------------------------------

def test_layer_kv_cache_append_to_empty_cache_stores_tensors():
    """Appending to an empty cache just adopts the tensors."""
    cache = LayerKVCache()
    k = torch.randn(2, 3, 1, 4)
    v = torch.randn(2, 3, 1, 4)
    cache.append(k, v)
    assert cache.length == 1
    assert torch.equal(cache.keys, k)
    assert torch.equal(cache.values, v)


def test_layer_kv_cache_append_grows_the_sequence_axis():
    """Repeated appends grow T_cache in (B, H, T_cache, head_dim) — dim=-2.

    If this reports a head_dim of 3 or a batch of 6, the concatenation is
    happening on the wrong axis.
    """
    cache = LayerKVCache()
    for _ in range(3):
        cache.append(torch.randn(2, 3, 1, 4), torch.randn(2, 3, 1, 4))
    assert cache.length == 3
    assert cache.keys.shape == (2, 3, 3, 4)
    assert cache.values.shape == (2, 3, 3, 4)


def test_layer_kv_cache_append_preserves_order():
    """Earlier tokens stay at earlier positions; the new token lands last."""
    cache = LayerKVCache()
    first_k, first_v = torch.randn(1, 2, 1, 4), torch.randn(1, 2, 1, 4)
    second_k, second_v = torch.randn(1, 2, 1, 4), torch.randn(1, 2, 1, 4)
    cache.append(first_k, first_v)
    cache.append(second_k, second_v)
    assert torch.equal(cache.keys[:, :, :1], first_k)
    assert torch.equal(cache.keys[:, :, 1:], second_k)
    assert torch.equal(cache.values[:, :, :1], first_v)
    assert torch.equal(cache.values[:, :, 1:], second_v)


def test_layer_kv_cache_append_accepts_multiple_positions():
    """A prefill pass appends T_new > 1 positions in one call."""
    cache = LayerKVCache()
    cache.append(torch.randn(2, 3, 5, 4), torch.randn(2, 3, 5, 4))
    assert cache.length == 5
    cache.append(torch.randn(2, 3, 1, 4), torch.randn(2, 3, 1, 4))
    assert cache.length == 6


def test_layer_kv_cache_append_returns_self():
    """append returns the cache so calls can chain."""
    cache = LayerKVCache()
    returned = cache.append(torch.randn(1, 1, 1, 2), torch.randn(1, 1, 1, 2))
    assert returned is cache


def test_layer_kv_cache_append_rejects_incompatible_head_dim():
    """Validation is provided — this passes before you implement append.

    The cache is populated via the constructor rather than `append` so the
    check runs even while `append` is still scaffolded.
    """
    cache = LayerKVCache(keys=torch.randn(2, 3, 1, 4), values=torch.randn(2, 3, 1, 4))
    with pytest.raises(ValueError):
        cache.append(torch.randn(2, 3, 1, 8), torch.randn(2, 3, 1, 8))


# ----------------------------------------------------------------------
# TransformerLM — forward
# ----------------------------------------------------------------------

def test_transformer_lm_forward_shape():
    """Output is (B, T, V) — one logit vector per position."""
    torch.manual_seed(0)
    m = TransformerLM(
        vocab_size=12,
        embedding_dim=8,
        num_layers=2,
        num_heads=2,
        max_seq_len=16,
    )
    ids = torch.randint(0, 12, (3, 5))
    out = m(ids)
    assert out.shape == (3, 5, 12)


def test_transformer_lm_forward_seq_len_too_long_raises():
    """Sequences longer than max_seq_len exceed the positional table
    and must raise.
    """
    m = TransformerLM(
        vocab_size=12,
        embedding_dim=8,
        num_layers=1,
        num_heads=2,
        max_seq_len=4,
    )
    ids = torch.randint(0, 12, (1, 5))
    with pytest.raises(ValueError):
        m(ids)


def test_transformer_lm_forward_routes_gradients():
    torch.manual_seed(0)
    m = TransformerLM(
        vocab_size=12,
        embedding_dim=8,
        num_layers=2,
        num_heads=2,
        max_seq_len=16,
    )
    ids = torch.randint(0, 12, (2, 4))
    out = m(ids)
    out.sum().backward()
    for p in m.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()


def test_transformer_lm_causality():
    """Headline test: changing a token at position t must not affect the
    LOGITS at positions < t.

    This is the property that makes the model trainable on next-token
    prediction. If position t's logits could see token t+1, the loss
    collapses to zero by trivially copying the answer.

    The LM-level version of `test_block_causality` — pins down that
    causality is preserved through the full embed→blocks→LN→unembed
    pipeline.
    """
    torch.manual_seed(0)
    m = TransformerLM(
        vocab_size=12,
        embedding_dim=8,
        num_layers=2,
        num_heads=2,
        max_seq_len=16,
    )
    ids1 = torch.randint(0, 12, (1, 6))
    ids2 = ids1.clone()
    # Mutate position 4 to a (different) token.
    new_token = (ids1[0, 4].item() + 1) % 12
    ids2[0, 4] = new_token
    out1 = m(ids1)
    out2 = m(ids2)
    # Logits at positions 0..3 must be unchanged.
    assert torch.allclose(out1[0, :4], out2[0, :4], atol=1e-5)
    # Logits at positions 4 and 5 SHOULD change.
    assert not torch.allclose(out1[0, 4:], out2[0, 4:], atol=1e-4)


def test_transformer_lm_forward_cached_matches_full_forward():
    """Cached one-token decoding must produce the same logits as full forward."""
    torch.manual_seed(0)
    m = TransformerLM(
        vocab_size=12,
        embedding_dim=8,
        num_layers=2,
        num_heads=2,
        max_seq_len=16,
    )
    ids = torch.randint(0, 12, (1, 6))
    full = m(ids)

    cache = m.empty_kv_cache()
    pieces = []
    for t in range(ids.shape[1]):
        logits_t, cache = m.forward_cached(ids[:, t:t + 1], cache)
        pieces.append(logits_t)

    cached = torch.cat(pieces, dim=1)
    assert cached.shape == full.shape
    assert cache.length == ids.shape[1]
    assert torch.allclose(cached, full, atol=1e-5)


def test_transformer_lm_forward_cached_validates_sequence_shape():
    m = TransformerLM(
        vocab_size=12,
        embedding_dim=8,
        num_layers=1,
        num_heads=2,
        max_seq_len=16,
    )
    with pytest.raises(ValueError):
        m.forward_cached(torch.randint(0, 12, (1, 2)), m.empty_kv_cache())


def test_transformer_lm_forward_cached_validates_context_length():
    m = TransformerLM(
        vocab_size=12,
        embedding_dim=8,
        num_layers=1,
        num_heads=2,
        max_seq_len=2,
    )
    cache = m.empty_kv_cache()
    m.forward_cached(torch.tensor([[1]]), cache)
    m.forward_cached(torch.tensor([[2]]), cache)
    with pytest.raises(ValueError):
        m.forward_cached(torch.tensor([[3]]), cache)


def test_transformer_lm_smoke_train():
    """Smoke test: a few SGD steps on a fixed batch should reduce loss.

    Not a quality check — just confirms the whole forward + backward +
    optimizer pipeline is plumbed correctly. A miswired residual or a
    missing gradient path would prevent loss from decreasing.
    """
    torch.manual_seed(0)
    m = TransformerLM(
        vocab_size=10,
        embedding_dim=8,
        num_layers=1,
        num_heads=2,
        max_seq_len=8,
    )
    loss_fn = CrossEntropyLoss()
    ids = torch.randint(0, 10, (4, 6))
    targets = torch.randint(0, 10, (4, 6))
    # Take an initial loss reading.
    logits = m(ids)
    initial_loss = loss_fn(logits.reshape(-1, 10), targets.reshape(-1)).item()
    # 30 steps of plain SGD on a fixed batch — should overfit easily.
    lr = 0.5
    for _ in range(30):
        for p in m.parameters():
            if p.grad is not None:
                p.grad = None
        logits = m(ids)
        loss = loss_fn(logits.reshape(-1, 10), targets.reshape(-1))
        loss.backward()
        with torch.no_grad():
            for p in m.parameters():
                p -= lr * p.grad
    final_loss = loss.item()
    assert final_loss < initial_loss * 0.5
