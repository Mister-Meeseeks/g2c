"""Tests for Module 11: sampling and decoding.

Suggested order to implement & turn green:

  1. apply_temperature              → test_apply_temperature_*
  2. top_k_filter                   → test_top_k_filter_*
  3. top_p_filter                   → test_top_p_filter_*
  4. apply_repetition_penalty       → test_apply_repetition_penalty_*
  5. generate                       → test_generate_*
  6. Optional cached generation      → test_generate_cached_*

Steps 1–4 are independent — you can do them in any order. Step 5
composes all four warpers into the autoregressive loop. Step 6 is the
optional KV-cache sibling introduced in Module 16. The remaining tests
exercise the pipeline end-to-end against a tiny pretrained-style transformer.

The four warper-test groups all use small synthetic logit tensors —
the suite runs in well under a second even with all five steps
implemented. The `generate` tests build a tiny `TransformerLM` (1
layer, embedding_dim=8, vocab=12) so they depend on Modules 03 / 05 /
07 / 08 / 09 being implemented. If your full Module 09 smoke-train
test (`test_transformer_lm_smoke_train`) isn't passing, finish that
first.
"""
from __future__ import annotations

import math

import pytest
import torch

from g2c.sampling import (
    apply_repetition_penalty,
    apply_temperature,
    generate,
    generate_cached,
    top_k_filter,
    top_p_filter,
)
from g2c.transformer import TransformerLM

# ----------------------------------------------------------------------
# apply_temperature — scaffolded
# ----------------------------------------------------------------------


def test_apply_temperature_identity_at_one():
    """temperature=1.0 should return logits unchanged (up to numerical noise)."""
    logits = torch.randn(3, 7)
    out = apply_temperature(logits, temperature=1.0)
    assert torch.allclose(out, logits)


def test_apply_temperature_shape_preserved():
    """The warper preserves shape: (..., V) → (..., V)."""
    logits = torch.randn(2, 4, 9)
    out = apply_temperature(logits, temperature=0.7)
    assert out.shape == logits.shape


def test_apply_temperature_low_t_sharpens_distribution():
    """t < 1 produces a sharper softmax than t = 1.

    'Sharper' means lower entropy and a higher peak probability on
    the argmax. Test by comparing entropy.
    """
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    probs_t1 = torch.softmax(apply_temperature(logits, 1.0), dim=-1)
    probs_t_low = torch.softmax(apply_temperature(logits, 0.1), dim=-1)
    # The top probability should be larger after sharpening.
    assert probs_t_low.max().item() > probs_t1.max().item()
    # Equivalently, entropy should be lower.
    entropy_t1 = -(probs_t1 * probs_t1.log()).sum().item()
    entropy_low = -(probs_t_low * (probs_t_low + 1e-12).log()).sum().item()
    assert entropy_low < entropy_t1


def test_apply_temperature_high_t_flattens_distribution():
    """t > 1 produces a flatter softmax."""
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    probs_t1 = torch.softmax(apply_temperature(logits, 1.0), dim=-1)
    probs_t_high = torch.softmax(apply_temperature(logits, 100.0), dim=-1)
    # At very high temperature the distribution approaches uniform.
    assert probs_t_high.max().item() < probs_t1.max().item()
    # Approximately uniform: every entry close to 1/V.
    V = logits.shape[-1]
    assert torch.allclose(
        probs_t_high, torch.full_like(probs_t_high, 1.0 / V), atol=1e-2
    )


def test_apply_temperature_preserves_argmax():
    """Temperature is monotone — it can't reorder which token is most likely."""
    logits = torch.randn(5, 20)
    am_native = logits.argmax(dim=-1)
    for t in (0.1, 0.5, 1.0, 2.0, 10.0):
        am_warped = apply_temperature(logits, t).argmax(dim=-1)
        assert torch.equal(am_native, am_warped)


def test_apply_temperature_zero_raises():
    """temperature == 0 is invalid — division by zero."""
    logits = torch.randn(2, 5)
    with pytest.raises(ValueError):
        apply_temperature(logits, 0.0)


def test_apply_temperature_negative_raises():
    """Negative temperature is meaningless and would invert the ranking."""
    logits = torch.randn(2, 5)
    with pytest.raises(ValueError):
        apply_temperature(logits, -0.5)


# ----------------------------------------------------------------------
# top_k_filter — scaffolded
# ----------------------------------------------------------------------


def test_top_k_filter_shape_preserved():
    logits = torch.randn(2, 4, 10)
    out = top_k_filter(logits, k=3)
    assert out.shape == logits.shape


def test_top_k_filter_k_equals_v_is_identity():
    """When k == V, every token survives — the filter is a no-op."""
    logits = torch.randn(3, 8)
    out = top_k_filter(logits, k=8)
    assert torch.equal(out, logits)


def test_top_k_filter_k_greater_than_v_clamps():
    """Asking for more tokens than V should not crash — just keep all."""
    logits = torch.randn(3, 5)
    out = top_k_filter(logits, k=99)
    assert torch.equal(out, logits)


def test_top_k_filter_k_one_keeps_only_argmax():
    """k=1 leaves exactly one finite logit per row — the argmax."""
    logits = torch.tensor([[1.0, 5.0, 3.0, 2.0]])
    out = top_k_filter(logits, k=1)
    finite_mask = torch.isfinite(out)
    assert finite_mask.sum().item() == 1
    assert finite_mask[0, 1].item()  # the argmax was at index 1
    assert out[0, 1].item() == 5.0
    # Every other entry is -inf.
    for i in (0, 2, 3):
        assert math.isinf(out[0, i].item()) and out[0, i].item() < 0


def test_top_k_filter_k_two_keeps_top_two():
    """k=2 keeps the top two; rest are -inf."""
    logits = torch.tensor([[1.0, 5.0, 3.0, 2.0]])
    out = top_k_filter(logits, k=2)
    finite_indices = torch.where(torch.isfinite(out[0]))[0].tolist()
    assert sorted(finite_indices) == [1, 2]  # values 5 and 3
    assert out[0, 1].item() == 5.0
    assert out[0, 2].item() == 3.0


def test_top_k_filter_per_row():
    """Filter is applied independently per row, not globally."""
    logits = torch.tensor(
        [
            [10.0, 0.0, 0.0],   # argmax at 0
            [0.0, 0.0, 10.0],   # argmax at 2
        ]
    )
    out = top_k_filter(logits, k=1)
    assert torch.isfinite(out[0, 0])
    assert torch.isfinite(out[1, 2])
    assert math.isinf(out[0, 1].item())
    assert math.isinf(out[1, 0].item())


def test_top_k_filter_softmax_sums_to_one_over_survivors():
    """After softmax, only the surviving k tokens have nonzero mass and
    they sum to 1 (the rest are exactly 0 because their logits are -inf).
    """
    logits = torch.randn(4, 20)
    out = top_k_filter(logits, k=5)
    probs = torch.softmax(out, dim=-1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(4))
    # Exactly 5 nonzero entries per row.
    assert (probs > 0).sum(dim=-1).tolist() == [5, 5, 5, 5]


def test_top_k_filter_zero_or_negative_raises():
    logits = torch.randn(2, 5)
    with pytest.raises(ValueError):
        top_k_filter(logits, k=0)
    with pytest.raises(ValueError):
        top_k_filter(logits, k=-3)


# ----------------------------------------------------------------------
# top_p_filter — scaffolded
# ----------------------------------------------------------------------


def test_top_p_filter_shape_preserved():
    logits = torch.randn(2, 3, 10)
    out = top_p_filter(logits, p=0.9)
    assert out.shape == logits.shape


def test_top_p_filter_p_one_keeps_everything():
    """p=1.0 means 'cumulative mass must reach 100%' — every token kept."""
    logits = torch.randn(3, 8)
    out = top_p_filter(logits, p=1.0)
    assert torch.allclose(out, logits)


def test_top_p_filter_keeps_smallest_prefix_reaching_p():
    """Construct a known distribution and check the survivor set.

    Logits chosen so the softmax probabilities are approximately
    [0.6, 0.3, 0.1, 0, 0, ...] in descending order. With p=0.7, the
    smallest prefix that reaches 0.7 is the top-1 token alone (0.6
    isn't enough, top-2 = 0.9 >= 0.7) — so positions 0 and 1 should
    survive. With p=0.5, only position 0 survives (its 0.6 already
    >= 0.5).
    """
    # Construct logits whose softmax is exactly [0.6, 0.3, 0.1].
    probs = torch.tensor([0.6, 0.3, 0.1])
    logits = probs.log().unsqueeze(0)        # (1, 3)

    out_07 = top_p_filter(logits, p=0.7)
    finite_07 = torch.where(torch.isfinite(out_07[0]))[0].tolist()
    assert sorted(finite_07) == [0, 1]

    out_05 = top_p_filter(logits, p=0.5)
    finite_05 = torch.where(torch.isfinite(out_05[0]))[0].tolist()
    assert finite_05 == [0]


def test_top_p_filter_argmax_always_survives():
    """Even if the argmax alone has probability > p, it must NOT be
    masked. A buggy implementation that 'drops everything once
    cumulative > p' would mask the argmax.
    """
    # The argmax has 95% of the mass; p=0.5 < 0.95.
    probs = torch.tensor([0.95, 0.025, 0.025])
    logits = probs.log().unsqueeze(0)
    out = top_p_filter(logits, p=0.5)
    assert torch.isfinite(out[0, 0])  # argmax survived


def test_top_p_filter_per_row():
    """Filter is applied independently per row."""
    # Row 0: very peaked (argmax has > p mass alone).
    # Row 1: very flat (need many tokens to reach p).
    logits = torch.tensor(
        [
            [10.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5, 0.5],
        ]
    )
    out = top_p_filter(logits, p=0.5)
    n_keep_row0 = torch.isfinite(out[0]).sum().item()
    n_keep_row1 = torch.isfinite(out[1]).sum().item()
    assert n_keep_row0 == 1            # only the argmax
    assert n_keep_row1 >= 2            # need multiple to reach 0.5


def test_top_p_filter_softmax_sums_to_one():
    """After softmax of the filtered logits, probs still sum to 1
    (the surviving tokens are renormalized to absorb the dropped mass).
    """
    logits = torch.randn(4, 30)
    out = top_p_filter(logits, p=0.8)
    probs = torch.softmax(out, dim=-1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5)


def test_top_p_filter_invalid_p_raises():
    logits = torch.randn(2, 5)
    with pytest.raises(ValueError):
        top_p_filter(logits, p=0.0)
    with pytest.raises(ValueError):
        top_p_filter(logits, p=-0.1)
    with pytest.raises(ValueError):
        top_p_filter(logits, p=1.5)


# ----------------------------------------------------------------------
# apply_repetition_penalty — scaffolded
# ----------------------------------------------------------------------


def test_apply_repetition_penalty_one_is_identity():
    logits = torch.randn(2, 5)
    prev = torch.tensor([[0, 1], [2, 3]])
    out = apply_repetition_penalty(logits, prev, penalty=1.0)
    assert torch.allclose(out, logits)


def test_apply_repetition_penalty_shape_preserved():
    logits = torch.randn(2, 5)
    prev = torch.tensor([[0, 1], [2, 3]])
    out = apply_repetition_penalty(logits, prev, penalty=1.5)
    assert out.shape == logits.shape


def test_apply_repetition_penalty_positive_logit_divided():
    """A positive logit at a previously-emitted token gets divided —
    smaller after, so lower probability.
    """
    logits = torch.tensor([[2.0, 3.0, 4.0, 5.0]])
    prev = torch.tensor([[0]])         # token 0 was emitted; its logit is 2.0
    out = apply_repetition_penalty(logits, prev, penalty=2.0)
    assert out[0, 0].item() == pytest.approx(2.0 / 2.0)
    # The other logits are untouched.
    assert out[0, 1].item() == 3.0
    assert out[0, 2].item() == 4.0
    assert out[0, 3].item() == 5.0


def test_apply_repetition_penalty_negative_logit_multiplied():
    """A negative logit at a previously-emitted token gets multiplied
    by the penalty — more negative, lower probability.
    """
    logits = torch.tensor([[-2.0, 1.0, 1.0, 1.0]])
    prev = torch.tensor([[0]])         # token 0; logit is -2.0
    out = apply_repetition_penalty(logits, prev, penalty=2.0)
    assert out[0, 0].item() == pytest.approx(-2.0 * 2.0)


def test_apply_repetition_penalty_untouched_tokens_unchanged():
    """Tokens NOT in `prev` keep their original logits exactly."""
    logits = torch.tensor([[2.0, 3.0, 4.0, 5.0]])
    prev = torch.tensor([[1]])
    out = apply_repetition_penalty(logits, prev, penalty=2.5)
    # Position 1 was penalized; the rest are unchanged.
    for i in (0, 2, 3):
        assert out[0, i].item() == logits[0, i].item()


def test_apply_repetition_penalty_lowers_probability():
    """The whole point: tokens in `prev` end up with LESS probability
    after softmax. This is the test that catches sign errors.
    """
    logits = torch.tensor([[2.0, 2.0, 2.0, 2.0]])
    prev = torch.tensor([[0, 1]])
    out = apply_repetition_penalty(logits, prev, penalty=1.5)
    probs = torch.softmax(out, dim=-1)
    # Tokens 0 and 1 were penalized; tokens 2 and 3 weren't.
    assert probs[0, 0].item() < probs[0, 2].item()
    assert probs[0, 1].item() < probs[0, 3].item()


def test_apply_repetition_penalty_one_d_prev_broadcasts():
    """A 1-D `prev` tensor broadcasts across all rows of `logits`."""
    logits = torch.tensor(
        [
            [2.0, 3.0, 4.0, 5.0],
            [5.0, 4.0, 3.0, 2.0],
        ]
    )
    prev = torch.tensor([0, 2])    # shared history
    out = apply_repetition_penalty(logits, prev, penalty=2.0)
    # Both rows: positions 0 and 2 get divided by 2 (they're positive).
    assert out[0, 0].item() == pytest.approx(1.0)
    assert out[0, 2].item() == pytest.approx(2.0)
    assert out[1, 0].item() == pytest.approx(2.5)
    assert out[1, 2].item() == pytest.approx(1.5)


def test_apply_repetition_penalty_invalid_raises():
    logits = torch.randn(1, 5)
    prev = torch.tensor([[0]])
    with pytest.raises(ValueError):
        apply_repetition_penalty(logits, prev, penalty=0.0)
    with pytest.raises(ValueError):
        apply_repetition_penalty(logits, prev, penalty=-1.0)


# ----------------------------------------------------------------------
# generate — depends on warpers + a tiny TransformerLM
# ----------------------------------------------------------------------


def _tiny_model() -> TransformerLM:
    """A miniature TransformerLM used by the generate tests."""
    torch.manual_seed(0)
    return TransformerLM(
        vocab_size=12,
        embedding_dim=8,
        num_layers=1,
        num_heads=2,
        max_seq_len=16,
    )


def test_generate_returns_correct_length():
    """Output length is `len(prompt) + max_new_tokens` (no EOS)."""
    m = _tiny_model()
    prompt = torch.tensor([1, 2, 3], dtype=torch.long)
    out = generate(m, prompt, max_new_tokens=5, temperature=1.0)
    assert out.shape == (3 + 5,)


def test_generate_starts_with_prompt():
    """The first len(prompt) tokens of the output equal the prompt."""
    m = _tiny_model()
    prompt = torch.tensor([4, 7, 1], dtype=torch.long)
    out = generate(m, prompt, max_new_tokens=4, temperature=1.0)
    assert torch.equal(out[:3], prompt)


def test_generate_outputs_valid_token_ids():
    """All generated tokens are in [0, vocab_size)."""
    m = _tiny_model()
    prompt = torch.tensor([0, 1, 2], dtype=torch.long)
    out = generate(m, prompt, max_new_tokens=10, temperature=1.0)
    assert (out >= 0).all() and (out < m.vocab_size).all()


def test_generate_rejects_prompt_ids_outside_model_vocab():
    """Bad tokenizer/model pairings should fail before an accelerator kernel runs."""
    m = _tiny_model()
    prompt = torch.tensor([0, 1, m.vocab_size], dtype=torch.long)
    with pytest.raises(ValueError, match="outside the model vocab"):
        generate(m, prompt, max_new_tokens=1, temperature=1.0)


def test_generate_greedy_is_deterministic():
    """temperature=0 (greedy) must produce identical outputs across calls,
    independent of any `generator` state.
    """
    m = _tiny_model()
    prompt = torch.tensor([0, 1, 2], dtype=torch.long)
    out1 = generate(m, prompt, max_new_tokens=8, temperature=0.0)
    out2 = generate(m, prompt, max_new_tokens=8, temperature=0.0)
    assert torch.equal(out1, out2)


def test_generate_seeded_sampling_is_reproducible():
    """Two calls with the same seeded generator produce the same output."""
    m = _tiny_model()
    prompt = torch.tensor([0, 1, 2], dtype=torch.long)
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    out1 = generate(m, prompt, max_new_tokens=8, temperature=1.0, generator=g1)
    out2 = generate(m, prompt, max_new_tokens=8, temperature=1.0, generator=g2)
    assert torch.equal(out1, out2)


def test_generate_different_seeds_produce_different_output():
    """Different seeds → (almost certainly) different sampled tokens."""
    m = _tiny_model()
    prompt = torch.tensor([0, 1, 2], dtype=torch.long)
    g1 = torch.Generator().manual_seed(1)
    g2 = torch.Generator().manual_seed(2)
    out1 = generate(m, prompt, max_new_tokens=10, temperature=1.0, generator=g1)
    out2 = generate(m, prompt, max_new_tokens=10, temperature=1.0, generator=g2)
    assert not torch.equal(out1, out2)


def test_generate_top_k_one_matches_greedy():
    """top_k=1 with any positive temperature is operationally greedy —
    only the argmax has nonzero mass after the filter, so multinomial
    deterministically picks it. Should match temperature=0 greedy.
    """
    m = _tiny_model()
    prompt = torch.tensor([0, 1, 2], dtype=torch.long)
    g = torch.Generator().manual_seed(0)
    out_topk1 = generate(
        m, prompt, max_new_tokens=8, temperature=1.0, top_k=1, generator=g
    )
    out_greedy = generate(m, prompt, max_new_tokens=8, temperature=0.0)
    assert torch.equal(out_topk1, out_greedy)


def test_generate_eos_stops_early():
    """If `eos_id` is sampled, generation halts. Force this by making
    the model effectively always pick token 0: bias the unembedding
    to give token 0 a huge positive logit.
    """
    m = _tiny_model()
    with torch.no_grad():
        m.head_bias.zero_()
        m.head_bias[0] = 100.0    # token 0 will dominate after softmax
    prompt = torch.tensor([1, 2, 3], dtype=torch.long)
    # Greedy → first generated token is 0; eos_id=0 stops immediately.
    out = generate(
        m, prompt, max_new_tokens=20, temperature=0.0, eos_id=0
    )
    # Output is prompt + the single eos token (which IS included).
    assert out.shape == (4,)
    assert out[-1].item() == 0


def test_generate_no_grad_does_not_populate_grads():
    """Generation must not leave `.grad` on parameters — `@torch.no_grad`
    prevents the autograd graph from being built. Forgetting it doesn't
    break correctness but quietly leaks memory on long generations.
    """
    m = _tiny_model()
    for p in m.parameters():
        p.grad = None
    prompt = torch.tensor([0, 1, 2], dtype=torch.long)
    generate(m, prompt, max_new_tokens=5, temperature=1.0)
    for p in m.parameters():
        assert p.grad is None


def test_generate_respects_max_seq_len_via_cropping():
    """When `len(prompt) + n_generated > model.max_seq_len`, generation
    must crop the context to the last `max_seq_len` tokens — not
    crash, not feed the model a too-long context.

    `_tiny_model` has max_seq_len=16. With a length-10 prompt and
    max_new_tokens=20, the running sequence eventually exceeds 16 and
    cropping kicks in.
    """
    m = _tiny_model()
    prompt = torch.arange(10, dtype=torch.long)   # length 10
    out = generate(m, prompt, max_new_tokens=20, temperature=0.0)
    # Total length = 10 + 20 = 30, exceeds max_seq_len=16. The function
    # must NOT raise — cropping happens inside.
    assert out.shape == (30,)


def test_generate_cached_greedy_matches_uncached_generate():
    """The KV-cache path should preserve greedy decoding exactly."""
    m = _tiny_model()
    prompt = torch.tensor([0, 1, 2], dtype=torch.long)
    cached = generate_cached(m, prompt, max_new_tokens=8, temperature=0.0)
    uncached = generate(m, prompt, max_new_tokens=8, temperature=0.0)
    assert torch.equal(cached, uncached)


def test_generate_cached_seeded_sampling_is_reproducible():
    m = _tiny_model()
    prompt = torch.tensor([0, 1, 2], dtype=torch.long)
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    out1 = generate_cached(m, prompt, max_new_tokens=8, generator=g1)
    out2 = generate_cached(m, prompt, max_new_tokens=8, generator=g2)
    assert torch.equal(out1, out2)


def test_generate_cached_stops_at_max_seq_len():
    """This simple cache has no rolling window, so it stops at max_seq_len."""
    m = _tiny_model()
    prompt = torch.arange(10, dtype=torch.long)
    out = generate_cached(m, prompt, max_new_tokens=20, temperature=0.0)
    assert out.shape == (m.max_seq_len,)


def test_generate_cached_rejects_prompt_that_fills_context():
    m = _tiny_model()
    prompt = torch.arange(m.max_seq_len, dtype=torch.long) % m.vocab_size
    with pytest.raises(ValueError):
        generate_cached(m, prompt, max_new_tokens=1, temperature=0.0)


def test_generate_invalid_temperature_raises():
    """Negative temperature is invalid. (Zero is reserved for greedy.)"""
    m = _tiny_model()
    prompt = torch.tensor([0, 1, 2], dtype=torch.long)
    with pytest.raises(ValueError):
        generate(m, prompt, max_new_tokens=3, temperature=-1.0)


def test_generate_empty_prompt_raises():
    """Generation needs at least one token of prompt to predict the next."""
    m = _tiny_model()
    empty = torch.tensor([], dtype=torch.long)
    with pytest.raises(ValueError):
        generate(m, empty, max_new_tokens=3, temperature=1.0)


def test_generate_repetition_penalty_discourages_repeats():
    """With a strong bias toward emitting the same token over and over,
    a high `repetition_penalty` should reduce the rate of repetition.

    Setup: bias the unembedding to favor token 0. Without penalty,
    greedy decoding emits 0 every step. With a large penalty, the bias
    against repeating 0 should override the unembedding bias eventually.
    """
    m = _tiny_model()
    with torch.no_grad():
        m.head_bias.zero_()
        m.head_bias[0] = 5.0   # mild bias toward token 0
    prompt = torch.tensor([0, 0, 0], dtype=torch.long)

    # No penalty: at temperature near 0 the model just keeps emitting 0.
    g_no_pen = torch.Generator().manual_seed(0)
    out_no_pen = generate(
        m, prompt, max_new_tokens=10, temperature=0.1,
        repetition_penalty=1.0, generator=g_no_pen,
    )
    # With a strong penalty, repeats of 0 should be discouraged.
    g_pen = torch.Generator().manual_seed(0)
    out_pen = generate(
        m, prompt, max_new_tokens=10, temperature=0.1,
        repetition_penalty=10.0, generator=g_pen,
    )
    n_zero_no_pen = (out_no_pen[3:] == 0).sum().item()
    n_zero_pen = (out_pen[3:] == 0).sum().item()
    # Penalty must reduce (or at least not increase) the rate of 0s.
    assert n_zero_pen <= n_zero_no_pen
    # And it should produce a different sequence than no-penalty.
    assert not torch.equal(out_no_pen, out_pen)
