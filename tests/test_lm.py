"""Tests for Module 06: language models — counts bigram, neural bigram, MLP.

Suggested order to implement & turn green:

  1. CountsBigramLM.fit                 → test_counts_fit_*
  2. CountsBigramLM.logits              → test_counts_logits_*
  3. NeuralBigramLM.forward             → test_neural_bigram_forward_*
  4. MLPLanguageModel.forward           → test_mlp_forward_*
  5. perplexity                         → test_perplexity_*
  6. sample                             → test_sample_*
  7. train_lm                           → test_train_lm_*

The construction tests for all three models pass from the start because
their `__init__`s are fully implemented.

`get_batch` is implemented for you and tested at the top of the helpers
section as a sanity check on the boilerplate.

Most tests use a tiny vocabulary (≤ 16 tokens) and short token sequences,
so the suite runs in well under a second.
"""
from __future__ import annotations

import math

import pytest
import torch

from g2c.lm import (
    CountsBigramLM,
    MLPLanguageModel,
    NeuralBigramLM,
    get_batch,
    perplexity,
    sample,
    train_lm,
)


# ----------------------------------------------------------------------
# CountsBigramLM — construction (boilerplate)
# ----------------------------------------------------------------------

def test_counts_construction_table_shape():
    m = CountsBigramLM(vocab_size=10)
    assert m.counts.shape == (10, 10)


def test_counts_construction_starts_zero():
    m = CountsBigramLM(vocab_size=5)
    assert m.counts.sum().item() == 0


def test_counts_construction_context_length():
    m = CountsBigramLM(vocab_size=10)
    assert m.context_length == 1


def test_counts_construction_default_smoothing():
    m = CountsBigramLM(vocab_size=10)
    assert m.smoothing == 1.0


# ----------------------------------------------------------------------
# CountsBigramLM.fit — counting bigrams
# ----------------------------------------------------------------------

def test_counts_fit_simple_pair():
    """The corpus [0, 1] has one bigram (0, 1)."""
    m = CountsBigramLM(vocab_size=4)
    m.fit(torch.tensor([0, 1]))
    assert m.counts[0, 1].item() == 1
    # All other entries should still be zero.
    assert m.counts.sum().item() == 1


def test_counts_fit_multiple_pairs():
    """The corpus [0, 1, 2, 0] has bigrams (0,1), (1,2), (2,0)."""
    m = CountsBigramLM(vocab_size=4)
    m.fit(torch.tensor([0, 1, 2, 0]))
    assert m.counts[0, 1].item() == 1
    assert m.counts[1, 2].item() == 1
    assert m.counts[2, 0].item() == 1
    assert m.counts.sum().item() == 3


def test_counts_fit_repeated_pair_accumulates():
    """The pair (0, 1) appears twice in [0, 1, 0, 1]."""
    m = CountsBigramLM(vocab_size=4)
    m.fit(torch.tensor([0, 1, 0, 1]))
    assert m.counts[0, 1].item() == 2
    assert m.counts[1, 0].item() == 1


def test_counts_fit_called_twice_accumulates():
    """Calling fit a second time should add to existing counts, not replace."""
    m = CountsBigramLM(vocab_size=4)
    m.fit(torch.tensor([0, 1]))
    m.fit(torch.tensor([0, 1]))
    assert m.counts[0, 1].item() == 2


def test_counts_fit_total_pairs():
    """A length-N sequence yields N-1 adjacent pairs."""
    m = CountsBigramLM(vocab_size=10)
    ids = torch.randint(0, 10, (100,))
    m.fit(ids)
    assert m.counts.sum().item() == 99


# ----------------------------------------------------------------------
# CountsBigramLM.logits — table lookup + smoothing + log
# ----------------------------------------------------------------------

def test_counts_logits_shape():
    m = CountsBigramLM(vocab_size=8)
    m.fit(torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]))
    out = m.logits(torch.tensor([0, 1, 2]))
    assert out.shape == (3, 8)


def test_counts_logits_are_log_probs():
    """Each row should sum (in probability space) to 1."""
    m = CountsBigramLM(vocab_size=4)
    m.fit(torch.tensor([0, 1, 2, 3, 0, 1]))
    log_probs = m.logits(torch.tensor([0, 1, 2, 3]))
    probs = log_probs.exp()
    row_sums = probs.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones(4), atol=1e-5)


def test_counts_logits_smoothing_no_zero_probability():
    """With smoothing > 0, no token has zero probability — even unseen ones."""
    m = CountsBigramLM(vocab_size=5, smoothing=1.0)
    # Train so that token 0 was seen, but never followed by token 4.
    m.fit(torch.tensor([0, 1, 0, 2, 0, 3]))
    log_probs = m.logits(torch.tensor([0]))
    # Every probability is finite.
    assert torch.isfinite(log_probs).all()


def test_counts_logits_no_smoothing_zero_for_unseen():
    """With smoothing = 0, an unseen successor should have prob 0 (log -inf)."""
    m = CountsBigramLM(vocab_size=3, smoothing=0.0)
    m.fit(torch.tensor([0, 1, 0, 1]))   # only (0,1) and (1,0) seen
    log_probs = m.logits(torch.tensor([0]))
    # P(2 | 0) = 0  →  log = -inf
    assert log_probs[0, 2].item() == float("-inf")
    # P(1 | 0) = 1  →  log = 0
    assert log_probs[0, 1].item() == pytest.approx(0.0, abs=1e-6)


def test_counts_logits_uniform_when_unfit_with_smoothing():
    """An unfit table + uniform smoothing → uniform distribution."""
    m = CountsBigramLM(vocab_size=8, smoothing=1.0)
    log_probs = m.logits(torch.tensor([0]))
    expected = math.log(1.0 / 8)
    assert torch.allclose(log_probs[0], torch.full((8,), expected), atol=1e-6)


def test_counts_logits_accepts_2d_context():
    """logits should accept ctx of shape (batch, 1) as well as (batch,)."""
    m = CountsBigramLM(vocab_size=4)
    m.fit(torch.tensor([0, 1, 2, 3, 0]))
    a = m.logits(torch.tensor([0, 1]))
    b = m.logits(torch.tensor([[0], [1]]))
    assert torch.allclose(a, b)


# ----------------------------------------------------------------------
# NeuralBigramLM — construction (boilerplate)
# ----------------------------------------------------------------------

def test_neural_bigram_construction():
    NeuralBigramLM(vocab_size=10, embedding_dim=4)


def test_neural_bigram_context_length():
    m = NeuralBigramLM(vocab_size=10, embedding_dim=4)
    assert m.context_length == 1


def test_neural_bigram_parameters_count():
    """Embedding (V*D) + Linear (D*V + V) parameter tensors: 1 + 2 = 3 entries."""
    m = NeuralBigramLM(vocab_size=10, embedding_dim=4)
    params = list(m.parameters())
    assert len(params) == 3


# ----------------------------------------------------------------------
# NeuralBigramLM.forward — embed → linear → logits
# ----------------------------------------------------------------------

def test_neural_bigram_forward_shape():
    m = NeuralBigramLM(vocab_size=10, embedding_dim=4)
    out = m(torch.tensor([0, 1, 2]))
    assert out.shape == (3, 10)


def test_neural_bigram_forward_accepts_2d_context():
    m = NeuralBigramLM(vocab_size=10, embedding_dim=4)
    a = m(torch.tensor([0, 1, 2]))
    b = m(torch.tensor([[0], [1], [2]]))
    assert a.shape == b.shape == (3, 10)


def test_neural_bigram_forward_routes_gradients():
    """A backward pass should populate grads on every parameter."""
    m = NeuralBigramLM(vocab_size=10, embedding_dim=4)
    out = m(torch.tensor([0, 1, 2]))
    out.sum().backward()
    for p in m.parameters():
        assert p.grad is not None


def test_neural_bigram_logits_alias():
    """`.logits` and `__call__` should return the same tensor."""
    m = NeuralBigramLM(vocab_size=10, embedding_dim=4)
    ctx = torch.tensor([0, 1, 2])
    assert torch.allclose(m(ctx), m.logits(ctx))


# ----------------------------------------------------------------------
# MLPLanguageModel — construction (boilerplate)
# ----------------------------------------------------------------------

def test_mlp_construction():
    MLPLanguageModel(vocab_size=10, context_length=2, embedding_dim=4, hidden_dim=8)


def test_mlp_construction_stores_args():
    m = MLPLanguageModel(vocab_size=10, context_length=3, embedding_dim=4, hidden_dim=8)
    assert m.vocab_size == 10
    assert m.context_length == 3
    assert m.embedding_dim == 4
    assert m.hidden_dim == 8


def test_mlp_parameters_count():
    """Embedding (1) + hidden Linear (2: W, b) + output Linear (2: W, b) = 5 tensors."""
    m = MLPLanguageModel(vocab_size=10, context_length=2, embedding_dim=4, hidden_dim=8)
    assert len(list(m.parameters())) == 5


# ----------------------------------------------------------------------
# MLPLanguageModel.forward — embed → flatten → tanh hidden → output
# ----------------------------------------------------------------------

def test_mlp_forward_shape():
    m = MLPLanguageModel(vocab_size=10, context_length=2, embedding_dim=4, hidden_dim=8)
    ctx = torch.tensor([[0, 1], [2, 3], [4, 5]])
    out = m(ctx)
    assert out.shape == (3, 10)


def test_mlp_forward_shape_context_length_3():
    m = MLPLanguageModel(vocab_size=10, context_length=3, embedding_dim=4, hidden_dim=8)
    ctx = torch.tensor([[0, 1, 2], [3, 4, 5]])
    out = m(ctx)
    assert out.shape == (2, 10)


def test_mlp_forward_routes_gradients():
    m = MLPLanguageModel(vocab_size=10, context_length=2, embedding_dim=4, hidden_dim=8)
    out = m(torch.tensor([[0, 1], [2, 3]]))
    out.sum().backward()
    for p in m.parameters():
        assert p.grad is not None


def test_mlp_forward_position_sensitive():
    """Concatenation should make (a, b) and (b, a) produce different logits.

    This is the headline reason we concatenate rather than sum or average
    the embeddings.
    """
    torch.manual_seed(0)
    m = MLPLanguageModel(vocab_size=10, context_length=2, embedding_dim=4, hidden_dim=8)
    out_ab = m(torch.tensor([[3, 7]]))
    out_ba = m(torch.tensor([[7, 3]]))
    assert not torch.allclose(out_ab, out_ba, atol=1e-3)


# ----------------------------------------------------------------------
# get_batch — implemented helper, tested as a sanity check
# ----------------------------------------------------------------------

def test_get_batch_shapes():
    ids = torch.arange(100)
    x, y = get_batch(ids, context_length=3, batch_size=16)
    assert x.shape == (16, 3)
    assert y.shape == (16,)


def test_get_batch_targets_immediately_follow_context():
    """The target should always be the token right after the context window."""
    ids = torch.arange(50)        # 0, 1, 2, ..., 49
    x, y = get_batch(ids, context_length=2, batch_size=20)
    # x[i] = [s, s+1], y[i] = s+2
    assert torch.equal(y, x[:, -1] + 1)


def test_get_batch_too_short_raises():
    with pytest.raises(ValueError):
        get_batch(torch.tensor([0, 1]), context_length=5, batch_size=1)


# ----------------------------------------------------------------------
# perplexity — exp(mean cross-entropy)
# ----------------------------------------------------------------------

def test_perplexity_returns_float():
    m = NeuralBigramLM(vocab_size=8, embedding_dim=4)
    ids = torch.randint(0, 8, (50,))
    p = perplexity(m, ids)
    assert isinstance(p, float)
    assert p > 0


def test_perplexity_uniform_model_matches_vocab_size():
    """A model whose distribution is uniform over vocab has perplexity = vocab_size.

    We construct this with a CountsBigramLM that has never been fit and
    smoothing = 1.0 — every row of `counts` is zero, smoothing makes every
    row uniform, so every prediction is uniform over `vocab_size` tokens.
    """
    V = 8
    m = CountsBigramLM(vocab_size=V, smoothing=1.0)
    ids = torch.randint(0, V, (200,))
    assert perplexity(m, ids) == pytest.approx(float(V), rel=1e-4)


def test_perplexity_perfect_model_is_one():
    """A model that puts all its mass on the actual next token has perplexity 1."""
    V = 4
    m = CountsBigramLM(vocab_size=V, smoothing=0.0)
    # Fit on a perfectly-predictable sequence: 0,1,0,1,0,1,...
    train = torch.tensor([0, 1] * 50)
    m.fit(train)
    eval_ids = torch.tensor([0, 1] * 20)
    # With smoothing=0 and the deterministic pattern, every prediction is correct.
    assert perplexity(m, eval_ids) == pytest.approx(1.0, abs=1e-3)


def test_perplexity_works_for_neural_bigram():
    m = NeuralBigramLM(vocab_size=8, embedding_dim=4)
    ids = torch.randint(0, 8, (50,))
    # Just check it runs and produces a finite positive number.
    p = perplexity(m, ids)
    assert math.isfinite(p)


def test_perplexity_works_for_mlp():
    m = MLPLanguageModel(vocab_size=8, context_length=2, embedding_dim=4, hidden_dim=8)
    ids = torch.randint(0, 8, (50,))
    p = perplexity(m, ids)
    assert math.isfinite(p)


# ----------------------------------------------------------------------
# sample — autoregressive decoding
# ----------------------------------------------------------------------

def test_sample_length():
    """Output is len(prompt) + num_tokens."""
    m = NeuralBigramLM(vocab_size=8, embedding_dim=4)
    out = sample(m, torch.tensor([0]), num_tokens=10)
    assert out.shape == (11,)


def test_sample_preserves_prompt():
    """The prompt should appear unchanged at the start of the output."""
    m = NeuralBigramLM(vocab_size=8, embedding_dim=4)
    prompt = torch.tensor([3, 1, 4])
    out = sample(m, prompt, num_tokens=5)
    assert torch.equal(out[: len(prompt)], prompt)


def test_sample_token_range():
    """All sampled tokens must be in [0, vocab_size)."""
    V = 8
    m = NeuralBigramLM(vocab_size=V, embedding_dim=4)
    out = sample(m, torch.tensor([0]), num_tokens=50)
    assert out.min().item() >= 0
    assert out.max().item() < V


def test_sample_works_for_counts_model():
    """Sampling should work with the counts model too — same interface."""
    V = 4
    m = CountsBigramLM(vocab_size=V)
    m.fit(torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]))
    out = sample(m, torch.tensor([0]), num_tokens=10)
    assert out.shape == (11,)
    assert out.min().item() >= 0
    assert out.max().item() < V


def test_sample_works_for_mlp():
    """Sampling should respect the MLP's context_length."""
    m = MLPLanguageModel(vocab_size=8, context_length=2, embedding_dim=4, hidden_dim=8)
    out = sample(m, torch.tensor([0, 1]), num_tokens=10)
    assert out.shape == (12,)


def test_sample_short_prompt_raises_for_mlp():
    """An MLP with context_length=2 needs a prompt of length >= 2."""
    m = MLPLanguageModel(vocab_size=8, context_length=2, embedding_dim=4, hidden_dim=8)
    with pytest.raises(ValueError):
        sample(m, torch.tensor([0]), num_tokens=5)   # prompt too short


# ----------------------------------------------------------------------
# train_lm — the full SGD training loop
# ----------------------------------------------------------------------

def test_train_lm_returns_dict_with_expected_keys():
    m = NeuralBigramLM(vocab_size=8, embedding_dim=4)
    ids = torch.randint(0, 8, (200,))
    out = train_lm(m, ids, num_steps=10, batch_size=4, log_every=5)
    assert "train_losses" in out
    assert "val_perplexities" in out


def test_train_lm_records_at_log_every():
    """We log at steps 0, log_every, 2*log_every, ..., plus the final step."""
    m = NeuralBigramLM(vocab_size=8, embedding_dim=4)
    ids = torch.randint(0, 8, (200,))
    out = train_lm(m, ids, num_steps=20, batch_size=4, log_every=5)
    # Steps 0, 5, 10, 15, plus the final 19 — but 15 is the last log_every checkpoint
    # before 19, so we record at 0, 5, 10, 15, 19 → 5 entries.
    # If step 19 happens to coincide with a log_every step, no double-counting.
    assert len(out["train_losses"]) >= 4
    assert len(out["train_losses"]) <= 5


def test_train_lm_loss_decreases_on_easy_corpus():
    """On a learnable pattern, training loss should drop noticeably.

    We use a very simple repeating pattern that a neural bigram can fit
    perfectly: token 0 is always followed by token 1, token 1 by token 0.
    """
    torch.manual_seed(0)
    m = NeuralBigramLM(vocab_size=4, embedding_dim=8)
    ids = torch.tensor([0, 1] * 200)
    out = train_lm(m, ids, num_steps=200, batch_size=16, lr=0.5, log_every=20)
    losses = out["train_losses"]
    assert losses[-1] < losses[0] * 0.5   # at least halved


def test_train_lm_records_val_perplexity():
    m = NeuralBigramLM(vocab_size=8, embedding_dim=4)
    train_ids = torch.randint(0, 8, (200,))
    val_ids = torch.randint(0, 8, (50,))
    out = train_lm(m, train_ids, val_ids=val_ids, num_steps=10,
                   batch_size=4, log_every=5)
    assert len(out["val_perplexities"]) >= 1
    assert all(math.isfinite(p) and p > 0 for p in out["val_perplexities"])


def test_train_lm_works_for_mlp():
    """train_lm should be model-agnostic — the MLP should train just as well."""
    torch.manual_seed(0)
    m = MLPLanguageModel(vocab_size=4, context_length=2, embedding_dim=8, hidden_dim=16)
    ids = torch.tensor([0, 1, 2] * 200)
    out = train_lm(m, ids, num_steps=200, batch_size=16, lr=0.5, log_every=20)
    losses = out["train_losses"]
    assert losses[-1] < losses[0] * 0.7
