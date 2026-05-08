"""Tests for Module 09B: pretraining setup.

Suggested order to implement & turn green:

  1. Read the `split_token_stream` and `get_lm_batch` tests. These
     helpers are implemented for you and establish the data contract.
  2. lm_cross_entropy             → test_lm_cross_entropy_*

`split_token_stream` and `get_lm_batch` are boilerplate. Their tests
pass from the start as a sanity check on the shape conventions. The
loss tests fail with `NotImplementedError` until you implement
`g2c.pretraining.lm_cross_entropy`.
"""
from __future__ import annotations

import math

import pytest
import torch

from g2c.pretraining import get_lm_batch, lm_cross_entropy, split_token_stream

# ----------------------------------------------------------------------
# split_token_stream — boilerplate (implemented)
# ----------------------------------------------------------------------

def test_split_token_stream_returns_contiguous_splits():
    ids = torch.arange(10)
    train_ids, val_ids = split_token_stream(ids, train_fraction=0.7)
    assert torch.equal(train_ids, torch.arange(7))
    assert torch.equal(val_ids, torch.arange(7, 10))


def test_split_token_stream_rejects_non_1d_input():
    with pytest.raises(ValueError, match="1-D"):
        split_token_stream(torch.arange(12).reshape(3, 4))


def test_split_token_stream_rejects_empty_split():
    with pytest.raises(ValueError, match="empty"):
        split_token_stream(torch.arange(2), train_fraction=0.01)


def test_split_token_stream_rejects_invalid_fraction():
    ids = torch.arange(10)
    for frac in (0.0, 1.0, -0.1, 1.1):
        with pytest.raises(ValueError, match="between 0 and 1"):
            split_token_stream(ids, train_fraction=frac)


# ----------------------------------------------------------------------
# get_lm_batch — boilerplate (implemented)
# ----------------------------------------------------------------------

def test_get_lm_batch_shapes():
    ids = torch.arange(50)
    x, y = get_lm_batch(ids, batch_size=4, context_length=6)
    assert x.shape == (4, 6)
    assert y.shape == (4, 6)


def test_get_lm_batch_targets_are_shifted_by_one():
    """The headline contract: y[b, t] is the token immediately AFTER
    x[b, t] in the corpus.

    A miswired shift (off-by-one in either direction) silently trains
    the wrong objective. Pinning this down is more important than it
    looks.
    """
    ids = torch.arange(100)  # ids[i] == i, so adjacency is trivial.
    x, y = get_lm_batch(ids, batch_size=8, context_length=5)
    # y[b, t] should equal x[b, t] + 1 because adjacent integers in the
    # corpus differ by 1.
    assert torch.equal(y, x + 1)


def test_get_lm_batch_no_overrun():
    """Sampling must never run off the end of the corpus."""
    ids = torch.arange(20)
    # With ctx=4, the highest valid `start` is `n - ctx - 1 = 15`. Calling
    # repeatedly should never produce a target index past `n - 1`.
    for _ in range(50):
        x, y = get_lm_batch(ids, batch_size=4, context_length=4)
        assert (y < 20).all()
        assert (x < 20).all()


def test_get_lm_batch_too_short_raises():
    ids = torch.arange(5)
    with pytest.raises(ValueError):
        get_lm_batch(ids, batch_size=2, context_length=10)


def test_get_lm_batch_reproducible_with_generator():
    ids = torch.arange(100)
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    x1, y1 = get_lm_batch(ids, batch_size=4, context_length=6, generator=g1)
    x2, y2 = get_lm_batch(ids, batch_size=4, context_length=6, generator=g2)
    assert torch.equal(x1, x2)
    assert torch.equal(y1, y2)


def test_get_lm_batch_delegates_to_disk_backed_stream():
    class FakeDiskBackedStream:
        def __len__(self):
            return 100

        def get_lm_batch(self, batch_size, context_length, *, generator=None):
            x = torch.zeros(batch_size, context_length, dtype=torch.long)
            return x, x + 1

    x, y = get_lm_batch(FakeDiskBackedStream(), batch_size=3, context_length=4)

    assert x.shape == (3, 4)
    assert torch.equal(y, x + 1)


# ----------------------------------------------------------------------
# lm_cross_entropy — scaffolded
# ----------------------------------------------------------------------

def test_lm_cross_entropy_returns_scalar():
    logits = torch.randn(2, 3, 5)
    targets = torch.randint(0, 5, (2, 3))
    out = lm_cross_entropy(logits, targets)
    assert out.shape == ()


def test_lm_cross_entropy_uniform_logits_equals_log_vocab():
    """Sanity value: with uniform logits (all zeros), the loss equals
    log(vocab_size) — the entropy of a uniform distribution.

    This is the reference value to keep in mind during training. At
    step 0, with random-init weights, your loss should be ≈ log(V).
    A loss much smaller than that suggests a shape bug aligning the
    loss against the wrong rows.
    """
    vocab_size = 7
    logits = torch.zeros(3, 4, vocab_size)
    targets = torch.randint(0, vocab_size, (3, 4))
    loss = lm_cross_entropy(logits, targets).item()
    assert loss == pytest.approx(math.log(vocab_size), abs=1e-5)


def test_lm_cross_entropy_perfect_predictions_near_zero():
    """If logits put almost all probability mass on the target token,
    the loss should be near zero.
    """
    vocab_size = 5
    targets = torch.tensor([[0, 1, 2], [3, 4, 0]])
    # Build logits with a huge spike at the target.
    logits = torch.zeros(2, 3, vocab_size)
    for b in range(2):
        for t in range(3):
            logits[b, t, targets[b, t]] = 100.0
    loss = lm_cross_entropy(logits, targets).item()
    assert loss < 1e-3


def test_lm_cross_entropy_per_position_average():
    """All B*T positions contribute equally — the loss is the mean,
    not the sum and not "the last position only."

    A miswired version that only computed loss at the last position
    (a Module-06-MLP throwback) would silently pass the shape and
    uniform-baseline tests. This pins down that EVERY position gets a
    gradient.
    """
    vocab_size = 4
    # Two positions, both predicted perfectly: loss should be ≈ 0.
    # If only the last position counted, we'd get the same answer for
    # this case — so add an asymmetric one to disambiguate.
    targets = torch.tensor([[0, 1]])
    logits = torch.zeros(1, 2, vocab_size)
    # Position 0: huge spike on target (0). Position 1: uniform.
    logits[0, 0, 0] = 100.0
    # Per-position losses: pos 0 ≈ 0, pos 1 = log(4).
    # Mean across positions: log(4) / 2 ≈ 0.6931...
    loss = lm_cross_entropy(logits, targets).item()
    assert loss == pytest.approx(math.log(vocab_size) / 2, abs=1e-4)


def test_lm_cross_entropy_routes_gradients():
    logits = torch.randn(2, 3, 5, requires_grad=True)
    targets = torch.randint(0, 5, (2, 3))
    loss = lm_cross_entropy(logits, targets)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
