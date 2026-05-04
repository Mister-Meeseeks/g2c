"""Tests for Module 10: pretraining a tiny GPT.

Suggested order to implement & turn green:

  1. lm_cross_entropy             → test_lm_cross_entropy_*
  2. Trainer.train_step           → test_trainer_train_step_*,
                                    test_trainer_evaluate_*,
                                    test_trainer_train_*

`get_lm_batch` is implemented for you; its tests pass from the start
as a sanity check on the boilerplate.

Trainer construction tests, attribute defaults, device-handling tests,
and Trainer.lr tests all pass without needing the model to forward
correctly once Module 03B's schedule is implemented. The
Trainer.train_step / evaluate / train tests are the only ones that
exercise a full TransformerLM forward+backward pass -- those depend on
Modules 03 / 03B / 05 / 08 / 09 being implemented in addition to this
module's language-modeling loss and trainer scaffold. If your
full TransformerLM smoke test from Module 09 isn't passing yet, finish
that first.

Most tests use very small dimensions (`vocab_size=12`,
`embedding_dim=8`, `num_layers=1`, `T=6`, batch=4) so the suite runs
in well under a second.
"""
from __future__ import annotations

import math

import pytest
import torch

from g2c.nn import AdamW, SGD
from g2c.training import (
    Trainer,
    cosine_with_warmup,
    get_lm_batch,
    lm_cross_entropy,
)
from g2c.transformer import TransformerLM


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


# ----------------------------------------------------------------------
# Trainer — construction (boilerplate)
# ----------------------------------------------------------------------

def _tiny_model() -> TransformerLM:
    """A miniature TransformerLM used by Trainer tests."""
    torch.manual_seed(0)
    return TransformerLM(
        vocab_size=12,
        embedding_dim=8,
        num_layers=1,
        num_heads=2,
        max_seq_len=16,
    )


def test_trainer_construction():
    Trainer(
        _tiny_model(),
        batch_size=4,
        context_length=6,
        max_steps=10,
        max_lr=1e-3,
    )


def test_trainer_construction_stores_args():
    m = _tiny_model()
    t = Trainer(
        m,
        batch_size=8,
        context_length=4,
        max_steps=200,
        max_lr=2e-3,
        min_lr=1e-5,
        warmup_steps=20,
        weight_decay=0.01,
        grad_clip=1.0,
        eval_every=50,
        eval_iters=5,
        log_every=2,
    )
    assert t.model is m
    assert t.batch_size == 8
    assert t.context_length == 4
    assert t.max_steps == 200
    assert t.max_lr == 2e-3
    assert t.min_lr == 1e-5
    assert t.warmup_steps == 20
    assert t.weight_decay == 0.01
    assert t.grad_clip == 1.0
    assert t.eval_every == 50
    assert t.eval_iters == 5
    assert t.log_every == 2


def test_trainer_defaults():
    """Defaults: min_lr=0, warmup_steps=0, weight_decay=0, grad_clip=None,
    device=auto.
    """
    t = Trainer(
        _tiny_model(),
        batch_size=4,
        context_length=6,
        max_steps=10,
        max_lr=1e-3,
    )
    assert t.min_lr == 0.0
    assert t.warmup_steps == 0
    assert t.weight_decay == 0.0
    assert t.grad_clip is None
    assert t.optimizer_name == "sgd"
    expected_device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    assert t.device == expected_device


def test_trainer_explicit_cpu_device_moves_model_params():
    m = _tiny_model()
    t = Trainer(
        m,
        batch_size=4,
        context_length=6,
        max_steps=10,
        max_lr=1e-3,
        device="cpu",
    )
    assert t.device == torch.device("cpu")
    for p in m.parameters():
        assert p.device.type == "cpu"


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS is only available on supported Apple Silicon machines",
)
def test_trainer_explicit_mps_device_moves_model_params_and_batches():
    torch.manual_seed(0)
    m = _tiny_model()
    t = Trainer(
        m,
        batch_size=4,
        context_length=6,
        max_steps=10,
        max_lr=1e-3,
        device="mps",
    )
    for p in m.parameters():
        assert p.device.type == "mps"
    ids = torch.randint(0, 12, (200,))  # corpus can stay on CPU
    metrics = t.train_step(ids)
    assert metrics["loss"] > 0
    for p in m.parameters():
        if p.grad is not None:
            assert p.grad.device.type == "mps"


def test_trainer_step_starts_at_zero():
    t = Trainer(
        _tiny_model(),
        batch_size=4,
        context_length=6,
        max_steps=10,
        max_lr=1e-3,
    )
    assert t.step == 0


def test_trainer_optimizer_is_sgd():
    t = Trainer(
        _tiny_model(),
        batch_size=4,
        context_length=6,
        max_steps=10,
        max_lr=1e-3,
        weight_decay=0.05,
    )
    assert isinstance(t.optimizer, SGD)
    assert t.optimizer.lr == 1e-3
    assert t.optimizer.weight_decay == 0.05


def test_trainer_can_construct_adamw_optimizer():
    t = Trainer(
        _tiny_model(),
        batch_size=4,
        context_length=6,
        max_steps=10,
        max_lr=3e-4,
        weight_decay=0.01,
        optimizer="adamw",
    )
    assert t.optimizer_name == "adamw"
    assert isinstance(t.optimizer, AdamW)
    assert t.optimizer.lr == 3e-4
    assert t.optimizer.weight_decay == 0.01


def test_trainer_rejects_unknown_optimizer():
    with pytest.raises(ValueError, match="optimizer"):
        Trainer(
            _tiny_model(),
            batch_size=4,
            context_length=6,
            max_steps=10,
            max_lr=1e-3,
            optimizer="rmsprop",
        )


# ----------------------------------------------------------------------
# Trainer.lr — depends on cosine_with_warmup
# ----------------------------------------------------------------------

def test_trainer_lr_explicit_step_uses_schedule():
    t = Trainer(
        _tiny_model(),
        batch_size=4,
        context_length=6,
        max_steps=110,
        max_lr=1e-3,
        warmup_steps=10,
    )
    # End of warmup = max_lr.
    assert t.lr(9) == pytest.approx(1e-3)
    # End of cosine = min_lr (default 0).
    assert t.lr(110) == pytest.approx(0.0, abs=1e-9)


def test_trainer_lr_default_uses_self_step():
    t = Trainer(
        _tiny_model(),
        batch_size=4,
        context_length=6,
        max_steps=20,
        max_lr=1e-3,
        warmup_steps=10,
    )
    t.step = 5
    expected = cosine_with_warmup(
        5, warmup_steps=10, max_steps=20, max_lr=1e-3
    )
    assert t.lr() == pytest.approx(expected)


# ----------------------------------------------------------------------
# Trainer.train_step — depends on lm_cross_entropy, schedule, clip
# ----------------------------------------------------------------------

def test_trainer_train_step_increments_step_counter():
    torch.manual_seed(0)
    t = Trainer(
        _tiny_model(),
        batch_size=4,
        context_length=6,
        max_steps=10,
        max_lr=1e-3,
    )
    ids = torch.randint(0, 12, (200,))
    assert t.step == 0
    t.train_step(ids)
    assert t.step == 1
    t.train_step(ids)
    assert t.step == 2


def test_trainer_train_step_returns_metrics_dict():
    torch.manual_seed(0)
    t = Trainer(
        _tiny_model(),
        batch_size=4,
        context_length=6,
        max_steps=10,
        max_lr=1e-3,
        warmup_steps=2,
        grad_clip=1.0,
    )
    ids = torch.randint(0, 12, (200,))
    metrics = t.train_step(ids)
    assert "loss" in metrics
    assert "lr" in metrics
    assert "grad_norm" in metrics
    assert metrics["loss"] > 0  # CE is non-negative; should be > 0 at init
    assert metrics["lr"] > 0    # First warmup step
    assert metrics["grad_norm"] > 0  # Clipping was on; norm was measured


def test_trainer_train_step_writes_lr_to_optimizer():
    """The trainer must update optimizer.lr from the schedule each step.
    Forgetting to do so is a silent bug — the model trains at the
    initial lr forever, ignoring the warmup ramp.
    """
    torch.manual_seed(0)
    t = Trainer(
        _tiny_model(),
        batch_size=4,
        context_length=6,
        max_steps=20,
        max_lr=1e-2,
        warmup_steps=10,
    )
    ids = torch.randint(0, 12, (200,))
    t.train_step(ids)
    # After step 0, optimizer.lr should equal the lr we used at step 0:
    expected_lr = cosine_with_warmup(
        0, warmup_steps=10, max_steps=20, max_lr=1e-2
    )
    assert t.optimizer.lr == pytest.approx(expected_lr)


def test_trainer_train_step_no_grad_clip_returns_zero_norm():
    """When grad_clip is None, the metric 'grad_norm' is reported as
    0.0 — a sentinel saying "not measured." (Pretraining runs without
    clipping are real but unusual; the metric should still be present.)
    """
    torch.manual_seed(0)
    t = Trainer(
        _tiny_model(),
        batch_size=4,
        context_length=6,
        max_steps=10,
        max_lr=1e-3,
        grad_clip=None,
    )
    ids = torch.randint(0, 12, (200,))
    metrics = t.train_step(ids)
    assert metrics["grad_norm"] == 0.0


def test_trainer_train_step_clips_grads():
    """With grad_clip set, post-step parameter gradients should have
    global L2 norm ≤ grad_clip.

    Verifies the clip actually fires, not just that the metric is
    populated. The test seeds with a model whose initial loss creates
    a large gradient; without clipping the norm exceeds 1.
    """
    torch.manual_seed(0)
    m = _tiny_model()
    # Inflate the head's weights so the loss + gradients are large.
    with torch.no_grad():
        m.head.W *= 100.0
    t = Trainer(
        m,
        batch_size=4,
        context_length=6,
        max_steps=10,
        max_lr=1e-3,
        grad_clip=0.5,
    )
    ids = torch.randint(0, 12, (200,))
    metrics = t.train_step(ids)
    # The pre-clip norm reported by the metric should be > 0.5 (the clip
    # threshold), confirming that clipping actually scaled grads down.
    assert metrics["grad_norm"] > 0.5
    # Post-step, the grads should reflect the clip — global norm ≤ 0.5.
    total_norm_sq = 0.0
    for p in m.parameters():
        if p.grad is not None:
            total_norm_sq += (p.grad ** 2).sum().item()
    total_norm = math.sqrt(total_norm_sq)
    assert total_norm <= 0.5 + 1e-3


def test_trainer_train_step_decreases_loss_on_fixed_batch():
    """End-to-end smoke test of train_step: stepping repeatedly on the
    same data should overfit and reduce the loss.

    This is the closest analog to Module 09's `test_transformer_lm_smoke_train`
    — confirms the full pipeline (forward, loss, backward, clip, step)
    is wired correctly through the Trainer rather than via a hand-
    written loop.
    """
    torch.manual_seed(0)
    t = Trainer(
        _tiny_model(),
        batch_size=4,
        context_length=4,
        max_steps=30,
        max_lr=0.5,                    # large lr for fast overfit
        warmup_steps=0,
    )
    # Tiny corpus of only 30 tokens — train_step samples random windows
    # but the entropy is bounded.
    ids = torch.randint(0, 12, (30,))
    initial_loss = t.train_step(ids)["loss"]
    for _ in range(29):
        last_loss = t.train_step(ids)["loss"]
    assert last_loss < initial_loss * 0.7


# ----------------------------------------------------------------------
# Trainer.evaluate — depends on lm_cross_entropy
# ----------------------------------------------------------------------

def test_trainer_evaluate_returns_float():
    torch.manual_seed(0)
    t = Trainer(
        _tiny_model(),
        batch_size=4,
        context_length=6,
        max_steps=10,
        max_lr=1e-3,
        eval_iters=3,
    )
    ids = torch.randint(0, 12, (200,))
    out = t.evaluate(ids)
    assert isinstance(out, float)
    assert out > 0
    assert math.isfinite(out)


def test_trainer_evaluate_does_not_populate_grads():
    """Evaluation should not leave `.grad` populated on parameters —
    `torch.no_grad()` should prevent the autograd graph from being
    built at all.

    Forgetting `no_grad` doesn't break correctness but wastes memory
    on long evals — the autograd graph for thousands of evaluation
    forward passes can blow out memory at modest model sizes.
    """
    torch.manual_seed(0)
    m = _tiny_model()
    t = Trainer(
        m, batch_size=4, context_length=6, max_steps=10,
        max_lr=1e-3, eval_iters=3,
    )
    # Zero any existing grads to be safe.
    for p in m.parameters():
        p.grad = None
    ids = torch.randint(0, 12, (200,))
    t.evaluate(ids)
    for p in m.parameters():
        assert p.grad is None


# ----------------------------------------------------------------------
# Trainer.train — the full loop
# ----------------------------------------------------------------------

def test_trainer_train_runs_max_steps():
    torch.manual_seed(0)
    t = Trainer(
        _tiny_model(),
        batch_size=4,
        context_length=6,
        max_steps=8,
        max_lr=1e-3,
    )
    ids = torch.randint(0, 12, (200,))
    t.train(ids)
    assert t.step == 8


def test_trainer_train_history_contains_keys():
    torch.manual_seed(0)
    t = Trainer(
        _tiny_model(),
        batch_size=4,
        context_length=6,
        max_steps=10,
        max_lr=1e-3,
        log_every=2,
    )
    ids = torch.randint(0, 12, (200,))
    history = t.train(ids)
    for key in ("step", "train_loss", "lr", "grad_norm", "val_step", "val_loss"):
        assert key in history
    # No val_ids passed → val arrays are empty.
    assert history["val_step"] == []
    assert history["val_loss"] == []
    # log_every=2 → metrics at steps 0, 2, 4, 6, 8 plus final step (9).
    assert len(history["train_loss"]) >= 5


def test_trainer_train_records_val_loss():
    torch.manual_seed(0)
    t = Trainer(
        _tiny_model(),
        batch_size=4,
        context_length=6,
        max_steps=10,
        max_lr=1e-3,
        eval_every=5,
        eval_iters=2,
    )
    train_ids = torch.randint(0, 12, (200,))
    val_ids = torch.randint(0, 12, (200,))
    history = t.train(train_ids, val_ids=val_ids)
    assert len(history["val_loss"]) > 0
    for v in history["val_loss"]:
        assert math.isfinite(v)


def test_trainer_smoke_train_decreases_val_loss():
    """End-to-end: a real (tiny) pretraining run should make val loss
    go down. The headline test for Module 10 — if any piece of the
    pipeline (loss, schedule, clipping, training loop) is misordered,
    this regresses.
    """
    torch.manual_seed(0)
    m = _tiny_model()
    # Tiny corpus of just 4 tokens → trivially memorizable.
    repeating = torch.tensor([0, 1, 2, 3] * 100, dtype=torch.long)
    t = Trainer(
        m,
        batch_size=8,
        context_length=4,
        max_steps=50,
        max_lr=0.3,
        warmup_steps=5,
        eval_every=10,
        eval_iters=4,
        log_every=10,
    )
    history = t.train(repeating, val_ids=repeating)
    # First and last logged val losses on the same corpus.
    assert history["val_loss"][-1] < history["val_loss"][0] * 0.7
