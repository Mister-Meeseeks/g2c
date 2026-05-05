"""Tests for Module 10: first LLM pretraining run.

Suggested order to implement & turn green:

  1. Confirm Module 09B setup    → pytest tests/test_pretraining_setup.py
  2. Trainer.train_step          → test_trainer_train_step_*,
                                   test_trainer_evaluate_*,
                                   test_trainer_train_*

Module 09B owns `split_token_stream`, `get_lm_batch`, and
`lm_cross_entropy`. This file starts after that setup work and focuses
on the top-level trainer loop.

Trainer construction tests, attribute defaults, device-handling tests,
and Trainer.lr tests all pass without needing the model to forward
correctly once Module 03B's schedule is implemented. The
Trainer.train_step / evaluate / train tests are the only ones that
exercise a full TransformerLM forward+backward pass -- those depend on
Modules 03 / 03B / 05 / 08 / 09 being implemented in addition to this
Module 10 trainer scaffold. If your full TransformerLM smoke test from
Module 09 isn't passing yet, finish that first.

Most tests use very small dimensions (`vocab_size=12`,
`embedding_dim=8`, `num_layers=1`, `T=6`, batch=4) so the suite runs
in well under a second.
"""
from __future__ import annotations

import math

import pytest
import torch

from g2c.nn import SGD
from g2c.pretraining import Trainer
from g2c.training import AdamW, cosine_with_warmup
from g2c.transformer import TransformerLM


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


def test_trainer_train_on_log_receives_progress_events():
    torch.manual_seed(0)
    t = Trainer(
        _tiny_model(),
        batch_size=4,
        context_length=6,
        max_steps=6,
        max_lr=1e-3,
        eval_every=3,
        eval_iters=1,
        log_every=2,
    )
    ids = torch.randint(0, 12, (200,))
    events: list[dict[str, float | int | None]] = []

    t.train(ids, val_ids=ids, on_log=events.append)

    assert events
    assert events[0]["step"] == 0
    assert events[-1]["step"] == 5
    assert any(event["val_loss"] is not None for event in events)
    for event in events:
        assert {"step", "train_loss", "val_loss", "lr", "grad_norm"} <= event.keys()
        assert event["elapsed_s"] is not None
        assert event["steps_per_s"] is not None


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
