"""Tests for Module 14: preference tuning (DPO).

Suggested order to implement & turn green:

  1. pad_and_collate_pref       → test_pad_and_collate_pref_*
  2. sequence_logprob           → test_sequence_logprob_*
  3. dpo_loss                   → test_dpo_loss_*
  4. DPOTrainer.train_step      → test_dpo_trainer_train_step_*,
                                   test_dpo_trainer_loss_decreases,
                                   test_dpo_trainer_ref_model_unchanged

Steps 1 and 3 are independent of each other — `dpo_loss` takes (B,)
log-prob tensors directly, so its tests don't depend on
`sequence_logprob`. Step 4 depends on all three. Boilerplate tests —
`PreferenceExample` shape, `DPOTrainer.__init__` validation,
`DPOTrainer.lr` — pass from the start as a sanity check on the test
file itself.

The end-to-end trainer tests pull in your full Module 03 / 05 / 07 /
08 / 09 stack via `TransformerLM`. If `test_trainer_train_runs_to_
completion` (Module 10) is passing, your prerequisites are in order.

Most tests use very small dimensions (`vocab_size=64`,
`embedding_dim=8`, `num_layers=1`, `T=10`, batch=2) so the suite runs
in well under a second.
"""
from __future__ import annotations

import copy
import math

import pytest
import torch

from g2c.dpo import (
    DPOTrainer,
    PreferenceExample,
    dpo_loss,
    pad_and_collate_pref,
    sequence_logprob,
)
from g2c.training import AdamW
from g2c.transformer import TransformerLM


# ----------------------------------------------------------------------
# PreferenceExample — boilerplate (NamedTuple)
# ----------------------------------------------------------------------


def test_preference_example_is_named_tuple():
    """PreferenceExample is a NamedTuple — three fields by name."""
    ex = PreferenceExample(
        prompt_ids=[1, 2, 3],
        chosen_ids=[4, 5],
        rejected_ids=[6, 7, 8],
    )
    assert ex.prompt_ids == [1, 2, 3]
    assert ex.chosen_ids == [4, 5]
    assert ex.rejected_ids == [6, 7, 8]


def test_preference_example_iterable_unpack():
    """NamedTuple unpacking works (used inside the trainer/collator)."""
    ex = PreferenceExample(prompt_ids=[1], chosen_ids=[2], rejected_ids=[3])
    p, c, r = ex
    assert p == [1]
    assert c == [2]
    assert r == [3]


# ----------------------------------------------------------------------
# pad_and_collate_pref — scaffolded
# ----------------------------------------------------------------------


def _ex(prompt: list[int], chosen: list[int], rejected: list[int]) -> PreferenceExample:
    return PreferenceExample(
        prompt_ids=prompt, chosen_ids=chosen, rejected_ids=rejected
    )


def test_pad_and_collate_pref_returns_six_tensors():
    examples = [_ex([1, 2], [3, 4], [5, 6, 7])]
    out = pad_and_collate_pref(examples, max_seq_len=6, pad_id=0)
    assert len(out) == 6
    for t in out:
        assert isinstance(t, torch.Tensor)


def test_pad_and_collate_pref_shapes():
    """All six tensors have shape (B, max_seq_len - 1)."""
    examples = [
        _ex([1, 2], [3, 4], [5, 6, 7]),
        _ex([10, 11], [12], [13, 14]),
    ]
    cx, cy, cm, rx, ry, rm = pad_and_collate_pref(
        examples, max_seq_len=6, pad_id=0
    )
    for t in (cx, cy, cm, rx, ry, rm):
        assert t.shape == (2, 5)


def test_pad_and_collate_pref_dtypes():
    examples = [_ex([1, 2], [3, 4], [5, 6])]
    cx, cy, cm, rx, ry, rm = pad_and_collate_pref(
        examples, max_seq_len=5, pad_id=0
    )
    assert cx.dtype == torch.long
    assert cy.dtype == torch.long
    assert rx.dtype == torch.long
    assert ry.dtype == torch.long
    assert cm.dtype in (torch.long, torch.float, torch.float32)
    assert rm.dtype in (torch.long, torch.float, torch.float32)


def test_pad_and_collate_pref_chosen_shifts_for_lm():
    """chosen_y[b, t] should equal chosen_full[b, t+1]: standard LM shift."""
    # prompt + chosen = [10, 11, 12, 20, 21]
    examples = [_ex([10, 11, 12], [20, 21], [30, 31])]
    cx, cy, cm, rx, ry, rm = pad_and_collate_pref(
        examples, max_seq_len=5, pad_id=0
    )
    assert torch.equal(cx[0], torch.tensor([10, 11, 12, 20]))
    assert torch.equal(cy[0], torch.tensor([11, 12, 20, 21]))


def test_pad_and_collate_pref_rejected_shifts_for_lm():
    """rejected_y[b, t] = rejected_full[b, t+1]: parallel to chosen."""
    # prompt + rejected = [10, 11, 12, 30, 31]
    examples = [_ex([10, 11, 12], [20, 21], [30, 31])]
    cx, cy, cm, rx, ry, rm = pad_and_collate_pref(
        examples, max_seq_len=5, pad_id=0
    )
    assert torch.equal(rx[0], torch.tensor([10, 11, 12, 30]))
    assert torch.equal(ry[0], torch.tensor([11, 12, 30, 31]))


def test_pad_and_collate_pref_chosen_mask_only_on_response():
    """chosen_mask[b, t] = 1 iff chosen_y[b, t] is part of the response.

    Concretely: the prompt is 3 tokens, the chosen response is 2 tokens,
    so chosen_full = [p0, p1, p2, c0, c1] (length 5). The mask aligned
    with full ids is [0, 0, 0, 1, 1]. After shift-by-one to align with
    y, it becomes [0, 0, 1, 1] — the 1s are exactly the positions where
    the model is being asked to score a response token (c0 or c1) given
    the prefix."""
    examples = [_ex([10, 11, 12], [20, 21], [30, 31])]
    cx, cy, cm, rx, ry, rm = pad_and_collate_pref(
        examples, max_seq_len=5, pad_id=0
    )
    assert cm[0].long().tolist() == [0, 0, 1, 1]


def test_pad_and_collate_pref_rejected_mask_only_on_response():
    """rejected_mask: parallel logic on rejected side."""
    examples = [_ex([10, 11, 12], [20, 21], [30, 31])]
    cx, cy, cm, rx, ry, rm = pad_and_collate_pref(
        examples, max_seq_len=5, pad_id=0
    )
    assert rm[0].long().tolist() == [0, 0, 1, 1]


def test_pad_and_collate_pref_pads_short_examples():
    """Short examples padded with pad_id; padded mask positions are 0."""
    # max_seq_len=6 but chosen example length is 4 (prompt=2 + chosen=2).
    examples = [_ex([10, 11], [20, 21], [30, 31, 32])]
    cx, cy, cm, rx, ry, rm = pad_and_collate_pref(
        examples, max_seq_len=6, pad_id=99
    )
    # chosen_full padded to [10, 11, 20, 21, 99, 99].
    # cx = full[:-1] = [10, 11, 20, 21, 99]
    # cy = full[1:]  = [11, 20, 21, 99, 99]
    # mask_full = [0, 0, 1, 1, 0, 0]; cm = mask_full[1:] = [0, 1, 1, 0, 0]
    assert torch.equal(cx[0], torch.tensor([10, 11, 20, 21, 99]))
    assert torch.equal(cy[0], torch.tensor([11, 20, 21, 99, 99]))
    assert cm[0].long().tolist() == [0, 1, 1, 0, 0]


def test_pad_and_collate_pref_pad_positions_have_loss_mask_zero():
    """Padded positions in BOTH chosen and rejected have mask = 0.

    Otherwise the implicit-reward signal depends on the pad-id's
    log-probability, which is meaningless."""
    examples = [_ex([10, 11], [20], [30])]
    cx, cy, cm, rx, ry, rm = pad_and_collate_pref(
        examples, max_seq_len=6, pad_id=0
    )
    # chosen_full = [10, 11, 20], padded to [10, 11, 20, 0, 0, 0]
    # mask_full   = [ 0,  0,  1, 0, 0, 0]
    # cm = mask_full[1:] = [0, 1, 0, 0, 0]
    # The last 3 positions are pad-targets and must be 0.
    assert cm[0].long().tolist() == [0, 1, 0, 0, 0]
    # Same for rejected.
    assert rm[0].long().tolist() == [0, 1, 0, 0, 0]


def test_pad_and_collate_pref_truncates_long_examples():
    """Examples whose prompt+response is longer than max_seq_len are
    head-truncated (keep the start, drop the tail)."""
    examples = [_ex(list(range(10)), [100, 101, 102], [200, 201, 202])]
    cx, cy, cm, rx, ry, rm = pad_and_collate_pref(
        examples, max_seq_len=6, pad_id=0
    )
    # chosen_full = [0..9, 100, 101, 102] truncated to [0,1,2,3,4,5].
    # cx = [0,1,2,3,4]; cy = [1,2,3,4,5]; cm should be all 0s
    # because the response tokens (100, 101, 102) were truncated away.
    assert torch.equal(cx[0], torch.tensor([0, 1, 2, 3, 4]))
    assert torch.equal(cy[0], torch.tensor([1, 2, 3, 4, 5]))
    assert cm[0].long().tolist() == [0, 0, 0, 0, 0]


def test_pad_and_collate_pref_chosen_and_rejected_independent():
    """Chosen and rejected are truncated and padded independently.

    A long chosen response should not affect rejected's shape, and
    vice versa. This is the contract: each completion is a separate
    `(x, y, mask)` triple sharing only the prompt prefix."""
    examples = [_ex([10, 11], [20], [30, 31, 32])]
    cx, cy, cm, rx, ry, rm = pad_and_collate_pref(
        examples, max_seq_len=6, pad_id=0
    )
    # rejected_full = [10, 11, 30, 31, 32], padded to [10, 11, 30, 31, 32, 0]
    # rx = [10, 11, 30, 31, 32]; ry = [11, 30, 31, 32, 0];
    # rmask_full = [0, 0, 1, 1, 1, 0]; rm = rmask_full[1:] = [0, 1, 1, 1, 0]
    assert torch.equal(rx[0], torch.tensor([10, 11, 30, 31, 32]))
    assert torch.equal(ry[0], torch.tensor([11, 30, 31, 32, 0]))
    assert rm[0].long().tolist() == [0, 1, 1, 1, 0]


def test_pad_and_collate_pref_batch_pads_to_common_length():
    """Mixed-length examples in one batch all get padded to max_seq_len."""
    examples = [
        _ex([1, 2], [3], [4]),
        _ex([10, 11, 12], [20, 21], [30, 31]),
    ]
    cx, cy, cm, rx, ry, rm = pad_and_collate_pref(
        examples, max_seq_len=5, pad_id=0
    )
    assert cx.shape == (2, 4)
    assert rx.shape == (2, 4)


# ----------------------------------------------------------------------
# sequence_logprob — scaffolded
# ----------------------------------------------------------------------


def test_sequence_logprob_returns_per_example_shape():
    """Returns shape (B,) — one scalar per example, NOT a single
    batch-mean scalar."""
    logits = torch.randn(3, 4, 7)
    targets = torch.randint(0, 7, (3, 4))
    mask = torch.ones(3, 4, dtype=torch.long)
    out = sequence_logprob(logits, targets, mask)
    assert out.shape == (3,)


def test_sequence_logprob_uniform_logits_full_mask():
    """With uniform logits (all zeros) and full mask, sequence_logprob
    returns -T * log(V) per example.

    Mathematically: log_softmax of all-zero logits is -log(V) at every
    vocab entry. Gathering any target gives -log(V); summing over T
    masked positions gives -T * log(V)."""
    B, T, V = 2, 5, 8
    logits = torch.zeros(B, T, V)
    targets = torch.randint(0, V, (B, T))
    mask = torch.ones(B, T, dtype=torch.long)
    out = sequence_logprob(logits, targets, mask)
    expected = -T * math.log(V)
    assert torch.allclose(out, torch.full((B,), expected), atol=1e-5)


def test_sequence_logprob_zero_mask_returns_zero():
    """With a zero mask, every example's log-prob sum is 0."""
    logits = torch.randn(2, 4, 7)
    targets = torch.randint(0, 7, (2, 4))
    mask = torch.zeros(2, 4, dtype=torch.long)
    out = sequence_logprob(logits, targets, mask)
    assert torch.allclose(out, torch.zeros(2))


def test_sequence_logprob_ignores_masked_positions():
    """Changing logits at mask=0 positions must not change the output.

    Strictest "is the mask actually masking" test — parallel to
    Module 13's masked_cross_entropy ignore test."""
    torch.manual_seed(7)
    logits = torch.randn(1, 4, 5)
    targets = torch.randint(0, 5, (1, 4))
    mask = torch.tensor([[0, 0, 1, 1]])
    before = sequence_logprob(logits, targets, mask).clone()
    logits[0, 0] = 1000.0
    logits[0, 1] = -1000.0
    after = sequence_logprob(logits, targets, mask)
    assert torch.allclose(before, after, atol=1e-5)


def test_sequence_logprob_accepts_float_mask():
    """The mask may be int or float; the result is identical."""
    torch.manual_seed(11)
    logits = torch.randn(1, 4, 5)
    targets = torch.randint(0, 5, (1, 4))
    mask_int = torch.tensor([[0, 1, 1, 1]], dtype=torch.long)
    mask_float = mask_int.to(torch.float32)
    a = sequence_logprob(logits, targets, mask_int)
    b = sequence_logprob(logits, targets, mask_float)
    assert torch.allclose(a, b, atol=1e-5)


def test_sequence_logprob_against_manual():
    """Pin to a hand-computed value. With B=1, T=2, V=3:
        logits  = [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]
        targets = [2, 0]
        mask    = [1, 1]
    log_softmax row 0: each entry minus log(e+e^2+e^3) = log(e^1+e^2+e^3) ≈ 3.4076
        log p(target=2) = 3 - 3.4076 = -0.4076
    log_softmax row 1: by symmetry log_sum_exp is also ≈ 3.4076
        log p(target=0) = 3 - 3.4076 = -0.4076
    sum = -0.8152.
    """
    logits = torch.tensor([[[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]])
    targets = torch.tensor([[2, 0]])
    mask = torch.ones(1, 2, dtype=torch.long)
    out = sequence_logprob(logits, targets, mask)
    expected = -2 * (math.log(math.exp(1) + math.exp(2) + math.exp(3)) - 3)
    assert torch.allclose(out, torch.tensor([expected]), atol=1e-5)


# ----------------------------------------------------------------------
# dpo_loss — scaffolded
# ----------------------------------------------------------------------


def test_dpo_loss_returns_scalar_and_metrics():
    """Returns (loss_tensor, metrics_dict). loss is scalar; metrics
    has the documented keys."""
    pc = torch.tensor([-3.0, -4.0])
    pr = torch.tensor([-5.0, -6.0])
    rc = torch.tensor([-3.0, -4.0])
    rr = torch.tensor([-5.0, -6.0])
    loss, metrics = dpo_loss(pc, pr, rc, rr, beta=0.1)
    assert loss.shape == ()
    for key in ("chosen_reward", "rejected_reward", "reward_margin", "accuracy"):
        assert key in metrics


def test_dpo_loss_initial_state_is_log2():
    """When policy log-probs equal reference log-probs (which is true
    on step 0 if the policy was just copied from the reference), the
    DPO logits are zero and the loss is exactly log(2) ≈ 0.6931.

    This is the canonical "DPO loss at step 0" sanity value."""
    pc = torch.tensor([-3.0, -4.0, -5.0])
    pr = torch.tensor([-7.0, -8.0, -9.0])
    rc = pc.clone()
    rr = pr.clone()
    loss, _ = dpo_loss(pc, pr, rc, rr, beta=0.1)
    assert math.isclose(loss.item(), math.log(2), abs_tol=1e-5)


def test_dpo_loss_initial_state_log2_at_various_betas():
    """The 'initial-state = log 2' invariant holds at any beta (because
    the DPO logits are still 0 when policy == ref)."""
    pc = torch.tensor([-3.0])
    pr = torch.tensor([-7.0])
    rc = pc.clone()
    rr = pr.clone()
    for beta in (0.01, 0.1, 0.5, 1.0):
        loss, _ = dpo_loss(pc, pr, rc, rr, beta=beta)
        assert math.isclose(loss.item(), math.log(2), abs_tol=1e-5), (
            f"beta={beta}: expected log(2), got {loss.item()}"
        )


def test_dpo_loss_strict_preference_below_log2():
    """When the policy prefers chosen MORE than the reference does
    (i.e., (pc - pr) > (rc - rr)), the DPO logits are positive and the
    loss is below log(2)."""
    pc = torch.tensor([-2.0])  # higher than ref
    pr = torch.tensor([-8.0])  # lower than ref
    rc = torch.tensor([-3.0])
    rr = torch.tensor([-7.0])
    loss, _ = dpo_loss(pc, pr, rc, rr, beta=0.5)
    assert loss.item() < math.log(2)


def test_dpo_loss_anti_preference_above_log2():
    """When the policy prefers rejected MORE than the reference does
    (i.e., (pc - pr) < (rc - rr)), the DPO logits are negative and the
    loss is above log(2). This is what 'gradient pushes you in the
    wrong direction' looks like."""
    pc = torch.tensor([-8.0])  # lower than ref
    pr = torch.tensor([-2.0])  # higher than ref
    rc = torch.tensor([-3.0])
    rr = torch.tensor([-7.0])
    loss, _ = dpo_loss(pc, pr, rc, rr, beta=0.5)
    assert loss.item() > math.log(2)


def test_dpo_loss_accuracy_metric():
    """accuracy is the fraction of examples where chosen reward >
    rejected reward. Construct a batch where 3/4 examples have the
    policy preferring chosen; expect accuracy = 0.75."""
    pc = torch.tensor([-2.0, -2.0, -2.0, -8.0])
    pr = torch.tensor([-8.0, -8.0, -8.0, -2.0])
    rc = torch.tensor([-3.0, -3.0, -3.0, -3.0])
    rr = torch.tensor([-7.0, -7.0, -7.0, -7.0])
    _, metrics = dpo_loss(pc, pr, rc, rr, beta=0.1)
    assert math.isclose(metrics["accuracy"].item(), 0.75, abs_tol=1e-5)


def test_dpo_loss_implicit_rewards():
    """chosen_reward = beta * (policy_chosen_logp - ref_chosen_logp)
    averaged over the batch."""
    pc = torch.tensor([-2.0, -3.0])
    pr = torch.tensor([-4.0, -5.0])
    rc = torch.tensor([-3.0, -4.0])
    rr = torch.tensor([-5.0, -6.0])
    beta = 0.5
    _, metrics = dpo_loss(pc, pr, rc, rr, beta=beta)
    expected_chosen = beta * ((-2.0 - -3.0) + (-3.0 - -4.0)) / 2
    expected_rejected = beta * ((-4.0 - -5.0) + (-5.0 - -6.0)) / 2
    assert math.isclose(
        metrics["chosen_reward"].item(), expected_chosen, abs_tol=1e-5
    )
    assert math.isclose(
        metrics["rejected_reward"].item(), expected_rejected, abs_tol=1e-5
    )
    assert math.isclose(
        metrics["reward_margin"].item(),
        expected_chosen - expected_rejected,
        abs_tol=1e-5,
    )


def test_dpo_loss_beta_scales_logits():
    """At fixed log-prob differences, doubling beta doubles the DPO
    logit. Concretely: with (pc - pr) - (rc - rr) = 4, the DPO logit
    at beta=0.25 is 1.0 and at beta=0.5 is 2.0; the losses are
    -log σ(1) ≈ 0.3133 and -log σ(2) ≈ 0.1269 respectively."""
    pc = torch.tensor([-1.0])
    pr = torch.tensor([-5.0])  # pc - pr = 4
    rc = torch.tensor([-3.0])
    rr = torch.tensor([-3.0])  # rc - rr = 0
    loss_25, _ = dpo_loss(pc, pr, rc, rr, beta=0.25)
    loss_50, _ = dpo_loss(pc, pr, rc, rr, beta=0.50)
    expected_25 = -math.log(1 / (1 + math.exp(-1.0)))
    expected_50 = -math.log(1 / (1 + math.exp(-2.0)))
    assert math.isclose(loss_25.item(), expected_25, abs_tol=1e-5)
    assert math.isclose(loss_50.item(), expected_50, abs_tol=1e-5)


def test_dpo_loss_gradient_flows_to_policy_only():
    """The loss has gradient w.r.t. policy log-probs but NOT w.r.t.
    reference log-probs (the reference is meant to be detached
    upstream, but the loss formula is symmetric — the test pins the
    contract via the gradient check rather than the implementation).

    We construct policy log-probs WITH grad and reference log-probs
    WITHOUT grad and check that backward() doesn't error and produces
    a finite gradient on the policy tensors."""
    pc = torch.tensor([-2.0], requires_grad=True)
    pr = torch.tensor([-5.0], requires_grad=True)
    rc = torch.tensor([-3.0])  # no grad
    rr = torch.tensor([-7.0])  # no grad
    loss, _ = dpo_loss(pc, pr, rc, rr, beta=0.5)
    loss.backward()
    assert pc.grad is not None
    assert pr.grad is not None
    assert torch.isfinite(pc.grad).all()
    assert torch.isfinite(pr.grad).all()
    # The chosen-side gradient should be NEGATIVE (we want to increase
    # pc, so dL/dpc < 0) and the rejected-side POSITIVE (we want to
    # decrease pr, so dL/dpr > 0).
    assert pc.grad.item() < 0
    assert pr.grad.item() > 0


# ----------------------------------------------------------------------
# DPOTrainer — construction and lr (boilerplate, implemented)
# ----------------------------------------------------------------------


def _make_tiny_model():
    """A tiny TransformerLM for the DPOTrainer end-to-end tests.

    1 layer, embedding_dim=8, vocab=64, max_seq_len=12. Small enough
    to forward+backward in milliseconds.
    """
    return TransformerLM(
        vocab_size=64,
        embedding_dim=8,
        num_layers=1,
        num_heads=2,
        max_seq_len=12,
    )


def _clone_model(m: TransformerLM) -> TransformerLM:
    """Make a deep copy of the model; used for building a frozen ref."""
    return copy.deepcopy(m)


def _make_tiny_dataset(n: int = 6) -> list[PreferenceExample]:
    """A handful of synthetic preference examples with explicit ids.

    Each example: prompt of 4 tokens, chosen of 3 tokens, rejected of
    3 tokens. The prompts and responses are independent random ids;
    there's no semantic structure — the test just checks that the
    DPO loss decreases as the policy learns to score the chosen
    responses higher than the rejected ones."""
    examples: list[PreferenceExample] = []
    rng = torch.Generator().manual_seed(42)
    for _ in range(n):
        prompt = torch.randint(1, 64, (4,), generator=rng).tolist()
        chosen = torch.randint(1, 64, (3,), generator=rng).tolist()
        rejected = torch.randint(1, 64, (3,), generator=rng).tolist()
        examples.append(
            PreferenceExample(
                prompt_ids=prompt, chosen_ids=chosen, rejected_ids=rejected
            )
        )
    return examples


def test_dpo_trainer_construction_defaults():
    model = _make_tiny_model()
    ref = _clone_model(model)
    examples = _make_tiny_dataset()
    trainer = DPOTrainer(
        model,
        ref_model=ref,
        examples=examples,
        max_seq_len=10,
        pad_id=0,
        beta=0.1,
        batch_size=2,
        max_steps=10,
        max_lr=1e-4,
    )
    assert trainer.beta == 0.1
    assert trainer.max_steps == 10
    assert trainer.batch_size == 2
    assert trainer.step == 0
    assert isinstance(trainer.optimizer, AdamW)
    expected_device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    assert trainer.device == expected_device


def test_dpo_trainer_explicit_cpu_device_moves_model_params():
    model = _make_tiny_model()
    ref = _clone_model(model)
    trainer = DPOTrainer(
        model,
        ref_model=ref,
        examples=_make_tiny_dataset(),
        max_seq_len=10,
        pad_id=0,
        beta=0.1,
        batch_size=2,
        max_steps=10,
        max_lr=1e-4,
        device="cpu",
    )
    assert trainer.device == torch.device("cpu")
    for p in model.parameters():
        assert p.device.type == "cpu"
    for p in ref.parameters():
        assert p.device.type == "cpu"


def test_dpo_trainer_optimizer_only_on_policy():
    """The optimizer's params come from `model.parameters()`, NOT
    from `ref_model.parameters()`. Otherwise AdamW would try
    to step the reference too."""
    model = _make_tiny_model()
    ref = _clone_model(model)
    trainer = DPOTrainer(
        model,
        ref_model=ref,
        examples=_make_tiny_dataset(),
        max_seq_len=10,
        pad_id=0,
        beta=0.1,
        batch_size=2,
        max_steps=10,
        max_lr=1e-4,
    )
    policy_param_ids = {id(p) for p in model.parameters()}
    ref_param_ids = {id(p) for p in ref.parameters()}
    optimizer_param_ids = {id(p) for p in trainer.optimizer.params}
    assert optimizer_param_ids == policy_param_ids
    assert optimizer_param_ids.isdisjoint(ref_param_ids)


def test_dpo_trainer_empty_examples_raises():
    model = _make_tiny_model()
    with pytest.raises(ValueError):
        DPOTrainer(
            model,
            ref_model=_clone_model(model),
            examples=[],
            max_seq_len=10,
            pad_id=0,
            beta=0.1,
            batch_size=2,
            max_steps=10,
            max_lr=1e-4,
        )


def test_dpo_trainer_bad_batch_size_raises():
    model = _make_tiny_model()
    with pytest.raises(ValueError):
        DPOTrainer(
            model,
            ref_model=_clone_model(model),
            examples=_make_tiny_dataset(),
            max_seq_len=10,
            pad_id=0,
            beta=0.1,
            batch_size=0,
            max_steps=10,
            max_lr=1e-4,
        )


def test_dpo_trainer_max_seq_len_too_small_raises():
    model = _make_tiny_model()
    with pytest.raises(ValueError):
        DPOTrainer(
            model,
            ref_model=_clone_model(model),
            examples=_make_tiny_dataset(),
            max_seq_len=1,
            pad_id=0,
            beta=0.1,
            batch_size=2,
            max_steps=10,
            max_lr=1e-4,
        )


def test_dpo_trainer_bad_beta_raises():
    """beta must be > 0 — beta=0 makes the loss constant log(2) and
    the gradient identically zero, which is a programming error not
    a trainable hyperparameter."""
    model = _make_tiny_model()
    with pytest.raises(ValueError):
        DPOTrainer(
            model,
            ref_model=_clone_model(model),
            examples=_make_tiny_dataset(),
            max_seq_len=10,
            pad_id=0,
            beta=0.0,
            batch_size=2,
            max_steps=10,
            max_lr=1e-4,
        )


def test_dpo_trainer_lr_uses_cosine_schedule():
    """DPOTrainer.lr should reproduce cosine_with_warmup at the
    current step. (Boilerplate test — passes without train_step.)"""
    from g2c.training import cosine_with_warmup

    model = _make_tiny_model()
    trainer = DPOTrainer(
        model,
        ref_model=_clone_model(model),
        examples=_make_tiny_dataset(),
        max_seq_len=10,
        pad_id=0,
        beta=0.1,
        batch_size=2,
        max_steps=100,
        max_lr=1e-4,
        warmup_steps=10,
    )
    assert math.isclose(
        trainer.lr(0),
        cosine_with_warmup(0, warmup_steps=10, max_steps=100, max_lr=1e-4),
    )
    assert math.isclose(
        trainer.lr(50),
        cosine_with_warmup(50, warmup_steps=10, max_steps=100, max_lr=1e-4),
    )


# ----------------------------------------------------------------------
# DPOTrainer.train_step — scaffolded
# ----------------------------------------------------------------------


def test_dpo_trainer_train_step_returns_metrics():
    """One step returns the documented metric dict; advances self.step."""
    torch.manual_seed(0)
    model = _make_tiny_model()
    ref = _clone_model(model)
    trainer = DPOTrainer(
        model,
        ref_model=ref,
        examples=_make_tiny_dataset(),
        max_seq_len=10,
        pad_id=0,
        beta=0.1,
        batch_size=2,
        max_steps=10,
        max_lr=1e-3,
        grad_clip=1.0,
    )
    metrics = trainer.train_step()
    for key in (
        "loss",
        "lr",
        "grad_norm",
        "chosen_reward",
        "rejected_reward",
        "reward_margin",
        "accuracy",
    ):
        assert key in metrics
        assert isinstance(metrics[key], float)
        assert math.isfinite(metrics[key])
    assert trainer.step == 1


def test_dpo_trainer_initial_loss_is_log2():
    """At step 0, with the reference model identical to the policy,
    the DPO loss is exactly log(2) — independent of the data, the
    architecture, or beta. This is the deepest sanity check on the
    whole pipeline (data → forward → log-prob → loss)."""
    torch.manual_seed(0)
    model = _make_tiny_model()
    ref = _clone_model(model)
    trainer = DPOTrainer(
        model,
        ref_model=ref,
        examples=_make_tiny_dataset(),
        max_seq_len=10,
        pad_id=0,
        beta=0.1,
        batch_size=4,
        max_steps=10,
        max_lr=0.0,  # don't update — we want to inspect step 0 only
    )
    metrics = trainer.train_step()
    # Equality holds before any update is applied; with max_lr=0 the
    # update is a no-op.
    assert math.isclose(metrics["loss"], math.log(2), abs_tol=1e-4)


def test_dpo_trainer_loss_decreases():
    """Headline end-to-end check. 30 DPO steps on a tiny synthetic
    dataset should drive the loss down meaningfully — final loss
    at least 5% below initial.

    If this fails: data, sequence_logprob, dpo_loss, optimizer, or
    trainer wiring is broken. The smaller tests in this file will
    tell you which."""
    torch.manual_seed(0)
    model = _make_tiny_model()
    ref = _clone_model(model)
    examples = _make_tiny_dataset(n=4)
    trainer = DPOTrainer(
        model,
        ref_model=ref,
        examples=examples,
        max_seq_len=10,
        pad_id=0,
        beta=0.1,
        batch_size=2,
        max_steps=30,
        max_lr=1e-2,  # higher than recommended; we want fast signal
        warmup_steps=2,
        grad_clip=1.0,
    )
    history = trainer.train()
    initial_loss = history["train_loss"][0]
    final_loss = history["train_loss"][-1]
    assert final_loss < initial_loss * 0.95, (
        f"DPO loss did not decrease: initial={initial_loss}, "
        f"final={final_loss}"
    )


def test_dpo_trainer_ref_model_unchanged():
    """The reference model's parameters must NOT change during
    training. This is the load-bearing invariant of DPO — without it
    the implicit reward `r̂(x,y) = β · log[pi(y|x)/pi_ref(y|x)]` drifts
    in lockstep with the policy and the preference signal vanishes."""
    torch.manual_seed(0)
    model = _make_tiny_model()
    ref = _clone_model(model)
    # Snapshot the reference parameters BEFORE training.
    ref_before = [p.detach().clone() for p in ref.parameters()]
    trainer = DPOTrainer(
        model,
        ref_model=ref,
        examples=_make_tiny_dataset(),
        max_seq_len=10,
        pad_id=0,
        beta=0.1,
        batch_size=2,
        max_steps=20,
        max_lr=1e-2,
        grad_clip=1.0,
    )
    trainer.train()
    # Snapshot AFTER. Every parameter must be identical.
    ref_after = list(ref.parameters())
    for before, after in zip(ref_before, ref_after, strict=True):
        assert torch.equal(before, after.detach().cpu()), (
            "Reference model parameter changed during DPO training. "
            "The optimizer must be attached only to the policy model."
        )


def test_dpo_trainer_evaluate_returns_metrics_dict():
    """evaluate() returns a dict with the documented keys, all
    Python floats."""
    torch.manual_seed(0)
    model = _make_tiny_model()
    trainer = DPOTrainer(
        model,
        ref_model=_clone_model(model),
        examples=_make_tiny_dataset(),
        max_seq_len=10,
        pad_id=0,
        beta=0.1,
        batch_size=2,
        max_steps=10,
        max_lr=1e-4,
        eval_iters=3,
    )
    val = trainer.evaluate(_make_tiny_dataset())
    for key in (
        "loss",
        "chosen_reward",
        "rejected_reward",
        "reward_margin",
        "accuracy",
    ):
        assert key in val
        assert isinstance(val[key], float)
        assert math.isfinite(val[key])


def test_dpo_trainer_evaluate_empty_raises():
    model = _make_tiny_model()
    trainer = DPOTrainer(
        model,
        ref_model=_clone_model(model),
        examples=_make_tiny_dataset(),
        max_seq_len=10,
        pad_id=0,
        beta=0.1,
        batch_size=2,
        max_steps=10,
        max_lr=1e-4,
    )
    with pytest.raises(ValueError):
        trainer.evaluate([])
