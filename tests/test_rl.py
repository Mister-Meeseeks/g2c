"""Tests for g2c/rl — Beyond module: reinforcement learning for LLMs.

Suggested order to implement & turn green:

1. `group_advantages` — unblocks `test_advantages_*`.
2. `completion_log_prob` — unblocks `test_log_prob_*`. These run
   against a deterministic stub model, so the math is checkable by
   hand — no BaseLM required.
3. `grpo_loss` — unblocks `test_grpo_loss_*`, including the
   gradient-direction test (positive advantage → gradient descent
   RAISES that completion's log-prob).
4. `GRPOTrainer.train_step` — unblocks the trainer smoke test, which
   drives a real (tiny) TransformerLM through one full update and so
   also needs Modules 09/11 implemented.

Verifier and sampler-validation tests are provided plumbing and pass
from the start.
"""
from __future__ import annotations

import math
import random

import pytest
import torch

from g2c.nn import Module
from g2c.rl import (
    GRPOTrainer,
    arithmetic_choice_task,
    completion_log_prob,
    format_task,
    group_advantages,
    grpo_loss,
    sample_group,
    verify_arithmetic,
    verify_arithmetic_sloppy,
    verify_format,
)


class _StubLM(Module):
    """Deterministic logits: `forward(ids)[b, t] = table[ids[b, t]]`.

    The next-token distribution depends only on the current token, so
    every log-prob is hand-computable from `table` alone.
    """

    def __init__(self, vocab_size: int = 8, max_seq_len: int = 32):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        g = torch.Generator().manual_seed(1234)
        table = torch.randn(vocab_size, vocab_size, generator=g)
        table.requires_grad_(True)
        self.table = table

    def parameters(self):
        return [self.table]

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.table[ids]


# ---------------------------------------------------------------------------
# Provided plumbing — green from the start.
# ---------------------------------------------------------------------------


def test_verify_arithmetic_scores_the_last_integer():
    task = {"prompt": "23+58=", "answer": 81}
    assert verify_arithmetic(task, "23+58 is 70 plus 11, so 81") == 1.0
    assert verify_arithmetic(task, "81 no wait, 91") == 0.0
    assert verify_arithmetic(task, "no numbers here") == 0.0


def test_verify_arithmetic_sloppy_is_exploitable():
    """The deliberately broken reward: digit salad gets full marks."""
    task = {"prompt": "23+58=", "answer": 81}
    salad = "10 20 30 40 50 60 70 80 81 90"
    assert verify_arithmetic_sloppy(task, salad) == 1.0
    assert verify_arithmetic(task, salad) == 0.0


def test_verify_format_checks_shape_not_value():
    task = {"prompt": "…", "answer": 81}
    assert verify_format(task, 'Sure: {"answer": 999}') == 1.0
    assert verify_format(task, "the answer is 81") == 0.0
    assert verify_format(task, "{broken json") == 0.0


def test_format_task_primes_a_discoverable_json_completion():
    task = format_task(random.Random(0))
    assert task["prompt"].endswith('{"answer":')
    assert verify_format(task, '123}') == 1.0
    assert verify_format(task, '123, "extra": true}') == 1.0
    assert verify_format(task, '123} trailing text') == 1.0
    assert verify_format(task, 'not JSON') == 0.0


def test_arithmetic_choice_task_has_one_correct_option():
    task = arithmetic_choice_task(random.Random(0))
    assert len(task["choices"]) == 2
    assert task["answer"] in task["choices"]
    assert task["choices"][0] != task["choices"][1]
    assert all(str(choice) in task["prompt"] for choice in task["choices"])


def test_sample_group_validates_inputs():
    with pytest.raises(ValueError):
        sample_group(None, None, "p", k=1)
    with pytest.raises(ValueError):
        sample_group(None, None, "p", k=4, temperature=0.0)


# ---------------------------------------------------------------------------
# group_advantages
# ---------------------------------------------------------------------------


def test_advantages_known_values():
    rewards = torch.tensor([1.0, 0.0, 0.0, 1.0])
    adv = group_advantages(rewards)
    assert torch.allclose(adv, torch.tensor([1.0, -1.0, -1.0, 1.0]), atol=1e-5)


def test_advantages_zero_mean_unit_std():
    torch.manual_seed(0)
    adv = group_advantages(torch.rand(16))
    assert abs(float(adv.mean())) < 1e-5
    assert abs(float(adv.std(unbiased=False)) - 1.0) < 1e-4


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_advantages_degenerate_group_is_all_zeros(value):
    adv = group_advantages(torch.full((8,), value))
    assert (adv == 0).all()


# ---------------------------------------------------------------------------
# completion_log_prob
# ---------------------------------------------------------------------------


def test_log_prob_matches_hand_computation():
    model = _StubLM()
    ids = torch.tensor([[3, 1, 4, 1, 5]])
    prompt_len = 2
    got = completion_log_prob(model, ids, prompt_len)

    lp = torch.log_softmax(model.table, dim=-1)
    # Completion tokens are ids[2:], predicted from positions 1..3.
    expected = lp[1, 4] + lp[4, 1] + lp[1, 5]
    assert torch.allclose(got, expected.reshape(1), atol=1e-5)


def test_log_prob_excludes_the_prompt():
    """Two rows with different PROMPTS but the same continuation-relevant
    suffix must score identically."""
    model = _StubLM()
    a = torch.tensor([[0, 2, 6, 7]])
    b = torch.tensor([[5, 2, 6, 7]])
    # prompt_len=2: completion log-prob reads logits at positions 1..2,
    # which depend only on tokens 2 and 6 — identical across rows.
    assert torch.allclose(
        completion_log_prob(model, a, 2), completion_log_prob(model, b, 2)
    )


@pytest.mark.parametrize("bad_len", [0, 5, 9])
def test_log_prob_rejects_bad_prompt_len(bad_len):
    with pytest.raises(ValueError):
        completion_log_prob(_StubLM(), torch.zeros(1, 5, dtype=torch.long), bad_len)


# ---------------------------------------------------------------------------
# grpo_loss
# ---------------------------------------------------------------------------


def test_grpo_loss_without_leash_is_pure_policy_gradient():
    logp = torch.tensor([-1.0, -2.0, -3.0], requires_grad=True)
    adv = torch.tensor([1.0, 0.0, -1.0])
    loss = grpo_loss(logp, logp.detach(), adv, kl_coef=0.0)
    assert torch.allclose(loss, -(adv * logp.detach()).mean(), atol=1e-6)


def test_grpo_loss_kl_is_zero_at_the_reference():
    logp = torch.tensor([-1.0, -2.0], requires_grad=True)
    adv = torch.tensor([1.0, -1.0])
    at_ref = grpo_loss(logp, logp.detach(), adv, kl_coef=10.0)
    no_kl = grpo_loss(logp, logp.detach(), adv, kl_coef=0.0)
    assert torch.allclose(at_ref, no_kl, atol=1e-6)


def test_grpo_loss_kl_penalizes_drift():
    logp = torch.tensor([-1.0, -2.0], requires_grad=True)
    ref = torch.tensor([-1.5, -1.0])
    adv = torch.tensor([1.0, -1.0])
    drifted = grpo_loss(logp, ref, adv, kl_coef=1.0)
    anchored = grpo_loss(logp, ref, adv, kl_coef=0.0)
    assert float(drifted.detach()) > float(anchored.detach())


def test_grpo_loss_gradient_direction():
    """Gradient DESCENT must raise the log-prob of above-average
    completions and lower below-average ones."""
    logp = torch.tensor([-1.0, -1.0], requires_grad=True)
    adv = torch.tensor([1.0, -1.0])
    loss = grpo_loss(logp, logp.detach(), adv, kl_coef=0.0)
    loss.backward()
    assert logp.grad[0] < 0  # descent step increases logp[0]
    assert logp.grad[1] > 0  # descent step decreases logp[1]


# ---------------------------------------------------------------------------
# GRPOTrainer — one full update on a real tiny model.
# ---------------------------------------------------------------------------


class _CharTokenizer:
    CHARS = "0123456789+= "

    def encode(self, text: str) -> list[int]:
        return [self.CHARS.index(c) for c in text if c in self.CHARS]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.CHARS[i] for i in ids if 0 <= i < len(self.CHARS))


def _tiny_policy():
    from g2c.transformer import TransformerLM

    torch.manual_seed(9)
    return TransformerLM(
        vocab_size=13,
        embedding_dim=16,
        num_layers=1,
        num_heads=2,
        max_seq_len=48,
    )


def test_trainer_smoke():
    trainer = GRPOTrainer(
        _tiny_policy(),
        _tiny_policy(),
        _CharTokenizer(),
        tasks=[{"prompt": "12+34=", "answer": 46}],
        verifier=lambda task, text: 1.0 if "1" in text else 0.0,
        group_size=4,
        max_new_tokens=6,
        seed=0,
    )
    metrics = trainer.train_step()
    assert set(metrics) == {
        "loss", "mean_reward", "kl", "sample_entropy", "skipped"
    }
    assert metrics["skipped"] in (0.0, 1.0)
    if metrics["skipped"] == 1.0:
        assert math.isnan(metrics["kl"])
        assert math.isnan(metrics["sample_entropy"])
    else:
        assert torch.isfinite(torch.tensor(metrics["loss"]))
        assert metrics["kl"] >= 0.0
        assert metrics["sample_entropy"] >= 0.0
        assert trainer.step_count == 1


def test_trainer_requires_tasks():
    with pytest.raises(ValueError):
        GRPOTrainer(
            None, None, _CharTokenizer(), tasks=[], verifier=verify_arithmetic
        )


def test_trainer_requires_unit_sampling_temperature():
    with pytest.raises(ValueError, match="temperature=1.0"):
        GRPOTrainer(
            _tiny_policy(),
            _tiny_policy(),
            _CharTokenizer(),
            tasks=[{"prompt": "12+34=", "answer": 46}],
            verifier=verify_arithmetic,
            temperature=0.8,
        )
