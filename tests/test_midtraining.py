"""Tests for g2c/midtraining — Beyond module: continued pretraining.

Suggested order to implement & turn green:

1. `TokenMixture.get_lm_batch` — unblocks `test_mixture_*`. Start with source
   assignment, then grouped sampling, row shuffling, and realized accounting.
2. `evaluate_domain_losses` — unblocks `test_evaluate_*`. The deterministic
   stub model makes the expected ordering visible without training.

Construction and validation tests pass from the start. No corpus download or
model checkpoint is required; all streams and models here are tiny fixtures.
"""
from __future__ import annotations

import pytest
import torch

from g2c.midtraining import TokenMixture, evaluate_domain_losses
from g2c.nn import Module


def _streams() -> dict[str, torch.Tensor]:
    # Distinct ranges make each sampled row's source visible.
    return {
        "python": torch.arange(1, 501),
        "general": torch.arange(10_001, 10_501),
    }


def test_mixture_constructor_normalizes_and_reports() -> None:
    mixture = TokenMixture(_streams(), {"python": 8, "general": 2})
    assert torch.allclose(mixture.weights, torch.tensor([0.8, 0.2], dtype=torch.float64))
    assert mixture.observed_fractions == {"python": 0.0, "general": 0.0}
    assert "python=80.0%" in repr(mixture)


@pytest.mark.parametrize(
    ("sources", "weights", "message"),
    [
        ({}, {}, "non-empty"),
        ({"a": torch.arange(10)}, {"b": 1.0}, "identical keys"),
        ({"a": torch.arange(10)}, {"a": -1.0}, "non-negative"),
        ({"a": torch.arange(10)}, {"a": 0.0}, "positive"),
    ],
)
def test_mixture_constructor_rejects_bad_configuration(sources, weights, message) -> None:
    with pytest.raises(ValueError, match=message):
        TokenMixture(sources, weights)


def test_mixture_batch_has_shifted_rows_from_real_sources() -> None:
    mixture = TokenMixture(_streams(), {"python": 0.8, "general": 0.2})
    x, y = mixture.get_lm_batch(
        batch_size=40,
        context_length=8,
        generator=torch.Generator().manual_seed(7),
    )
    assert x.shape == y.shape == (40, 8)
    assert torch.equal(y[:, :-1], x[:, 1:])
    assert all(bool((row < 1_000).all() or (row > 10_000).all()) for row in x)
    assert sum(mixture.example_counts.values()) == 40


def test_mixture_is_reproducible_with_the_same_generator_seed() -> None:
    a = TokenMixture(_streams(), {"python": 0.8, "general": 0.2})
    b = TokenMixture(_streams(), {"python": 0.8, "general": 0.2})
    ax, ay = a.get_lm_batch(32, 12, generator=torch.Generator().manual_seed(11))
    bx, by = b.get_lm_batch(32, 12, generator=torch.Generator().manual_seed(11))
    assert torch.equal(ax, bx)
    assert torch.equal(ay, by)
    assert a.example_counts == b.example_counts


def test_mixture_realized_fraction_converges_to_weights() -> None:
    mixture = TokenMixture(_streams(), {"python": 0.8, "general": 0.2})
    generator = torch.Generator().manual_seed(19)
    for _ in range(100):
        mixture.get_lm_batch(20, 4, generator=generator)
    assert mixture.observed_fractions["python"] == pytest.approx(0.8, abs=0.03)
    assert mixture.observed_fractions["general"] == pytest.approx(0.2, abs=0.03)


def test_mixture_rejects_bad_batch_dimensions() -> None:
    mixture = TokenMixture(_streams(), {"python": 1.0, "general": 0.0})
    with pytest.raises(ValueError, match="batch_size"):
        mixture.get_lm_batch(0, 4)
    with pytest.raises(ValueError, match="context_length"):
        mixture.get_lm_batch(2, 0)


class _ConstantLogitsLM(Module):
    """Always favors token zero, independently at every position."""

    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.zeros(1, requires_grad=True)

    def parameters(self):
        return [self.anchor]

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(*ids.shape, 3, device=ids.device) + self.anchor * 0
        logits[..., 0] = logits[..., 0] + 5.0
        return logits


def test_evaluate_domain_losses_separates_domains() -> None:
    domains = {
        "easy": torch.zeros(200, dtype=torch.long),
        "hard": torch.ones(200, dtype=torch.long),
    }
    losses = evaluate_domain_losses(
        _ConstantLogitsLM(),
        domains,
        batch_size=4,
        context_length=8,
        eval_iters=3,
        device="cpu",
        seed=3,
    )
    assert list(losses) == ["easy", "hard"]
    assert losses["easy"] < 0.1
    assert losses["hard"] > 4.0


def test_evaluate_domain_losses_is_reproducible() -> None:
    domains = {"mixed": torch.arange(600, dtype=torch.long) % 3}
    kwargs = dict(batch_size=3, context_length=7, eval_iters=4, device="cpu", seed=8)
    first = evaluate_domain_losses(_ConstantLogitsLM(), domains, **kwargs)
    second = evaluate_domain_losses(_ConstantLogitsLM(), domains, **kwargs)
    assert first == second


@pytest.mark.parametrize(
    "kwargs",
    [
        {"domains": {}},
        {"domains": {"x": torch.arange(20)}, "batch_size": 0},
        {"domains": {"x": torch.arange(20)}, "context_length": 0},
        {"domains": {"x": torch.arange(20)}, "eval_iters": 0},
    ],
)
def test_evaluate_domain_losses_validates_inputs(kwargs) -> None:
    defaults = dict(batch_size=2, context_length=4, eval_iters=2, device="cpu")
    defaults.update(kwargs)
    with pytest.raises(ValueError):
        evaluate_domain_losses(_ConstantLogitsLM(), **defaults)
