"""Weighted token-stream mixtures for continued pretraining.

Midtraining keeps the language-model objective and changes the data
distribution. `TokenMixture` makes that distribution explicit while
remaining compatible with Module 10's `Trainer`: it implements the same
`get_lm_batch` protocol as a single token stream.

Construction and accounting are provided. Sampling one mixed batch is the
scaffold because that operation *is* the experiment's intervention.
"""
from __future__ import annotations

from collections.abc import Mapping

import torch

from g2c.pretraining.data import (  # noqa: F401 - helper used by scaffold
    SupportsLMBatch,
    get_lm_batch,
)

TokenSource = torch.Tensor | SupportsLMBatch


class TokenMixture:
    """A weighted collection of language-model token streams.

    Args:
        sources: non-empty mapping from stable domain name to a 1-D token
            tensor or disk-backed source implementing `get_lm_batch`.
        weights: non-negative sampling weights with exactly the same keys as
            `sources`. They need not sum to one; the constructor normalizes
            them. At least one weight must be positive.

    Each batch row independently chooses a source according to `weights`, then
    samples one contiguous next-token window from that source. Consequently,
    an 80/20 mixture is exact in expectation, not necessarily in every small
    batch. `example_counts` records the realized allocation so the notebook
    can report what training actually consumed.
    """

    def __init__(
        self,
        sources: Mapping[str, TokenSource],
        weights: Mapping[str, float],
    ) -> None:
        if not sources:
            raise ValueError("sources must be non-empty")
        if set(sources) != set(weights):
            raise ValueError("sources and weights must have identical keys")
        if any(not isinstance(name, str) or not name for name in sources):
            raise ValueError("source names must be non-empty strings")

        names = tuple(sources)
        raw_weights = torch.tensor(
            [float(weights[name]) for name in names], dtype=torch.float64
        )
        if not torch.isfinite(raw_weights).all():
            raise ValueError("weights must be finite")
        if (raw_weights < 0).any():
            raise ValueError("weights must be non-negative")
        total = float(raw_weights.sum())
        if total <= 0:
            raise ValueError("at least one weight must be positive")

        self.sources = dict(sources)
        self.names = names
        self.weights = raw_weights / total
        self.example_counts = {name: 0 for name in names}

    def __len__(self) -> int:
        """Total tokens available across sources (sampling may revisit them)."""
        return sum(len(source) for source in self.sources.values())

    def __repr__(self) -> str:
        mixture = ", ".join(
            f"{name}={weight:.1%}"
            for name, weight in zip(self.names, self.weights.tolist(), strict=True)
        )
        return f"TokenMixture({mixture})"

    @property
    def observed_fractions(self) -> dict[str, float]:
        """Realized example fraction by source; zero before the first batch."""
        total = sum(self.example_counts.values())
        if total == 0:
            return {name: 0.0 for name in self.names}
        return {name: self.example_counts[name] / total for name in self.names}

    def get_lm_batch(
        self,
        batch_size: int,
        context_length: int,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample one weighted `(x, y)` language-model batch.

        Returns tensors of shape `(batch_size, context_length)`. Every row
        comes from exactly one source and every target is the corresponding
        input shifted by one token.

        Recipe:
            1. Validate `batch_size > 0` and `context_length > 0`.
            2. Draw `batch_size` source indices with replacement using
               `torch.multinomial(self.weights, ..., generator=generator)`.
            3. For each source represented in the draw, call `get_lm_batch`
               once with the number of rows assigned to it.
            4. Concatenate those source batches, then shuffle rows with
               `torch.randperm(..., generator=generator)` so domains are not
               grouped inside the optimizer batch.
            5. Increment `example_counts` by the realized rows per source.

        Do not concatenate token streams before window sampling: a window
        crossing that boundary would train on a fake Python→prose transition.
        """
        # TODO
        raise NotImplementedError
