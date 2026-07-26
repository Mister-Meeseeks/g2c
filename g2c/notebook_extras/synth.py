"""Notebook glue for Module 16B (synthetic data). Not pedagogical.

Progress display around `synthesize_dataset` and a seeded
pair-sampler for the audit exercise. The concepts live in
`g2c/synth/`; this file is IPython furniture.
"""
from __future__ import annotations

import random

from IPython.display import Markdown, display

from g2c.synth import synthesize_dataset


def synthesize_with_progress(backend, seeds, **kwargs):
    """Run `synthesize_dataset` with a live progress line and a funnel
    table at the end. Returns `(pairs, funnel)` unchanged."""
    handle = display(Markdown("*starting generation...*"), display_id=True)

    def progress(message: str) -> None:
        handle.update(Markdown(f"*{message}*"))

    pairs, funnel = synthesize_dataset(backend, seeds, progress=progress, **kwargs)

    rows = ["| stage | count |", "| --- | --- |"]
    rows += [f"| {stage} | {count} |" for stage, count in funnel.items()]
    handle.update(Markdown("\n".join(rows)))
    return pairs, funnel


def show_pair_sample(pairs: list[dict], *, k: int = 10, seed: int = 0) -> None:
    """Print a reproducible random sample of pairs for hand-auditing."""
    rng = random.Random(seed)
    chosen = rng.sample(pairs, min(k, len(pairs)))
    for i, pair in enumerate(chosen, 1):
        print(f"--- {i} ---")
        print("user:     ", pair["user"])
        print("assistant:", pair["assistant"])
