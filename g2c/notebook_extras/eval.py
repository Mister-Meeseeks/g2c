"""Notebook display helpers for Module 15 evaluation.

The pedagogical implementation lives in ``g2c.eval``. This module keeps the
Module 15 notebook focused on designing eval sets and interpreting results
rather than reimplementing report printing and plotting in every cell.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt

from g2c.eval import EvalReport, reliability_curve

__all__ = [
    "print_eval_report",
    "print_generation_results",
    "print_multiple_choice_misses",
    "plot_reliability",
]


def print_eval_report(report: EvalReport, *, title: str | None = None) -> None:
    """Print the headline metrics from one ``EvalReport``."""
    label = title or report.task_name
    confidence = _fmt(report.mean_confidence)
    ece = _fmt(report.ece)
    print(label)
    print("-" * max(24, len(label)))
    print(f"n:                {report.n}")
    print(f"accuracy:         {report.accuracy:.3f}")
    print(f"mean confidence:  {confidence}")
    print(f"ECE:              {ece}")


def plot_reliability(
    report: EvalReport,
    *,
    n_bins: int = 5,
    title: str | None = None,
) -> None:
    """Plot a reliability curve for reports with per-example confidence."""
    confidences = [r.confidence for r in report.results if r.confidence is not None]
    if len(confidences) != len(report.results):
        print("Reliability curve skipped: this report has no complete confidences.")
        return

    conf, acc, count = reliability_curve(
        [float(c) for c in confidences],
        [bool(r.correct) for r in report.results],
        n_bins=n_bins,
    )

    xs = [c for c in conf if not math.isnan(c)]
    ys = [a for a in acc if not math.isnan(a)]
    sizes = [max(40, 18 * n) for c, n in zip(conf, count, strict=True) if not math.isnan(c)]

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="perfect calibration")
    if xs:
        ax.scatter(xs, ys, s=sizes, color="#4c78a8", alpha=0.8, label="model bins")
        ax.plot(xs, ys, color="#4c78a8", alpha=0.7)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("mean confidence")
    ax.set_ylabel("empirical accuracy")
    ax.set_title(title or f"{report.task_name} reliability")
    ax.legend()
    fig.tight_layout()
    plt.show()

    print("bin  mean_conf  accuracy  count")
    for idx, (c, a, n) in enumerate(zip(conf, acc, count, strict=True)):
        c_text = "nan" if math.isnan(c) else f"{c:.3f}"
        a_text = "nan" if math.isnan(a) else f"{a:.3f}"
        print(f"{idx:>3}  {c_text:>9}  {a_text:>8}  {n:>5}")


def print_multiple_choice_misses(
    examples: Sequence[Any],
    report: EvalReport,
    *,
    limit: int = 8,
) -> None:
    """Print a compact table of wrong multiple-choice predictions."""
    misses = [
        (example, result)
        for example, result in zip(examples, report.results, strict=True)
        if not result.correct
    ]
    if not misses:
        print("No misses.")
        return

    for index, (example, result) in enumerate(misses[:limit], start=1):
        gold = result.metadata.get("gold_idx")
        pred = int(result.prediction)
        confidence = _fmt(result.confidence)
        print("=" * 72)
        print(f"miss {index}: predicted {pred}, gold {gold}, confidence {confidence}")
        prompt = getattr(example, "prompt", "")
        print(_one_line(prompt, max_len=120))
        for choice_index, choice in enumerate(getattr(example, "choices", [])):
            marker = "PRED" if choice_index == pred else "GOLD" if choice_index == gold else "    "
            print(f"  {marker} [{choice_index}] {_one_line(choice, max_len=96)}")


def print_generation_results(
    examples: Sequence[Any],
    report: EvalReport,
    *,
    limit: int = 12,
) -> None:
    """Print generated predictions next to their references."""
    for index, (example, result) in enumerate(
        zip(examples, report.results, strict=True),
        start=1,
    ):
        if index > limit:
            remaining = len(report.results) - limit
            if remaining > 0:
                print(f"... {remaining} more result(s) omitted")
            break
        status = "ok" if result.correct else "miss"
        print("=" * 72)
        print(f"{index}. {status}")
        print("prompt:")
        print(_one_line(getattr(example, "prompt", ""), max_len=180))
        print("prediction:")
        print(_one_line(str(result.prediction), max_len=220))
        print("references:")
        for ref in getattr(example, "references", []):
            print(f"  - {_one_line(ref, max_len=160)}")


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def _one_line(text: str, *, max_len: int) -> str:
    line = text.replace("\n", "\\n")
    if len(line) <= max_len:
        return line
    return line[: max_len - 1] + "..."
