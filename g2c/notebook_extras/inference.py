"""Notebook helpers for Module 16 inference experiments.

These helpers keep the inference notebook focused on the measurement itself
rather than on timing plumbing and chart formatting.

Timing a generation call is a correctness trap unrelated to the concept under
study: MPS and CUDA both queue work asynchronously, so a bare `perf_counter()`
around the call measures how long it took to *submit* the work, not how long
the work took. `time_generation` inserts the device barrier and runs an untimed
warmup first. The sweep that calls it belongs in the notebook — that loop is
the experiment, and it is meant to be read.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from statistics import median
from typing import Any

import matplotlib.pyplot as plt
import torch


def _synchronize(device: torch.device | str | None) -> None:
    """Block until queued accelerator work has actually finished."""
    if device is None:
        return
    dev = torch.device(device)
    if dev.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif dev.type == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


def time_generation(
    fn: Callable[[], Any],
    *,
    device: torch.device | str | None = None,
    warmup: int = 1,
    repeats: int = 3,
) -> float:
    """Return median wall-clock seconds for one `fn()` call.

    Args:
        fn: zero-argument callable running one complete generation.
        device: the device the model lives on, so timing can synchronize.
        warmup: untimed calls made first. The first call on MPS pays Metal
            kernel JIT compilation; without a warmup that one-time cost lands
            entirely inside your first measurement and looks like real work.
        repeats: timed calls. The median is returned so a single scheduling
            hiccup doesn't dominate the result.
    """
    for _ in range(max(0, warmup)):
        fn()
    _synchronize(device)

    samples: list[float] = []
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        fn()
        _synchronize(device)
        samples.append(time.perf_counter() - start)
    return median(samples)


def plot_cache_speedup(rows: Sequence[dict[str, Any]]) -> None:
    """Plot cached vs uncached decode cost, and the resulting speedup.

    Args:
        rows: dicts with keys `new_tokens`, `uncached_s`, and `cached_s`,
            one per point in the sweep.
    """
    if not rows:
        print("no benchmark rows to plot")
        return

    xs = [r["new_tokens"] for r in rows]
    uncached = [r["uncached_s"] for r in rows]
    cached = [r["cached_s"] for r in rows]
    speedup = [
        (u / c if c else float("nan")) for u, c in zip(uncached, cached, strict=True)
    ]

    fig, (ax_time, ax_speed) = plt.subplots(1, 2, figsize=(11, 4))

    ax_time.plot(xs, uncached, marker="o", label="uncached (recompute)")
    ax_time.plot(xs, cached, marker="o", label="cached (KV cache)")
    ax_time.set_xlabel("new tokens generated")
    ax_time.set_ylabel("wall-clock seconds")
    ax_time.set_title("Decode cost")
    ax_time.legend()
    ax_time.grid(alpha=0.3)

    ax_speed.plot(xs, speedup, marker="o", color="tab:green")
    ax_speed.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax_speed.set_xlabel("new tokens generated")
    ax_speed.set_ylabel("speedup (uncached / cached)")
    ax_speed.set_title("Speedup from the cache")
    ax_speed.grid(alpha=0.3)

    fig.tight_layout()
    plt.show()


def print_cache_speedup_table(rows: Sequence[dict[str, Any]]) -> None:
    """Print the raw sweep numbers behind `plot_cache_speedup`."""
    if not rows:
        print("no benchmark rows")
        return
    header = f"{'new tokens':>11} {'uncached s':>11} {'cached s':>10} {'speedup':>8}"
    print(header)
    print("-" * len(header))
    for r in rows:
        cached_s = r["cached_s"]
        ratio = r["uncached_s"] / cached_s if cached_s else float("nan")
        print(
            f"{r['new_tokens']:>11} {r['uncached_s']:>11.3f} "
            f"{cached_s:>10.3f} {ratio:>7.2f}x"
        )
