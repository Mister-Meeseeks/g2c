"""Tiny benchmark utility for any `Backend`.

Given a `Backend` and a list of prompts, run the backend over each
prompt and aggregate the per-call timings into a `BenchmarkResult`.
The result captures latency (mean / p50 / p90), throughput
(tokens/sec — both per-request and overall), and the raw per-request
arrays for plotting.

The point of running this on a SUITE of prompts rather than a single
one is variance: the first call to a freshly-loaded model is slower
than the steady-state (model load, page faults, JIT compilation on
MPS), and short prompts have artificially-high tokens/sec because
fixed per-call overhead dominates. A 5–20 prompt suite at a similar
length-budget tells you the steady-state throughput; a single call
tells you a single noisy number.

What lives where:

  * `BenchmarkResult` (dataclass) — fully implemented. The aggregated
    record returned by `benchmark`.

  * `benchmark(backend, prompts, ...)` — scaffolded. Takes a backend
    and a list of prompts, calls `backend.complete(p, ...)` for each,
    aggregates timings into a `BenchmarkResult`. The recipe is in the
    docstring.

The aggregation math is light: percentiles via `statistics.quantiles`
(stdlib), means via `statistics.fmean`, sums via `sum`. No numpy
dependency for this — keeps the imports tight, and the test fixtures
don't need to spin up tensors.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

from .backend import Backend, BackendInfo, InferenceResult


@dataclass
class BenchmarkResult:
    """Aggregated timing record for a benchmark run.

    Attributes:
        backend: identity of the backend that produced these numbers.
            Carried alongside so a JSON dump tells you which backend
            ran. (Redundant with each result's `backend` field, but
            useful for the headline-level record.)
        n: number of prompts run. Equal to `len(results)`.
        latency_ms_total: sum of per-request `latency_ms`. The total
            wall time the benchmark spent in `complete` calls. Note
            this is sum of per-call latencies, NOT wall time of the
            outer loop — the two agree when calls are sequential
            (which they are here) but differ if you parallelize.
        latency_ms_mean: arithmetic mean of per-request latencies.
        latency_ms_p50: median of per-request latencies.
        latency_ms_p90: 90th-percentile per-request latency. The
            "tail" — useful for spotting that one slow call.
        completion_tokens_total: sum of per-request
            `completion_tokens`. `None` if any call returned None for
            that field (in which case throughput is incomputable).
        tokens_per_second_overall: total completion tokens divided
            by total wall time (in seconds). The aggregate throughput.
            `None` if `completion_tokens_total` is None.
        per_request_latency_ms: list of per-request latencies, in
            order. Length n.
        per_request_tokens_per_second: list of per-request
            tokens-per-second, in order. Length n. Elements may be
            `None` for individual requests that didn't report
            `completion_tokens`.
        results: the underlying `InferenceResult` records. Same order
            as `per_request_latency_ms`.

    Note: percentiles are computed via `statistics.quantiles` with
    `n=100` (so `p50` is the 50th percentile, `p90` is the 90th).
    This is interpolated quantile — for n < 2 the percentile fields
    fall back to the single value.

    The boilerplate here is fully implemented — the benchmark is
    just iteration plus statistics, and the lesson is the iteration,
    not the statistics.
    """

    backend: BackendInfo
    n: int
    latency_ms_total: float
    latency_ms_mean: float
    latency_ms_p50: float
    latency_ms_p90: float
    completion_tokens_total: int | None
    tokens_per_second_overall: float | None
    per_request_latency_ms: list[float]
    per_request_tokens_per_second: list[float | None]
    results: list[InferenceResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        tps = (
            f"{self.tokens_per_second_overall:.2f}"
            if self.tokens_per_second_overall is not None
            else "—"
        )
        ct = (
            f"{self.completion_tokens_total}"
            if self.completion_tokens_total is not None
            else "—"
        )
        return (
            f"BenchmarkResult(backend={self.backend.name!r}/"
            f"{self.backend.model_id!r}, n={self.n}, "
            f"latency_ms_mean={self.latency_ms_mean:.2f}, "
            f"latency_ms_p50={self.latency_ms_p50:.2f}, "
            f"latency_ms_p90={self.latency_ms_p90:.2f}, "
            f"completion_tokens_total={ct}, "
            f"tokens_per_second_overall={tps})"
        )


def _percentile(values: list[float], pct: float) -> float:
    """Interpolated percentile via `statistics.quantiles(n=100)`.

    For lists of length 0, raises. For length 1, returns the single
    value (no interpolation possible). For length >= 2, uses the
    stdlib's exclusive-quantile method (Python's default), which
    matches numpy's `np.percentile(..., method='inclusive')` to
    within rounding for our use.

    `pct` is on `[0, 100]`. `_percentile(xs, 50)` is the median.
    """
    if len(values) == 0:
        raise ValueError("_percentile undefined on empty list")
    if len(values) == 1:
        return values[0]
    qs = statistics.quantiles(values, n=100, method="inclusive")
    # `qs` has length 99 — the 1st through 99th percentiles. Index
    # 0 is the 1st percentile; index 49 is the 50th; index 89 is
    # the 90th. Clamp pct to [1, 99] for safety.
    idx = int(round(pct)) - 1
    idx = max(0, min(idx, len(qs) - 1))
    return qs[idx]


def benchmark(
    backend: Backend,
    prompts: list[str],
    *,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> BenchmarkResult:
    """Run `backend.complete(p, ...)` over each prompt and aggregate.

    Args:
        backend: the `Backend` to benchmark.
        prompts: list of prompt strings. Must be non-empty. Calls
            are made sequentially in list order.
        max_new_tokens: forwarded to every `complete` call.
        temperature: forwarded.
        top_k: forwarded.
        top_p: forwarded.
        metadata: optional dict copied into `result.metadata` for
            recording context (suite name, run id, etc.).

    Returns:
        `BenchmarkResult` with per-request and aggregate timings.

    Raises:
        ValueError: on empty prompts list.

    Recipe:
        1. # Validate.
           if len(prompts) == 0:
               raise ValueError("benchmark needs at least one prompt")

        2. # Run all prompts, collecting per-request results.
           results: list[InferenceResult] = []
           for p in prompts:
               r = backend.complete(
                   p,
                   max_new_tokens=max_new_tokens,
                   temperature=temperature,
                   top_k=top_k,
                   top_p=top_p,
               )
               results.append(r)

        3. # Per-request latencies (always present).
           latencies = [r.latency_ms for r in results]
           latency_total = sum(latencies)
           latency_mean  = statistics.fmean(latencies)
           latency_p50   = _percentile(latencies, 50)
           latency_p90   = _percentile(latencies, 90)

        4. # Per-request and aggregate completion-token counts.
           # If ANY request had None for completion_tokens, the
           # aggregate is None — throughput across heterogeneous
           # requests isn't meaningful.
           ct_each = [r.completion_tokens for r in results]
           if any(c is None for c in ct_each):
               completion_tokens_total = None
               tokens_per_second_overall = None
           else:
               completion_tokens_total = sum(ct_each)
               total_seconds = latency_total / 1000.0
               if total_seconds > 0:
                   tokens_per_second_overall = (
                       completion_tokens_total / total_seconds
                   )
               else:
                   tokens_per_second_overall = None

        5. # Per-request tokens-per-second — uses the result's
           # property. Yields None for individual requests that
           # didn't report completion_tokens.
           per_req_tps = [r.tokens_per_second for r in results]

        6. return BenchmarkResult(
               backend=backend.info,
               n=len(results),
               latency_ms_total=latency_total,
               latency_ms_mean=latency_mean,
               latency_ms_p50=latency_p50,
               latency_ms_p90=latency_p90,
               completion_tokens_total=completion_tokens_total,
               tokens_per_second_overall=tokens_per_second_overall,
               per_request_latency_ms=latencies,
               per_request_tokens_per_second=per_req_tps,
               results=results,
               metadata=dict(metadata) if metadata is not None else {},
           )

    Implementation notes:

      * Calls are sequential. A parallel benchmark would need either
        threads (for I/O-bound HTTP backends) or a process pool (for
        CPU/MPS-bound local backends, since PyTorch on Apple silicon
        doesn't release the GIL inside model forward).
        Sequential keeps the timing math honest — total wall time
        equals sum of per-call latencies.

      * The first call typically dominates p90 because of model-
        load latency. For a steady-state measurement, run a "warmup
        prompt" through the backend before calling `benchmark`,
        OR drop the first request's number from your analysis.

      * `metadata` is COPIED, not held by reference. A caller can
        mutate their original dict after calling `benchmark` without
        affecting the result.

      * `tokens_per_second_overall` is `total_completion_tokens /
        total_wall_seconds`. This is sometimes higher and sometimes
        lower than the mean of per-request tokens-per-second
        depending on the spread of latencies and lengths. Both
        readouts are useful; they answer slightly different questions.

    Sanity values:

      * Single prompt: latency_mean == latency_p50 == latency_p90.
        completion_tokens_total == this one request's count.

      * All requests reporting completion_tokens=N at latency=L ms:
        tokens_per_second_overall = N / (L/1000), which simplifies
        to 1000*N/L. Same as the per-request mean (which is the
        same as the median, since all values are equal).

      * Any request with completion_tokens=None:
        completion_tokens_total and tokens_per_second_overall both
        fall back to None.
    """
    # TODO
    raise NotImplementedError
