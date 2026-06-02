# Hand-maintained solutions mirror — canonical, lives in `main` (see AGENTS.md).
"""Solutions for g2c.inference.benchmark pedagogical scaffolds.

Patched onto the scaffold targets by g2c.solutions.apply().
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any
from g2c.inference.backend import Backend, BackendInfo, InferenceResult


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
    if len(prompts) == 0:
        raise ValueError("benchmark needs at least one prompt")

    results: list[InferenceResult] = []
    for prompt in prompts:
        results.append(
            backend.complete(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
        )

    latencies = [result.latency_ms for result in results]
    latency_total = sum(latencies)
    completion_counts = [result.completion_tokens for result in results]
    if any(count is None for count in completion_counts):
        completion_tokens_total = None
        tokens_per_second_overall = None
    else:
        completion_tokens_total = sum(int(count) for count in completion_counts)
        total_seconds = latency_total / 1000.0
        tokens_per_second_overall = (
            completion_tokens_total / total_seconds if total_seconds > 0 else None
        )

    return BenchmarkResult(
        backend=backend.info,
        n=len(results),
        latency_ms_total=latency_total,
        latency_ms_mean=statistics.fmean(latencies),
        latency_ms_p50=_percentile(latencies, 50),
        latency_ms_p90=_percentile(latencies, 90),
        completion_tokens_total=completion_tokens_total,
        tokens_per_second_overall=tokens_per_second_overall,
        per_request_latency_ms=latencies,
        per_request_tokens_per_second=[
            result.tokens_per_second for result in results
        ],
        results=results,
        metadata=dict(metadata) if metadata is not None else {},
    )
