# Module 16 Rubric

Use this rubric to grade submitted answers in the student's working notebook (`notebooks/solutions/16-inference.ipynb`, falling back to `notebooks/clean/16-inference.ipynb` if no solutions copy exists yet). Each exercise has `Question:` / `Answer:` string cells; treat a blank `"Answer: "` as not submitted (skip it). If the student wrote a hint or help request inside the answer string, tutor first before grading and avoid giving away the full solution unless they explicitly ask. Give feedback exercise by exercise and avoid replacing the student's work with a full solution unless explicitly asked.

Latency and quality numbers are machine- and model-dependent. Grade cited evidence from the student's own runs and the correctness of the causal story, not specific values.

## Exercise 16.01 — Smoke-test capability gap

A correct answer should include:

- A concrete capability gap cited from the smoke outputs — typically the artifact backend wanders off-topic, gets facts or arithmetic wrong, or ignores instruction shape ("list three..."), where ProdLM answers correctly and concisely.
- The latency observation from their run, usually that the small in-process artifact is faster per request than a 3B model over HTTP — capability and speed are separate axes.

Common issues:

- No cited completions; a verdict asserted from priors instead of the run.
- Attributing the latency difference mainly to HTTP overhead when model size dominates.
- Generalizing from a single prompt instead of the four-prompt sweep.

## Exercise 16.02 — Reading the benchmark distribution

A correct answer should include:

- Mean, p50, and p90 cited from the table, with the relationship explained: a cold-start first request drags the mean up while p50 barely moves.
- Whether the first request was an outlier, checked against the per-request latency plot.
- p50 named as the typical-request statistic (with p90 for the tail), because the mean is outlier-sensitive.

Common issues:

- Calling the mean "typical" despite a visible outlier.
- Asserting cold start without looking at the per-request series.
- Comparing absolute numbers across machines or model tags as if portable.

## Exercise 16.03 — Generation eval misses: model vs matcher

A correct answer should include:

- Per miss, a classification: the model's answer was wrong, vs `contains_match` too strict (correct answer phrased without any reference substring), vs too loose (a reference accidentally present as a substring in a hedged or verbose answer).
- The specific prompt and model output cited for each miss. "Zero misses" is a fine answer if stated with the report as evidence.

Common issues:

- Not separating "wrong answer" from "right answer that failed the match".
- Forgetting `contains_match` is substring-based and case-insensitive, so verbose answers pass easily — the loose direction is the quiet one.
- Blaming the model for a matcher artifact (or vice versa) without quoting the output.

## Exercise 16.04 — The quantization cliff

A correct answer should include:

- Where degradation started in their table — typically fp16 ≈ int8, visible degradation at 4-bit, collapse to gibberish/repetition at 2-bit — and that the fall-off is a cliff, not linear.
- The mechanism: each bit removed doubles the quantization step. At 8 bits there are 256 levels so rounding error is small relative to the weights; at 2 bits there are only four levels per tensor, and with per-tensor max-abs scaling one outlier weight sets the scale, so most weights collapse onto one or two levels — error comparable to the weights themselves, compounding through the layers.

Common issues:

- Expecting linear degradation per bit removed.
- Claiming the fake quantization saves memory — the cell prints that it does not; weights stay float tensors, this is a quality experiment only.
- Conflating this per-tensor fake quant with GGUF per-block K-quants — the per-block scales are why real Q4 hurts less than fake int4.

## Exercise 16.05 — Routing on prompt length

A correct answer should include:

- A cited misroute (or near-miss) from their run: the router keys on token count, but difficulty is not length — short-but-hard prompts like "What is 13 + 28?" go to the weak artifact. With the default threshold, all four smoke prompts are short enough to route to the artifact.
- An alternative routing signal — a task/topic classifier, keyword rules (numbers → strong model), a cheap confidence probe, or try-cheap-then-escalate cascading — and its cost: extra inference or classifier calls, added latency, and maintenance.

Common issues:

- Proposing "route by difficulty" without saying how difficulty would be measured.
- Answering the signal half and skipping the cost half.
- Concluding routing is useless rather than that the signal was wrong.

## Exercise 16.06 — Why the KV cache must match

A correct answer should include:

- Greedy decoding is deterministic argmax, and the KV cache is a pure reorganization of the same computation — K/V rows for earlier positions are identical whether recomputed or read from cache — so logits, and therefore token IDs, must match exactly.
- A mismatch means a correctness bug in `forward_cached` — wrong positional-encoding offset for the appended token, causal-mask misalignment, or cache written/read at the wrong index — not acceptable numerical drift.

Common issues:

- "Small float differences are fine" — the contract is identical token IDs under greedy decoding.
- Treating the cache as an approximation that trades accuracy for speed.
- Attributing a mismatch to sampling randomness when temperature is 0.

## Exercise 16.07 — Choosing the default backend

A correct answer should include:

- A named backend (almost always ProdLM) and one strongest piece of evidence from this notebook — e.g., the generation-eval accuracy gap or a smoke-prompt capability gap — weighed against acceptable benchmark latency.
- If they choose a course artifact instead, an explicit justification for accepting its capability ceiling in Modules 17-20.

Common issues:

- Citing throughput alone and ignoring quality (or the reverse).
- A decision with no number or output from the notebook behind it.
- "It's bigger so it's better" without pointing at observed behavior.
