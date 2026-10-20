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

## Exercise 16.03 — Prefill versus decode on ProdLM

A correct answer should include:

- The two rates cited from their long-prompt row, with prefill (prompt tokens per second) many times faster than decode (output tokens per second). The exact ratio is machine-dependent; an order of magnitude is typical.
- The mechanism: prefill processes all prompt tokens in one pass, so each weight read from memory is reused across the whole prompt and the step is limited by arithmetic. Decode produces one token per step, so every weight is streamed again for a single token and the step is limited by memory bandwidth.
- The second identical send: prefill time collapses even though `prompt_eval_count` still shows the full prompt, because Ollama reused the KV cache for the shared prefix. In a multi-turn chat each new turn shares the earlier transcript, so with a warm cache time to first token is spent mostly on the new tokens; without one, the whole growing transcript is prefilled again every turn.

Common issues:

- Reading the cache hit as the model "warming up" or getting faster.
- Comparing a prefill rate from the short prompt, where a handful of tokens makes the rate noisy, instead of the long one.
- Explaining the gap with "prompt tokens are easier" rather than weight reuse versus weight streaming.

## Exercise 16.04 — Generation eval misses: model vs matcher

A correct answer should include:

- Per miss, a classification: the model's answer was wrong, vs `contains_match` too strict (correct answer phrased without any reference substring), vs too loose (a reference accidentally present as a substring in a hedged or verbose answer).
- The specific prompt and model output cited for each miss. "Zero misses" is a fine answer if stated with the report as evidence.

Common issues:

- Not separating "wrong answer" from "right answer that failed the match".
- Forgetting `contains_match` is substring-based and case-insensitive, so verbose answers pass easily — the loose direction is the quiet one.
- Blaming the model for a matcher artifact (or vice versa) without quoting the output.

## Exercise 16.05 — The quantization cliff

A correct answer should include:

- Where degradation started in their table — typically fp16 ≈ int8, visible degradation at 4-bit, collapse to gibberish/repetition at 2-bit — and that the fall-off is a cliff, not linear.
- The mechanism: each bit removed doubles the quantization step. At 8 bits there are 256 levels so rounding error is small relative to the weights; at 2 bits there are only four levels per tensor, and with per-tensor max-abs scaling one outlier weight sets the scale, so most weights collapse onto one or two levels — error comparable to the weights themselves, compounding through the layers.

Common issues:

- Expecting linear degradation per bit removed.
- Claiming the fake quantization saves memory — the cell prints that it does not; weights stay float tensors, this is a quality experiment only.
- Conflating this per-tensor fake quant with GGUF per-block K-quants — the per-block scales are why real Q4 hurts less than fake int4.

## Exercise 16.06 — Routing on prompt length

A correct answer should include:

- A cited misroute (or near-miss) from their run: the router keys on token count, but difficulty is not length — short-but-hard prompts like "What is 13 + 28?" go to the weak artifact. With the default threshold, all four smoke prompts are short enough to route to the artifact.
- An alternative routing signal — a task/topic classifier, keyword rules (numbers → strong model), a cheap confidence probe, or try-cheap-then-escalate cascading — and its cost: extra inference or classifier calls, added latency, and maintenance.

Common issues:

- Proposing "route by difficulty" without saying how difficulty would be measured.
- Answering the signal half and skipping the cost half.
- Concluding routing is useless rather than that the signal was wrong.

## Exercise 16.07 — Why the KV cache must match

A correct answer should include:

- Greedy decoding is deterministic argmax, and the KV cache is a pure reorganization of the same computation — K/V rows for earlier positions are identical whether recomputed or read from cache — so logits, and therefore token IDs, must match exactly.
- A mismatch means a correctness bug in `forward_cached` — wrong positional-encoding offset for the appended token, causal-mask misalignment, or cache written/read at the wrong index — not acceptable numerical drift.

Common issues:

- "Small float differences are fine" — the contract is identical token IDs under greedy decoding.
- Treating the cache as an approximation that trades accuracy for speed.
- Attributing a mismatch to sampling randomness when temperature is 0.

## Exercise 16.08 — Reading the cache speedup curve

A correct answer should include:

- Whether the cached/uncached gap widened with generated-token count in their table, cited with at least two rows. A widening gap is the O(T²) versus O(T) signature, independent of the headline multiplier.
- A break-even point from their run (on small models usually somewhere around 16–32 new tokens) and the reason below it: per-step Python and kernel-dispatch overhead dominates both paths, and with a short context there is little recomputation for the cache to save.

Common issues:

- Judging the implementation by the absolute speedup and expecting production's 5–20× on a few-million-parameter model.
- Calling the cache broken because it is slower at 16 tokens.
- Citing one length only, which can't show a trend.

## Exercise 16.09 — Prefill two ways

A correct answer should include:

- The scaling from their table: one-pass time grows slowly with prompt length (on MPS it can stay nearly flat), while token-by-token time grows roughly linearly, so the ratio widens with length.
- Why the logits match: the causal mask means each position in a full forward pass sees exactly the tokens before it, which is the same history the cached loop has accumulated at that step, so the K/V rows and the last position's logits are the same computation in a different order (equal up to floating-point tolerance).
- What would change: `forward_cached` would need to accept a block of several new tokens, apply a causal mask within the block, and append all of their K/V rows at once. For a 4,000-token RAG prompt, prefill is most of time to first token, so a token-by-token prefill pays decode's per-step cost thousands of times before the first output token.

Common issues:

- Saying the loop is slow "because of Python" without noting that it also gives up the batched matrix multiply.
- Expecting bit-identical logits; the contract is equality within tolerance, because the summation order differs.
- Missing the causal-mask-within-the-block requirement for multi-token cached steps.

## Exercise 16.10 — Choosing the default backend

A correct answer should include:

- A named backend (almost always ProdLM) and one strongest piece of evidence from this notebook — e.g., the generation-eval accuracy gap or a smoke-prompt capability gap — weighed against acceptable benchmark latency.
- If they choose a course artifact instead, an explicit justification for accepting its capability ceiling in Modules 17-20.

Common issues:

- Citing throughput alone and ignoring quality (or the reverse).
- A decision with no number or output from the notebook behind it.
- "It's bigger so it's better" without pointing at observed behavior.
