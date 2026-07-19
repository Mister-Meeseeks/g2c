# Module 11 Rubric

Use this rubric to grade submitted answers in the student's working notebook (`notebooks/solutions/11-sampling.ipynb`, falling back to `notebooks/clean/11-sampling.ipynb` if no solutions copy exists yet). Each exercise has `Question:` / `Answer:` string cells; treat a blank `"Answer: "` as not submitted (skip it). If the student wrote a hint or help request inside the answer string, tutor first before grading and avoid giving away the full solution unless they explicitly ask. Give feedback exercise by exercise and avoid replacing the student's work with a full solution unless explicitly asked.

## Exercise 11.01 — Hardest sampling test

A correct answer should include:

- An actual test named from `tests/test_sampling.py` and the bug it caught in their implementation (run-dependent). Plausible pairings: a top-p test catching the off-by-one (the first token crossing the cumulative threshold must be *kept*), a repetition-penalty test catching the sign-asymmetric divide/multiply rule, a filter test catching `-1e9` masking instead of `-inf`, or a generate test catching the crop-to-`max_seq_len` or slice-before-warp ordering.

Common issues:

- Naming a test with no description of the bug it caught.
- Describing a bug with no tie to the test that exposed it.

## Exercise 11.02 — Which artifact the loader chose and why

A correct answer should include:

- The artifact actually selected in their run (machine-dependent), plus the reasoning: the loader ranks the available artifacts (ShakespeareLM-1M → StoryLM-5M → StoryLM-10M → StoryLM-30M → TinyLLM-30M → TinyLLM-100M) and picks the strongest self-trained one, because sampling intuition develops best against the best model the student personally trained.

Common issues:

- Restating the artifact name with no reasoning about why the ranking makes it the right choice.
- Confusing the auto-selected self-trained artifact with the optional `BaseLM` override.

## Exercise 11.03 — Which Module 10 save cell still needs to run

A correct answer should include:

- The Module 10 `save_model_artifact` cell for the stronger run was never executed, so the stronger artifact doesn't exist under `artifacts/models/` — go back and run that save cell, then re-run the load cell here.

Common issues:

- Proposing to retrain the model instead of just running the missing save-artifact cell.
- Blaming the Module 11 loader when the artifact simply isn't on disk.

## Exercise 11.04 — What changes as temperature rises

A correct answer should include:

- More diversity/variety as temperature rises, with the failure modes shifting from repetition/looping (low T) toward derailment — off-topic words, broken syntax, long-tail tokens (high T). Grounded in at least one observation from their own sweep samples; exact outputs are run-dependent.

Common issues:

- Claiming temperature changes *which* token is most likely — temperature is a monotone rescaling and never reorders tokens.
- Describing the tradeoff abstractly with no cited sample behavior.

## Exercise 11.05 — Why temperature=0 ignores the seed

A correct answer should include:

- `temperature=0` is the greedy path: the implementation takes `argmax` of the logits and never draws from `multinomial`, so the random generator is never consulted — the output is fully deterministic.

Common issues:

- Explaining it as "dividing by zero rounds to the max" — the code branches to argmax; there is no division at T=0.
- Saying the output is only "mostly" deterministic — the generator is untouched, so the match is exact.

## Exercise 11.06 — Best coherence vs most variety

A correct answer should include:

- A specific setting from their run named for best local coherence (typically a tighter filter, e.g. `top_p=0.7` or `top_k=10`) and one for most variety (looser: `no filter` or `top_p=0.95`) — run-dependent; grade the cited evidence and the direction of reasoning.

Common issues:

- Restating the filter definitions without any observation from their own samples.
- Comparing settings sampled with different seeds/temperatures and attributing the difference to the filter.

## Exercise 11.07 — Is top-p more adaptive than top-k?

A correct answer should include:

- The adaptivity distinction with one concrete observation from their samples: top-k keeps a fixed count regardless of confidence, while top-p's surviving set shrinks when the model is confident and expands when it is uncertain — spending the candidate budget where it matters.

Common issues:

- Claiming top-k or top-p can change the top-ranked token — both always preserve the argmax.
- Giving the textbook contrast with no concrete observation, when the question explicitly asks for one.

## Exercise 11.08 — Dominant vs spread next-token mass

A correct answer should include:

- A description of what their table/histogram actually shows (run-dependent): either one dominant token or mass spread over many plausible tokens.
- The correct implication for random sampling: a dominant token makes sampling nearly deterministic at that step (almost always picks it); spread mass makes samples diverse — and raises the chance of drawing a derailing lower-ranked token.

Common issues:

- Answering in general terms without reading their own distribution.
- Treating a spread distribution as a model bug rather than genuine uncertainty over plausible continuations.

## Exercise 11.09 — Repetition penalty and type-token ratio

A correct answer should include:

- Whether the penalty raised their type-token ratio (typically yes, roughly monotonically across 1.0 → 1.1 → 1.3), citing their measured numbers or bar chart; exact values are run-dependent.
- The metric-vs-quality distinction: a higher TTR means less repetition, not necessarily better text — the penalty also suppresses legitimately repeated tokens (names, articles), so "reads better" must be judged separately by reading the samples.

Common issues:

- Equating "less repetitive" with "better" without reading the output.
- Comparing the greedy row (different temperature settings) directly against the penalty rows as if only the penalty changed.
- Not reporting any numbers when the exercise computed them.

## Exercise 11.10 — Why top_k=1 matches greedy

A correct answer should include:

- `top_k=1` sets every logit except the argmax to `-inf`, so after softmax the surviving token has probability 1; `multinomial` over a one-hot distribution returns that token every time, regardless of the generator/seed — hence token-for-token identical to the explicit argmax path.
- (Implicitly or explicitly) both paths preserve the argmax; the randomness is only ever over the surviving set, and here that set has size 1.

Common issues:

- Saying the two "usually" match or match because the seeds were chosen well — the equality is exact and seed-independent.
- Attributing the match to `temperature=1.0` in the top-k path rather than to the degenerate one-token distribution.

## Exercise 11.11 — Prompt playground

A correct answer should include:

- At least three prompts tried, with a pattern identified: the artifact handles in-distribution prompts best (text shaped like its training corpus — Shakespearean dialogue or story openings, depending on the artifact) and reliably breaks on out-of-distribution material — modern vocabulary, questions/instructions (it's a base LM, not an instruct model), unusual formats.
- Concrete cited behavior from their runs (run-dependent).

Common issues:

- Fewer than three prompts, or three prompts of the same kind that can't reveal a pattern.
- Grading the model as "bad" for failing instruction-following — a pretrained-only artifact continuing text is expected behavior, previewing why post-training (Modules 13–14) exists.

## Exercise 11.12 — Forbidden token IDs

A correct answer should include:

- The mechanism: setting a token's logit to `-inf` gives it probability exactly 0 after softmax, so it can never be sampled; the loop reroutes probability to the remaining tokens.
- Which tokens they forbade and what changed in the continuation (run-dependent — e.g., the model paraphrases around the banned token, picks the next-best candidate, or degrades if the banned token was central to the continuation).

Common issues:

- Using a large-negative-constant framing ("very unlikely") instead of exactly zero probability — `-inf` is a hard constraint, not a soft nudge.
- Forbidding tokens that never appeared in the top candidates and concluding biasing "does nothing."
