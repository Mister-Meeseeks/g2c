# Module 12 Rubric

Use this rubric to grade submitted answers in the student's working notebook (`notebooks/solutions/12-scaling.ipynb`, falling back to `notebooks/clean/12-scaling.ipynb` if no solutions copy exists yet). Each exercise has `Question:` / `Answer:` string cells; treat a blank `"Answer: "` as not submitted (skip it). If the student wrote a hint or help request inside the answer string, tutor first before grading and avoid giving away the full solution unless they explicitly ask. Give feedback exercise by exercise and avoid replacing the student's work with a full solution unless explicitly asked.

## Exercise 12.01 — Why the scaling plot is TinyStories-only

A correct answer should include:

- Restricting to `StoryLM-1M/5M/30M` holds the corpus family, tokenizer (`StoryTokenizer`, vocab 4096), and objective fixed, so parameter count is the only knob varying along the curve — the comparison isolates scale.
- Mixing in `ShakespeareLM` or `TinyLLM` would confound model size with corpus and tokenizer changes; in particular, per-token loss/perplexity is not comparable across different vocabularies.

Common issues:

- Answering "they're different models" without naming what specifically confounds (corpus, tokenizer/vocab).
- Missing the cross-vocab perplexity problem (raw per-token numbers shift with tokenization granularity).
- Framing the restriction as a compute limitation rather than an experimental-design choice.

## Exercise 12.02 — Reading the samples

A correct answer should include:

- One *concrete* behavior cited from their samples where the larger model succeeds and the smaller one reliably doesn't — e.g., keeping a character or situation consistent across sentences, completing a simple story beat, sustained grammaticality. The specific behavior and the size where it appears are run-dependent; grade the citation and the size-ordered comparison.
- The comparison read from the same prompt and sampling settings across models (the notebook fixes both), not from cherry-picked different setups.

Common issues:

- Vague answers ("the big one is better") with no named behavior from the outputs.
- Judging from a single sample per model rather than the qualitative pattern.
- Attributing a difference to sampling luck when the seed/prompt were held fixed — or, conversely, treating one lucky small-model sample as refuting the trend.

## Exercise 12.03 — Next-token distributions across sizes

A correct answer should include:

- What their probe plots actually show (run-dependent), interpreted with the right distinction: "merely more confident" means the same token ranking with mass more concentrated; "qualitatively different" means the larger model promotes contextually plausible tokens the smaller model barely ranks. Typically larger models do some of both — sharper *and* better-placed mass.
- Evidence tied to the probe prompt (`"the boy opened the "`): which tokens gained or lost mass across the ladder.

Common issues:

- Reading only the top-1 token and ignoring the shape of the rest of the distribution.
- Equating higher confidence with correctness — a small model can be confidently wrong.
- No reference to the actual plotted tokens/probabilities.

## Exercise 12.04 — Capacity vs compute (iso-step vs iso-FLOP)

A correct answer should include:

- Recognition that model size is confounded with compute in this ladder: FLOPs per token scale with parameter count (`≈ 6·N`), and the larger StoryLM models also trained for *more steps* (the 1M anchor runs ~5k steps; the Module 10 payoff runs are far longer) — so the 30M model's lower loss bundles capacity *and* much more compute. A good answer cites the table's steps / tokens-seen / FLOPs columns as the evidence rather than assuming equal budgets.
- A concrete change toward iso-FLOP: equalize total compute across sizes — give smaller models proportionally more steps (or larger models fewer) so `6·N·steps·batch·context` matches, and/or compare models by plotting loss against FLOPs rather than parameter count.

Common issues:

- Answering that the result is "all capacity" or "all compute" — the honest answer is that this experiment cannot split them.
- Proposing to equalize *steps* or *epochs* as the fix — equal steps still spend `≈ 30×` more FLOPs on the 30M model; the budget to equalize is FLOPs (tokens weighted by model size).
- Ignoring the artifact metadata (steps, tokens seen, approximate FLOPs) the exercise printed for exactly this purpose.
