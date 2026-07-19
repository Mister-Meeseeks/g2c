# Module 06 Rubric

Use this rubric to grade submitted answers in the student's working notebook (`notebooks/solutions/06-language-models.ipynb`, falling back to `notebooks/clean/06-language-models.ipynb` if no solutions copy exists yet). Each exercise has `Question:` / `Answer:` string cells; treat a blank `"Answer: "` as not submitted (skip it). If the student wrote a hint or help request inside the answer string, tutor first before grading and avoid giving away the full solution unless they explicitly ask. Give feedback exercise by exercise and avoid replacing the student's work with a full solution unless explicitly asked.

## Exercise 06.01 — Next failing test

Run-dependent — grade the reasoning shape.

A correct answer should include:

- A concrete test name from `pytest tests/test_lm.py -x` at the time they ran it, mapped to the implementation it points at (`CountsBigramLM.fit`/`.logits`, `NeuralBigramLM.forward`, `MLPLanguageModel.forward`, `perplexity`, `sample`, `train_lm`). "All passing" is a complete answer.

Common issues:

- Naming a file with no test, or a test with no target function.

## Exercise 06.02 — (context, target) examples

A correct answer should include:

- Exactly three examples: `[4, 7] → 1`, `[7, 1] → 3`, `[1, 3] → 9`.

Common issues:

- Including a window with no next token (`[3, 9]` has nothing to predict).
- Off-by-one targets (predicting the last token *of* the window instead of the token after it).

## Exercise 06.03 — Why "self-supervised"

A correct answer should include:

- The labels come from the data itself: the target for each window is just the next token already sitting in the corpus — no human annotation step.
- The consequence: any raw text becomes training pairs automatically, which is what makes internet-scale pretraining possible.

Common issues:

- Calling it "unsupervised" — there *is* a supervised loss; the labels are just free.
- Missing the scale consequence entirely.

## Exercise 06.04 — `get_batch` shapes

A correct answer should include:

- `x: (4, 3)` and `y: (4,)` — a batch of context windows, and one scalar next-token target per window (not a window of targets).

Common issues:

- `y` as `(4, 3)` or `(4, 1)` — the target is the single token after each window.

## Exercise 06.05 — Counts for `[0, 1, 0, 1, 2]`

A correct answer should include:

- `counts[0, 1] = 2` (the pair `(0, 1)` occurs twice) and `counts[1, 0] = 1`.
- (Implicitly) rows index the previous token, columns the next.

Common issues:

- Counting unordered pairs so `counts[0, 1] = counts[1, 0] = 3`.
- Transposing the table (previous/next swapped).

## Exercise 06.06 — Why add-one smoothing works

A correct answer should include:

- The smoothing constant is added to *every* cell before row-normalizing, so even a never-seen pair gets probability `smoothing / rowsum > 0`.
- Why it matters: an unseen pair with probability 0 gives `log(0) = −inf`, which turns the whole perplexity into inf.

Common issues:

- Saying smoothing "fixes" the estimates rather than flooring them away from zero.
- Missing the `log(0) → −inf → inf perplexity` consequence.

## Exercise 06.07 — Why `logits` returns log-probabilities

A correct answer should include:

- Downstream consumers work in log space: cross-entropy and perplexity need log-probs, and raw counts are unnormalized.
- It keeps the interface interchangeable with the neural models' logits: CrossEntropyLoss applies log-softmax internally, and log-softmax of already-normalized log-probs is (numerically) the identity — so the same evaluation and sampling code serves all three models.

Common issues:

- "Logs avoid underflow" alone — true but not the interface reason the docstring stresses.
- Claiming CountsBigramLM returns unnormalized logits like the neural models (it returns genuine normalized log-probs).

## Exercise 06.08 — Why squeeze `(B, 1)` context

A correct answer should include:

- `get_batch` with `context_length=1` yields `(B, 1)`; embedding that directly gives `(B, 1, D)` and the projection `(B, 1, V)` — an extra dim that breaks the `(B, V)` logits contract every downstream helper assumes.
- Squeezing to `(B,)` makes embedding return `(B, D)` and the projection `(B, V)`.

Common issues:

- Saying the code would crash immediately — it mostly runs; the wrong-shaped logits break things downstream (loss, sampling).
- Squeezing the wrong axis in the explanation.

## Exercise 06.09 — Flattened hidden input width

A correct answer should include:

- `3 × 8 = 24` — context_length × embedding_dim, from flattening `(B, N, D)` to `(B, N·D)`.

Common issues:

- Answering 8 (forgetting the concat) or including hidden_dim in the arithmetic.

## Exercise 06.10 — Concat vs. average

A correct answer should include:

- Concatenation gives each context position its own slot in the flattened vector, so `[e_a, e_b] ≠ [e_b, e_a]` — order survives.
- Averaging (or summing) is permutation-invariant: different orderings collapse to the same vector, the bag-of-tokens failure. The notebook's `[3, 7]` vs `[7, 3]` logit difference is the direct evidence.

Common issues:

- Saying averaging "loses information" without naming order/permutation-invariance specifically.
- Not connecting to the demonstrated nonzero `max_difference` in the cell.

## Exercise 06.11 — Uniform model perplexity = V

A correct answer should include:

- A uniform model assigns every token probability `1/V`, so per-token cross-entropy is `−log(1/V) = log V`, and perplexity is `exp(log V) = V`.
- (Implicitly or explicitly) the branching-factor reading: it's exactly as uncertain as choosing among V options.

Common issues:

- Asserting the fact without the `exp(−log(1/V))` derivation.
- Mixing log bases inconsistently mid-derivation.

## Exercise 06.12 — Perfect model perplexity = 1

A correct answer should include:

- Probability 1 on the true next token every step gives loss `−log 1 = 0`, and `exp(0) = 1` — an effective branching factor of one option.

Common issues:

- Claiming perplexity 0 for the perfect model (1 is the floor).

## Exercise 06.13 — Why feed the sampled token back

A correct answer should include:

- The model only defines `P(next | context)`. To generate beyond one token, the sampled token must become part of the next context — predict, append, repeat.
- Without appending, every step would sample from the identical distribution — no sequence, just repeated draws from one conditional.

Common issues:

- Describing the loop mechanics without the "otherwise the distribution never changes" point.
- Confusing feeding back the *sampled token* with feeding back logits/probabilities.

## Exercise 06.14 — Why the same token stream

A correct answer should include:

- Perplexity depends on the tokenization: a different vocab changes both how many predictions there are and how hard each one is, so numbers aren't comparable across tokenizers.
- Fairness requires all three models to be trained and scored on the identical tokenized train/val streams — then differences reflect the models, not the encoding.

Common issues:

- Reducing it to "same data is fairer" without the vocab-size/perplexity mechanism.
- Believing perplexity is tokenizer-invariant in general (it's comparable only on the same token stream).

## Exercise 06.15 — Prediction: more context

A correct answer should include:

- Validation perplexity should drop: the MLP with context 3 should land below both bigrams, because more preceding tokens carry more information about the next one.
- (After running) reconciled against the plot: the MLP curve falls below the counts-bigram baseline line.

Common issues:

- Expecting the neural bigram to beat the counts bigram by a wide margin — same context, same distribution; they converge to roughly the same floor.
- Predicting unbounded improvement with context, ignoring data/capacity limits.

## Exercise 06.16 — Val perplexity stalls while train loss falls

A correct answer should include:

- The model is starting to fit training-stream specifics that don't transfer — memorization/overfitting; the generalizable signal at this capacity/context is exhausted, so held-out perplexity flatlines (or rises) while train loss keeps creeping down.

Common issues:

- Calling any flat val curve a bug rather than a generalization ceiling.
- Not distinguishing "stops improving" (plateau) from "gets worse" (overfitting proper) — either is acceptable if the mechanism is right.

## Exercise 06.17 — Most locally plausible sample

Run-dependent — grade the reasoning shape.

A correct answer should include:

- A named model — expected: the MLP (context 3) — with quoted evidence from their own samples: better word-like spellings, plausible short character sequences or word fragments relative to the bigrams.

Common issues:

- A verdict with no quoted sample.
- Judging by global coherence (none of them have it) instead of *local* plausibility.

## Exercise 06.18 — Failure modes in the best sample

A correct answer should include:

- Concrete observed failures: repetition loops (repeated words/fragments), locally fine but drifting text, made-up or fractured words, no sentence- or discourse-level grammar.

Common issues:

- "It's bad" with no specific failure mode cited from the sample.

## Exercise 06.19 — Why no long-range coherence

A correct answer should include:

- These models literally cannot see past their context window (1–3 tokens): anything earlier has zero influence on the next-token distribution.
- Coherent prose needs long-range dependencies, which requires a mechanism for wide context — the motivation for attention in the upcoming modules.

Common issues:

- Blaming under-training or model size alone — more steps cannot give a bigram a memory.
- Vague "they're too simple" without the hard context-window ceiling.

## Exercise 06.20 — Which curve is smoother

A correct answer should include:

- Validation perplexity is smoother: each point averages cross-entropy over the whole validation stream, while train loss is a single random 64-window minibatch per step — minibatch sampling noise makes it jagged.

Common issues:

- Picking train loss because it "has more points" — density isn't smoothness.
- Missing the minibatch-variance vs. full-set-average mechanism.

## Exercise 06.21 — Val perplexity rises while train loss falls

A correct answer should include:

- Overfitting: the model is memorizing the training stream at the expense of held-out prediction.
- (Optionally) a sensible response: stop earlier, add data, or regularize (weight decay).

Common issues:

- Reading it as a broken evaluation rather than the classic divergence signature.
- Confusing this with the plateau case (06.16) — rising val is the stronger signal.

## Exercise 06.22 — Testing whether more context helps

A correct answer should include:

- A controlled sweep: train MLPLanguageModel at several `context_length` values (e.g., 1, 3, 5, 8) with everything else held fixed — same token stream, vocab, embedding/hidden dims, steps, batch size — and compare validation perplexity.
- One knob at a time as the explicit design principle.

Common issues:

- Changing context length together with model size, steps, or tokenizer, confounding the comparison.
- Comparing on training loss instead of validation perplexity.
