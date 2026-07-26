# Module 16B Rubric

Use this rubric to grade submitted answers in the student's working notebook (`notebooks/solutions/16b-synthetic-data.ipynb`, falling back to `notebooks/clean/16b-synthetic-data.ipynb` if no solutions copy exists yet). Each exercise has `Question:` / `Answer:` string cells; treat a blank `"Answer: "` as not submitted (skip it). If the student wrote a hint or help request inside the answer string, tutor first before grading and avoid giving away the full solution unless they explicitly ask. Give feedback exercise by exercise and avoid replacing the student's work with a full solution unless explicitly asked.

Note: answers here depend on the student's own generation run — grade the *reading* of their evidence, not agreement with any fixed expected output.

## Exercise 16B.01 — Read one batch

A correct answer should include:

- An editor's count: how many of the eight proposals are genuinely new task ideas versus recombinations/rephrasings of the seeds (any honest count is fine; the grading target is that they distinguished the two).
- At least one quoted good proposal and one quoted reject, with a reason for each.
- Awareness of the answerable-by-a-small-model criterion — flagging proposals needing outside context, long outputs, or capabilities a small model lacks.

Common issues:

- Grading grammar/fluency instead of novelty and trainability.
- No quotes from their own batch.
- Treating "on-topic with my seeds" as automatically good — topical clustering is exactly the collapse the next exercise measures.

## Exercise 16B.02 — Mode collapse as a number

A correct answer should include:

- Their per-batch duplicate rates reported, and the (typical) observation that the rate climbs as the pool grows — each batch competes with everything already proposed.
- The mechanism: the teacher's proposal distribution has a few modes; as the pool fills them, fresh samples land on already-seen ideas more often.
- For the fixed-few-shot counterfactual: anchored examples narrow the proposal distribution further, so duplicate rates would climb faster / diversity would drop.

Common issues:

- Reporting the aggregate rate but not the trend.
- Attributing duplicates to randomness or "temperature too low" rather than to the shape of the proposal distribution.
- Missing that the measurement itself depends on their `ngram_overlap` threshold (0.5 here) — strong answers note the number moves with the yardstick.

## Exercise 16B.03 — Audit ten pairs

A correct answer should include:

- A per-pair verdict count (e.g., "7 clean, 2 wrong facts, 1 style drift") and an extrapolated error-rate estimate for the dataset.
- At least one named failure mode with the offending pair quoted or paraphrased.
- An explicit ship/no-ship call with a reason tied to the intended use (SFT data for a small model), not to perfection.

Common issues:

- Auditing shape (which the filters already gated) instead of *sense* — factual correctness, answerability, instruction/answer mismatch.
- No extrapolation from sample to dataset.
- Refusing to make the vendor call.

## Exercise 16B.04 — The three-way table

A correct answer should include:

- The winner on held-out hand-val loss named, with the actual numbers, and a sample-quality reading that either corroborates or disputes the loss ranking (disagreement is a fine finding if stated).
- An explicit answer to the LIMA question as measured: did ~3× synthetic beat the hand set on *hand-authored* evaluation?
- A reading of the mixed run: if mixed ≈ best single source, the sources are largely redundant; if mixed wins, they're complementary — either conclusion accepted with the numbers behind it.
- (Implicitly or explicitly) why the referee is hand data: evaluating on teacher-written data would grade teacher-style imitation, not human intent.

Common issues:

- Reading a 0.01 val-loss gap as a decisive result.
- Comparing datasets of different sizes without acknowledging the size confound (strong answers mention steps/epochs-per-example).
- Forgetting the referee rationale when the question or discussion touches it.

## Exercise 16B.05 — Style imprinting (optional)

A correct answer should include:

- At least one concrete teacher fingerprint cited from the synthetic-trained model's outputs — an opener, hedge, length habit, or formatting tic absent from their hand data.
- Ideally traced to a synthetic training pair containing the same habit (the causal chain: teacher → data → student).

Common issues:

- Citing a difference that also exists in their hand data.
- Stopping at the length statistics without quoting actual phrasing.

## Exercise 16B.06 — Name what you did

A correct answer should include:

- Sequence-level (hard-label) distillation performed: teacher-written text trained into the student — Kim & Rush's sense, and the modern colloquial meaning.
- Classic soft-logit distillation not performed, with at least one of the two structural blockers named precisely: ProdLM's API returns text, not logits/distributions; and BaseLM vs. course models have mismatched vocabularies, so a KL over the same token set is undefined.
- The callback: the logits-are-server-side limitation is the same one behind Module 18's `format: "json"` (and Module 16's text-only `Backend` interface).

Common issues:

- Vague "we didn't have access to the model internals" without naming logits/the distribution.
- Claiming classic distillation is merely expensive here rather than structurally blocked.
- Missing the callback the question explicitly asks for.
