# Module 09B Rubric

Use this rubric to grade submitted answers in `answers/module-09b.md`. If `Help request / hint request` is filled, tutor before grading and avoid giving away the full solution unless explicitly asked. Blank `Student answer` sections are not wrong; skip them unless the student asks for a completeness check. Give feedback by exercise and avoid replacing the student's work with a full solution unless explicitly asked.

## Exercise 09B.01 — Shift a toy stream by hand

A correct answer should include:

- `x = [11, 12, 13]`.
- `y = [12, 13, 14]`.
- `y[t]` is the token immediately after `x[t]` in the original stream.

Common issues:

- Starting at index 0 instead of index 1.
- Making `y` equal to `x`.
- Treating only the last token as the target.

## Exercise 09B.02 — Explain why the causal mask matters

A correct answer should include:

- Without a causal mask, position `t` can attend to token `t+1`.
- If it can see the target token, next-token prediction becomes cheating.
- The loss would collapse for the wrong reason and would not teach autoregressive generation.

Common issues:

- Saying the mask is only for speed.
- Confusing padding masks with causal masks.
- Saying every token can only see itself.

## Exercise 09B.03 — Inspect `get_lm_batch`

A correct answer should include:

- In the `torch.arange` toy stream, adjacent corpus tokens differ by 1.
- Therefore every `y` value should be the matching `x` value plus 1.
- This makes off-by-one errors visible.

Common issues:

- Checking only shapes and not the shifted values.
- Assuming the relationship holds for arbitrary token streams.

## Exercise 09B.04 — Implement `lm_cross_entropy`

A correct answer should include:

- `pytest tests/test_pretraining_setup.py -k lm_cross_entropy` passes.
- The implementation reshapes logits from `(B, T, V)` to `(B*T, V)`.
- The implementation reshapes targets from `(B, T)` to `(B*T,)`.
- The implementation reuses `CrossEntropyLoss`.

Common issues:

- Computing loss at only the final position.
- Summing instead of averaging.
- Flattening logits and targets in inconsistent orders.

## Exercise 09B.05 — Flatten shapes explicitly

A correct answer should include:

- `logits` shape is `(2, 4, 7)`.
- `targets` shape is `(2, 4)`.
- `flat_logits` shape is `(8, 7)`.
- `flat_targets` shape is `(8,)`.
- The batch contains `2 * 4 = 8` classification examples.

Common issues:

- Answering `(B, V)` as if there is one prediction per sequence.
- Losing the vocabulary dimension during flattening.

## Exercise 09B.06 — Compute the baseline

A correct answer should include:

- `log(256) ~= 5.545`, perplexity `256`.
- `log(512) ~= 6.238`, perplexity `512`.
- `log(1024) ~= 6.931`, perplexity `1024`.
- Larger vocabularies have larger uniform-loss baselines because there are more possible classes.
- A larger vocab can still be useful if it reduces sequence length or produces better token units.

Common issues:

- Treating higher `log(V)` as automatically worse.
- Using base-10 log without saying so.

## Exercise 09B.07 — Random model sanity check

A correct answer should include:

- Random-init loss should be near `log(V)`, not near zero.
- Possible explanations for a large mismatch include a shape bug, wrong targets, trivial repeated data, non-random loaded weights, or broken model output.
- The check is a sanity test before a longer Module 10 run.

Common issues:

- Expecting a random model to produce good loss.
- Diagnosing every mismatch as optimizer failure before checking data/loss shapes.
