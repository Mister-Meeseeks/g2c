# Module 02 Rubric

Use this rubric to grade submitted answers in the student's working notebook (`notebooks/solutions/02-tensors.ipynb`, falling back to `notebooks/clean/02-tensors.ipynb` if no solutions copy exists yet). Each exercise has `Question:` / `Answer:` string cells; treat a blank `"Answer: "` as not submitted (skip it). If the student wrote a hint or help request inside the answer string, tutor first before grading and avoid giving away the full solution unless they explicitly ask. Give feedback exercise by exercise and avoid replacing the student's work with a full solution unless explicitly asked.

Note: several cells batch multiple `Question:` strings, and the Exercise 6 shape-prediction questions sit below a `batch = 4` header in their cell — grade every `Question:` string, not just cells that open with one.

## Exercise 02.01 — Shape of `A @ B`

A correct answer should include:

- `(2, 4)`.
- The inner dims (3 and 3) match and collapse; the outer dims survive.

Common issues:

- Reporting `(3, 3)` (confusing which dims collapse).

## Exercise 02.02 — Shape of `B @ A`

A correct answer should include:

- Invalid: `(3, 4) @ (2, 3)` fails because the inner dims are 4 and 2.
- Names the mismatched dimensions (4 vs. 2), per the exercise instruction.

Common issues:

- Assuming matmul commutes because `A @ B` was valid.
- Marking it invalid but blaming the wrong pair of dims.

## Exercise 02.03 — Shape of `x @ W + b`

A correct answer should include:

- `(5, 6)`: `(5, 2) @ (2, 6) = (5, 6)`, then `b` of shape `(6,)` broadcasts.
- How the broadcast works: `(6,)` right-aligns against `(5, 6)`, pads to `(1, 6)`, and the size-1 dim stretches across the 5 rows — the bias is added to every row.

Common issues:

- Correct shape but no account of how `(6,)` broadcasts.
- Saying `b` is "copied" 5 times in memory (broadcasting re-reads, it doesn't copy).

## Exercise 02.04 — Shape of `x @ W + b_bad`

A correct answer should include:

- Invalid: `(5, 6) + (5,)` right-aligns 6 against 5; neither is 1, so broadcasting errors.
- The intuition trap named: `(5,)` looks like it matches the batch dim, but broadcasting aligns by the *last* dim, not the first.

Common issues:

- Declaring it valid because "5 matches the number of rows."
- Marking it invalid at the matmul step (the matmul is fine; the addition fails).

## Exercise 02.05 — Benchmark ordering vs. prediction

Machine-dependent — grade the reasoning shape, not specific timings.

A correct answer should include:

- An honest comparison of predicted vs. observed ordering, citing the plot or timing rows.
- The broadly expected picture: Python loops slowest by orders of magnitude; NumPy and torch CPU comparable; MPS fastest at large sizes but not necessarily at small ones.
- If the prediction missed, a mechanism for the surprise (e.g., MPS kernel-launch overhead at small `n`).

Common issues:

- Restating the prediction without citing any observed evidence.
- Treating a small-size MPS loss as "MPS is slower than CPU" in general.

## Exercise 02.06 — Best constant factor

A correct answer should include:

- The understanding that all implementations do the same O(n³) work — "constant factor" means a lower curve at the same size, i.e. a vertical offset between roughly parallel lines on the log-log plot.
- A specific winner from the student's own run, with the plot offset (or timing rows) as evidence.

Common issues:

- Confusing constant factor with asymptotic complexity ("NumPy is O(n²)").
- Naming a winner with no cited evidence.

## Exercise 02.07 — Where MPS beats CPU

Machine-dependent — grade the reasoning shape.

A correct answer should include:

- A crossover size read off their own plot (typically somewhere around n ≥ 512 per the lesson), or an honest "it didn't."
- The mechanism either way: at small sizes per-call kernel-launch overhead dominates and CPU wins; at large sizes the GPU's throughput wins.

Common issues:

- No mechanism, just a size.
- Ignoring that the benchmark warms up and synchronizes MPS — attributing the small-size loss to "MPS being broken."

## Exercise 02.08 — Why Python-loop matmul is unusable

A correct answer should include:

- Matmul is O(n³) operations, and the loop version pays per-element Python interpreter overhead (dynamic dispatch, boxed floats) on every one of them.
- NumPy/PyTorch execute the same O(n³) flops in compiled, vectorized native code (BLAS: SIMD, cache blocking, parallelism), so the per-operation cost is thousands of times smaller.
- The cubic growth is what makes the constant-factor gap fatal — doubling `n` is 8× the work.

Common issues:

- "Python is slow" with no per-element-overhead mechanism.
- Claiming NumPy has better asymptotic complexity (it's the same O(n³); the constant differs).

## Exercise 02.09 — Shape of `example_a + example_b`

A correct answer should include:

- `(2, 3)`: `(3,)` right-aligns with `(2, 3)`, pads to `(1, 3)`, and the padded leading dim broadcasts from 1 to 2.
- In words: the vector is added to each row of the matrix.

Common issues:

- Saying the last dim (3) is "the one that broadcasts" — the 3s *match*; the stretched dim is the padded size-1 leading dim.

## Exercise 02.10 — Predicted nested list

A correct answer should include:

- `[[11, 22, 33], [14, 25, 36]]` — rows `[1, 2, 3]` and `[4, 5, 6]` each plus `[10, 20, 30]`.

Common issues:

- Adding the vector to columns instead of rows.
- Misreading the flat data layout (row-major: `[1,2,3]` then `[4,5,6]`).

## Exercise 02.11 — What happened to `naive_large`

A correct answer should include:

- The output is NaN: `exp(1000)` overflows float32 to `inf` (float32 `exp` overflows for arguments ≳ 88), so the division becomes `inf / inf = NaN`.

Common issues:

- Saying it overflowed without following through to why the *result* is NaN rather than inf.
- Blaming the sum rather than the individual `exp` calls.

## Exercise 02.12 — Why subtracting the max avoids overflow

A correct answer should include:

- After subtracting the row max, every logit is ≤ 0, so every `exp` lands in `(0, 1]` — the largest becomes exactly `exp(0) = 1`. Nothing can overflow.
- (Bonus, not required) very negative shifted logits underflow to 0, which is harmless.

Common issues:

- Saying the subtraction "makes the numbers smaller" without the ≤ 0 / `(0, 1]` bound.
- Claiming it changes the resulting probabilities (that's the next question — it doesn't).

## Exercise 02.13 — Why the shift preserves probabilities

A correct answer should include:

- The algebra: `exp(x_i − c) / Σ_j exp(x_j − c) = e^{−c}·exp(x_i) / (e^{−c}·Σ_j exp(x_j))` — the `e^{−c}` factors cancel, so softmax is shift-invariant.

Common issues:

- Asserting shift invariance without the cancellation argument.
- Confusing shift invariance (true) with scale invariance (false — multiplying logits changes softmax).

## Exercise 02.14 — Predicted shape of `x`

A correct answer should include:

- `(4, 3)` — `(batch, in_features)`.

Common issues:

- Transposing to `(3, 4)`.

## Exercise 02.15 — Predicted shape of `W`

A correct answer should include:

- `(3, 5)` — this module's convention is `W: (in_features, out_features)` so that `x @ W` works directly.

Common issues:

- `(5, 3)` from the `torch.nn.Linear` `(out, in)` convention — wrong under this module's `x @ W + b` convention.

## Exercise 02.16 — Predicted shape of `b` and its broadcast

A correct answer should include:

- `(5,)`; it right-aligns against the `(4, 5)` matmul result, pads to `(1, 5)`, and stretches across the batch dim — one bias per class, added to every row.

Common issues:

- `(4,)` (a per-example bias) — repeats the Exercise 02.04 trap.

## Exercise 02.17 — Predicted shape of `logits`

A correct answer should include:

- `(4, 5)` — `(batch, num_classes)`.

Common issues:

- Carrying a wrong `W` prediction through without re-deriving.

## Exercise 02.18 — Predicted shape of `probs`

A correct answer should include:

- `(4, 5)` — softmax along `dim=-1` normalizes each row but never changes shape.

Common issues:

- Predicting a reduced shape like `(4,)` (confusing softmax with argmax or a reduction).

## Exercise 02.19 — Which helper produces logits vs. probabilities

A correct answer should include:

- `linear` produces the logits (`x @ W + b`); `softmax` turns each logits row into probabilities.

Common issues:

- Reversing the two, or inserting a nonlinearity that isn't in `classifier_forward`.

## Exercise 02.20 — Observed shapes vs. predictions

A correct answer should include:

- An honest comparison against the printed `shape_report`, and for any miss, the specific convention that caused it (usually `W` as `(in, out)` vs. `(out, in)`).

Common issues:

- "Yes they matched" with earlier predictions left visibly wrong and unreconciled.

## Exercise 02.21 — Why each row of `probs` sums to 1

A correct answer should include:

- Softmax along `dim=-1` divides each row's exponentials by that same row's sum, so each row is normalized by construction.
- One row = one batch example's probability distribution over the 5 classes.

Common issues:

- Saying the *columns* sum to 1, or that the whole matrix sums to 1.
- Restating "softmax makes probabilities" without the row-wise normalization mechanism.

## Exercise 02.22 — Why there's no training step

A correct answer should include:

- This module's subject is the vectorized forward pass (matmul, broadcasting, stable softmax); the weights are random and there are no labels or loss here.
- Training — loss, backward, parameter update — is Module 03's job, wiring Module 01's gradients to Module 02's tensor ops.

Common issues:

- Claiming the random classifier is somehow already trained or meaningful.
- Saying training is impossible here rather than deliberately out of scope.
