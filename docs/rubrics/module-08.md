# Module 08 Rubric

Use this rubric to grade submitted answers in the student's working notebook (`notebooks/solutions/08-multi-head-attention.ipynb`, falling back to `notebooks/clean/08-multi-head-attention.ipynb` if no solutions copy exists yet). Each exercise has `Question:` / `Answer:` string cells; treat a blank `"Answer: "` as not submitted (skip it). If the student wrote a hint or help request inside the answer string, tutor first before grading and avoid giving away the full solution unless they explicitly ask. Give feedback exercise by exercise and avoid replacing the student's work with a full solution unless explicitly asked.

## Exercise 08.01 — Next failing multi-head test

A correct answer should include:

- An actual test named from `tests/test_multi_head_attention.py` that was failing in their run (run-dependent), mapped to the implementation detail it pins — e.g., the `view + transpose(1, 2)` reshape, the `√head_dim` scaling, `attention_weights` returning `(B, H, T, T)`, `.contiguous()` before the head-merging view, or the mask broadcasting over `(B, H, T, T)` scores.
- If the suite is already green, "none — all passing" is a valid answer; grade the test-to-contract mapping.

Common issues:

- Naming a test without saying what implementation detail it points at.
- Treating the construction/`causal_mask`/parameter-count tests (passing from the start) as work still to do.

## Exercise 08.02 — head_dim with D=4, H=2

A correct answer should include:

- `head_dim = D / H = 4 / 2 = 2`.

Common issues:

- Treating `head_dim` as a free hyperparameter rather than `D // H`.
- Forgetting the divisibility constraint (`embedding_dim` not divisible by `num_heads` is a constructor `ValueError`).

## Exercise 08.03 — Why the scale is sqrt(head_dim), not sqrt(D)

A correct answer should include:

- Each per-head dot product is over `head_dim`-dimensional vectors — the score variance grows with the dot-product length, which is `head_dim`, not `D`.
- (Ideally) the failure mode of using `sqrt(D)`: scores are under-scaled by a factor of `sqrt(H)`, the softmax sits in a too-flat, high-temperature regime, and training is sluggish without crashing.

Common issues:

- Justifying the scale by "that's the formula" without the variance-of-the-dot-product argument.
- Getting the direction wrong — dividing by the *larger* `sqrt(D)` makes attention too flat (under-confident), not too sharp.
- Saying using `sqrt(D)` crashes or breaks shapes — it's a silent quality bug, which is why it's the most common multi-head mistake.

## Exercise 08.04 — Which channels head 0 receives

A correct answer should include:

- The contiguous channel slice `[0, 1]` — head `h` gets channels `[h·head_dim, (h+1)·head_dim)`; in the notebook's `arange` demo that is the values `[0.0, 1.0]`.

Common issues:

- Answering with the head's *shape* instead of *which channels* it receives.
- An answer that contradicts the printed canonical-layout output in the same cell.

## Exercise 08.05 — Why a wrong reshape preserves shapes but changes the model

A correct answer should include:

- `view(B, T, head_dim, H)` produces tensors with valid-looking shapes, but the channel-to-head assignment changes — heads see different (interleaved) slices of the embedding. Shape checks can't catch it; the model silently computes something different.
- (Implicitly or explicitly) `view` semantics assign memory order to axes — reinterpreting the same memory with axes in a different order reassigns which numbers land in which head.

Common issues:

- Believing that if the shapes match, the computation must match.
- Thinking the wrong reshape raises an error — the failure is silent, which is the point.

## Exercise 08.06 — Why fixed-D comparison is fairer

A correct answer should include:

- Fixing `D = 64` while varying `H` keeps the attention parameter count identical across `H = 1, 4, 8`, so any validation-curve difference is attributable to the *structure* (number of parallel attention patterns), not to a bigger parameter budget.
- Changing `D` and `H` together confounds capacity with head count.

Common issues:

- Saying more heads means more parameters (the parameter-count invariance is the premise of the experiment).
- Framing "fairness" as equal compute or wall-clock rather than equal parameters.

## Exercise 08.07 — Failure mode when head_dim is too small

A correct answer should include:

- With large `H` at fixed `D`, each head's Q/K subspace is too low-dimensional to represent enough distinctions — per-head attention becomes coarse/noisy and quality can degrade despite the same total parameters.

Common issues:

- Framing the tradeoff as compute cost rather than per-head expressiveness.
- Expecting a clean monotone "more heads is better" result from one tiny run.

## Exercise 08.08 — Spotting a previous-token head

A correct answer should include:

- A previous-token head would appear as a bright sub-diagonal stripe: nearly all of each row's mass on position `t-1`. Whether one appears in their run is run-dependent — grade the description of the pattern, not the presence of the head.

Common issues:

- Describing the causal-mask triangle as if it were a learned specialization.
- Confusing sub-diagonal (previous token) with the main diagonal (self-attention).

## Exercise 08.09 — Why averaging over heads hides the point

A correct answer should include:

- Averaging would blur distinct per-head patterns into a near-uniform mush; the entire point of the visualization (and of `attention_weights` returning `(B, H, T, T)` rather than `(B, T, T)`) is that heads differ.

Common issues:

- Saying averaging is numerically wrong rather than that it destroys the per-head visualization.
- Concluding heads "failed" because individual patterns are partial or noisy at this tiny scale.

## Exercise 08.10 — Which head deviates most from causal-uniform

A correct answer should include:

- A specific head cited from their baseline-adjusted plot as deviating most, with the pattern described (where the red/blue mass sits relative to the `1/(t+1)` causal-uniform baseline). The exact head index is run-dependent; grade the cited evidence.

Common issues:

- Reading the raw heatmap (dominated by the mask) instead of the baseline-subtracted one.
- No reference to their own plots at all.

## Exercise 08.11 — Nearly-white adjusted heatmaps

A correct answer should include:

- If all adjusted heatmaps are nearly white, the heads are all behaving close to uniform-over-allowed-positions — i.e., little head specialization has emerged, which is an expected outcome for a tiny model with a short training budget, not a bug.

Common issues:

- Treating near-white heatmaps as evidence the implementation is broken.
- Over-claiming specialization from faint noise in an otherwise white plot.

## Exercise 08.12 — Same parameters, different behavior

A correct answer should include:

- The parameters are the same four `(D, D)` projections regardless of `H`; what changes is how the projection output is *interpreted*: `H` independent softmax attention distributions over disjoint `head_dim` subspaces instead of one distribution over all of `D`.
- The behavioral consequence: more heads means more distinct attention patterns computed in parallel per layer (room to specialize), at the cost of lower-dimensional similarity per head — a structural/expressiveness difference, not a capacity difference.

Common issues:

- Falling back to "more heads, more parameters" — directly contradicted by the parameter-count table in the same exercise.
- Saying the difference is only speed/implementation, missing that H changes which functions the layer can express per-pattern.
- Not mentioning softmax at all — the per-head softmax is what makes the H patterns genuinely independent distributions.
