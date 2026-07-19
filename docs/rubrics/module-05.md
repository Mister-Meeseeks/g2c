# Module 05 Rubric

Use this rubric to grade submitted answers in the student's working notebook (`notebooks/solutions/05-embeddings.ipynb`, falling back to `notebooks/clean/05-embeddings.ipynb` if no solutions copy exists yet). Each exercise has `Question:` / `Answer:` string cells; treat a blank `"Answer: "` as not submitted (skip it). If the student wrote a hint or help request inside the answer string, tutor first before grading and avoid giving away the full solution unless they explicitly ask. Give feedback exercise by exercise and avoid replacing the student's work with a full solution unless explicitly asked.

## Exercise 05.01 — Next failing test

Run-dependent — grade the reasoning shape.

A correct answer should include:

- A concrete test name from `pytest tests/test_embeddings.py -x` at the time they ran it, mapped to the implementation it points at (`TokenEmbedding.forward`, a positional table `__init__`, `RotaryEmbedding`, skip-gram helpers). "All passing" is a complete answer.

Common issues:

- Naming a file with no test, or a test with no target function.

## Exercise 05.02 — Shape of `weight[ids]`

A correct answer should include:

- `(B, T, C)` — indexing replaces each integer ID with its C-dim row, so the output is the ids shape with the embedding dim appended.

Common issues:

- `(B*T, C)` (flattening that indexing doesn't do).
- Involving V in the output shape.

## Exercise 05.03 — Why indexing, not matmul

A correct answer should include:

- The forward pass just selects rows of the table by integer ID.
- It's mathematically equivalent to `one_hot(ids) @ weight`, but the one-hot matmul wastes O(V) multiply-adds per token to select one row that indexing grabs directly.
- (Bonus) autograd handles indexing fine — gradients flow only to the rows that were touched.

Common issues:

- Missing the one-hot equivalence entirely (the answer should show the two are the *same* operation, one efficient).
- Claiming a matmul would give different values.

## Exercise 05.04 — Why `(T, C)` adds to `(B, T, C)`

A correct answer should include:

- Broadcasting: right-align the shapes, pad `(T, C)` to `(1, T, C)`, and the size-1 batch dim stretches across B.
- The effect: the same positional vectors are added to every sequence in the batch — which is exactly what positions should be.

Common issues:

- "PyTorch just handles it" with no alignment rule.
- Thinking the T dims are what broadcast (they match; the *padded batch dim* stretches).

## Exercise 05.05 — Position 0 of the sinusoidal table

A correct answer should include:

- Every sine slot holds `sin(0) = 0`; every cosine slot holds `cos(0) = 1` — the angle `pos · freq` is 0 for every frequency at position 0. (The notebook asserts exactly this on `weight[0, 0::2]` and `weight[0, 1::2]`.)

Common issues:

- Swapping the two (sines 1, cosines 0).
- Reasoning about which slots are even/odd instead of the angle being zero.

## Exercise 05.06 — Why even `embedding_dim`

A correct answer should include:

- The table is built in (sin, cos) pairs — one pair per frequency, `dim/2` frequencies. An odd dim would leave one slot without its partner, so the constructor raises `ValueError`.

Common issues:

- Vague "the math needs it" without the pairing structure.
- Claiming odd dims produce silently wrong values (the implementation refuses, by design).

## Exercise 05.07 — Why no learnable parameters

A correct answer should include:

- The table is fully determined by the formula — there's nothing to learn; it must be excluded from `parameters()` so the optimizer never updates it (in code: `requires_grad_(False)`).
- (Bonus) the formula evaluates at any position, so a fixed table extrapolates beyond trained lengths — a property learned tables lack.

Common issues:

- Saying learning the table would break it (learned positional tables exist — GPT-2; the point is this *scheme* is defined as fixed).
- No mention of keeping it out of the optimizer/`parameters()`.

## Exercise 05.08 — Split-halves pairing at dim 8

A correct answer should include:

- Dim `i` pairs with dim `d/2 + i`: (0, 4), (1, 5), (2, 6), (3, 7).

Common issues:

- Interleaved pairing (0,1), (2,3), (4,5), (6,7) — that's the original RoPE paper's convention, not this repo's split-halves `_rotate_half` convention.

## Exercise 05.09 — Why cos/sin tables are full-width

A correct answer should include:

- There are only `dim/2` distinct frequencies, but the implementation duplicates them — `emb = cat([freqs, freqs], dim=-1)` — so both members of each pair see the same angle.
- The payoff: the forward pass is a single element-wise `x * cos + rotate_half(x) * sin` across all dims, with no per-pair indexing or reshaping.

Common issues:

- Claiming there are `dim` distinct frequencies.
- Right shape, no connection to the element-wise forward-pass recipe.

## Exercise 05.10 — Contents of `cos[0]` and `sin[0]`

A correct answer should include:

- `cos[0]` is all ones, `sin[0]` is all zeros: the angle is `position × inv_freq = 0` for every frequency at position 0.
- Consequence: position 0 is the identity rotation — `RoPE(x, position=0) = x` (the notebook asserts both).

Common issues:

- Values swapped.
- Not connecting the values to the identity-rotation consequence when the question asks "why."

## Exercise 05.11 — Why rotation preserves L2 norm

A correct answer should include:

- Each pair `(a, b)` maps to `(a cos θ − b sin θ, a sin θ + b cos θ)`, whose squared norm is `(a² + b²)(cos²θ + sin²θ) = a² + b²` — rotations are orthogonal; they change direction, never length.

Common issues:

- Asserting norm preservation without the `sin² + cos² = 1` argument.
- Confusing preserving each vector's norm with preserving dot products between *differently*-positioned vectors (those change — that's the point).

## Exercise 05.12 — Why position 0 is the identity

A correct answer should include:

- The rotation angle is `m · θᵢ`, which is 0 at `m = 0`; `cos 0 = 1, sin 0 = 0`, so `x·1 + rotate_half(x)·0 = x`.

Common issues:

- Restating "cos[0]=1, sin[0]=0" without tying it back to the angle being `m · θᵢ`.

## Exercise 05.13 — Why relative position beats absolute

A correct answer should include:

- With RoPE, `q'·k' = qᵀR(n−m)k` — attention scores depend only on the offset `n − m`.
- Why that's what we want: linguistic relationships are about relative offsets ("the token two words back"), not absolute indices, so the same learned pattern works at every position in the sequence.
- (Contrast) additive absolute encodings leave `q·p_n + p_m·k + p_m·p_n` cross-terms that depend on absolute m and n.

Common issues:

- Restating the algebra with no linguistic/generalization payoff.
- Claiming absolute position information is useless (it's *less* useful, not useless).

## Exercise 05.14 — Skip-gram input vs. target

A correct answer should include:

- The center token is the input; a nearby context token (within the window) is the target — the model predicts context from center.

Common issues:

- Reversing the roles (that's CBOW's direction, not skip-gram's).

## Exercise 05.15 — Why co-occurrence shapes geometry

A correct answer should include:

- Tokens that appear in similar contexts must produce similar predicted context distributions; the only way the model can do that is by giving them similar embedding vectors, so co-occurrence statistics turn into geometric proximity.

Common issues:

- "Training makes similar words close" with no mechanism (shared prediction targets → shared vectors).
- Claiming the model is told about similarity — the geometry emerges from the objective alone.

## Exercise 05.16 — Structure TinyShakespeare should reveal

Run-dependent — grade the reasoning shape, citing their neighbor lists or 2D plot.

A correct answer should include:

- With a real corpus (~1MB, real vocabulary) the neighbor lists should show usage-based clusters — character names near names, function words together, morphologically/thematically similar words adjacent — versus a tiny hand-written corpus which lacks enough co-occurrence signal for any of it.
- At least one concrete observation from their own output.

Common issues:

- Expecting clean semantic analogies (that's Exercise 6's contrast, explicitly deferred).
- No cited neighbor or cluster at all.

## Exercise 05.17 — Why still weaker than pretrained vectors

A correct answer should include:

- Scale: TinyShakespeare is ~1MB / a single narrow domain vs. billions of diverse tokens for GloVe — orders of magnitude fewer co-occurrence observations per token.
- Consequence: coarse, noisy geometry; not enough signal for fine-grained semantic structure like analogies.

Common issues:

- Blaming the algorithm or embedding dim as the primary gap rather than corpus scale/diversity.
- "It trained for fewer steps" alone — more steps on the same tiny corpus wouldn't close the gap.

## Exercise 05.18 — The queen analogy expression

A correct answer should include:

- `king − man + woman ≈ queen`.

Common issues:

- Sign errors (`king + man − woman`).
- Writing it with the answer on the wrong side (expression should *produce* queen).

## Exercise 05.19 — Cosine vs. raw dot product

A correct answer should include:

- Raw dot product conflates direction with magnitude: frequent tokens tend to have larger norms, so they dominate nearest-neighbor lookups regardless of meaning.
- Cosine normalizes both vectors, comparing direction only — which is where the semantic signal lives.

Common issues:

- "Cosine is bounded to [−1, 1]" as the whole answer — boundedness isn't the reason; norm bias is.
- Claiming dot product is wrong rather than biased.

## Exercise 05.20 — Pretrained works, tiny doesn't

A correct answer should include:

- The analogy structure isn't a property of the algorithm but of the training signal: linear offset regularities only emerge with massive, diverse co-occurrence evidence.
- So the tiny model's failure indicts its corpus scale/coverage, not the method — same mechanism, insufficient data.

Common issues:

- Concluding the tiny implementation is buggy because the analogy fails.
- Restating "more data is better" without the point that analogy offsets specifically need broad evidence to line up.

## Exercise 05.21 — Reading the 2D GloVe plot

Run-dependent — grade the reasoning shape.

A correct answer should include:

- Specific clusters visible in their plot (e.g., royalty terms, country–capital groups, animal terms).
- The caveat that 50D → 2D projection destroys most structure: some analogy offsets won't look parallel and some clusters smear, even though they're real in 50D.

Common issues:

- Treating the 2D picture as the full geometry — concluding an offset "doesn't exist" because the projection hides it.
- No specific cluster named.

## Exercise 05.22 — Why the learned table looks noisy

A correct answer should include:

- It's randomly initialized and untrained here — its values are noise until gradient updates give positions structure; the plot is intentionally showing the pre-training state.

Common issues:

- Thinking the noise indicates a bug, or that learned tables stay unstructured after training.

## Exercise 05.23 — Which scheme has learnable parameters

A correct answer should include:

- Only the learned positional embedding. Sinusoidal and RoPE tables come from fixed formulas with no gradient updates.

Common issues:

- Counting RoPE's cos/sin buffers as parameters.

## Exercise 05.24 — Shared visual pattern

A correct answer should include:

- Both show multi-frequency banded/striped oscillation: values oscillate along the position axis, fast in the low-index (high-frequency) dimensions and slowly in the high-index (low-frequency) ones — because both tables are sin/cos of `position × exponentially-decaying frequency`.

Common issues:

- "They both look wavy" without the frequency-varies-by-dimension structure.
- Attributing the pattern to training (both are formula-fixed).

## Exercise 05.25 — Why RoPE lives on Q and K in attention

A correct answer should include:

- The relative-position property is a property of the *dot product*: `(R(m)q)·(R(n)k) = qᵀR(n−m)k`. To get it, the rotation must be applied to the two vectors being dotted — queries and keys, at the point of comparison inside attention.
- Rotating token embeddings (or adding RoPE like an additive scheme) would not make attention scores a function of `n − m`; the identity only holds when both sides of the score are position-rotated.

Common issues:

- "That's just where Llama puts it" with no dot-product argument.
- Saying values must also be rotated (they aren't — only the score computation needs the property).
