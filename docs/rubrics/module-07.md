# Module 07 Rubric

Use this rubric to grade submitted answers in the student's working notebook (`notebooks/solutions/07-attention.ipynb`, falling back to `notebooks/clean/07-attention.ipynb` if no solutions copy exists yet). Each exercise has `Question:` / `Answer:` string cells; treat a blank `"Answer: "` as not submitted (skip it). If the student wrote a hint or help request inside the answer string, tutor first before grading and avoid giving away the full solution unless they explicitly ask. Give feedback exercise by exercise and avoid replacing the student's work with a full solution unless explicitly asked.

## Exercise 07.01 — Next failing attention test

A correct answer should include:

- An actual test named from `tests/test_attention.py` that was failing in their run (run-dependent), mapped to the implementation detail it pins — e.g., a `forward` test pointing at the `1/√D` scaling, mask-before-softmax, `softmax(dim=-1)`, or the output projection; an `attention_weights` test pointing at sharing the same score/mask path as `forward`.
- If the suite is already green, "none — all passing" is a valid answer; grade the test-to-contract mapping, not which test they picked.

Common issues:

- Naming a test without saying what implementation detail it points at.
- Treating the construction/`causal_mask` tests (passing from the start) as work still to do — they cover implemented boilerplate.

## Exercise 07.02 — Shape of Q @ K.T

A correct answer should include:

- `(1, 3, 3)` — one score per (query position, key position) pair; the score matrix is `(T, T)` regardless of `D`.

Common issues:

- Answering `(1, 3, 2)` or `(3, 2)` — confusing the score matrix with Q/K/V shapes.
- Dropping the batch dim, or transposing the wrong dims (a 3D tensor needs `transpose(-2, -1)`, not `.T`).

## Exercise 07.03 — Softmax dimension

A correct answer should include:

- Softmax normalizes over the last dim (`dim=-1`, the key positions): each query row must become a probability distribution over the positions it attends to, summing to 1.

Common issues:

- Saying `dim=-2` (normalizing over query positions per key) — a different, wrong computation; symptom: rows no longer sum to 1.
- Justifying with "last dim by convention" without the rows-are-distributions reason.

## Exercise 07.04 — Why causal=False eases the hand check

A correct answer should include:

- With `causal=False` there is no `-inf` masking step to replicate — every row is a plain softmax over all three dot products, so the manual pipeline (project, score, scale, softmax, mix, project) matches `forward` exactly.

Common issues:

- Explaining `causal=False` as "makes the model better" — the question is about the manual computation, not model quality.
- Thinking the mask changes tensor shapes (it changes values, not shapes).

## Exercise 07.05 — Why future-token attention is zero before training

A correct answer should include:

- The causal mask is structural, not learned: `-inf` is written above the diagonal before softmax, so those entries are exactly 0 no matter what the (random, untrained) weights are.

Common issues:

- Attributing the zeroed future weights to training or to the data rather than to the mask.
- Saying the mask is applied after softmax (it must be `-inf` before softmax so masked weights are exact zeros and rows still sum to 1).

## Exercise 07.06 — Why random attention can't resolve "it"

A correct answer should include:

- The Q/K projections are untrained — the scores are arbitrary similarities of random projections with no linguistic structure; coreference patterns only come from training.

Common issues:

- Reading random-looking weights as a bug — this plot is the negative control; only the mask structure should be visible.
- Confusing the masked-zero structure (which does appear) with semantic structure (which shouldn't yet).

## Exercise 07.07 — What to hope for after training

A correct answer should include:

- Attention becoming structured and context-dependent inside the allowed prefix — e.g., "it" placing more mass on "animal" vs "street" depending on "tired"/"wide" — instead of near-uniform noise.

Common issues:

- Expecting the tiny probes in this course to actually show clean coreference heads (the answer is a hope, not a promised observation).
- Describing only "sharper" without the content-dependence that makes the pattern meaningful.

## Exercise 07.08 — Before vs after training attention maps

A correct answer should include:

- Training makes attention sharper: rows concentrate mass on fewer positions (lower entropy) and deviate further from the causal-uniform baseline (`1/(t+1)` over allowed positions), instead of staying near-uniform.

Common issues:

- Describing the change only as "the numbers changed" without the uniform-vs-concentrated framing.
- Not referencing their own before/after plots at all.

## Exercise 07.09 — Sharper attention is not semantic understanding

A correct answer should include:

- Concentration can reflect positional or frequency statistics (e.g., previous-token bias), and an attention map is correlational — claims about "understanding" need behavioral evidence, not a heatmap from one tiny head.

Common issues:

- Reading a sharp trained head as proof the model "knows grammar" or "understands the sentence."
- Treating the heatmap as mechanistic evidence rather than a visualization of one layer's mixing weights.

## Exercise 07.10 — Trained ShakespeareLM map still not proof of semantics

A correct answer should include:

- The same epistemic point as 07.09 at higher fidelity: a trained map shows attention moved away from the random near-uniform baseline; it does not show *why* — sharp patterns can encode positional/statistical regularities, and semantic claims require behavioral tests.

Common issues:

- Treating one interpretable-looking head in a tiny model as established mechanistic evidence.
- Answering "because the model is small" alone, without the correlation-vs-behavior argument (the point survives at any scale).

## Exercise 07.11 — Why the revisit belongs after Module 10

A correct answer should include:

- It needs the saved `ShakespeareLM` artifact, which only exists once Module 10's training run has produced and saved it; on the first Module 07 pass there is no trained full model and the cell safely skips.

Common issues:

- Answering "because Module 10 is harder" instead of the artifact dependency.
- Proposing to train the full model inside Module 07 rather than reusing the saved artifact.

## Exercise 07.12 — Why the non-causal model can see the answer

A correct answer should include:

- The objective is sequence-wide: input positions `0..T-1` predict targets `1..T`, so row `t`'s target is the token sitting at input position `t+1`. With `causal=False`, position `t` can attend directly to that position and copy it — the answer is inside the input window.
- (Ideally) the observed collapse: the non-causal loss falls far below the causal one, toward ~0, while the causal model on this *random* token stream stays near the `log(8) ≈ 2.08` baseline (random data is genuinely unpredictable without leakage).

Common issues:

- Saying the non-causal model is "smarter" or "learned faster" rather than that it cheats via leakage.
- Missing why the causal model *can't* beat the baseline here (the stream is random, so honest next-token prediction has nothing to learn).

## Exercise 07.13 — Why a perfect training curve is a red flag

A correct answer should include:

- Loss below the data's entropy means the model can see its target (leakage), not that it learned — this bug silently looks like training success.
- (Ideally) the tie to the mask: the standard all-positions LM training shape is exactly where a missing causal mask produces this failure.

Common issues:

- Treating any low loss as good news without asking whether the data could honestly support it.
- Failing to name leakage as the mechanism behind the "too good" curve.

## Exercise 07.14 — Why parameter count doesn't depend on T

A correct answer should include:

- The parameters are the four `Linear(D, D)` projections — `4·(D² + D)` — and the mixing weights over positions are *computed* from dot products at run time, not stored per position; the same parameters serve any sequence length.

Common issues:

- Forgetting the bias terms or counting only three projections.
- Thinking there are learned per-position attention weights (that's Module 06's MLP limitation, which attention removes).

## Exercise 07.15 — What still gets expensive as T grows

A correct answer should include:

- The `(T, T)` score/attention matrix — `O(T²)` memory for activations and `O(T²·D)` compute per forward pass.

Common issues:

- Confusing parameter count with activation memory/compute (this question pair is exactly that distinction).
- Saying attention is "cheap" because parameters don't grow — the quadratic cost is the famous bottleneck.

## Exercise 07.16 — Why the log-log plot bends toward slope 2

A correct answer should include:

- Wall time is asymptotically dominated by the `O(T²)` score and mixing matmuls, so `time ≈ c·T²`; on log-log axes `log(time) = 2·log(T) + const` — a straight line of slope 2.

Common issues:

- Saying the curve "is always slope 2" — the bend *toward* slope 2 as T grows is the point.
- Attributing the slope to parameter growth (parameters stay fixed; 07.14 established this).

## Exercise 07.17 — Why small T doesn't show a clean quadratic

A correct answer should include:

- At small `T` other terms dominate: fixed per-call overhead (Python/framework dispatch, kernel launch), the `O(T·D²)` projection matmuls which are only linear in `T`, and timing noise on sub-millisecond runs.

Common issues:

- Attributing small-T flatness only to "noise" without mentioning fixed overheads or the linear-in-T projection cost.
- Concluding from small-T data that the quadratic claim is wrong.
