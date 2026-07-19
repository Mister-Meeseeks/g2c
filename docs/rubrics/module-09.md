# Module 09 Rubric

Use this rubric to grade submitted answers in the student's working notebook (`notebooks/solutions/09-transformer-block.ipynb`, falling back to `notebooks/clean/09-transformer-block.ipynb` if no solutions copy exists yet). Each exercise has `Question:` / `Answer:` string cells; treat a blank `"Answer: "` as not submitted (skip it). If the student wrote a hint or help request inside the answer string, tutor first before grading and avoid giving away the full solution unless they explicitly ask. Give feedback exercise by exercise and avoid replacing the student's work with a full solution unless explicitly asked.

## Exercise 09.01 — Next failing transformer test

A correct answer should include:

- An actual test named from `tests/test_transformer.py` that was failing in their run (run-dependent), mapped to the exact contract it pins — e.g., a LayerNorm test pointing at last-dim statistics, `unbiased=False`, or the `eps`/constant-input guard; an FFN test pointing at per-position independence; a Block test pointing at the pre-norm residual order `x = x + sublayer(LN(x))`; a TransformerLM test pointing at `(B, T, V)` logits, the final LayerNorm, or the tied head.
- If the suite is already green, "none — all passing" is a valid answer; grade the test-to-contract mapping.

Common issues:

- Naming a test without stating the contract it points at.
- Treating the 22 initially-passing construction/parameter-count/init tests as work still to do.

## Exercise 09.02 — How many means LayerNorm computes

A correct answer should include:

- For input `(2, 3, 8)`: 6 separate means — one per `(batch, position)` pair; statistics pool over the 8-channel dim only, never across batch or sequence. (This per-token pooling is why batch size never affects LN output.)

Common issues:

- Answering 2, 8, or 48 means — pooling over the wrong dims.
- Describing BatchNorm-style per-channel statistics pooled over the batch.

## Exercise 09.03 — Why unbiased=False

A correct answer should include:

- `unbiased=False` is the population variance (divide by `N = D`, not `D - 1`): it matches the standard LayerNorm definition, so a normalized vector has variance exactly 1 (the cell asserts `var(unbiased=False) == 1`). The unbiased estimator would be off by `D/(D-1)`.

Common issues:

- Explaining `unbiased=False` as a performance optimization rather than a definition/normalization-exactness point.
- Missing that the error is subtle — a small constant factor that's hard to notice and debug, not a crash.

## Exercise 09.04 — Which FFN outputs change when position 2 mutates

A correct answer should include:

- Only position 2's output changes; positions 0, 1, 3 are bit-identical. The FFN applies the same two-layer MLP to each position independently — no cross-position mixing (that's attention's job).

Common issues:

- Saying nearby positions also change "a little" — the independence is exact, and the cell asserts it.
- Confusing "same weights at every position" with "positions influence each other."

## Exercise 09.05 — Why the FFN expands to 4× before projecting down

A correct answer should include:

- The wider intermediate gives the GELU room to carve nonlinear features before projecting back to `D`; it's the standard convention and the reason most of a transformer's parameters live in the FFN.

Common issues:

- Explaining the 4× as required for shape correctness rather than as an expressiveness convention.
- Claiming 4 is a derived optimum rather than an empirical convention.

## Exercise 09.06 — Why zeroed projections make each sublayer output zero

A correct answer should include:

- Zeroing `attn.out_proj` and `ffn.fc2` (weights and biases) makes the *final linear* of each sublayer map every input to exactly zero, so each sublayer contributes a zero update and `x + 0 + 0 = x` — the block is the identity. Upstream pieces (LN, softmax, fc1/GELU) still run; their output is annihilated by the zeroed projection.

Common issues:

- Claiming the attention weights or hidden activations are zero — only the final projections are zeroed; the point is *where* in the sublayer the zeroing happens.
- Missing the framing this cell pins: sublayers are *contributions added to* the stream, and zero contribution means identity.

## Exercise 09.07 — What the cell would print without residuals

A correct answer should include:

- The block would output all zeros (the attention sublayer returns 0, and the FFN of that zero stream is also 0 through the zeroed `fc2`), so the printed "max absolute difference from identity" would be the input's own magnitude — large, not ~0 — and the assert would fail.

Common issues:

- Answering "it would raise an error" — nothing crashes; the output is just zeros instead of `x`.
- Saying the printed difference would still be near zero.

## Exercise 09.08 — Which curve is smoother (pre vs post norm)

A correct answer should include:

- Which curve was smoother in *their* run (run-dependent; typically pre-norm), tied to the mechanism: in pre-norm the residual path is a clean identity — LN sits inside the branch, not on the stream — so gradients flow undisturbed; post-norm routes the residual *through* LN, perturbing that identity path and making early training touchier at a fixed LR without warmup.

Common issues:

- Reporting which curve won without connecting it to the residual gradient path.
- Inverting the definitions (pre-norm is `x + f(LN(x))`; post-norm is `LN(x + f(x))`).

## Exercise 09.09 — Does warmup make pre-norm unnecessary?

A correct answer should include:

- No: warmup is a workaround that compensates for post-norm's early-training instability, while pre-norm removes the underlying sensitivity — it trains stably at depth without special scheduling, which is why it's the modern default.

Common issues:

- Treating similar *final* losses as evidence the two are equivalent — the question is about stability/robustness, not the endpoint of one toy run.
- Framing warmup as fixing the architecture rather than working around its early instability.

## Exercise 09.10 — Depth where residual-free training stalls

A correct answer should include:

- The depth at which their residual-free model stalls, cited from their plot (run-dependent; shallow depths 1–2 usually still learn, and by depth 4–8 the curve typically flattens near the `log(50)` uniform baseline).

Common issues:

- Blaming the stall on too few steps or too small a model rather than the missing identity path.
- Confusing the uniform baseline `log(50)` (model vocab) with `log(10)` (tokens actually used in the stream).

## Exercise 09.11 — Connection to the residual-stream explanation

A correct answer should include:

- Without residuals, each sublayer *replaces* the stream rather than adding an update, and the gradient must multiply through every sublayer Jacobian — shrinking exponentially with depth (vanishing gradients). The residual turns each factor into `1 + ∂f/∂x`, so the identity path keeps gradients alive at depth.

Common issues:

- Reciting the gradient-flow story without anchoring it to the depth threshold observed in 09.10 (or vice versa).
- Attributing the failure to overfitting or capacity rather than gradient flow.

## Exercise 09.12 — Does removing LayerNorm fail the same way?

A correct answer should include:

- The failure differs in kind: removing residuals gives a flat stall (no gradient signal at depth), while removing LayerNorm tends to produce *instability* — spiky or diverging loss, possibly NaN, worsening with depth — because unnormalized sublayer outputs accumulate on the residual stream and activation scale grows unchecked.

Common issues:

- Saying both ablations "just fail" without distinguishing stall from blow-up — the distinction is the entire question.
- Attributing no-norm instability to vanishing gradients (the residual path is still intact in `NoNormBlock`).

## Exercise 09.13 — Symptom pointing at scale instability

A correct answer should include:

- Loss spikes / divergence / NaN, or exploding activation magnitudes, point to scale instability; a smooth flat plateau points to missing gradient flow instead.

Common issues:

- Treating every bad training curve as the same failure mode.
- Over-generalizing one lucky stable shallow run to "LayerNorm doesn't matter."

## Exercise 09.14 — Which config tying saves the largest fraction

A correct answer should include:

- The third config (`V=500, D=64, L=2`): tying saves `V·D` parameters, so the largest *fractional* saving is where `V·D` is biggest relative to everything else — the bigger vocab makes the embedding table a large share of the total.

Common issues:

- Comparing absolute savings instead of fractions (the question says fraction).
- Ignoring the printed per-config numbers computed in the same cell.

## Exercise 09.15 — Relative tying saving as D grows

A correct answer should include:

- It *shrinks*: the saving grows linearly in `D` while block parameters grow quadratically (`≈ L·12·D²`), so `V·D / total → 0`.

Common issues:

- Getting the trend backwards by forgetting blocks scale as `D²` versus the embedding's `D`.
- Claiming tying changes model behavior for free — the question here is parameter accounting, not expressiveness.

## Exercise 09.16 — When embeddings vs blocks dominate

A correct answer should include:

- Embeddings (`V·D + max_seq_len·D`) dominate when the vocab is large relative to the stack — small `D`, few layers; blocks (`≈ L·12·D²`) dominate as `D` and `L` grow, since they scale quadratically in `D`. Anchored to the three printed configs (the `V=500` config shifts weight toward embeddings; the deeper/wider configs toward blocks).

Common issues:

- Vague "bigger models have more block params" without the `D²` vs `D` scaling argument.
- Ignoring the printed config numbers when the question says "in the configs above."

## Exercise 09.17 — Why the FFN outweighs attention inside one block

A correct answer should include:

- The 4× hidden expansion: FFN ≈ `8·D²` (two `D↔4D` matrices) vs attention's `4·D²` (four `D×D` projections) — a factor of 2.

Common issues:

- Attributing the FFN's dominance to it having more layers rather than to the widened hidden dim.
- Assuming attention dominates because it's the headline mechanism.
