# Module 13B Rubric

Use this rubric to grade submitted answers in the student's working notebook (`notebooks/solutions/13b-lora.ipynb`, falling back to `notebooks/clean/13b-lora.ipynb` if no solutions copy exists yet). Each exercise has `Question:` / `Answer:` string cells; treat a blank `"Answer: "` as not submitted (skip it). If the student wrote a hint or help request inside the answer string, tutor first before grading and avoid giving away the full solution unless they explicitly ask. Give feedback exercise by exercise and avoid replacing the student's work with a full solution unless explicitly asked.

## Exercise 13B.01 — Count before you build

A correct answer should include:

- Per-layer arithmetic: `q_proj` (960→960) gets `8 × (960 + 960) = 15,360`; `v_proj` (960→320) gets `8 × (960 + 320) = 10,240`; per layer `25,600`.
- Across 32 layers: `819,200` trainable parameters.
- The percentage: roughly `0.23%` of ~362M (accept anything in the 0.2–0.25% range with the arithmetic shown).
- The GQA point: `v_proj`'s output is 320, not 960, because only 5 of 15 heads carry KV.

Common issues:

- Treating `v_proj` as square (gives 983,040 / ~0.27%) — the exact mistake the question warns about.
- Computing `r × in × out` (the full-matrix count) instead of `r × (in + out)`.
- Forgetting the factor of 2 matrices, or the factor of 32 layers.

## Exercise 13B.02 — Why B starts at zero

A correct answer should include:

- With `B = 0` the delta `A @ B` is exactly zero, so the injected model computes the identical function — fine-tuning starts *from* the pretrained model, not near it.
- The chain-rule asymmetry: `∂L/∂A` routes through `B`, so `A`'s gradient is exactly zero on the first backward pass; `∂L/∂B` routes through `A` (random, nonzero), so `B` moves first and unfreezes `A` from step 2 on.
- Zero/zero kills the adapter permanently: both gradients route through the other's zeros, so nothing ever moves (the two-matrix version of Module 03's symmetry/dead-init problem).

Common issues:

- Saying `A`'s gradient is "small" at init rather than exactly zero.
- Explaining the no-op property but not the gradient-flow reason the zero must be one-sided.
- Claiming random/random would be fine (it breaks the no-op start: fine-tuning would begin from a perturbed model).

## Exercise 13B.03 — The four memory tenants

A correct answer should include:

- LoRA shrinks gradients and optimizer state (they exist only for trainable parameters); weights are unchanged (the full frozen model still loads) and activations are essentially unchanged (the forward pass is the same computation).
- The wall-clock consequence: the forward still runs the whole model and the backward still propagates through every layer to reach the adapters, so per-step time lands near full fine-tuning — LoRA saves memory, not FLOPs.

Common issues:

- Claiming LoRA makes training much *faster* — the near-universal misconception the lesson flags.
- Forgetting activations entirely, or claiming they shrink.
- Not connecting the savings to `requires_grad` / the trainable set.

## Exercise 13B.04 — LoRA vs full-SFT samples

A correct answer should include:

- A concrete side-by-side observation against their Module 13 run (format compliance, stopping on `<|end|>`, answer style) with at least one cited sample.
- The expected headline: at this data scale the two land close — format is learnable in a rank-8 subspace, which is the intrinsic-rank story in miniature.
- Any honest observed difference is acceptable (LoRA slightly less damaging on the prose continuation is a common and correct finding); what matters is that the comparison was actually made.

Common issues:

- Grading the model ("it's bad at facts") instead of comparing the two fine-tunes.
- No citation from their own samples.
- Overclaiming that LoRA is categorically better/worse from one prompt.

## Exercise 13B.05 — Merged vs unmerged difference

A correct answer should include:

- The two paths compute the same function through different float operation orders: `(x @ W.T) + (x @ A) @ B · s` versus `x @ (W + (A@B).T · s).T` — float addition/multiplication is not associative, so results differ in the last bits.
- The difference is bounded rounding noise (~1e-6 scale), not a bug — contrast with Exercise 2, where exact zero was required because the delta itself was exactly zero.

Common issues:

- Calling it a bug or a precision "loss" to be fixed.
- Confusing this with the Exercise 2 bit-identical check (different claim: there the delta is exactly zero; here two nonzero computations are reassociated).
- Vague "floating point error" with no mechanism (reassociation/rounding).

## Exercise 13B.06 — The round trip and zero forgetting

A correct answer should include:

- Merge-then-unmerge adds and subtracts the same matrix in floats, leaving last-bit residue in the base weights — close, not bit-identical.
- The workflow conclusion: never merge the training copy; keep the base artifact pristine and the fine-tune in the adapter file. Then "zero forgetting" is structural — removing the adapter *is* the base model — rather than a hyperparameter hope, unlike Module 13's full SFT where forgetting happens inside the weights with no undo.

Common issues:

- Expecting bitwise restoration from the float round trip.
- Missing the contrast with Module 13's forgetting failure modes (the point of the question).
- Treating `unmerge()` as the recovery mechanism while still training on merged weights.

## Exercise 13B.07 — The adapter-file contract

A correct answer should include:

- What must match: the injection targets (same `target_names`), the rank/shapes, and the base model's weights (the delta was trained *relative to* those weights).
- Wrong targets or rank fail loudly (`load_lora_state_dict` is strict; keys/shapes mismatch).
- Different base weights fail *silently*: the adapter loads fine mechanically but the behavior is wrong, because `ΔW` only means something on top of the `W` it was trained against.

Common issues:

- Only listing the loud failures and missing the silent one (the interesting half).
- Claiming an adapter is portable across different base models of the same architecture.

## Exercise 13B.08 — Rank sweep (optional)

A correct answer should include:

- Their measured result: final val losses and a sample comparison across ranks (rank 1 vs 8 typically land surprisingly close on this task).
- The reading: the format shift SFT teaches needs very few directions of weight change — a firsthand miniature of the paper's rank ablation and the intrinsic-dimensionality claim.

Common issues:

- Expecting rank to matter like model size does and reading noise as signal.
- No connection back to "SFT teaches format" from Module 13.

## Exercise 13B.09 — Closing paragraph

A correct answer should include:

- What changes: which parameters are trainable (a low-rank bypass per adapted layer), and therefore gradient + optimizer memory.
- What stays exactly the same: the data, the masked loss, the trainer, the base weights (bit-for-bit), and essentially the per-step compute.
- The one flag: `requires_grad` — every memory saving traces to parameters autograd no longer tracks.

Common issues:

- Describing LoRA as a new training objective or new loss.
- Missing the "one flag" half of the question.
