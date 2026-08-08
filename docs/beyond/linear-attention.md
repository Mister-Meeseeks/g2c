# Beyond — Linear attention and efficient sequence models

> **Question this module answers:** *Must attention remember every token it has ever seen?*

<!-- TODO(hero pipeline): asset not yet generated -->
![Full attention drawn as a key-value cache growing with every token beside linear attention's fixed-size state matrix, feeding a hybrid stack where many recurrent layers alternate with an occasional global attention layer.](linear-attention/BeyondLinearAttention-Hero.png)

Module 07's attention consults every past token, and Module 16's KV cache is the bill for that: memory that grows with every token generated. Several current long-context families avoid paying it in every layer — Qwen3.5 mixes recurrent and full attention, Kimi K3 reports 69 KDA layers and 24 global-attention layers, and Nemotron mixes Mamba and attention blocks. This module builds the smallest inspectable version of that fixed-state/full-attention trade. It is an ancestor of those production mechanisms, not a reproduction of their token-dependent gates, delta rules, or fused kernels.

> **This is a Beyond module.** Beyond modules sit outside the numbered course: nothing in Modules 00–20 depends on them, and they are not part of finishing the course. Come here in any order, whenever a model card or paper names the idea and you want the load-bearing version — built, trained, and broken on your own machine.

---
## Before you start

* *Review*
	* [07-attention](../modules/07-attention.md) for the Q/K/V computation this module rearranges
	* [16-inference](../modules/16-inference.md) for the KV cache and why its growth is the problem
* *Finish*
	* `g2c/attention` ([07](../modules/07-attention.md) and [08](../modules/08-multi-head-attention.md))
	* `g2c/transformer` ([09-transformer-block](../modules/09-transformer-block.md))
	* `g2c/training` ([03b-training](../modules/03b-training.md))
* *Run*
	* `./datasets.sh --tiny` if the TinyStories corpus or `StoryTokenizer` artifact is missing
	* `G2C_APPLY_SOLUTIONS=01-16 ./notebook.sh linear-attention` instead of the plain launch if you're entering without your own implementations through cached inference

---
## Where this fits in

Module 07 made a specific promise: every position can consult every other position, exactly. Module 16 showed what that promise costs at inference — a KV cache holding two vectors per token per layer, growing without bound. At million-token context lengths, that cache can become a dominant memory object.

There are two escape routes, and reading a 2026 model card requires recognizing both:

* **Sparse attention** keeps exact attention but only over a *subset* of past tokens — a learned or structured selection (DeepSeek's DSA is this family). The cache survives; fewer entries are touched.
* **Linear / recurrent attention** — this module — replaces the growing cache with a **fixed-size state** that is updated as each token arrives. Constant memory, constant per-token compute, at the price of storing a *summary* of the past instead of the past itself.

Pure recurrent stacks trade exact token access for a compressed state. A common practical response is the **hybrid**: mostly-recurrent stacks with periodic full-attention layers that restore exact access at selected depths. The ratios differ (3:1, 69:24); this module builds one small `[linear, linear, linear, full]` instance and measures what it does rather than assuming the hybrid always wins.

## The big idea

Start from Module 07's attention for a single query position `t`, and delete the softmax for a moment:

```
   with softmax:      out_t = Σ_i  softmax(q_t·k_i) · v_i      ← must visit every i ≤ t

   without softmax:   out_t = q_t · ( Σ_i  k_i v_iᵀ )
                              ─────   ───────────────
                              (D,)       (D × D)
```

That reassociation is the entire trick. `Σ k_i v_iᵀ` doesn't depend on the query — it's a running sum you can maintain **incrementally**, one rank-one update per token:

```
   S_t = S_{t-1} + k_t v_tᵀ         # (D, D) state — the "memory"
   z_t = z_{t-1} + k_t              # (D,)   normalizer
   out_t = (φ(q_t) · S_t) / (φ(q_t) · z_t)
```

`φ` is a small positive feature map (we use `elu(x) + 1`) standing in for what the softmax's exponential was doing — keeping scores positive so the normalizer behaves. The result is attention as an RNN: the whole past is compressed into `S`, a matrix whose size never changes.

The equations above describe one head and use `D` for readability. In the multi-head implementation, each head uses `d_h = D/H`, so the complete state is `H` matrices of shape `(d_h, d_h)` plus `H` normalizer vectors — `D²/H + D` values, not `D²`.

The same computation now has **two forms**, and implementing both — and proving they agree — is the core deliverable:

```
   PARALLEL REFERENCE:                 RECURRENT (inference):

   all T tokens at once,               one token at a time,
   explicit T × T decay mask,          carry (S, z) forward
   full teacher forcing, but           O(D²) per token,
   still quadratic in T                O(1) memory in T

          └────────── same numbers, to floating-point tolerance ──────────┘
```

This mirrors a distinction you've met before: Module 09's `forward` trains in parallel while Module 16's `forward_cached` generates incrementally. Linear attention makes the incremental form *cheap* — the "cache" is one fixed matrix instead of `T` growing entries.

There is an important implementation honesty line here. The course's `forward` constructs the full `T × T` score and decay matrices because that makes the equivalence with `step` visible. It therefore **does not provide linear-time or linear-memory training**. Production systems use associative scans, chunking, and fused kernels to expose parallelism without materializing the full matrix. This module demonstrates the fixed-state inference result directly and explains, but does not implement, the production training kernel.

### What the compression costs

`S_t` is a compressed summary. Every key-value pair is superimposed into the same fixed matrix, so associations can interfere. Plausible consequences include:

* **Exact recall may degrade with distance.** Full attention retains an addressable key/value row for every token; the recurrent state superimposes them.
* **Copying and induction may weaken.** The induction pattern from Module 08 asks for an exact lookup, which is a demanding use of a fixed-capacity state.
* **Average next-token loss may hide the difference.** If most predictions in a corpus depend on nearby context, a relatively rare retrieval failure contributes little to the average. Exercises 2 and 3 test whether that story actually appears at StoryLM scale.

### Fixed decay versus an input-dependent gate

The raw accumulator `S_t = S_{t-1} + k_t v_tᵀ` never forgets — the state's magnitude can grow and early tokens never fade. This module adds the smallest useful forgetting mechanism:

```
   S_t = γ_h · S_{t-1} + k_t v_tᵀ      # one learned γ per head
```

With `γ_h` below 1, every token handled by head `h` decays that head's old state by the same amount. This is a **fixed learned decay**, not a content-dependent gate: it cannot decide that punctuation should be forgotten quickly while a name should persist. Modern gated recurrent layers compute a decay or write strength from the current token, and delta-rule variants also subtract what the state already predicts before writing. Those mechanisms are more selective; this module keeps one scalar per head so the recurrence and its parallel reference remain inspectable.

### The hybrid pattern

Linear attention compresses history; full attention everywhere pays the full cache. A useful middle point is to stack mostly-linear layers with an occasional full-attention layer:

```
   layer 12:  linear      ┐
   layer 11:  linear      │  cheap, O(1)-state mixing
   layer 10:  linear      ┘
   layer  9:  FULL        ←  exact token access available; KV cache paid here
   layer  8:  linear      ┐
   ...                    ┘
```

The full layer can re-ground the residual stream in exactly addressable token identities; whether one such layer repairs a given task is empirical. The cache bill scales with the number of full layers. Exercise 5 computes the contrast: at `D=4096`, one fp16 full-attention layer needs about 16.4 GB of KV cache at one million tokens, while one linear layer's `(S, z)` state is about 1.1 MB. A 3:1 hybrid pays the growing cache in one quarter of its layers.

## Concepts to internalize

- **Linear attention is attention reassociated.** Drop the softmax, regroup `(qKᵀ)V` as `q(KᵀV)`, and each head's past collapses into a running `(d_h, d_h)` state.
- **One computation, two forms.** The readable parallel reference and recurrent inference form are equal to floating-point tolerance. The equivalence test is the module's anchor; the reference is not an efficient training kernel.
- **The state is a compressed summary.** Average loss and targeted recall measure different things; neither experiment's outcome should be assumed in advance.
- **A fixed decay is not a selective gate.** This module learns one forgetting rate per head. Token-dependent gates can choose what to retain based on content.
- **Hybrids expose a dial.** Full layers retain exact token access and growing caches; recurrent layers use fixed state. The layer pattern chooses where to pay each cost.
- **Model-card literacy:** "hybrid," "recurrent," "linear," "SSM," and ratios like 3:1 often signal a choice along the growing-cache versus fixed-state axis, though their update equations are not interchangeable.

### What we don't cover

- **State-space model derivations.** Mamba's selective-scan formulation arrives at a similar place from control theory. The convergence is the interesting fact; the HiPPO math is its own course.
- **The sparse-attention family in depth.** DSA-style learned token selection is the *other* escape route; it deserves its own treatment if the brief evidence keeps accumulating, and this module only positions it.
- **Hardware-aware scan and chunked kernels.** Real implementations balance within-chunk parallelism against recurrent state passing. Our readable parallel reference materializes `T × T` scores, so it proves equivalence but does not realize efficient training.
- **Input-dependent gates and delta-rule writes.** They let the model choose what to forget and correct an existing association before writing. We keep a fixed per-head decay to isolate the recurrent-state idea.
- **Position embeddings under recurrence.** How RoPE-style rotations interact with state-based attention is a genuinely fiddly topic we sidestep by keeping Module 09's learned positions.

---
## What you'll build

Package: `g2c/linear_attention/`

```python
def feature_map(x): ...                  # implemented — elu(x) + 1

class LinearAttention(Module):
    embedding_dim: int
    num_heads: int
    # same Q/K/V/output projections as Module 08
    gamma_logit: torch.Tensor            # (H,) learned parameter
    decay: torch.Tensor                  # property: sigmoid(gamma_logit)

    def parameters(self): ...                                   # implemented
    def forward(self, x): ...                                   # SCAFFOLDED
        # parallel reference: explicit decayed T × T scores, all
        # positions at once; readable, but not an efficient train kernel
    def step(self, x_t, state): ...                             # SCAFFOLDED
        # recurrent/inference form: one token in, (S, z) carried
        # forward, one token out — must match forward() exactly

class HybridBlock(Module):               # Block with attn ∈ {full, linear}   # implemented
class HybridTransformerLM(Module):       # TransformerLM over a layer pattern  # implemented
    # pattern like ["linear", "linear", "linear", "full"] * repeats
```

Total scaffolded code: roughly 35 lines across two methods. `forward` and `step` are two views of one computation; when the equivalence test passes, you've understood the module.

## How to run the tests

Tests live in `tests/test_linear_attention.py`. Initial state: 8 passed (the provided feature map, construction, state shapes, accounting, and hybrid stacking), 7 failed.

```bash
source .venv/bin/activate

pytest tests/test_linear_attention.py              # all module tests
pytest tests/test_linear_attention.py -x           # stop at first failure (recommended)
pytest tests/test_linear_attention.py -k equiv     # parallel-vs-recurrent equivalence
pytest tests/test_linear_attention.py -k causal    # causality tests
pytest tests/test_linear_attention.py -v           # verbose
```

The anchor test: `test_parallel_matches_recurrent` runs the same sequence through `forward` and through repeated `step` calls and asserts the outputs agree. Nearly every implementation bug — normalizer dropped, state updated in the wrong order, decay applied twice — fails this one test with a large, readable error.

## Exercises

To launch the exercise notebook run:

```bash
./notebook.sh linear-attention
```

If at any point you want to archive the work in your current notebook and restart fresh:

```bash
./notebook.sh linear-attention --fresh
```

Written exercises live in the notebook as `Question:` / `Answer:` cells; ask a coding agent for hints or grading when you're ready, and partial submissions are fine — blank answers are skipped.

1. **Prove the equivalence, then benchmark decoding.** Verify parallel/recurrent equality. Compare one cached full-attention decoding step against one linear recurrent step over growing history lengths, with MPS synchronization and memory arithmetic. Report the observed crossover, including “none in this range.”
2. **Train the three-way comparison.** Train StoryLM-1M full attention, pure linear, and `[linear, linear, linear, full]` models on the 100MB TinyStories tier with near-identical parameter counts, batches, and step budgets. Compare validation loss and generated stories.
3. **The recall probe.** Train the same three architectural patterns on a synthetic key-value task and measure answer accuracy by retrieval distance. Treat every curve as data: explain what occurred rather than repairing it into the expected narrative.
4. **Inspect decay.** Compare state norms under fixed decay values, then inspect the per-head decays learned by the TinyStories models. Explain what a fixed per-head decay cannot do that an input-dependent gate can.
5. **Written: the 1M-token bill.** Compute fp16 state per layer at `T=1M`, `D=4096`, `H=32`: full KV cache, linear `(S, z)`, and the four-layer 3:1 pattern.
6. **StoryLM-5M rerun (optional).** Repeat the TinyStories comparison at the StoryLM-5M architecture. It remains laptop-sized but makes the three-run block substantially longer.

## Pitfalls to expect

- **Dropping the normalizer `z`.** Outputs drift in scale with sequence length; training limps. The equivalence test won't catch it (both forms drift identically) — the state-norm plot in Exercise 4 will.
- **State update order.** Whether you update `(S, z)` before or after computing the current token's output is the causality seam: update-first lets a token attend to itself (correct, matches the masked parallel form); output-first silently shifts everything by one. `test_causality` pins the convention.
- **Applying decay in one form but not the other.** The parallel form must implement the *same* geometric decay as `step` — via distance-dependent weights `γ^(t-i)` — or the equivalence test fails only for `γ < 1`, which reads as a mystery.
- **Timing asynchronous MPS work without synchronization.** The Python timer stops before Metal finishes and produces fiction. The notebook synchronizes before and after every measured region.
- **Comparing training and decoding timings.** A full teacher-forced pass and one recurrent token answer different questions. Exercise 1 compares one-token decode paths; the lesson separately explains why this module's training reference remains quadratic.
- **Promising a crossover or recall curve.** Kernel overhead, context range, optimization, and random seed all matter. “No crossover observed” and “hybrid did not recover the gap” are valid results when supported by the measurement.
- **Judging by average loss alone.** Even if losses are close, a targeted probe may separate the models; if it does not, that is also evidence about this model and task scale.

## M-series notes

- **Train with the parallel reference only.** The recurrent `step` in a Python loop throws away teacher-forcing parallelism. The reference is still quadratic in `T`; efficient linear-attention training needs a scan/chunk kernel outside this module's scope.
- **The equivalence check is CPU-comfortable; the full recall probe is a several-minute training block.** Its models and sequences are small, but the required path trains three four-layer variants for 600 steps. MPS is recommended for that cell.
- **Exercise 2's three StoryLM-1M runs are the compute cost of the required path.** StoryLM-5M is an optional rerun. Plug in for either sustained training block.
- **MPS and the parallel reference.** Its explicit score and decay matrices can be memory-hungrier than standard attention at the same `T`. If memory pressure bites, halve the batch before shrinking the model.

---
## Reading

Primary:

- **Katharopoulos, Vyas, Pappas, Fleuret, "Transformers are RNNs" (2020).** The reassociation trick, the feature map, and the parallel/recurrent duality — this module is §3 of this paper, built. Read it first.
- **Schlag, Irie, Schmidhuber, "Linear Transformers Are Secretly Fast Weight Programmers" (2021).** Reframes the state as a fast-weight memory and introduces the delta-rule update — the error-correcting write behind the DeltaNet/KDA lineage.
- **Gu, Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (2023).** The state-space route to the same destination. Read for the convergence, skim the derivations.

Secondary:

- **Yang, Kautz, Hatamizadeh, "Gated DeltaNet" (2024).** Decay gating plus the delta rule, in the form Qwen3.5 adopted; the cleanest bridge from this module's code to a current model card.
- **Sun, Dong, Huang et al., "Retentive Network" (2023).** An independent derivation of decay-gated linear attention; useful second angle on why the decay is load-bearing.
- **Yuan, Gao, Dai et al., "Native Sparse Attention" (2025).** The other escape route — learned sparse selection with the cache intact. Read to position DSA-family model cards against this module's family.

Optional:

- **The current release lineage.** The Kimi K3 (69:24 KDA hybrid) and Qwen3.5 (3:1 DeltaNet hybrid) technical reports (2026) are where the ratios in the lecture notes come from. After this module, their architecture sections read as configuration, not invention.
- **Gu, Goel, Ré, "Efficiently Modeling Long Sequences with Structured State Spaces" (S4, 2021).** Where the SSM lineage began, if the control-theory route appeals.

## Deliverable checklist

- [ ] All tests in `tests/test_linear_attention.py` pass — the parallel/recurrent equivalence especially.
- [ ] Notebook: honest cached-decoding benchmark (Exercise 1), StoryLM-1M three-way comparison with generated stories (Exercise 2), and empirical recall-vs-distance probe (Exercise 3).
- [ ] Written answer for the 1M-token memory bill (Exercise 5).
- [ ] You can explain — out loud, without notes — what reassociating `(qKᵀ)V` into `q(KᵀV)` buys, and what the softmax's removal costs.
- [ ] You can explain — out loud, without notes — why hybrid stacks exist, in terms of what the recurrent state cannot do.
- [ ] You can explain — out loud, without notes — why this module's recurrent inference state is fixed-size while its readable parallel training reference is still quadratic.
- [ ] You can explain — out loud, without notes — how a fixed per-head decay differs from an input-dependent gate.
