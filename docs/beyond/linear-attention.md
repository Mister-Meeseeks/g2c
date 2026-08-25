# Beyond — Linear and efficient attention

> **Question this module answers:** *What makes long-context attention expensive—and which of those costs can we remove?*

<!-- TODO(hero pipeline): asset not yet generated -->
![Dense attention branching into exact tiled kernels, compressed caches, sparse token access, and fixed recurrent state, with the recurrent branch feeding a hybrid stack.](linear-attention/BeyondLinearAttention-Hero.png)

Module 07's attention consults every past token, and Module 16's KV cache showed one bill for that promise. Modern systems attack several different bills: some compute exact attention with less memory traffic, some store fewer key/value channels, some visit fewer tokens, and some replace token-addressable history with fixed recurrent state. This module maps those families, then builds the smallest inspectable version of the fixed-state/full-attention trade.

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

In Module 07 we learned attention is all you need. And then we built the contract that entails: every token attends to every previous token in the sequence. In Module 16 we saw the bill that comes due with that. As context length grows, inference must choose between quadratic computation costs, or a huge memory footprint in the form of a KV cache (almost always the latter). 

Modern LLM workloads are increasingly loaded on complex tasks and persistent agents that require long-running trajectories. That necessitates long context windows. On today's frontier models, context windows of 1 million tokens or longer is common. That has leads to a situation where the bottleneck has largely shifted from compute to memory (in the form of KV cache). Against that backdrop, there's been tremendous efforts to build "efficient attention". 

Keep in mind that there is no free lunch. The success of LLMs was built on all-to-all token attention. Relaxing that contract can lead to significant efficiency gains, but also significantly degraded and fragile intelligence if done wrong, fragile intelligence. Modern alternative attention mechanisms have experimented with many different approaches around this tradeoff. 

Therefore “efficient attention” is more of an umbrella than a single mechanism. The most prominent (and what we'll primarily focus on in this module) is *linear attention*. It replaces token to token attention, with attention intermediated through an intermediate running state of fixed size. Along similar lines is *sparse attention*, which maintains direct token-to-token attention, but relaxes the *all-to-all* part of token attention. A contrasting approach, which covers a variety of specific techniques, is to not reduce the scale of attention, but to increase the efficiency of each entry or comparison. 

Regardless, production systems typically combine *multiple* efficient attention techniques.  
So reading a model card requires understanding both the technical impact and the economics of each change.  

## The big idea

Attention does not send one bill. For a given sequence of length `T`, ordinary multi-head attention pays costs in several dimensions:

```
   dense training / prefill     compare T queries with T keys
                                └── O(T²) arithmetic and naive score storage

   autoregressive decoding      keep K and V for every previous token
                                └── O(T) state and O(T) work per new token

   hardware execution           move projections, scores, and values
                                between memory levels
                                └── wall-clock cost can be dominated by I/O
```

When evaluating efficient attention mechanisms, one should start by framing which of these cost dimensions the mechanism mitigates and how. Of the major families:

| Family                                  | What changes                                        | What remains                                                    |
| --------------------------------------- | --------------------------------------------------- | --------------------------------------------------------------- |
| **FlashAttention**                      | Avoid full score matrix. Reduce memory traffic      | Dense attention, quadratic pairs, and decode cache              |
| **Smaller token cache** (MQA, GQA, MLA) | Share or compress stored key/value representations  | All tokens remain available and cache size still grows with `T` |
| **Sparse attention**                    | Softmax over a structured subset of tokens          | Selection itself has a cost, limited tokens available           |
| **Linear attention**                    | Replace token rows with fixed-size running state    | The past is compressed rather than exactly addressable          |
| **Hybrids**                             | Pay different costs in different layers or branches | The costs of every retained mechanism still apply               |

 The right question to ask is **“which object stopped growing, and what did the model give up?”** The notes in this module situates the whole map from above. The exercises primarily focus on linear (hybrid) attention. The lesson cuts across all forms of attention variants. Regardless of the exact mechanism in question, the critical skills are understanding and measuring the economics and intelligence tradeoffs.

### FlashAttention

*FlashAttention* computes ordinary softmax attention in tiles and carries the normalization statistics needed to combine those tiles. The output is mathematically exact, but the implementation avoids writing the entire `T × T` score matrix to high-bandwidth memory. That can dramatically reduce activation memory and memory traffic.

Nothing about the connectivity changes: every query still interacts with every key. The pairwise arithmetic remains quadratic, and autoregressive decoding still needs a KV cache. FlashAttention is essentially a free lunch in terms of efficiency. It makes dense attention execute better; it does not turn dense attention into a fixed-state model.

### Store less for each token

Module 08's multi-head attention gives every query head its own K and V head, so its cache stores `2 · T · H · d_h = 2 · T · D` values per layer. *Multi-query attention (MQA)* shares one K/V head across all query heads. *Grouped-query attention (GQA)* uses an intermediate number of K/V heads. With `H_kv` K/V heads, the cache becomes `2 · T · H_kv · d_h` values. Empirical evidence generally indicates the impact on capabilities is minimal in most cases.

*Multi-head latent attention (MLA)* goes further by caching a lower-dimensional latent representation from which the needed key/value information is recovered. These methods preserve token-addressable history while reducing its width and bandwidth. Their caches are much smaller than the course's MHA baseline, but still grow linearly with `T`. 

### Attend to fewer tokens

*Sparse attention* reduces attention to a subset of sequence history. *Structured sparsity* fixes the pattern: a sliding window, blocks, global tokens, or combinations of them. A fixed sliding window can bound each layer's live KV cache and changes dense `T²` pairwise work to roughly `T · W` for window size `W`, but it removes direct access to tokens outside the window.

*Content-selected sparsity* uses the current query or a learned indexer to choose relevant tokens. It can preserve long-range retrieval better than a fixed window, but the system may still retain or index the full history, and finding the subset has its own cost. Native Sparse Attention combines local, compressed, and selected branches; DeepSeek Sparse Attention is a later learned-indexing design. “Sparse” therefore does not imply one cache policy or one complexity bound.

### Summarize token history

The rest of this module, and the exercises, will primarily focus on *linear attention*. Start with classic attention on a single query position `t`. Consider deleting the softmax for a moment:

```
   with softmax:      out_t = Σ_i  softmax(q_t·k_i) · v_i      ← must visit every i ≤ t

   without softmax:   out_t = q_t · ( Σ_i  k_i v_iᵀ )
                              ─────   ───────────────
                              (D,)       (D × D)
```

The entire trick *is* the reassociation. `Σ k_i v_iᵀ` doesn't depend on the query. It's just a running sum you can maintain incrementally, with one rank-one update per token:

```
   S_t = S_{t-1} + φ(k_t) v_tᵀ      # (D, D) state — the "memory"
   z_t = z_{t-1} + φ(k_t)           # (D,)   normalizer
   out_t = (φ(q_t) · S_t) / (φ(q_t) · z_t)
```

`φ` is a small positive feature map (we use `elu(x) + 1`) that provides a factorizable positive similarity in place of softmax's dot products. The result is attention becomes a *recurrent neural network (RNN)*. The whole past, regardless of sequence length, is compressed into a fixed size matrix `S`.

The equations above cover a single head. In the multi-head implementation, each head uses `d_h = D/H`. The complete state is `H` matrices of shape `(d_h, d_h)` plus `H` normalizer vectors. 

The same computation now has **two forms**, and implementing both. Proving they agree is the module's core deliverable:

```
   PARALLEL REFERENCE:                 RECURRENT (inference):

   all T tokens at once,               one token at a time,
   explicit T × T decay mask,          carry (S, z) forward
   full teacher forcing, but           O(D²) per token,
   still quadratic in T                O(1) memory in T

          └────────── same numbers, to floating-point tolerance ──────────┘
```

This mirrors a distinction you previously covered. In class attention training you use `forward` to trains in parallel, while in inference you use `forward_cached` to generate incrementally. Linear attention makes the incremental form cheap. The "cache" turns into one fixed matrix instead of `T` growing entries.

(One wrinkle for what we'll build in this module. The course's `forward` uses the full `T × T` score because that makes the equivalence with `step` visible. It does not provide linear-time or linear-memory training. Actual production systems use associative scans, chunking, and fused kernels to expose parallelism without materializing the full matrix. This module demonstrates the fixed-state inference result but does not implement the production training kernel.)

### What the compression costs

`S_t` is a compressed summary. Every key-value pair is superimposed into the same fixed matrix, so associations can interfere. Plausible consequences include:

* **Exact recall may degrade with distance.** Full attention retains an addressable key/value row for every token; the recurrent state superimposes them.
* **Copying and induction may weaken.** The induction pattern from Module 08 asks for an exact lookup, which is a demanding use of a fixed-capacity state.
* **Average next-token loss may hide the difference.** If most predictions in a corpus depend on nearby context, a relatively rare retrieval failure contributes little to the average. Exercises 2 and 3 test whether that story actually appears at StoryLM scale.

### Fixed decay versus an input-dependent gate

The raw accumulator `S_t = S_{t-1} + φ(k_t) v_tᵀ` never forgets. The magnitudes inside the recurrent state can grow indefinitely, because early tokens never fade. This module adds the smallest useful forgetting mechanism:

```
   S_t = γ_h · S_{t-1} + φ(k_t) v_tᵀ   # one learned γ per head
```

With `γ_h` below 1, every token handled by head `h` decays the state by a fixed proportion. This is a *fixed learned decay*, not a content-dependent gate. Alone it cannot decide that punctuation should be forgotten quickly, while a name should persist. Modern gated recurrent layers will compute a decay based on the current token. Delta-rule variants also residualize by what the state already predicting before the token. Those mechanisms are more selective, but this module keeps one scalar per head so the recurrence and its parallel reference remain inspectable.

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

The full layer can re-ground the residual stream in exactly addressable token identities. The cache bill scales with the number and cache geometry of the full layers. A 3:1 hybrid pays a growing cache in one quarter of its layers, but the size of that cache still depends on whether those layers use MHA, GQA, or another compression scheme.

## Concepts to internalize

- **There is more than one attention bill.** Pairwise arithmetic, activation memory, KV-cache capacity, bandwidth, and wall-clock execution are related but not interchangeable targets.
- **Every efficiency claim needs a denominator.** Ask which object stopped growing, during which phase, and whether the method changed exact token access.
- **Exact kernels, compressed caches, sparsity, and recurrence solve different problems.** FlashAttention executes dense attention better; MQA/GQA/MLA store less per token; sparse methods visit fewer tokens; recurrent methods replace token rows with fixed state.
- **Linear attention uses a factorizable similarity.** Replace softmax with a positive feature kernel, regroup the computation, and each head's past collapses into a running `(d_h, d_h)` state.
- **One computation, two forms.** The readable parallel reference and recurrent inference form are equal to floating-point tolerance. The equivalence test is the module's anchor; the reference is not an efficient training kernel.
- **The state is a compressed summary.** Average loss and targeted recall measure different things; neither experiment's outcome should be assumed in advance.
- **A fixed decay is not a selective gate.** This module learns one forgetting rate per head. Token-dependent gates can choose what to retain based on content.
- **Hybrids expose a dial.** Full layers retain exact token access and growing caches; recurrent layers use fixed state. The layer pattern chooses where to pay each cost.
- **Model-card literacy:** "hybrid," "recurrent," "linear," "SSM," and ratios like 3:1 often signal a choice along the growing-cache versus fixed-state axis, though their update equations are not interchangeable.

### What we don't cover

- **State-space model derivations.** Mamba's selective-scan formulation arrives at a similar place from control theory. The convergence is the interesting fact; the HiPPO math is its own course.
- **Efficient dense-attention kernels.** We explain FlashAttention's I/O idea but do not build a tiled softmax kernel.
- **MQA, GQA, and MLA implementations.** We include their cache geometry in the survey and memory exercise, not new projection or latent-cache scaffolds.
- **Sparse-attention implementations.** We distinguish local, structured, and learned selection conceptually without building indexing or block-sparse kernels.
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

The anchor test: `test_parallel_matches_recurrent` runs the same sequence through `forward` and through repeated `step` calls and asserts the outputs agree. Mismatched normalization, state updated in the wrong order, or decay applied in only one form fails this test with a large, readable error. `test_forward_matches_undecayed_reference_at_gamma_one` separately prevents both forms from agreeing on the same unnormalized computation.

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
5. **Written: four versions of the 1M-token bill.** At `T=1M`, `D=4096`, `H=32`, compare the course's MHA cache, GQA with eight KV heads, an MHA layer retaining a 4096-token sliding window, and linear `(S, z)` state. Then total the four-layer all-MHA and `[L,L,L,F]` toy stacks and identify which states still grow with `T`.
6. **StoryLM-5M rerun (optional).** Repeat the TinyStories comparison at the StoryLM-5M architecture. It remains laptop-sized but makes the three-run block substantially longer.

## Pitfalls to expect

- **Dropping the normalizer `z`.** Outputs drift in scale with sequence length; training limps. Parallel/recurrent equivalence alone cannot catch an omission made in both forms, but `test_forward_matches_undecayed_reference_at_gamma_one` pins the normalized result explicitly.
- **State update order.** Whether you update `(S, z)` before or after computing the current token's output is the alignment seam: update-first lets a token attend to itself (correct, matching the parallel form); output-first silently shifts everything by one. The equivalence test pins this convention. Causality alone only proves that the future cannot affect the past.
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

Efficient-attention map:

- **Dao, Fu, Ermon, Rudra, Ré, ["FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"](https://arxiv.org/abs/2205.14135) (2022).** Exact dense attention with tiled, I/O-aware execution; the clearest example of improving the kernel without changing the attention graph.
- **Ainslie et al., ["GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"](https://arxiv.org/abs/2305.13245) (2023).** The continuum from one K/V head to one per query head.
- **DeepSeek-AI, ["DeepSeek-V2"](https://arxiv.org/abs/2405.04434) (2024).** Multi-head latent attention as learned KV-cache compression.
- **Yuan et al., ["Native Sparse Attention"](https://arxiv.org/abs/2502.11089) (2025).** A hardware-aware combination of compressed, selected, and sliding-window branches.

Linear and recurrent core:

- **Katharopoulos, Vyas, Pappas, Fleuret, "Transformers are RNNs" (2020).** The reassociation trick, the feature map, and the parallel/recurrent duality — this module is §3 of this paper, built. Read it first for the implementation branch.
- **Schlag, Irie, Schmidhuber, "Linear Transformers Are Secretly Fast Weight Programmers" (2021).** Reframes the state as a fast-weight memory and introduces the delta-rule update — the error-correcting write behind the DeltaNet/KDA lineage.
- **Gu, Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (2023).** The state-space route to the same destination. Read for the convergence, skim the derivations.

Further bridges:

- **Yang, Kautz, Hatamizadeh, "Gated DeltaNet" (2024).** Decay gating plus the delta rule, in the form Qwen3.5 adopted; the cleanest bridge from this module's code to a current model card.
- **Sun, Dong, Huang et al., "Retentive Network" (2023).** An independent derivation of decay-gated linear attention; useful second angle on why the decay is load-bearing.
- **DeepSeek-AI, ["DeepSeek-V3.2"](https://arxiv.org/abs/2512.02556) (2025).** DeepSeek Sparse Attention's learned indexer and selected-token attention, distinct from Native Sparse Attention above.

Optional:

- **2026 release examples.** The Kimi K3 (69:24 KDA/Gated-MLA hybrid) and Qwen3.5 (3:1 DeltaNet/gated-attention hybrid) technical reports show how production systems combine the fixed-state branch with compressed full-attention layers.
- **Gu, Goel, Ré, "Efficiently Modeling Long Sequences with Structured State Spaces" (S4, 2021).** Where the SSM lineage began, if the control-theory route appeals.

## Deliverable checklist

- [ ] All tests in `tests/test_linear_attention.py` pass — the parallel/recurrent equivalence especially.
- [ ] Notebook: honest cached-decoding benchmark (Exercise 1), StoryLM-1M three-way comparison with generated stories (Exercise 2), and empirical recall-vs-distance probe (Exercise 3).
- [ ] Written answer for the 1M-token memory bill (Exercise 5).
- [ ] You can identify whether an efficiency claim reduces dense-attention I/O, pairwise work, per-token KV width, sequence-length cache growth, or some combination.
- [ ] You can explain — out loud, without notes — what reassociating `(qKᵀ)V` into `q(KᵀV)` buys, and what the softmax's removal costs.
- [ ] You can explain — out loud, without notes — why hybrid stacks exist, in terms of what the recurrent state cannot do.
- [ ] You can explain — out loud, without notes — why this module's recurrent inference state is fixed-size while its readable parallel training reference is still quadratic.
- [ ] You can explain — out loud, without notes — how a fixed per-head decay differs from an input-dependent gate.
