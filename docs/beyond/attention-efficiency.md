# Beyond guide — The efficient-attention map

> **Question this guide answers:** *When a release says “efficient attention,” which bill did it actually make smaller?*

<!-- TODO(hero pipeline): asset not yet generated -->
![Dense attention branching into exact tiled execution, narrower per-token caches, selected token access, and fixed recurrent state.](attention-efficiency/BeyondAttentionEfficiency-Hero.png)

Efficient attention is not one replacement for softmax attention. It is a map of techniques that change memory traffic, cache width, token connectivity, or the representation of history. This guide keeps those mechanisms separate, shows how production systems combine them, and routes the recurrent branch to a focused build in [Beyond Linear Attention](linear-attention.md).

> **Guide status.** This is supporting lecture material, not a standalone coding module. It has no package, notebook, tests, rubric, or deliverable, and nothing in Modules 00–20 depends on it.

---
## Before you start

* *Review* [07-attention](../modules/07-attention.md) for causal Q/K/V attention
* *Review* [16-inference](../modules/16-inference.md) for cached decoding
* Read [Beyond Hardware-aware Model Architecture](hardware.md) alongside this guide when a claim is about wall-clock speed rather than tensor size

---
## Where this fits in

Module 07 builds exact all-to-all attention. Module 16 caches each token's keys and values so decoding does not recompute the past, but the cache still grows with history. Long contexts expose multiple bills: quadratic pairwise work in dense prefill, growing state and bandwidth in decode, and the hardware cost of moving intermediates.

No single “post-transformer attention” module should own every response to those bills. This guide is the map. [Beyond Linear Attention](linear-attention.md) is the build for fixed recurrent state; [Beyond Long Context](long-context.md) owns position and extension training; [Beyond Hardware](hardware.md) owns execution.

## The big idea

| Family | What changes | What remains |
|---|---|---|
| **Exact tiled kernels** — FlashAttention | Memory traffic and score materialization | Exact dense connectivity, quadratic pairs, growing decode cache |
| **Narrower per-token cache** — MQA, GQA, MLA | K/V heads or cached representation width | Token-addressable state grows with `T` |
| **Sparse/local attention** | The subset of token pairs evaluated | Restricted access or a selection/indexing cost |
| **Linear/recurrent attention and SSMs** | Token rows become fixed running state | History is compressed rather than exactly addressable |
| **Hybrids** | Different layers or branches pay different bills | Every retained mechanism keeps its own cost |

Read every efficiency claim with four questions:

1. Which phase—training, prefill, or decode?
2. Which object—scores, activations, payload cache, index, recurrent state, or bandwidth?
3. What still grows with context length?
4. What access or quality contract changed?

### Exact dense attention, executed differently

FlashAttention tiles ordinary softmax attention and carries online normalization statistics between tiles. It avoids materializing the full `T × T` score matrix in high-bandwidth memory while preserving the mathematical dense-attention result. Pairwise arithmetic remains quadratic and autoregressive decoding still retains K/V history.

This distinction is load-bearing: an algorithm can have identical model semantics and very different wall-clock behavior because the kernel moves fewer bytes. The [hardware module](hardware.md) develops that execution layer.

### Store less for every token

With `H` query heads and head width `d_h`, ordinary MHA stores K and V for all `H` heads. MQA shares one K/V head; GQA uses `H_kv` K/V heads:

```text
MHA cache values per layer = 2 · T · H · d_h
GQA cache values per layer = 2 · T · H_kv · d_h
```

MLA caches a lower-dimensional latent from which key/value information is recovered. All three retain a representation for each token, so capacity and bandwidth improve while cache length remains `O(T)`.

### Visit fewer token rows

Structured sparsity fixes an access pattern such as a sliding window, blocks, or global tokens. A window of width `W` changes pairwise work toward `T · W` and can bound the live cache, but removes direct access outside the window.

Content-selected sparsity uses the query or a learned indexer to retrieve relevant history. It can preserve long-range access, but a complete accounting separates:

```text
payload cache       representations eventually attended to
index               keys or summaries used to find candidates
selection work      ranking, top-k, routing, and gathering
```

“Sparse” alone tells you neither which objects grow nor how candidates are found.

### Replace token rows with state

Factorizable linear attention reassociates computation so past keys and values accumulate in a fixed matrix. Gated delta rules, KDA, recurrent models, and SSMs refine how fixed state is updated and forgotten. They bound history state in `T`, but exact token rows are no longer available. [Beyond Linear Attention](linear-attention.md) builds the parallel/recurrent equivalence, fixed decay, and a recurrent/full hybrid.

### GLM-5.3-Flash: several compressions at once

The published text-layer weave is:

```text
[KDA, KDA, KDA, DSA] × 11, then KDA
 ─── fixed state ───   ─ selected token-addressable history
```

Its 34 KDA layers carry recurrent state. Its 11 DSA layers retain MLA-compressed token history and select a reported top-k 2,048 entries. IndexPool pools four index keys into one weighted key before selection:

```text
token history
   ├─ MLA latent payload ────────────────────────────┐
   └─ indexer keys → pool 4 into 1 → rank candidates ┤
                                                     ▼
                                      attend to selected payload
```

IndexPool shrinks the index, not the payload it points to. KDA, MLA, sparse selection, and pooled indexing therefore reduce four different bills. Z.ai's reported aggregate ratios describe the full GLM system and its comparison baseline; they are not constants for any individual mechanism.

## Concepts to internalize

- **Efficient attention is a map, not a lineage.** Exact kernels, narrower caches, sparsity, and recurrence preserve different contracts.
- **A cache can have payload, index, and selection bills.** Compressing one does not eliminate the others.
- **Asymptotics need dimensions and phases.** `O(T)` may describe state capacity, per-token decode work, or total sequence work.
- **Hybrids compose costs.** A mostly recurrent stack still pays growing cache in every retained token-addressable layer.
- **Wall-clock claims require a kernel and hardware context.** Tensor accounting identifies opportunities, not guaranteed speedups.

### What this guide does not turn into modules yet

- **Sparse and compressed token-addressable attention.** The mechanisms recur across model families, but the course still needs a laptop-scale scaffold that teaches selection rather than library plumbing.
- **Custom attention kernels.** Kernel concepts belong in [Beyond Hardware](hardware.md); a Metal implementation remains a possible lab.
- **State-space derivations.** The recurrent destination is relevant, while the full control-theory lineage is broader than this map.

---
## Reading routes

Exact execution:

- **Dao et al., [“FlashAttention”](https://arxiv.org/abs/2205.14135) (2022).** Exact dense attention reorganized around memory traffic.

Per-token cache compression:

- **Ainslie et al., [“GQA”](https://arxiv.org/abs/2305.13245) (2023).** The continuum between one and many K/V heads.
- **DeepSeek-AI, [“DeepSeek-V2”](https://arxiv.org/abs/2405.04434) (2024).** MLA as learned latent cache compression.

Sparse selection:

- **Yuan et al., [“Native Sparse Attention”](https://arxiv.org/abs/2502.11089) (2025).** Local, compressed, and selected branches.
- **DeepSeek-AI, [“DeepSeek-V3.2”](https://arxiv.org/abs/2512.02556) (2025).** A learned sparse indexer and selected-token attention.

Fixed recurrent state:

- **Katharopoulos et al., [“Transformers are RNNs”](https://arxiv.org/abs/2006.16236) (2020).** Continue into the [linear-attention build](linear-attention.md).

---
## Comprehension route

After reading, you should be able to take an attention claim and annotate its phase, stored state, work per new token, context-dependent objects, retained access contract, and hardware assumptions. If the source does not disclose one of those, mark it unknown rather than filling it in from a neighboring architecture.
