# G2C Brief — Qwen3.8-27B: a frontier recipe at workstation scale

> **Release question:** *What has to change inside a 27B transformer before a million-token context is affordable?*

Qwen3.8-27B is the small open sibling of the first Qwen-Max-class generation
to ship open weights. It is a dense 27.8B-parameter model that sees images and
video, thinks by default, and claims a quarter-million-token native context
extensible to a million — and three quarters of its "attention" layers are not
attention at all. They are Gated DeltaNet, a recurrent linear-attention
design whose state does not grow with the sequence. The
[Beyond Linear Attention](../beyond/linear-attention.md) module builds exactly
this hybrid pattern at toy scale; this release is that lesson shipped as a
flagship product decision.

The release is also a different kind of reading exercise than a DeepSeek-style
report. **There is no technical report.** The model card's own citation block
points at a blog post. The ground truth for this model is its published
`config.json` — a file in which, by this point in the course, essentially
every field names something you have built. This Brief reads the release the
way the course equips you to: card first, config as arbiter, blog last and
labeled.

## Snapshot and evidence policy

| Field | Snapshot |
|---|---|
| Release | Qwen3.8 generation announced August 3, 2026 (hosted Qwen3.8-Max); Qwen3.8-27B open weights published August 14, 2026 (UTC) |
| Models | Qwen3.8-27B / -FP8 (Apache-2.0); Qwen3.8-2.4T-A95B / -FP8 (custom "Qwen3.8-Max License"); hosted Qwen3.8-Max (closed; adds vision to the 2.4T, 1M default context, built-in tools) |
| Release source | [Official blog: "Qwen3.8-Max: A New Bar for Coding and Cowork"](https://qwen.ai/blog?id=qwen3.8) |
| Primary source | [Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B) and its [`config.json`](https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/config.json) |
| Technical report | **None exists.** The card's BibTeX cites the blog post; there is no arXiv paper and no generation GitHub repo at the snapshot date |
| Last verified | August 15, 2026 |

The labels below matter:

- **Reported** means Qwen reports it. It is not an independent replication.
- **Derived** means arithmetic from reported values.
- **G2C interpretation** means this Brief is connecting the evidence to the
  course.
- **Not disclosed** means the source does not support a stronger conclusion.

One subject-discipline rule this release forces: the blog's showcase results
describe **Qwen3.8-Max**, the closed hosted model. The 27B's numbers live on
its model card. This Brief never lets a Max claim stand in for the 27B.

## The release in one page

**Reported.** Qwen3.8-27B is a dense causal LM of 27,781,427,952 parameters
(BF16, from the published safetensors metadata) with an attached vision
encoder — "a native vision-language model that understands images and
videos." Its 64 text layers follow a strict weave the card writes as
`16 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))`: three
recurrent linear-attention layers, then one full attention layer, sixteen
times. Native context is 262,144 tokens, extensible to 1M with a published
YaRN recipe. The vocabulary is 248,320 tokens. A one-layer multi-token
-prediction head is trained in. Thinking mode is on by default with
`reasoning_effort` levels `xhigh` (default), `medium`, `low`, and can be
disabled per request. Weights are Apache-2.0, with an official FP8 twin.

**Reported.** The MoE sibling, Qwen3.8-2.4T-A95B, scales the same weave to
2.4T total / 95B active parameters (512 experts, 10 routed + 1 shared) — and,
notably, is text-only: in this generation the *small* open model got the
vision tower, while vision at the top end stays in the closed Max.

**Derived — what the weave buys.** From the config, only the 16 full-attention
layers build a KV cache: 4 KV heads × 256 dims for K and V at BF16 is
`16 × 2 × 4 × 256 × 2 bytes = 64KiB` per token — 16GiB at the native 262K
context, 64GiB at 1M. Had all 64 layers been full attention at the same GQA
geometry, those figures would be 4× larger (256GiB at 1M). The 48 DeltaNet
layers instead carry a fixed-size state: one 128×128 matrix per value head —
`48 heads × 128 × 128` per layer, held in float32 per the config — about
144MiB *total*, independent of context length. Past roughly the first
thousand tokens, the recurrent layers are already storing less than full
attention would; at a million tokens they store about three orders of
magnitude less. That is the entire economic argument for the hybrid, and it
is arithmetic you can do from the config alone.

**Derived — what it costs to hold.** 27.78B parameters is ~55.6GB at BF16 and
~27.8GB at FP8. This is the rare Brief subject that touches the course's own
hardware ladder: the FP8 checkpoint fits the 64GB Stretch track outright, and
community 4-bit quantizations (~14–16GB) reach ordinary Standard-track
machines. Nothing in this Brief requires you to run it — but unlike a
trillion-parameter release, nothing about physics forbids it either.

## Architecture: three recurrent layers for every exact one

### The weave, read from the config

```text
config.json:  layer_types (64 entries)
  [ linear, linear, linear, full,   ← repeated 16 times
    ... ]                              full_attention_interval: 4

per-token sequence memory:
  48 × Gated DeltaNet   state: 48 × (128×128) fp32 / layer   ≈ constant 144MiB
  16 × Gated Attention  cache: 4 KV heads × 256 × BF16       64KiB / token, growing
```

[Beyond Linear Attention](../beyond/linear-attention.md) builds both halves of
this design at mechanism scale — the linear-attention state update that makes
sequence memory constant, and the hybrid interleave that keeps a few exact
-attention layers for the retrieval tasks pure recurrence handles poorly. That
core is **built in g2c**. What the course version does not have is DeltaNet's
specifics: the delta-rule state update (write a correction, not a raw
accumulation), the gating that decays state, and the short convolution in
front (`linear_conv_kernel_dim: 4`). Those refinements are a **conceptual
bridge** — same state-versus-cache question, sharper answer.

**G2C interpretation.** The 3:1 ratio is a claim about workload: most layers'
work survives compression into a running summary, but some fraction of
language-model behavior — exact copying, long-range retrieval — measurably
needs an uncompressed look at the history. Qwen committing its flagship
generation to this layout (both the 27B and the 2.4T use it) is strong
recurrence evidence that hybrids, not pure linear attention, are the shape
that survives contact with production.

### The attention that remains is gated, grouped, and barely rotated

The 16 full layers are grouped-query attention — 24 query heads sharing 4 KV
heads — with an output gate (`attn_output_gate: true`) modulating what leaves
the layer. [Module 08](../modules/08-multi-head-attention.md) builds the
all-heads-equal baseline; GQA is the cache-shrinking variant you met
concretely in [Module 13B](../modules/13b-lora.md), where SmolLM's
non-square `v_proj` is GQA showing up in a weight shape. The output gate
itself is a small **not yet covered** refinement.

Positions are handled with partial RoPE: only a quarter of each head's
dimensions rotate (`partial_rotary_factor: 0.25`), at a base frequency of
10,000,000, with the multimodal M-RoPE variant interleaving position across
text, image height, and width. [Module 05](../modules/05-embeddings.md) builds
the position problem this answers; the specific long-context tuning — high
theta, partial rotation, YaRN extension to 1M — is a **conceptual bridge**.

### A 248,320-token vocabulary is a quarter-billion-row commitment

**Derived.** With untied embeddings at hidden size 5120, the input table and
output head are `2 × 248,320 × 5120 ≈ 2.54B` parameters — about 9% of the
whole model spent on the vocabulary interface [Module 04](../modules/04-tokenizer.md)
taught you to price. The tokenizer is byte-level BPE of the familiar kind.

### The vision tower rides in front

A 27-layer, 1152-wide encoder patches images at 16×16, merges 2×2 spatially,
and projects into the 5120-dim text stream; video is sampled at 2 fps with a
budget the card sizes for "hour-scale" input. [Beyond Multimodal](../beyond/multimodal.md)
builds the load-bearing move — patches becoming tokens in a shared sequence —
at MNIST scale, so this is a **conceptual bridge** with the mechanism already
in hand. The encoder's provenance and training are **not disclosed**.

### MTP, third sighting

`mtp_num_hidden_layers: 1`, sharing the model's embeddings. The cards say the
model is "trained with multiple steps" of MTP but never quantify decoding
speedup or acceptance. [Module 06](../modules/06-language-models.md) builds
one-token prediction; MTP remains **not yet covered** — and this is now the
third independent family shipping it, which is exactly the recurrence the
roadmap's alternative/parallel-generation watchlist is waiting on.

## Training: a card that says almost nothing

**Reported.** "Training Stage: Pre-training & Post-training." That sentence is
nearly the complete disclosure. **Not disclosed:** token count, data
composition, staging, optimizer, training precision, hardware, and any
distillation relationship between the 2.4T/Max and the 27B — the card claims
none in either direction, so treat "the 27B is distilled from Max" as rumor
shape, not fact.

The generation blog describes post-training only at the systems level:
RL scaled by "continuously scaling decoupled real environments" across task,
workspace, and harness axes; a "Universal Reward System" spanning
execution-based checking, rubric-conditioned adjudication over text and
rendered visual output, and agentic inspection; and an online data balancer
suppressing inter-batch gradient variance. No RL algorithm is named anywhere.
[Beyond RL](../beyond/rl.md) gives you the vocabulary these claims assume —
environments, verifiable rewards, rubric judges — as **conceptual bridges**,
but there is no disclosed mechanism here to map more precisely.

**G2C interpretation.** The three reward modes are the course's reward
taxonomy at production scale: execution-based checking is
[Beyond RL](../beyond/rl.md)'s verifiable reward, rubric adjudication is the
LLM-judge pattern [Module 15](../modules/15-evaluation.md) warns you to
calibrate, and "agentic inspection" is a judge that acts before scoring. The
blog treats reward *plumbing*, not reward *math*, as the scaling frontier.

### Thinking is a default with a cache argument

Thinking mode is on unless disabled per request; `preserve_thinking` keeps
prior turns' thinking blocks in context by default, which the card notes
"improves KV cache utilization" — retained blocks keep the prefix stable, so
[Module 16](../modules/16-inference.md)'s prefix-reuse logic keeps its hits.
Recommended sampling splits by mode: temperature 1.0 / top-p 0.95 thinking,
temperature 0.7 / top-p 0.80 with presence penalty 1.5 non-thinking — the
[Module 11](../modules/11-sampling.md) warper stack, shipped as a per-mode
serving contract.

## The harness is part of the measurement

**Reported.** The 27B card's headline numbers include Terminal Bench 2.1
(Terminus) 73.0, SWE-bench Pro 61.7, LiveCodeBench v6 90.3, GPQA Diamond
89.2, OSWorld-Verified 84.3, and MathVision 90.0 (94.6 with a code
interpreter) — with the card itself noting most agentic evaluations ran "with
the Claude Code harness," and that several suites use corrected annotations
with re-evaluated baselines.

**G2C interpretation.** A model-card number of the form
model × harness × scaffold is a system measurement, which is
[Beyond Harness](../beyond/harness.md)'s central claim arriving on a first-party
card: the vendor's own results are declared harness-dependent. The blog's
spectacular long-autonomy showcases — a 16-day autonomous coding run, a
125-hour paper reproduction, a contest placing above 87% of human teams —
are all **Max** results and say nothing measured about this checkpoint.

## Coverage map

| Qwen3.8-27B mechanism | G2C connection | Coverage |
|---|---|---|
| Transformer, causal objective, decoding | Modules 07–11 | **Built in g2c** |
| Linear-attention state + hybrid interleave | Beyond Linear Attention | **Built in g2c** at mechanism scale |
| Gated DeltaNet specifics (delta rule, gates, conv) | Beyond Linear Attention | **Conceptual bridge** |
| GQA KV-head sharing | Module 08 baseline; seen in Module 13B's SmolLM | **Conceptual bridge** |
| Attention output gating | Module 08 baseline | **Not yet covered** |
| Partial RoPE / M-RoPE / YaRN extension | Module 05 | **Conceptual bridge** |
| 248K BPE vocabulary and its parameter price | Module 04 | **Built in g2c** at mechanism scale |
| Vision patches as tokens | Beyond Multimodal | **Conceptual bridge** |
| KV-cache economics, prefix reuse, FP8 serving | Module 16 | **Conceptual bridge** |
| Thinking budgets and per-mode sampling contracts | Modules 11, 13 | **Conceptual bridge** |
| RL environments, verifiable + rubric rewards | Beyond RL, Module 15 | **Conceptual bridge** (mechanism not disclosed) |
| Harness-dependent evaluation | Beyond Harness | **Built in g2c** at mechanism scale |
| MTP / parallel generation | Module 06 baseline | **Not yet covered** |
| MoE at 2.4T (sibling) | Beyond MoE | **Built in g2c** at mechanism scale |

## Claims versus demonstrated evidence

### What the primary sources establish directly

- The weights, config, tokenizer, FP8 twin, and Apache-2.0 license are public
  and downloadable; every architectural number in this Brief is checkable in
  the published files, and the parameter count comes from safetensors
  metadata rather than marketing copy.
- The hybrid layout, GQA geometry, partial RoPE, vocabulary, MTP head, and
  native context length are facts of the artifact, not claims about it.

### What remains reported rather than independently established here

- All benchmark rows are Qwen-run, several under a third-party coding harness
  and several against re-annotated versions of public suites; no independent
  evaluation existed at the snapshot date (the weights were hours old).
- "Extensible up to 1,000,000 tokens" is a supported configuration plus a
  YaRN recipe, not evidence that quality holds across that window; the card
  publishes no long-context evaluation for the 27B.
- The vision claims ("hour-scale videos") are capability descriptions with
  configuration guidance, not benchmarked video results on this checkpoint.
- Pretraining and post-training are essentially undisclosed: no token count,
  no data statement, no optimizer, no RL algorithm, no ablations — the
  absence is broader than for any release with a technical report, and it is
  permanent unless a report appears.
- FP8 "nearly identical" quality is the vendor's characterization.

## A practical reading route

No report exists, so the route runs through the artifact:

1. **The [`config.json`](https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/config.json), top to bottom.**
   Nearly every key names something you built: `layer_types` is the hybrid
   weave, `num_key_value_heads` is GQA, `partial_rotary_factor` is Module
   05's rotation applied to a quarter of each head, `mtp_num_hidden_layers`
   is the parallel-generation head, `vision_config.patch_size` is Beyond
   Multimodal's patching. Reading this file cold is the course's real exam.
2. **The [model card](https://huggingface.co/Qwen/Qwen3.8-27B)** — for the
   layout formula, context/YaRN recipe, sampling contracts, and the benchmark
   table's own harness caveats.
3. **The KV arithmetic**, by hand, from the config — reproduce this Brief's
   64KiB-per-token figure and the 4× hybrid saving before trusting either.
4. **The [blog](https://qwen.ai/blog?id=qwen3.8) last**, with the subject rule
   in force: tag each claim Max or generation-wide, and let none of the
   showcase results attach to the 27B.

## After this Brief, you should be able to

- compute, from the config alone, why 48 recurrent layers plus 16 attention
  layers make a 262K–1M context affordable where 64 attention layers would
  not;
- explain what a Gated DeltaNet layer stores instead of a KV cache, and why
  its memory does not grow with the sequence;
- read `layer_types`, GQA head counts, partial RoPE, and the MTP head out of
  a raw config file without a paper to interpret them for you;
- price a 248K vocabulary in parameters and say where that cost lives;
- state which of the release's numbers are properties of the downloadable
  artifact and which are vendor-run evaluations;
- explain why "evaluated with the Claude Code harness" makes a benchmark a
  system measurement, and why the Max showcases say nothing about this
  checkpoint;
- name the two mechanisms in this release still accumulating watchlist
  evidence (parallel generation via MTP; hybrid recurrence as the production
  answer to long context) and what would promote them.
