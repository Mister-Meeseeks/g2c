# G2C Brief — GLM-5.3-Flash: hybrid attention becomes the product

> **Release question:** *What changes when serving cost—not maximum parameter scale—is the architecture's first-class constraint?*

GLM-5.3-Flash is easy to misread from its name. It is not GLM-5.3 with fewer
experts, a smaller quantization, or a faster serving tier. Z.ai describes a
newly trained 320B-total / 18B-active base with a new 45-layer architecture, a
30T-token multimodal corpus, and public weights on release day. Three quarters
of its text layers replace growing token caches with recurrent linear-attention
state; the remaining quarter retain sparse, compressed access to the full
history. The serving system then attacks the same bill again with quantization,
state replay, and disaggregated multimodal execution.

That makes this release a compact test of the course's efficiency thesis. The
model is not cheap because of one mechanism. MoE reduces active weights,
linear attention reduces sequence-length state, sparse attention preserves
selected global retrieval, mHC changes the residual topology, and the serving
stack is co-designed around the resulting workload. This Brief separates
those layers and marks what the artifact proves from what the announcement
only reports.

## Snapshot and evidence policy

| Field | Snapshot |
|---|---|
| Release | GLM-5.3-Flash, announced August 26, 2026 |
| Release source | [Official announcement: “GLM-5.3-Flash: Frontier Intelligence, Flash Cost”](https://z.ai/blog/glm-5.3-flash) |
| Weight source | [Official FP8 checkpoint](https://huggingface.co/zai-org/GLM-5.3-Flash) at commit [`3f1971b7`](https://huggingface.co/zai-org/GLM-5.3-Flash/tree/3f1971b7b5f7a528c9c4ef6212c8785298a8c24a) · [official BF16 checkpoint](https://huggingface.co/zai-org/GLM-5.3-Flash-BF16) at commit [`b1967181`](https://huggingface.co/zai-org/GLM-5.3-Flash-BF16/tree/b1967181a3917ae70a437f4884748f6b8e3a1f4d) |
| Architecture source | Published [`config.json`](https://huggingface.co/zai-org/GLM-5.3-Flash/raw/main/config.json), processor config, and safetensors index |
| Lineage sources | [GLM-5 technical report, v1](https://arxiv.org/abs/2602.15763) · [mHC paper](https://arxiv.org/abs/2512.24880) · [ReplaySSM explanation](https://dao-lab.ai/blog/2026/replayssm/) |
| License | MIT, from the checkpoint's published `LICENSE` |
| Last verified | August 26, 2026 |

The labels below matter:

- **Reported** means Z.ai reports it. It is not an independent replication.
- **Derived** means arithmetic from disclosed values, with the inputs shown.
- **G2C interpretation** means this Brief is connecting the evidence to the
  course.
- **Not disclosed** means the available primary sources do not support a
  stronger conclusion.

**There is no GLM-5.3-Flash technical report at this snapshot.** The model card
cites the February GLM-5 report, but that report predates and does not describe
this architecture. It is useful lineage for DSA, MLA, MoE, MTP, and the broader
training stack; it is not evidence that Flash reused GLM-5's exact data mix,
optimizer, or post-training recipe. The announcement and published artifact
are the primary sources for Flash itself.

One more name guardrail:

```text
GLM-5.2 base (744B / 40B active)
      │
      └─ same base + scaled post-training ─► GLM-5.3 (August 14)

new 320B / 18B multimodal base
      └─ hybrid KDA + sparse attention ────► GLM-5.3-Flash (August 26)
```

The earlier [GLM-5.3 Brief](glm-5.3.md) reads the frozen-base post-training
experiment. This Brief reads a different model. Z.ai's Flash benchmark table
compares against GLM-5.2, not against the August 14 GLM-5.3, so “Flash” should
not be read as a smaller checkpoint with an otherwise controlled delta.

## The release in one page

**Reported.** GLM-5.3-Flash is the first natively multimodal GLM-5-series
model. Z.ai names 320B total parameters, 18B active parameters, 45 text layers,
a one-million-token context, and a 30T-token multimodal pretraining corpus. It
claims performance above GLM-5.2 across its published coding and agentic table
at what it calls one-tenth the price, and reports 3× lower attention compute
and 4.4× less KV cache than GLM-5.3 under its per-head/per-layer comparison.
Those are vendor measurements, not consequences of the parameter counts alone;
the announcement does not give the “one-tenth” claim a precise price
denominator.

**Artifact fact.** The text config contains 34 `linear_attention` layers and
11 `deepseek_sparse_attention` layers in this exact order:

```text
[KDA, KDA, KDA, DSA] × 11, then one final KDA layer
```

The same config specifies a 4096-wide residual stream; 288 routed experts with
8 selected per token and one shared expert; the first three FFN layers dense;
MLA rank 512 and sparse top-k 2048; four-way mHC residual streams; one
multi-token-prediction layer; and a vocabulary of 154,880 entries. The vision
tower is a 24-layer, 1024-wide encoder using 14×14 spatial patches and
two-frame temporal patches.

**Derived.** `34 / 45 = 75.6%` of the text layers use recurrent linear
attention, while `11 / 45 = 24.4%` retain token-addressable sparse attention.
The linear config names 64 heads of width 128, so the KDA matrix state implied
by that geometry is `64 × 128 × 128 = 1,048,576` values per linear layer,
independent of context length. Across all 34 KDA layers that is about 35.7M
matrix-state values. Runtime state precision and additional convolution/gate
state are **not disclosed**, so converting this count into a serving-memory
claim would require an assumption.

**Derived.** `18 / 320 = 5.6%` is the ratio of named active to total
parameters. It is a useful headline for conditional computation, not a direct
expert-selection fraction: every token also uses embeddings, attention,
routers, shared experts, normalization, and the output head.

**Artifact fact, laptop boundary.** The published safetensors metadata totals
328.3GB for the official FP8 artifact and 642.6GB for the BF16 artifact before
runtime state. Even an idealized 4-bit copy of 320B parameters is roughly
160GB before metadata and cache. This model is far outside the course's
16–64GB M-series boundary. Open weights make the design inspectable; they do
not make local reproduction or useful local inference a course requirement.

### The vendor's same-table delta

These rows are **Reported** from Z.ai's release table. The delta is **Derived**
arithmetic against the vendor's own GLM-5.2 column:

| Benchmark | GLM-5.2 | GLM-5.3-Flash | Delta |
|---|---:|---:|---:|
| Terminal Bench 2.1 | 81.0 | 84.3 | +3.3 |
| DeepSWE v1.1 | 46.2 | 63.4 | +17.2 |
| Toolathlon Verified | 59.9 | 78.4 | +18.5 |
| AutomationBench v1.0.6 | 26.2 | 48.8 | +22.6 |
| HLE with tools | 54.7 | 55.3 | +0.6 |
| GDPval-AA v2 | 1504 | 1773 | +269 |

The clustered gains are suggestive—large on several environment-heavy agentic
suites, almost flat on HLE with tools—but this is not a clean architecture
ablation. Flash changes parameter scale, architecture, modalities, pretraining
data, and post-training at once. The table supports “this released system
scored differently,” not “linear attention caused the gain.”

## Architecture: three summaries, then one retrieval layer

### KDA carries fixed state through most layers

The config calls the 34 recurrent layers `kda_layers`, and their position in
the layer list matches the release's linear-attention description. Kimi Delta
Attention is a gated delta-rule descendant of linear attention: instead of
retaining one K/V row per token, it updates a fixed matrix state. A short
convolution (`short_conv_kernel_size: 4`) and learned gates add local and
content-dependent structure around that recurrence.

[Beyond Linear Attention](../beyond/linear-attention.md) builds the
load-bearing economics directly: factorizable similarity, a fixed recurrent
state, and a hybrid stack that periodically restores exact token access. That
is **built in g2c** at mechanism scale. KDA's delta-rule correction,
token-dependent gates, short convolution, and fused production kernels are a
**conceptual bridge**, not the code students built.

The release blog says linear attention “captures local dependencies through
state modeling.” Read that as the releasing organization's description, not a
formal receptive-field limit. A recurrent state can carry information over
arbitrarily many steps; the constraint is that all such information must
survive compression into a fixed-capacity state.

### DSA keeps selected global access in eleven layers

Every fourth layer is `deepseek_sparse_attention`, with one extra KDA layer at
the end. These layers combine three ideas:

```text
full token history
   → MLA latent cache (rank 512 per token)
   → IndexPool (four indexer keys → one pooled key)
   → learned sparse selection (top 2048)
   → attention over selected history
```

**Reported.** IndexPool compresses four indexer key vectors into one through
weighted pooling, reducing indexer latency and memory at one-million-token
context. **Artifact fact.** The config pins `index_kpool: 4`,
`index_topk: 2048`, and `kv_lora_rank: 512`.

This is not fixed-state attention. The eleven DSA layers still retain a
token-addressable latent history that grows with `T`; they merely store a
narrower representation and consult a selected subset. The 34 KDA layers stop
paying that sequence-length cache entirely. Z.ai's reported 4.4× average KV
reduction versus GLM-5.3 is therefore the combined result of fewer cached
layers, MLA geometry, IndexPool, and the comparison model's own layout—not a
generic “linear attention saves 4.4×” constant.

[Module 08](../modules/08-multi-head-attention.md) builds the projections,
[Module 16](../modules/16-inference.md) builds the cache whose growth matters,
and [Beyond Linear Attention](../beyond/linear-attention.md) separates
compressed caches, sparse selection, and recurrence. MLA + DSA + IndexPool is
a **conceptual bridge**; g2c does not implement a learned index or pooled
indexer cache.

### mHC makes the residual stream four lanes wide

The checkpoint sets `mhc: true`, `hc_mult: 4`, and 20 Sinkhorn iterations.
Ordinary Transformer residual connections carry one stream from layer to
layer. Manifold-Constrained Hyper-Connections carry several streams and learn
how to mix them, while constraining the mixing matrices to be doubly
stochastic. The constraint restores a bounded, identity-like information path
that unconstrained hyper-connections can lose.

[Module 09](../modules/09-transformer-block.md) provides the ordinary residual
baseline. The draft [Beyond mHC](../beyond/mhc.md) now specifies a laptop-scale
build around expanded residual streams, Sinkhorn projection, and propagation
diagnostics, but its scaffold has not shipped. mHC therefore remains **not yet
covered** by executable course work. Its appearance here after DeepSeek V4 is
the cross-family evidence that promoted the topic from the watchlist to that
module draft.

### MoE shrinks the active model without shrinking capacity as far

The first three FFN layers are dense; the other 42 are sparse MoE layers. Each
has 288 routed experts, selects 8 per token, and adds one shared expert. That
is the capacity-versus-per-token-compute split built in
[Beyond MoE](../beyond/moe.md): **built in g2c** at mechanism scale. The
production design adds sigmoid routing, expert-score correction biases, expert
parallelism, and a much finer expert panel than the toy.

The release's 18B active count does not mean inference behaves like a dense
18B model. All 320B weights must be stored or distributed, routing and expert
dispatch move tokens across devices, and attention/state costs sit outside
the expert count. “Active parameters” is one line of the bill, not latency.

### Native multimodality still has a vision frontend

The artifact contains a 24-layer vision tower with 14×14 patches, 2×2 spatial
merging, and two-frame temporal patches, projected into the 4096-wide text
stream. The processor allows 16–8,000 image tokens and up to 240,000 video
tokens at 2 fps. Images and video therefore become long sequences of learned
vectors before the text backbone reasons over them.

[Beyond Multimodal](../beyond/multimodal.md) builds the load-bearing interface:
patches become vectors, a projector matches the language width, and those
vectors enter a shared sequence. That mechanism is **built in g2c** at toy
scale; this learned vision tower, temporal patching, resolution policy, and
large token budgets are a **conceptual bridge**. “Native multimodal” means the
release trained the system as a multimodal model; it does not mean raw pixels
enter an unchanged language transformer.

## Training: 30T multimodal tokens, few recipe details

**Reported.** GLM-5.3-Flash begins from a newly trained base and uses a
30T-token multimodal pretraining corpus. Z.ai says the architecture and corpus
were co-optimized for capability and efficiency, and shows base-model results
against earlier GLM bases.

**Not disclosed.** The release does not provide the text/image/video token
mix, deduplication or filtering recipe, sequence-length curriculum, optimizer,
learning-rate schedule, batch size, training hardware, total FLOPs, or an
architecture ablation. The GLM-5 report used Muon and disclosed staged
long-context work, but this release does not say Flash inherited either choice.
Those facts must not be imported by family resemblance.

### Visual coding is trained as an observe–act–verify loop

**Reported.** Z.ai developed visual-coding data synthesis around self-visual
judgment and test-time improvement. Trajectories interact with environments,
inspect rendered output, and revise it. Frontend coding also uses reinforcement
learning with environment feedback and agent-based verification grounded in
user flows.

**G2C interpretation.** [Module 16B](../modules/16b-synthetic-data.md)'s
propose–filter funnel has become a trajectory generator, while
[Module 15](../modules/15-evaluation.md)'s separation between artifact and
evaluator has become render-and-inspect feedback. [Beyond RL](../beyond/rl.md)
provides the environment/reward vocabulary. The exact RL algorithm, reward
weights, judge model, environment counts, rejection rates, and amount of
improvement attributable to this pipeline are **not disclosed**, so the
mechanism is a **conceptual bridge** rather than a production recipe the
course could reproduce.

## Serving: the architecture is only half the efficiency claim

**Reported.** Z.ai anonymously served the model as `ox-alpha` before release,
then served the traffic on a large cluster of Chinese AI accelerators. Its
dedicated SGLang-based engine combines:

- tensor parallelism for linear attention and the language-model head;
- ReplaySSM;
- W8A8 compute quantization;
- hybrid INT8/FP8/BF16 cache quantization;
- Layer Split; and
- Encode–Prefill–Decode (EPD) disaggregation across separately scheduled
  multimodal encoding, prompt prefill, and token-by-token decode workers.

Z.ai reports a 3× end-to-end improvement over its initial baseline on the same
hardware, reaching per-token cost comparable to mainstream NVIDIA GPUs. The
hardware models, baseline software, workload mix, latency distribution,
throughput, cluster size, power, and absolute cost are **not disclosed**.
The 3× figure therefore establishes an internal before/after claim, not a
portable speedup for the public checkpoint.

[Module 16](../modules/16-inference.md) supplies the phase distinction: encode,
prefill, and decode have different shapes and bottlenecks, so assigning them
to different worker pools can improve utilization. ReplaySSM adds a production
twist for recurrent layers: it caches recent recurrence inputs and can replay
them to reconstruct state, trading compute for state movement under the needs
of serving and speculative execution. Those serving mechanisms are
**conceptual bridges**; distributed scheduling, device communication, and
mixed cache precision are **not yet covered**.

**G2C interpretation.** “Flash cost” is a stack claim:

```text
fewer active expert weights
  × fewer layers
  × fixed-state recurrence in 34/45 layers
  × sparse compressed caches in 11/45 layers
  × low-precision compute and cache
  × phase-specific cluster scheduling
  = reported serving economics
```

No single row licenses the final claim by itself. In particular, publishing a
hybrid-attention config does not prove the API price or throughput, while a
serving benchmark does not isolate the architecture's contribution.

## Evaluation: model × harness × context policy

The release is unusually explicit about several evaluation settings:

- Terminal Bench 2.1 runs inside Claude Code 2.1.207 with a six-hour timeout
  and up to 65,536 generated tokens.
- DeepSWE uses `mini-swe-agent`, a six-hour timeout, and 400K context.
- Agents' Last Exam uses the Claude Code harness, maximum reasoning effort,
  one-million-token context, and 64K maximum output.
- HLE with tools permits 163,840 generated tokens, manages a 300K context,
  and uses GPT-5.6-luna (medium) as judge.
- Toolathlon Verified reports pass@1 averaged over three independent runs
  through the official evaluation service.
- Vision evaluations use different context limits and image/video ingestion
  policies; raw video is used where supported and frame extraction otherwise.

That disclosure is valuable precisely because it prevents the score from
being mistaken for a model-only property. [Beyond Harness](../beyond/harness.md)
builds this lesson: tools, timeouts, retry policy, context management, and judge
configuration all participate in the measured system. The cross-vendor table
is **Reported** and useful for hypothesis generation; it is not an independent
leaderboard with a universal harness.

## Coverage map

| GLM-5.3-Flash mechanism | G2C connection | Coverage |
|---|---|---|
| Transformer, causal objective, decoding | Modules 07–11 | **Built in g2c** |
| Recurrent linear-attention state + hybrid interleave | Beyond Linear Attention | **Built in g2c** at mechanism scale |
| KDA delta rule, token gates, short convolution | Beyond Linear Attention | **Conceptual bridge** |
| MLA latent cache + DSA learned sparse selection | Modules 08, 16 + Beyond Linear Attention | **Conceptual bridge** |
| IndexPool pooled indexer keys | Module 16 cache reasoning | **Conceptual bridge** |
| MoE total/active split and routing | Beyond MoE | **Built in g2c** at mechanism scale |
| 4-stream mHC residual topology | Module 09 + draft Beyond mHC | **Not yet covered** — lesson drafted, executable build not shipped |
| Vision patches projected into the shared stream | Beyond Multimodal | **Built in g2c** at mechanism scale |
| Production vision tower, temporal patches, token budgets | Beyond Multimodal | **Conceptual bridge** |
| Visual trajectory synthesis and environment feedback | Modules 15, 16B + Beyond RL | **Conceptual bridge** |
| MTP auxiliary prediction layer | Modules 06, 11, 16 | **Not yet covered** |
| Quantized recurrent/sparse cache serving | Module 16 | **Conceptual bridge** |
| ReplaySSM, Layer Split, EPD cluster scheduling | Module 16 baseline | **Not yet covered** |
| Harness-dependent agent evaluation | Beyond Harness | **Built in g2c** at mechanism scale |

## Claims versus demonstrated evidence

### What the primary sources establish directly

- The MIT-licensed FP8 and BF16 weights are public and ungated. Their configs,
  processors, tokenizer, and safetensors indices make the 45-layer weave, MoE
  geometry, mHC settings, context ceiling, vision tower, MTP layer, and stored
  checkpoint sizes inspectable facts.
- The layer list independently confirms the release's central architecture
  claim: 34 fixed-state linear-attention layers interleaved with 11 sparse
  token-addressable layers.
- The processor config confirms that “multimodal” means concrete image and
  video ingestion machinery, not a text model wrapped by an undocumented API.
- The GLM-5.2 → Flash delta table is reproducible arithmetic from Z.ai's
  published table, with several harness details disclosed in its footnotes.

### What remains reported rather than independently established here

- The 30T-token corpus size, “one-tenth the price,” 3× attention-compute
  reduction, 4.4× KV-cache reduction, and 3× serving-stack improvement are
  vendor claims. The release does not publish enough absolute measurements or
  ablations to reproduce their attribution, and it does not state the exact
  denominator for “one-tenth.”
- “Preserving precise long-context capabilities” is not established by the
  one-million-token config field. The benchmark footnotes use contexts from
  164K to 1M and context-management policies that vary by task; no controlled
  quality-vs-distance curve is published.
- The new base is public, but its pretraining recipe is mostly not: corpus
  composition, optimizer, compute, hardware, long-context staging, and
  modality mixture are absent.
- Post-training details are narrower still. Visual environment feedback is
  described, but the general SFT/RL pipeline and its relationship to the
  August 14 GLM-5.3 recipe are not stated.
- The anonymous `ox-alpha` user-feedback claim and Chinese-accelerator serving
  story have no released traffic distribution, cluster inventory, or external
  audit.
- No evidence isolates KDA, IndexPool, mHC, the 30T corpus, smaller active
  count, or systems work as the cause of any individual benchmark delta.

At hundreds of gigabytes before runtime state, GLM-5.3-Flash can be inspected
from a laptop but not usefully reproduced on one. This Brief adds no cloud,
paid-API, checkpoint, or local-serving requirement. The course-scale build is
the relevant artifact: it lets you understand why this production stack has
the shape it does.

## A practical reading route

1. **The published [`config.json`](https://huggingface.co/zai-org/GLM-5.3-Flash/raw/main/config.json).** Count `layer_types` before reading any prose. Then locate `linear_attn_config`, `index_kpool`, `index_topk`, `kv_lora_rank`, the MoE fields, `mhc`, and `vision_config`.
2. **The [release architecture section](https://z.ai/blog/glm-5.3-flash).** For each efficiency sentence, label the object it shrinks: active weights, number of layers, recurrent state, indexer keys, selected tokens, or serving precision.
3. **The cache split, by hand.** Verify that 34 layers are fixed-state and 11 retain growing history. Do not turn that 34:11 count into the reported 4.4× figure; list the additional MLA and IndexPool factors needed.
4. **The [mHC paper](https://arxiv.org/abs/2512.24880), abstract and method overview.** Read only far enough to distinguish a four-stream residual topology from a wider hidden dimension.
5. **The serving section last.** Separate artifact mechanisms from deployment mechanisms, then mark which reported speedups lack an absolute denominator.
6. **The benchmark footnotes.** Treat each score as `model × harness × context policy × judge`, not a scalar property of the weights.

## After this Brief, you should be able to

- explain why GLM-5.3-Flash is not a quantized or distilled serving variant of
  the August 14 GLM-5.3, despite the shared generation name;
- read the 34:11 KDA/DSA weave directly from the config and identify which
  layers carry fixed state versus a growing token cache;
- distinguish KDA recurrence, MLA cache compression, DSA selection, and
  IndexPool index compression without collapsing them into “linear attention”;
- explain why 18B active parameters neither determine latency nor make 320B
  weights fit where a dense 18B model fits;
- trace an image or video from patches through the vision tower into the text
  residual stream, and name what “native multimodal” leaves unspecified;
- state what mHC changes relative to Module 09's ordinary residual connection;
- separate architecture, quantization, and distributed serving contributions
  inside the reported “Flash cost” claim; and
- label the release's benchmark, pretraining, and infrastructure statements as
  artifact fact, reported claim, derived arithmetic, interpretation, or not
  disclosed.
