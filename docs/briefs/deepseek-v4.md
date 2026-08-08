# G2C Brief — DeepSeek V4: reading a frontier model stack

> **Release question:** *What has to change—in the model, training recipe, and serving system—to make a sparse trillion-parameter model use a million-token context economically?*

DeepSeek V4 is a good test of whether the course has paid off. Its report is
full of unfamiliar names—CSA, HCA, mHC, Muon, OPD, DSec—but the stack underneath
them is recognizable. Tokens still enter a Transformer; attention communicates,
FFNs compute, next-token loss builds the base, post-training shapes behavior,
and an inference system maintains state while an agent acts. The novelty is in
how aggressively every layer of that stack has been redesigned together.

This Brief uses V4-Flash as the numerical anchor and brings in V4-Pro where
scale changes the conclusion. It does not attempt to run either model locally:
the smaller core target has 284 billion parameters before its attached
speculative-decoding module and sits far outside the course's laptop-scale
contract.

## Snapshot and evidence policy

| Field | Snapshot |
|---|---|
| Release | DeepSeek V4 Preview, announced April 24, 2026; Flash-0731 update, July 31, 2026 |
| Models | V4-Flash / V4-Flash-Base; V4-Pro / V4-Pro-Base; V4-Flash-0731 |
| Primary source | [DeepSeek V4 technical report, v1](https://arxiv.org/html/2606.19348v1) |
| Release source | [Official preview announcement](https://api-docs.deepseek.com/news/news260424/) |
| July update source | [Official V4-Flash-0731 model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) |
| Weight source | [Official DeepSeek V4 collection](https://huggingface.co/collections/deepseek-ai/deepseek-v4) |
| Last verified | August 4, 2026 |

The labels below matter:

- **Reported** means DeepSeek reports it. It is not an independent replication.
- **Derived** means arithmetic from reported values.
- **G2C interpretation** means this Brief is connecting the evidence to the
  course.
- **Not disclosed** means the source does not support a stronger conclusion.

## Update note — July 31, 2026

DeepSeek-V4-Flash-0731 is the official Flash release that supersedes the April
preview. It is not a new core V4 architecture. Its published configuration
retains the Flash target's 43 layers, CSA/HCA schedule, 256 routed experts with
six selected per token, mHC residual streams, and one-million-token context.
The update changes two other layers of the stack.

First, the target weights changed. **Reported:** DeepSeek describes 0731 as
having “substantially enhanced agentic capabilities” and publishes much higher
scores than Flash Preview and Pro Preview across Terminal-Bench 2.1, NL2Repo,
CyberGym, DeepSWE, Toolathlon-Verified, Agents' Last Exam, and two internal
DSBench sets. The code-agent evaluations use `max` reasoning effort,
`temperature = 1.0`, `top_p = 0.95`, and an announced but not-yet-released
minimal DeepSeek Harness.

**Not disclosed:** the model card does not give the additional post-training
data, token count, SFT/RL/OPD schedule, reward design, or ablations behind that
change. The evidence supports “same core architecture, new trained weights”; it
does not support reconstructing the update recipe or assigning the gains to one
intervention.

Second, the distributed checkpoint now includes **DSpark**, an attached
speculative-decoding module. A small semi-autoregressive drafter proposes a
block of future tokens; the full V4 target verifies several proposals in one
pass and accepts the valid prefix. This is intended to reduce the number of
expensive target-model passes without changing the target distribution. The
0731 configuration identifies a five-token draft block, a Markov head, and the
last three target layers as drafter inputs; the supplied vLLM recipe requests
up to seven speculative tokens. DeepSeek's separate
[DSpark technical report](https://arxiv.org/abs/2607.05147) reports 60–85%
faster per-user generation than its MTP-1 production baseline at matched
throughput.

The release also exposes `low`, `high`, and `max` reasoning-effort levels and
recommends a maximum output length of 384K tokens for `high` and `max`. Those
are post-training and serving controls, not evidence that the attention or MoE
architecture changed.

```text
April Flash Preview target
          │
          ├─ additional, undisclosed post-training ──► 0731 target weights
          │                                                │
          └─ same core V4 architecture                     + attached DSpark drafter
                                                           │
                                                           ▼
                                                V4-Flash-0731 package
```

This is exactly why Briefs pin snapshots. Capability can move substantially
without a new architecture, while latency can move through an attached serving
module without changing the target model's semantics.

## The release in one page

**Reported.** V4-Flash has 284B total parameters, 13B activated parameters,
43 Transformer layers, and a one-million-token context. V4-Pro has 1.6T total
parameters, 49B activated parameters, and the same context limit. Flash was
pretrained on 32T tokens and Pro on 33T. Both retain DeepSeekMoE and
multi-token prediction while adding hybrid compressed attention,
manifold-constrained hyper-connections, and a Muon/AdamW optimizer split. See
the report's [architecture](https://arxiv.org/html/2606.19348v1#S2) and
[pretraining setup](https://arxiv.org/html/2606.19348v1#S4).

**Derived.** One token in Flash activates at most a rough `13 / 284 = 4.6%` of
the model's named parameter count; for Pro the corresponding ratio is
`49 / 1,600 = 3.1%`. These ratios explain why total and active parameters are
both printed on the model card. They do **not** say that inference costs exactly
4.6% or 3.1% of a dense model: attention, routing, embeddings, memory movement,
precision, and hardware utilization remain active costs.

**Reported.** At a one-million-token context, DeepSeek estimates that Pro uses
27% of the single-token inference FLOPs and 10% of the KV cache of V3.2; Flash
uses 10% and 7%, respectively. Against a conventional BF16 GQA8 cache baseline,
the report estimates the V4 KV cache at roughly 2%. These are comparisons from
DeepSeek's own implementation and assumptions, not universal ratios for every
server.

The whole stack can be read as one pipeline:

```text
32T / 33T-token corpus
  └─ long documents, code, math, multilingual and midtraining agent data
                         │
                         ▼
                 V4 base checkpoint
  ┌─────────────────────────────────────────────────────────────┐
  │ Transformer blocks                                         │
  │                                                             │
  │ attention: SWA bootstrap → interleaved CSA / HCA            │
  │ FFN:       fine-grained routed experts + shared experts     │
  │ residual:  manifold-constrained hyper-connections           │
  │ training:  next-token + multi-token-prediction objectives   │
  └─────────────────────────────────────────────────────────────┘
                         │
            specialist SFT → specialist GRPO
                         │
              more than ten teacher policies
                         │
              on-policy logit distillation
                         ▼
                 unified instruct model
                         │
  ┌─────────────────────────────────────────────────────────────┐
  │ serving and agent system                                    │
  │ fused expert parallelism · heterogeneous/on-disk KV cache   │
  │ DSpark speculation · preemptible rollouts · real sandboxes  │
  └─────────────────────────────────────────────────────────────┘
```

The report's efficiency story is not one box in this diagram. It is the claim
that the boxes have been co-designed.

## Architecture: familiar Transformer, unfamiliar plumbing

### MoE separates capacity from per-token compute

V4 keeps the DeepSeekMoE pattern: many fine-grained routed experts plus shared
experts in place of each dense FFN. The router chooses only a small subset for
each token. That is the production-scale version of the distinction built in
[Beyond MoE](../beyond/moe.md): total parameters describe available capacity;
active parameters are a better first proxy for per-token computation.

V4 adds refinements that the toy implementation does not reproduce. It uses an
auxiliary-loss-free balancing strategy with a small sequence-level balance
term, changes the routing-score activation, and uses fixed token-id hash
routing in the first MoE blocks. Those are **conceptual bridges**, not details
we should quietly collapse into top-k routing.

### CSA and HCA compress the history in two different ways

The top-level V4 attention design is **Compressed Sparse Attention (CSA) plus
Heavily Compressed Attention (HCA)**. DeepSeek Sparse Attention is a component
inside CSA, not the name of the whole architecture.

CSA first compresses each group of `m` historical tokens into one KV entry. A
learned indexer then selects `k` compressed entries for each query; a small
sliding window restores fine local detail. HCA compresses much more aggressively
(`m' >> m`) but attends densely over the resulting short sequence. V4
interleaves the two layer types after two initial sliding-window layers.

```text
CSA:  full history → group compression → learned sparse selection → attention
HCA:  full history → much heavier compression ────────────────→ dense attention
SWA:  recent local window ────────────────────────────────────→ exact attention
```

[Module 07](../modules/07-attention.md) supplies the query/key/value mechanism,
[Module 16](../modules/16-inference.md) supplies the KV-cache reason for caring,
and [Beyond Linear Attention](../beyond/linear-attention.md) supplies the central
question: what state must survive from an ever-growing history? V4's answer is
not recurrent linear attention. It keeps an explicit—but compressed and partly
sparse—cache. That makes it a **conceptual bridge** to the other major escape
route from dense quadratic attention.

### mHC widens the residual highway

Ordinary residual connections carry one vector from block to block. V4's
manifold-constrained hyper-connections carry several residual streams and learn
how to mix them around each layer. The residual mixing matrix is constrained to
be doubly stochastic, bounding its spectral norm and making deep signal
propagation less prone to expansion. The report implements the constraint with
Sinkhorn normalization.

G2C teaches why residual connections and normalization stabilize the
Transformer in [Module 09](../modules/09-transformer-block.md), but it does not
build hyper-connections or constrained matrix manifolds. This mechanism is
**not yet covered**.

### Muon changes how most matrices are updated

V4 does not simply replace AdamW everywhere. The report applies Muon to most
matrix parameters and retains AdamW for embeddings, the prediction head,
normalization weights, and some mHC parameters. Muon approximately
orthogonalizes momentum updates with Newton–Schulz iterations before applying
them.

[Module 03B](../modules/03b-training.md) gives the optimizer contract and makes
the update inspectable, but Muon's matrix geometry and distributed state are
**not yet covered**. One release is not enough to promote it automatically;
the Brief records it as evidence for the roadmap's alternative-optimizer
watchlist.

## Training: the base and the assistant are different artifacts

### Pretraining includes long data and a midtraining capability mix

**Reported.** The corpus contains web data, mathematics, programming,
multilingual material, and deliberately curated long documents. DeepSeek also
says it adds agentic data during midtraining. The report does not disclose the
full source mixture or enough data provenance to reproduce it.

This maps cleanly to the course's progression. [Module 09B](../modules/09b-pretraining.md)
builds causal pretraining; [Module 10](../modules/10-tinyllm.md) makes the base
artifact durable; [Beyond Midtraining](../beyond/midtraining.md) changes the
distribution while retaining the causal objective and explains why long
context requires training exposure, not just a larger configuration value.
V4 demonstrates both forms at production scale.

### Specialists are trained, then consolidated

The post-training recipe is easier to understand if “expert” is used carefully.
The MoE experts above are internal FFNs selected token by token. The
post-training specialists here are complete model policies trained for domains
such as mathematics, coding, agents, and instruction following.

**Reported.** Each specialist receives fine-tuning followed by GRPO with
domain-specific prompts and rewards. More than ten resulting teacher models
are then consolidated into one student through on-policy distillation. The
student generates trajectories and minimizes a weighted reverse-KL objective
against full teacher vocabulary distributions.

```text
one base
  ├─ math SFT  → math GRPO  ─┐
  ├─ code SFT  → code GRPO  ─┤
  ├─ agent SFT → agent GRPO ─┼─ on-policy, multi-teacher distillation → one model
  └─ other specialists ──────┘
```

[Module 13](../modules/13-sft.md) covers response-masked SFT and [Beyond RL](../beyond/rl.md)
builds the GRPO loop. [Module 16B](../modules/16b-synthetic-data.md) builds
sequence-level distillation from teacher-written text. V4's OPD is only a
**conceptual bridge** to that module: it uses student-generated trajectories
and full teacher logits, which the course implementation does not have.

## Infrastructure: where the efficiency claim becomes real

At course scale, a clear Python loop is the right implementation. At V4 scale,
the same loop can spend most of its time moving tokens, waiting for experts,
managing caches, or recovering interrupted rollouts. DeepSeek's report is
unusually useful because it treats those systems as part of the model release.

### Expert parallelism must hide the network

Routing an MoE token to an expert on another accelerator creates a dispatch,
expert computation, and combine path. DeepSeek divides experts into waves and
fuses communication and computation into one pipeline: while one wave computes,
the next moves tokens and a completed wave returns outputs. The report claims
1.50–1.73× speedups over its non-fused baselines for general inference and up
to 1.96× in latency-sensitive cases.

**G2C interpretation.** “13B active” does not itself guarantee a fast 284B
model. It becomes useful only if dispatch overhead is hidden well enough that
expert computation remains the bottleneck. This is the production counterpart
to Beyond MoE's warning that its Python expert loop demonstrates conditional
computation, not wall-clock acceleration.

### Reproducibility constrains kernel design

DeepSeek reports batch-invariant and deterministic kernels intended to align
pretraining, post-training, and inference bit for bit. That requires preserving
accumulation order across alternate attention kernels and deterministic token
ordering in MoE communication and backward accumulation.

This is more than debugging polish. On-policy training repeatedly moves data
between generation and optimization systems. If the same token's result changes
with batch packing, reproducing a failed trajectory or comparing serving and
training behavior becomes harder. The report presents determinism as a system
property purchased alongside throughput—not something a random seed alone
provides.

### Hybrid attention creates a hybrid cache

CSA, HCA, and sliding-window layers produce different cache entry sizes,
compression schedules, and eviction rules. Incomplete groups must retain
uncompressed tail state until another compression block can be formed. DeepSeek
therefore co-designs a heterogeneous cache layout with its sparse-attention
kernels instead of treating the KV cache as one uniform tensor.

For shared prefixes, compressed CSA/HCA entries can be stored on disk and
reused. Sliding-window state is larger, so the report describes a configurable
tradeoff: cache it fully, checkpoint it periodically and recompute a tail, or
store none and recompute more. This is a useful production extension of
[Module 16's](../modules/16-inference.md) cache: cache design is now a policy over
storage, I/O, and recomputation.

### Speculative decoding attacks serial generation

The original V4 report concentrates on making each target-model step cheaper.
The 0731 package also tries to take fewer serial target steps. DSpark drafts a
short block with an attached smaller module, verifies that block with the V4
target in parallel, and keeps only the prefix accepted by the target. Because
verification is defined against the target distribution, speculation is an
inference acceleration rather than a new decoding policy.

The system problem is deciding how much to speculate. A long proposal is useful
when many tokens survive and wasteful when the target rejects the suffix.
DSpark predicts prefix-survival confidence and schedules verification length
against the serving engine's measured throughput. That is why its technical
report describes both a draft architecture and a hardware-aware scheduler:
acceptance quality alone does not maximize served tokens per second.

[Module 11](../modules/11-sampling.md) builds autoregressive generation and
[Module 16](../modules/16-inference.md) makes cached decoding explicit. DSpark is
a **conceptual bridge** from those mechanisms to speculative decoding; the
course does not yet build a drafter or rejection-verification loop.

### Long-context training needs parallelism and memory policy

The attention architecture reduces theoretical work; it does not make a
million-token training example fit automatically. The training stack adds
context parallelism for compressed attention, selective activation
recomputation for mHC, and a hybrid ZeRO strategy adapted to Muon's need to see
whole logical matrices. These are **not yet covered** by the course. They belong
to distributed training and performance engineering, not to the mechanism
students need in order to understand what Muon or compressed attention computes.

### Interrupted rollouts are a statistical problem

For RL and on-policy distillation, DeepSeek logs each generated token to a
write-ahead log and saves KV state when work is preempted. If hardware fails,
persisted tokens can reconstruct the cache. The report makes a subtle point:
restarting every interrupted generation from scratch creates length bias,
because short responses are more likely to finish before interruption.

That connects directly to [Beyond Harness](../beyond/harness.md). Its event log
persists intent and outcome around tool actions; DeepSeek's rollout WAL persists
generation progress around unreliable compute. They protect different state,
but both reject the fiction that recovery is just “call the function again.”

### Agent evaluation needs actual isolation

DeepSeek Elastic Compute exposes function calls, containers, Firecracker
microVMs, and QEMU full VMs behind one interface. The report says one production
cluster manages hundreds of thousands of concurrent sandbox instances.

This closes an important teaching loop. Beyond Harness deliberately labels its
in-process permission checks as **not sandboxing**. V4's system selects stronger
execution substrates when security or operating-system fidelity requires them.
The model's tool-call tokens are only the protocol; safe and repeatable agent
training depends on the environment beneath that protocol.

## The efficiency stack

No single row explains the headline. The release combines savings at several
levels:

| Layer | Mechanism | What it tries to save |
|---|---|---|
| Model capacity | Sparse MoE | Compute per token relative to total capacity |
| Attention | CSA + HCA + local SWA | Attention work and retained KV entries |
| Precision | FP4/FP8 mixture and post-training QAT | Memory traffic and arithmetic cost |
| Distributed execution | Fused, wave-pipelined expert parallelism | Communication exposed on the critical path |
| Training | Context parallelism, recomputation, hybrid ZeRO | Device memory and idle time |
| Serving | Heterogeneous plus on-disk prefix cache | Repeated prefill and accelerator memory |
| Decoding | DSpark draft-and-verify speculation | Serial target-model generation steps |
| Post-training | Preemptible rollout WAL | Lost generation work and biased recovery |
| Agents | Tiered sandbox substrates | Cold-start cost, density, and isolation tradeoffs |

**G2C interpretation.** Architecture determines what can be saved; infrastructure
determines whether the saving survives contact with hardware and workloads.
That is why active parameters are a useful first line on a model card and an
incomplete latency prediction.

## Coverage map

| V4 mechanism | G2C connection | Coverage |
|---|---|---|
| Transformer and causal objective | Modules 07–10 | **Built in g2c** |
| DeepSeekMoE routing and total/active arithmetic | Beyond MoE | **Built in g2c** at mechanism scale |
| CSA sparse selection and HCA compression | Module 16 + Beyond Linear Attention | **Conceptual bridge** |
| Million-token data and continuation | Beyond Midtraining | **Conceptual bridge**; lecture-only long context |
| Specialist SFT | Module 13 | **Built in g2c** |
| Specialist GRPO | Beyond RL | **Built in g2c** at toy scale |
| Multi-teacher on-policy logit distillation | Module 16B | **Conceptual bridge** |
| Tool schema and interleaved agent state | Modules 18–19 + Beyond Harness | **Conceptual bridge** |
| Real container/microVM sandboxing | Beyond Harness | **Conceptual bridge**, explicitly outside its build |
| DSpark speculative decoding | Modules 11 and 16 provide target decoding | **Conceptual bridge** |
| mHC | Module 09 provides the residual baseline | **Not yet covered** |
| Muon | Module 03B provides the optimizer baseline | **Not yet covered** |
| Multi-token prediction | Module 06 provides one-token prediction | **Not yet covered** |
| FP4 quantization-aware training | Module 16 motivates inference precision | **Not yet covered** |
| Expert/context parallelism and custom kernels | No distributed-training track | **Not yet covered** |

The map is the point of the Brief. A student does not need to have implemented
V4 to identify which claims are familiar, which are extensions, and which
require genuinely new machinery.

## Claims versus demonstrated evidence

### What the primary sources establish directly

- Model weights, configuration files, inference code, and an MIT license are
  publicly listed in the official model collection.
- The report discloses total/active parameter counts, architecture settings,
  token counts, training stages, benchmark protocols, and several systems
  designs in substantial detail.
- The 284B/13B and 1.6T/49B arithmetic is reproducible from those disclosed
  counts.
- The 0731 model card publishes a separate checkpoint, its attached DSpark
  configuration, serving examples, and an agent-benchmark comparison.

### What remains reported rather than independently established here

- Benchmark superiority and the 1M-context quality results come from DeepSeek's
  evaluation framework unless a table explicitly cites an external evaluation.
- FLOP, KV-cache, and kernel-speedup ratios depend on DeepSeek's baselines,
  precision accounting, implementation, and hardware.
- A supported one-million-token input limit is not by itself proof that every
  task can use evidence anywhere in that window. The report provides long-context
  evaluations, but this Brief does not replicate them.
- The report presents many coupled changes and does not isolate every component
  with a component-by-component ablation. It therefore does not justify assigning
  the total capability or efficiency gain to one named mechanism.
- The full pretraining mixture and data provenance are not disclosed, so the
  training corpus cannot be reconstructed from the report.
- The 0731 card does not disclose its incremental post-training recipe, and its
  code-agent results depend on a DeepSeek Harness mode that was not public at
  the snapshot date.

The disciplined reading is neither “marketing” nor “proven fact.” It is:
*this is what the release reports, this is what follows arithmetically, this is
what the disclosed mechanism plausibly explains, and this is what remains
unknown.*

## A practical reading route

The report is long. Read it in this order:

1. **Abstract and §1.** Capture the claimed problem, model sizes, and efficiency
   baselines.
2. **§2.1 and §2.3.** Read MoE and CSA/HCA first. They explain the total/active
   split and million-token headline.
3. **§3.1 and §3.5.** See what expert routing and compressed attention demand
   from communication and cache management.
4. **§4.1–4.2.** Separate corpus construction, midtraining choices, and model
   configuration from post-training.
5. **§5.1.** Trace complete-model specialists through SFT, GRPO, and OPD. Do not
   confuse these specialists with internal MoE experts.
6. **§5.2.3–5.2.5.** Read rollout recovery and sandboxing as part of the agent
   training method, not deployment trivia.
7. **Evaluation tables last.** Check whose harness produced each result, the
   reasoning mode, context budget, and whether the comparison is base-to-base
   or instruct-to-instruct.
8. **Then read the 0731 model card and DSpark report.** Keep the target-weight
   update separate from the attached speculative-decoding system.

## After this Brief, you should be able to

- explain why 284B total / 13B active describes capacity and conditional
  computation but does not predict latency by itself;
- distinguish CSA, HCA, DSA, and sliding-window attention without calling the
  whole architecture “linear attention”;
- trace V4 from causal pretraining through specialist SFT/GRPO and on-policy
  distillation into one instruct model;
- name at least three infrastructure mechanisms required to realize the model's
  efficiency claims;
- explain why a token write-ahead log prevents biased rollout recovery;
- explain how 0731 can change capability through post-training and latency
  through DSpark without changing the core V4 architecture;
- distinguish a permission check from container, microVM, and full-VM isolation;
- label a release statement as reported, derived, interpreted, or not disclosed.
