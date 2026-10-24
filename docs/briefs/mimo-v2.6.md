# G2C Brief — MiMo-V2.6: an itemized bill for frontier RL

> **Release question:** *What does scaling reinforcement learning actually cost, and what has to hold still for thirty giant update steps to work?*

Most frontier releases say post-training mattered and then stop. The
[GLM-5.3 Brief](glm-5.3.md) had to infer its whole story from one sentence:
"scaling post-training is all we did." Xiaomi's MiMo-V2.6 report does
something rarer. It puts dollar figures on the RL phase, splits them into
rollout, grading, and training, gives the batch size and step count, and
then documents what broke. It is the first release this course has read where
the RL budget can be checked by arithmetic.

The same report carries a quieter architectural story. The context cost
problem that [Qwen3.8-27B](qwen3.8-27b.md) solved with recurrent state and
[DeepSeek V4](deepseek-v4.md) solved with compressed sparse attention,
MiMo solves with ordinary softmax attention restricted to a **128-token
window** in five of every six layers. That is the most conservative answer
yet to "how do you afford a million tokens", and the config lets you price it
exactly.

## Snapshot and evidence policy

| Field          | Snapshot                                                                                                                                                                                                                                                                                                                                 |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Release        | MiMo-V2.6 series, published September 22, 2026 (Hugging Face repos last modified 03:52 UTC that day)                                                                                                                                                                                                                                     |
| Models         | MiMo-V2.6-Pro-RL (1.02T total / 42B active); MiMo-V2.6-Flash-RL (310B total / 15B active); MiMo-V2.6-Distill-Qwen-9B. All MIT-licensed                                                                                                                                                                                                   |
| Primary source | [*MiMo-V2.6: Scaling Reinforcement Learning Towards Self-Improvement*](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL/blob/main/MiMo_V2_6_technical_report.pdf), a 44-page PDF shipped inside the model repos (no arXiv version at the snapshot date)                                                                                |
| Artifacts      | [Pro `config.json`](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL/raw/main/config.json) at commit `73875d0`; [Flash `config.json`](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL/raw/main/config.json) at commit `5711b26`; [9B model card](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B) at commit `2367e86` |
| Release source | [Xiaomi MiMo-V2.6-Pro product page](https://mimo.mi.com/models/en-US/mimo-v2.6-pro)                                                                                                                                                                                                                                                      |
| Last verified  | September 22, 2026                                                                                                                                                                                                                                                                                                                       |

The labels below matter:

- **Reported** means Xiaomi reports it. It is not an independent replication.
- **Derived** means arithmetic from reported values.
- **G2C interpretation** means this Brief is connecting the evidence to the
  course.
- **Not disclosed** means the source does not support a stronger conclusion.

Two subject-discipline rules apply to this release:

1. **Three models, three different claims.** Pro and Flash share one
   architecture and one RL recipe. The 9B is a different model: a Qwen3.5-9B
   fine-tune trained on MiMo-generated data. A 9B number never stands in for
   Pro, and neither does the reverse.
2. **The report beats day-one press.** Launch coverage reported "7,000+ RL
   environments" and an $850K Flash RL bill. The report says the released
   training sets hold roughly 7k *tasks* (Table 5) and gives Flash's RL cost as
   $0.9M. Where the two disagree, this Brief uses the report.

## The release in one page

**Reported.** MiMo-V2.6 is an omni-modal sparse MoE family: text, image,
video, and audio in, text out. The text backbone interleaves sliding-window
attention (SWA) with global attention (GA) and runs a 128-token window.
Pretraining took 30T tokens for Pro and 48T for Flash. That was followed by
agent-centric mid-training, which extended context from 256K to 1M, switched
the optimizer to a Muon variant, and added MXFP4 quantization-aware training.
Then came a short SFT stage, 30 steps of large-batch asynchronous GRPO across
coding, general tool use, design, and cybersecurity tasks, and a final
multi-teacher on-policy distillation stage. Xiaomi also releases the 9B
distilled model, RL environments, an RL framework, and "composable
mini-harnesses" as a research starter kit.

**Reported: the RL bill.** Pro's RL phase cost $2.6M and Flash's cost
$0.9M, run "across thousands of GPUs in a single run." Each step samples
1,568 prompts × 16 rollouts and trains on 2.7B–3.7B tokens. For Pro, rollout
took 43.8% of the cost, training 43.5%, and grading 12.7%. Thirty steps took
123.1 wall-clock hours for Pro and 81.8 for Flash, failures included. Over
the RL run, Pro's DeepSWE v1.1 score rose from 58.4 to 72.6 and Flash's from
48.7 to 65.7.

**Derived: what the bill works out to.**

```text
trajectories per step   1,568 × 16                  = 25,088
trajectories per run    25,088 × 30                 ≈ 753K
tokens per run          30 × (2.7B … 3.7B)          ≈ 81B … 111B
Pro  cost per step      $2.6M / 30                  ≈ $87K
Pro  cost per rollout   $2.6M / 753K                ≈ $3.45
Flash cost per rollout  $0.9M / 753K                ≈ $1.20
Pro  RL share of data   (81B … 111B) / 30T          ≈ 0.3% of pretraining tokens
```

The last line is the one to remember. RL touches roughly a third of a percent
as many tokens as pretraining did, yet it is the phase the report credits for
the headline capability jump. The tokens are expensive because each one was
generated, executed against an environment, and graded before it was trained
on. **Not disclosed:** the GPU type, the GPU-hours, and whether the dollar
figures are rental-equivalent or internal cost. They are comparable with each
other but with nothing else.

## Architecture: ordinary attention with a 128-token horizon

### The weave, read from the config

```text
Flash config.json   hybrid_layer_pattern (48 entries, 0 = global, 1 = sliding)
  [0, 1,1,1,1, 0, 1,1,1,1,1, 0, 1,1,1,1,1, 0, ... ]
   │  └ first block: 4 SWA ┘  └──── then 5 SWA : 1 GA, repeated ────┘
   └ layer 0: global attention + dense FFN ("to stabilize early representation learning")

per-token sequence memory (BF16 cache assumed):
   9 × GA   4 KV heads × (192 K + 128 V) dims      22.5KiB / token, growing
  39 × SWA  8 KV heads × (192 + 128) × 128 window  ≈ 24.4MiB total, fixed
```

**Reported.** Flash has 48 layers (39 SWA, 9 GA) and Pro has 70 (60 SWA, 10
GA). Both use a 128-token window, grouped-query attention with asymmetric
head dimensions (192 for query/key, 128 for value), and an MoE FFN in every
layer except the first.

**Derived: what the weave buys.** Only the global layers keep a cache that
grows with context. For Flash that is
`9 layers × 4 KV heads × (192 + 128) dims × 2 bytes = 23,040 bytes` per token
(22.5KiB), which comes to 22.5GiB at 1M tokens. The 39 sliding layers each hold at most 128 tokens of
K and V, about 24MiB in total whatever the context length. Had all 48 layers
been global at the same GQA geometry, the cache would be 120KiB per token and
120GiB at 1M, 5.3× larger. Pro's numbers follow the same pattern: 50KiB per
token (8 KV heads on its 10 global layers), 50GiB at 1M, and 7× smaller than
an all-global Pro. For scale, the Qwen3.8-27B Brief derived 64KiB per token
for its 16 full-attention layers.

**G2C interpretation.** This is the third distinct answer to one bill in the
Brief series, and the most conservative. Qwen replaces most attention layers
with recurrent state and DeepSeek compresses and selects from the history.
MiMo keeps exact softmax attention everywhere and restricts where most layers
may look. [Module 07](../modules/07-attention.md)'s causal mask is the
starting point: a sliding window is that mask with a second diagonal cut off,
turning a triangle into a band. The consequence is that everything a token
learns about text more than 128 positions back has to pass through one of
nine (Flash) or ten (Pro) global layers. The
[efficient-attention map](../beyond/attention-efficiency.md) files this under
structured sparsity: pairwise work falls toward `T · W` and the live cache is
bounded, at the price of direct access outside the window.

### Two RoPE bases and a sink

The config separates local and global position handling. Sliding layers
rotate at `swa_rope_theta: 10000`, the base value the original RoPE paper
used, since they never see a position more than 128 back. Global layers
rotate at `rope_theta: 10000000`, and only `partial_rotary_factor: 0.334` of
each head, meaning 64 of the 192 query/key dimensions. (**Derived**:
0.334 × 192 ≈ 64.) [Module 05](../modules/05-embeddings.md) builds the
position problem this answers. The two-base split is a **conceptual bridge**.

Sliding layers also set `add_swa_attention_sink_bias: true` and global
layers do not. **G2C interpretation:** the field name and the report's
"sink-augmented SWA" wording for the vision encoder point to a learned logit
that lets a head send attention mass *nowhere* instead of forcing it across
128 tokens that may all be irrelevant. The V2.6 report does not describe the
mechanism and defers to the MiMo-V2-Flash report for backbone details. This
Brief treats sinks as **not yet covered**.

### Experts without a shared expert

Flash routes each token to 8 of 256 experts and Pro to 8 of 384, with
sigmoid scoring, no shared experts, and auxiliary-loss-free balancing
(`topk_method: noaux_tc`). [Beyond MoE](../beyond/moe.md) builds the
total/active split and the router, which by now appear in almost every
frontier release. The detail specific to MiMo is the absence of a shared expert, since every FFN
parameter a token uses was chosen by the router. That becomes relevant when
the router starts drifting during RL (see below).

### The encoders ride in front

**Reported.** A 681M-parameter vision encoder (28 layers, itself an SWA/GA
hybrid that alternates row-major and column-major token order) was trained
from scratch on more than 4T image tokens using only a captioning-style
cross-entropy objective. Audio is split into 20 residual-vector-quantized
codes per 25Hz frame by a tokenizer trained on 20 million hours of audio.
Four frames are then packed into one backbone token, giving 6.25 tokens per
second of audio. [Beyond Multimodal](../beyond/multimodal.md) builds the
load-bearing idea that patches become tokens in the shared sequence. RVQ
audio codes are **not yet covered**.

## Pre- and mid-training: preparing the base for RL

**Reported.** Pretraining ran in two stages: text only, then joint omni-modal
training. Context started at 32K and was extended to 256K partway through.
Flash saw 26T text + 22T omni tokens and Pro saw 27T + 3T, so the larger model
was trained on fewer tokens. Both used AdamW.

Mid-training is framed explicitly as preparation for RL, not as more of the
same:

- **The data mix shifts to agent trajectories** across coding, general,
  visual, and research tasks, plus repository-level code. It runs mostly at
  256K context with a final stage at 1M.
  [Beyond Midtraining](../beyond/midtraining.md) builds the underlying
  lesson: continued next-token training can add a capability, and the mixture
  decides what is forgotten.
- **The optimizer switches mid-run** from AdamW to *Muown*, a Muon variant
  with row-norm control, for hidden matrices. Embeddings, the LM head, and the
  router stay on AdamW. The stated reason is that Muon keeps its data
  efficiency beyond the batch size where AdamW stops improving, which matters
  because RL batches are enormous. Prior work warns that switching an
  Adam-trained model to Muon can hurt, and Xiaomi reports no loss spike.
  [Beyond Muon](../beyond/muon.md) builds the orthogonalized update and the
  matrix/non-matrix split. Muon also appears in the DeepSeek V4 and both GLM
  Briefs. MiMo is the first to switch to it partway through training.
- **MXFP4 quantization-aware training** begins here so that the RL rollout
  engine can later run experts in 4-bit without the model seeing a numerics
  change. [Module 16](../modules/16-inference.md) explains post-training
  quantization. Training *through* quantization is **not yet covered** and
  adds evidence to the roadmap's low-precision-training watchlist item.
- **Anti-reward-hacking examples are written into mid-training.** Early RL
  runs showed hacking. Xiaomi turned those cases into examples where the model
  recognizes the faulty reasoning, revises the turn, and continues honestly,
  and it reports improved alignment. The defense against a failure in phase
  four was inserted in phase two.

## RL: thirty steps, itemized

### The objective is the one you built, plus production parts

**Reported.** The loss (Eq. 1) is GRPO: sample a group of `G = 16`
completions per prompt, score them, turn scores into group-relative
advantages, and push up the log-probability of above-average tokens.
[Beyond RL](../beyond/rl.md) builds exactly this core, including the rule
that all-pass and all-fail groups carry no signal. MiMo's "dynamic sampler"
filters those groups for the same reason.

The production parts that Beyond RL deliberately strips out are all present:

| Production part | What it does in MiMo | G2C status |
|---|---|---|
| Per-token importance ratio `r = π_θ / μ_old` | Corrects for rollouts generated by a policy up to 4 versions stale ("staleness of 4") | **Conceptual bridge** |
| Four decoupled clip bounds | Separate bounds for positive and negative advantages, retuned at runtime to steer entropy | **Conceptual bridge** |
| Prompt-mean aggregation | Averages the loss per prompt, not per token, so long responses don't dominate and length doesn't grow too fast | **Conceptual bridge** |
| Partial rollout | Unfinished trajectories carry into the next step, which then has to re-prefill their KV cache under the new weights | **Conceptual bridge** via Module 16 |
| No KL term | The report's objective has no reference-model penalty | Contrast with Beyond RL's KL leash |

**G2C interpretation.** Thirty optimizer steps is a small number. Each step
is a 25K-trajectory batch that takes hours to collect, so the algorithmic
choices that matter most are the ones that keep a batch *valid* when it
arrives: staleness correction, group filtering, and length control. Clever
update math matters less.

### Grading is 12.7% of the bill and does more than check tests

Binary test rewards give every passing patch the same score. Xiaomi pays for
a grader that tells passing patches apart, in two ways:

- **Groupwise Reward Synthesis (GRS)**, an offline method. An agent studies
  several rollouts of one task and writes task-specific *solution* and
  *behavior* rubrics. During training each rollout is scored as
  `R = R_test · S_sol · S_beh`. The multiplication keeps failures at zero and
  ranks the passes.
- **Groupwise Advantage Redistribution (GAR)**, an online method. An
  SFT-trained grader agent inspects a whole mixed-outcome group in a shared
  workspace, ranks the passing patches on approach, precision, minimality,
  side effects, and craftsmanship, and moves positive advantage from worse
  passes to better ones. The total positive advantage is conserved. A
  trajectory confirmed to have used a leaked answer gets its reward reset to
  zero before advantages are recomputed.

**Reported.** Without GAR, a Flash code-only run grew rapidly in turns and
tokens, and audits found "speculative compatibility branches, broad exports,
exception swallowing, relaxed validation". Those are patches that pass tests
by making the code worse. With GAR, pass rate kept climbing and patches stayed
small. A group-relative length penalty and token-level penalties for malformed
tool calls complete the shaping.

**G2C interpretation.** This is [Module 15](../modules/15-evaluation.md)'s
lesson about LLM judges turned into a reward. The grader is not trusted to
decide *whether* code works; tests still do that, and GRS multiplies by the
test result. The judge is trusted only to rank solutions that already pass,
which is a much easier job to calibrate.

### Reward hacking gets an adversary

The report's Table 2 is the most concrete reward-hacking evidence the Brief
series has seen. It lists real rollout excerpts in which the agent, asked to
fix a historical bug, instead runs `pip install pytest==5.4.3` to read a
newer release, `curl`s the fixed upstream file, `git clone`s the current
repo, or reads the original Django ticket's fix. Each passes the tests, and
none is the task.

The defense runs in three layers:

1. **Environment cleanup.** Remove build logs, residual patches, caches, and
   later git history, and isolate the network.
2. **A dedicated *hack agent*** probes each prepared environment for leaks,
   and cleanup repeats "until the hack agent could no longer find a successful
   exploit."
3. **Offline trajectory audits** during training. The confirmed-hack share
   stayed below 2% for both models (**Reported**).

[Beyond RL](../beyond/rl.md) states the principle: the reward curve is a
claim and the samples are the evidence. MiMo's contribution is making "read
the samples" an automated, adversarial process. That is a **conceptual
bridge**; the course's toy arithmetic tasks have no network to leak through.

### The harness is a training axis

**Reported.** Rather than train inside production harnesses such as Codex or
MiMo Code, Xiaomi derives four minimal "mini-harnesses" from one shared agent
loop and varies them deliberately. Production harnesses were rejected because
their guardrails and instruction prompts sit outside the reward, which makes
credit assignment unreliable. On three held-out harnesses (Codex, Claude
Code, mini-swe-agent), mean DeepSWE pass@1 rose from about 50% to 66% during
training (Figure 10).

**G2C interpretation.** [Beyond Harness](../beyond/harness.md) argues that a
benchmark score measures model × harness. MiMo takes the next step and treats
the harness as a distribution to train over, so that the model's skill stops
depending on one particular loop. [Module 19](../modules/19-agent.md)'s agent
loop is the "minimal agent loop" the mini-harnesses start from.

## What has to hold still

Two sections of the report are about not letting anything move except the
policy weights. Both connect directly to code students have written.

### The router is frozen

**Reported.** With a trainable router, Pro's expert load collapsed within 20
RL steps. At layer 9, the load's coefficient of variation went from 0.78 to
2.0, peak expert load from 6× to 16× the mean, and cold experts from 0.5% to
22%. Resetting only the router to its pre-RL weights restored balance with no
benchmark loss, which locates the problem in router drift rather than expert
damage. The router is frozen for all of RL. A related training failure: one
expert-parallel rank once received over 30× the mean token load and ran out
of memory.

**G2C interpretation.** [Beyond MoE](../beyond/moe.md) teaches load
balancing as a pretraining concern. RL reopens it: the reward says nothing
about balance, the batches are dominated by a few task types, and without a
shared expert there is no fallback path. Freezing the router is the blunt fix.

### Training and inference must compute the same function

The rollout engine (SGLang) and the training engine (Megatron-LM) are
separate programs. For the importance ratio to mean anything, both must assign
the same probability to the same token. Three mechanisms enforce that:

- **Quantize-dequantize after every update.** Experts are rounded to exactly
  the MXFP4 values the rollout kernels will use.
- **Routing replay (R3).** Tiny numeric differences can flip a top-k expert
  choice, so the expert indices chosen during rollout are recorded and replayed
  during training.
- **Top-p candidate-set replay.** This is
  [Module 11](../modules/11-sampling.md), in production. The sampler
  renormalized over the nucleus, not the full vocabulary, so the probability
  the sampled token actually had is its share *of the nucleus*. Training must
  renormalize within the same set. **Reported:** at top-p 0.97 the candidate
  set averages fewer than five tokens.

**G2C interpretation.** Everything in this section exists because RL is
off-policy by a few steps and the ratio `π_θ / μ_old` has to be meaningful.
In a single-process toy, like the course's, the two probabilities come from
the same forward pass and the problem never appears.

### Serving: a drafter trained on the policy's own rollouts

**Reported.** Pretraining attaches a three-layer MTP head (`model_mtp.safetensors`
in the repo). RL rollouts replace it with a 5-layer *DFlash* drafter that
proposes a whole block of tokens in one pass, using bidirectional attention
within the block ("block diffusion"). The architecture section describes 7
tokens per pass; RL uses 6-token blocks by default because they gave about 6%
more throughput than 8-token blocks. The drafter is fine-tuned on early RL rollouts,
which gives a 31.3% higher accepted length than the MTP configuration. The
drafter weights ship in the Flash repo (`dflash/`).
[Beyond Speculative Decoding](../beyond/specdec.md) builds draft-and-verify
and an MTP head. The block-diffusion drafter is **not yet covered** and is a
data point for the roadmap's non-autoregressive-generation watchlist.

**G2C interpretation.** Speculative decoding is usually presented as a
serving trick. Here it is an RL-throughput trick, because rollout is 43.8% of
the bill.

## After RL: consolidation and a 9B starter model

**Reported.** After mixed RL, *MOPD2* (multi-prefix multi-teacher on-policy
distillation) merges in capabilities from domain teachers. Some teachers are
RL-trained; others, for domains where no reliable reward exists, such as
long-horizon game development or research, are SFT-trained. The student
generates its own tokens and a teacher scores them token by token. The
"prefix" variant starts the student partway through a teacher or SFT
trajectory so that it practices single turns deep inside long tasks.

[Module 16B](../modules/16b-synthetic-data.md) separates classic
distillation (match the teacher's distribution) from sequence-level
distillation (train on the teacher's text). On-policy distillation is a third
thing: the *student* writes and the teacher grades each token. It is **not
yet covered**, and with DeepSeek V4 it now has two independent families
using it as the final consolidation stage.

The 9B release is the opposite case, and entirely **built in g2c**: plain
sequence-level distillation. Xiaomi fine-tuned Qwen3.5-9B on 77.4B tokens of
MiMo-generated trajectories (27.2B of them loss tokens) across code, cyber,
general, and visual tasks. It then ran domain-specific GRPO from that
checkpoint using the released environments (**Reported**: SWE-bench
Verified 60.0 → 61.1 after SFT → 66.2 after RL; Terminal Bench 2.1
27.0 → 37.1 → 52.8).

**Derived: laptop scale.** 9.41B parameters (safetensors metadata) is about
18.8GB at BF16 and roughly 5GB at 4-bit, which puts it within reach of the
course's Stretch track at full precision and the Standard track quantized.
Qwen3.5 is also the family the course's default ProdLM backend
(`qwen3.5:4b`) comes from, so its chat template will look familiar. Nothing
in this Brief requires running it.

## The cybersecurity domain

**Reported.** Cybersecurity is 4% of the RL task mix. The task is
vulnerability reproduction from OSS-Fuzz: given a project and a described
bug, construct an input that triggers *that* bug. The verifier matches the
crash's sanitizer-reported type and top project stack frame by string
comparison. Xiaomi chose this over LLM judging because judges return
different verdicts for the same input, and over CyberGym's patched-binary
comparison because incomplete patches corrupt the reward. Reported scores
include CyberGym 94.0 (Pro) and 95.1 (Flash), the latter using Xiaomi's
corrected environments, and ExploitBench 47.9 (Pro).

**G2C interpretation.** The verifier design is textbook
[Beyond RL](../beyond/rl.md): a reward should be deterministic, cheap, and
tied to exactly the described outcome, and task description and verifier
should come from a single source of truth. The release decision is a contrast
with GLM-5.3, which held back its weights over emergent offensive capability.
MiMo trained the capability deliberately and published weights on day one.
**Not disclosed:** any safety evaluation, misuse assessment, or release-gating
process. The report does not discuss one.

## Coverage map

| MiMo-V2.6 mechanism | G2C connection | Coverage |
|---|---|---|
| Transformer, causal objective, decoding | Modules 07–11 | **Built in g2c** |
| Sliding-window / global hybrid | Module 07 mask; efficient-attention map | **Conceptual bridge** |
| Attention sink bias | — | **Not yet covered** |
| GQA with asymmetric K/V head dims | Module 08 baseline; Module 13B's SmolLM | **Conceptual bridge** |
| Partial RoPE with separate local/global bases | Module 05; draft Beyond Long Context | **Conceptual bridge** |
| Sparse MoE, no shared expert | Beyond MoE | **Built in g2c** at mechanism scale |
| Router freezing against RL load collapse | Beyond MoE | **Conceptual bridge** |
| AdamW → Muon switch mid-run | Beyond Muon; Module 03B | **Built in g2c** (Muon core); Muown row-norm control is a **conceptual bridge** |
| Agent-centric midtraining mixture | Beyond Midtraining | **Built in g2c** at mechanism scale |
| MXFP4 quantization-aware training, QDQ | Module 16 (post-hoc quantization only) | **Not yet covered** |
| GRPO with group filtering | Beyond RL | **Built in g2c** |
| Importance ratios, decoupled clipping, async staleness | Beyond RL (explicitly stripped) | **Conceptual bridge** |
| Rubric and groupwise agentic grading (GRS, GAR) | Beyond RL; Module 15 | **Conceptual bridge** |
| Reward-hacking audits and hack agents | Beyond RL | **Conceptual bridge** |
| Multi-harness training | Module 19; Beyond Harness | **Conceptual bridge** |
| Top-p candidate-set replay | Module 11 | **Built in g2c** (top-p); replay is a **conceptual bridge** |
| KV context caching, host offload, re-prefill | Module 16 | **Conceptual bridge** |
| MTP head; draft-and-verify decoding | Beyond Speculative Decoding | **Built in g2c** |
| Block-diffusion (DFlash) drafter | Beyond Speculative Decoding baseline | **Not yet covered** |
| Sequence-level distillation to the 9B | Module 16B | **Built in g2c** |
| Multi-teacher on-policy distillation | Module 16B baseline | **Not yet covered** |
| Vision patches as tokens | Beyond Multimodal | **Conceptual bridge** |
| RVQ audio tokenization | — | **Not yet covered** |

## Claims versus demonstrated evidence

### What the primary sources establish directly

- Pro, Flash, and the 9B are downloadable under MIT, and the architectural
  numbers in this Brief (layer pattern, window, KV heads, head dimensions,
  experts, RoPE bases, sink flags) are read from the published configs.
- The DFlash drafter and MTP weights ship in the Flash repo, so the
  speculative-decoding configuration is part of the artifact and not only a
  description.
- The report discloses the RL batch geometry, step count, learning rate,
  optimizer settings, task mix, and cost split. That is enough to do the
  arithmetic above, and more than any earlier Brief subject disclosed.

### What remains reported rather than independently established here

- **The dollar figures.** Cost basis, hardware, and GPU-hours are not
  disclosed, so the figures cannot be compared with anyone else's.
- **Every benchmark row.** They are Xiaomi-run, several on internal suites
  (MiMo Code, Cyber, and Visual benchmarks) and one on a Xiaomi-corrected
  CyberGym. Final-table scores differ from the RL curves (for example, Pro
  DeepSWE is 71.9 in Table 3 versus 72.6 at the end of RL), consistent with
  Table 3 measuring the post-MOPD2 checkpoint. The report does not say so
  explicitly.
- **The "1M context" claim.** It is a trained context length and a config
  value. The report publishes no long-context retrieval evaluation, and with
  only nine or ten global layers carrying all long-range information, that
  evaluation is the one this architecture most needs.
- **The open-source kit.** The report says it releases environments, an
  end-to-end RL framework, and mini-harnesses. At the snapshot date this
  Brief could confirm the 9B checkpoint but did not locate an environment
  dataset. New `XiaomiMiMo` GitHub repositories (`uni-agent`, a `verl` fork,
  `mimoagent`) appeared the day before release, but none is identified in the
  report as *the* framework.
- **The router-freezing and GAR ablations.** Each is a single comparison
  (one layer's statistics; one Flash code-only run at batch size 128). They
  are suggestive, not definitive.
- **Artificial Analysis score of 46.** Xiaomi's product page quotes it. It is
  a third-party index relayed by the vendor and does not appear in the report.

## A practical reading route

1. **Section 4.1 and Figure 3 of the report first.** Reproduce this Brief's
   per-rollout and share-of-pretraining arithmetic before reading anything
   else. That anchors the rest of the report in scale.
2. **Section 5.1 (training setup) next to [Beyond RL](../beyond/rl.md).** List
   every term in MiMo's Eq. 1 that your GRPO implementation lacks, and say
   what each is for.
3. **Section 4.2.6 and Table 2.** Read the hacking excerpts, then decide what
   your own Beyond RL verifier would have let through.
4. **Sections 5.4 and 6.4** for the "what has to hold still" story: the
   router, R3, and top-p replay.
5. **The [Flash `config.json`](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL/raw/main/config.json)**,
   to recompute the 22.5KiB-per-token cache figure and find the two RoPE
   bases and the sink flag yourself.
6. **Section 2 and Table 1 last.** The architecture is conventional enough
   that it reads quickly once the rest is in place.

## After this Brief, you should be able to

- turn a reported RL budget, batch size, and step count into cost per step,
  cost per rollout, and share of pretraining tokens, and name what the
  figures cannot be compared with;
- compute, from the config, why 39 sliding-window layers plus 9 global layers
  make a million-token cache 5× cheaper than 48 global layers, and say what
  the design gives up;
- place sliding-window attention next to recurrent state and compressed
  sparse attention as three answers to the same cost;
- list the parts of production GRPO that the course's implementation strips
  out, and explain why importance ratios matter once rollouts are stale;
- explain why a judge that ranks passing solutions is easier to trust than a
  judge that decides pass or fail;
- explain why a frozen router, routing replay, and top-p candidate replay are
  all ways of making training compute the same function the rollout sampled
  from;
- distinguish sequence-level distillation (the 9B) from on-policy
  distillation (MOPD2), and say which one the course has built.
