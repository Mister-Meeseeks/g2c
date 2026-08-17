# G2C Brief — GLM-5.3: scaling post-training on a frozen base

> **Release question:** *How much capability can a lab add without touching the base model at all?*

GLM-5.3 is the closest thing a frontier release has offered to a controlled
experiment. Z.ai states the base model is byte-identical to GLM-5.2's and says,
verbatim, "Scaling post-training is all we did for GLM-5.3." Every number that
moved on its benchmark table therefore moved for reasons the second half of
this course studies: supervised fine-tuning, reinforcement learning, reward
design, and the environments the model practices in. Part II's thesis — that a
language model and an assistant are different artifacts built from the same
weights — is here demonstrated at 744 billion parameters.

The release carries a second story the course has not met before: Z.ai reports
that scaled agentic RL produced *emergent offensive-security capability*, and
it is holding the open weights for roughly two weeks of "safety evaluation and
hardening" as a consequence. This Brief reads both stories with the usual
evidence labels.

## Snapshot and evidence policy

| Field           | Snapshot                                                                                                                                                                                                                                                           |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Release         | GLM-5.3, announced August 14, 2026; API / Coding Plan access only at the snapshot date                                                                                                                                                                             |
| Models          | GLM-5.3 (single variant; no Air or base checkpoint announced)                                                                                                                                                                                                      |
| Release source  | [Official announcement blog](https://z.ai/blog/glm-5.3)                                                                                                                                                                                                            |
| Docs source     | [Official developer docs](https://docs.z.ai/guides/llm/glm-5.3)                                                                                                                                                                                                    |
| Lineage sources | [GLM-5 technical report, v1](https://arxiv.org/abs/2602.15763) · [GLM-5.2 model card](https://huggingface.co/zai-org/GLM-5.2) and its `config.json` · [SAO paper, v1](https://arxiv.org/abs/2607.07508) · [IndexCache paper, v1](https://arxiv.org/abs/2603.12201) |
| Weight source   | **Not yet published.** Z.ai: weights ship "in two weeks after launch, once safety evaluation and hardening are complete"                                                                                                                                           |
| License         | Not stated for GLM-5.3; GLM-5 and GLM-5.2 weights are MIT (lineage precedent, not a commitment)                                                                                                                                                                    |
| Last verified   | August 15, 2026                                                                                                                                                                                                                                                    |

The labels below matter:

- **Reported** means Z.ai reports it. It is not an independent replication.
- **Derived** means arithmetic from reported values.
- **G2C interpretation** means this Brief is connecting the evidence to the
  course.
- **Not disclosed** means the source does not support a stronger conclusion.

**The evidence chain is unusually indirect.** GLM-5.3 has no technical report
of its own. Its architecture facts come from a *different release's*
disclosures — the GLM-5 report and the GLM-5.2 `config.json` — carried across
by the blog's single sentence "It uses the same base model as GLM-5.2." Until
the 5.3 weights are published, nothing about the model itself can be checked
independently; every quality claim rides on Z.ai's own evaluation of an
API-only artifact. This Brief marks inherited facts as **Reported (lineage)**.

```text
GLM-5 technical report (Feb 2026)     GLM-5.2 card + config.json (Jun 2026)
   architecture, pretraining,             1M context, IndexShare,
   post-training pipeline                 critic PPO, published weights
              └───────────────┬───────────────────┘
                              │  "same base model"
                              ▼
                 GLM-5.3 blog + docs (Aug 2026)
                 post-training delta only; weights pending
```

## The release in one page

**Reported (lineage).** The base is GLM-5.2's: a mixture-of-experts
Transformer with 744B total and 40B activated parameters, 78 layers, MLA
attention with DeepSeek Sparse Attention on top, 256 routed experts with 8
selected per token plus one shared expert, a one-million-token native context,
one multi-token-prediction layer, and a 154,880-entry vocabulary. It was
pretrained on 28.5T tokens with the Muon optimizer. GLM-5.3 changes none of
this.

**Reported.** What changed is post-training scale: more synthesized RL
environments, more tasks, more compute spent training in them, and systems
work that raised end-to-end RL training throughput by more than 2.3×. The
product surface also changed: three thinking-effort levels (`low`, `high`,
`max`), and disabling thinking is no longer supported — a breaking change from
5.2.

**Derived.** One token activates roughly `40 / 744 = 5.4%` of the named
parameter count. More interesting is the delta arithmetic below: because the
base is fixed, the *entire* movement in each benchmark row is attributable to
post-training plus evaluation harness, with zero architecture confound. That
attribution is exactly what multi-change releases (compare DeepSeek V4's
everything-at-once redesign) cannot offer.

| Benchmark | GLM-5.2 | GLM-5.3 | Delta |
|---|---|---|---|
| Terminal Bench 2.1 | 81.0 | 88.2 | +7.2 |
| Terminal Bench 3.0 | 4.6 | 28.3 | +23.7 |
| DeepSWE v1.1 | 46.2 | 66.9 | +20.7 |
| SWE-Marathon v1.1 | 19.4 | 42.5 | +23.1 |
| Toolathlon Verified | 59.9 | 73.0 | +13.1 |
| AutomationBench v1.0.6 | 26.2 | 48.2 | +22.0 |
| CyberGym | 77.2 | 84.5 | +7.3 |
| ExploitBench | 24.4 | 54.4 | +30.0 |
| HLE with tools | 54.7 | 62.5 | +7.8 |
| GDPval-AA v2 | 1508 | 1769 | +261 |

All rows are Z.ai's own evaluations of both models. The official table also
places GLM-5.3 alongside current commercial frontier models — near parity on
several agentic suites, behind on most coding-depth suites — but those
cross-vendor rows inherit every harness and configuration caveat at once, so
this Brief leans on the same-base delta column, where the comparison is
cleanest.

**G2C interpretation.** The deltas cluster. The long-horizon,
environment-heavy suites (SWE-Marathon, AutomationBench, Terminal Bench 3.0,
ExploitBench) roughly doubled or better; the shorter-horizon suites moved
single digits. That pattern is what you would predict if the intervention were
"more and harder RL environments" rather than a general capability jump — the
gains sit where the new training distribution sits.

## What didn't change: the inherited base

The frozen half of the experiment is worth reading on its own, because it
collects several mechanisms the course builds or tracks. Everything in this
section is **Reported (lineage)** — disclosed for GLM-5/5.2, inherited by 5.3's
"same base model" assertion, and checkable in the published
[GLM-5.2 `config.json`](https://huggingface.co/zai-org/GLM-5.2/raw/main/config.json).

### MoE: the total/active split, again

256 fine-grained routed experts, 8 active per token, one shared expert, with
the first three layers dense. This is the same capacity-versus-per-token-compute
separation built in [Beyond MoE](../beyond/moe.md) — **built in g2c** at
mechanism scale. The 5.4% activation ratio describes conditional computation,
not latency; the course's standing warning about that gap applies unchanged.

### MLA plus DSA: compress the cache, then select from it

GLM-5's attention stacks two ideas. Multi-head Latent Attention stores a
compressed 512-dimensional latent per token instead of full per-head keys and
values (the config's `kv_lora_rank: 512`), decompressing on use. DeepSeek
Sparse Attention then puts a learned indexer on top: for each query it selects
the `index_topk: 2048` most relevant cached tokens and attends only to those,
so attention work stops growing with the full history even at a
million-token context.

GLM-5.2 added **IndexShare**: every four layers share one indexer, with
intermediate layers reusing the nearest full indexer's top-k selections. Z.ai
reports this removes 75% of indexer compute and cuts per-token FLOPs 2.9× at
1M context. One naming footnote a careful reader should know: the blog calls
the mechanism IndexShare while the underlying paper is titled *IndexCache* —
the same design under two names, a small live example of why Briefs pin
sources.

[Module 08](../modules/08-multi-head-attention.md) builds the projections
being compressed, [Module 16](../modules/16-inference.md) builds the KV cache
that makes the memory problem concrete, and
[Beyond Linear Attention](../beyond/linear-attention.md) frames the question
every long-context design answers: what state must survive from the history?
GLM's answer — keep everything, but compressed, and consult only a selected
subset — is a **conceptual bridge**; the course does not build a latent cache
or a learned indexer.

### Muon, second sighting

The GLM-5 report confirms pretraining used Muon with a "Muon Split"
modification (per-head orthogonalization). With
[DeepSeek V4](deepseek-v4.md) this makes a second independent frontier family
training on Muon rather than AdamW. [Module 03B](../modules/03b-training.md)
provides the optimizer contract; Muon itself remains **not yet covered**. Two
families is recurrence evidence for the roadmap's alternative-optimizer
watchlist — still evidence, not yet a module.

### Multi-token prediction, also again

One parameter-shared MTP layer, used for speculative decoding at serving time;
the report cites an acceptance length of 2.76 tokens. The same mechanism
family appears across DeepSeek and Qwen releases. [Module 06](../modules/06-language-models.md)
builds one-token prediction; MTP is **not yet covered** and is accumulating
the same watchlist evidence as Muon, under alternative/parallel generation.

### A million tokens took staged exposure

GLM-5's base context was built out in stages — 32K (1T tokens), 128K (500B),
200K (50B), plus a ~20B-token DSA adaptation phase — and GLM-5.2 extended to
1M. Long context arrived through *training exposure*, not through editing a
configuration value. That is [Beyond Midtraining](../beyond/midtraining.md)'s
central claim demonstrated at production scale — a **conceptual bridge** from
the course's lecture-level treatment.

## What changed: post-training at scale

### The pipeline 5.3 inherits

The GLM-5 report describes the fullest disclosed version of the pipeline: SFT
with "interleaved thinking" traces at a 202,752-token context, then reasoning
RL with GRPO plus a stabilization scheme it names IcePop, then asynchronous
agentic RL, then general RL, consolidated by an on-policy cross-stage
distillation step. GLM-5.2 swapped in critic-based PPO for long-horizon tasks
and added an anti-reward-hacking module — a rule filter plus an LLM judge —
in the reward path.

```text
GLM-5 / 5.2 pipeline (Reported, lineage):

  base ─ SFT (interleaved thinking)
           ├─ reasoning RL (GRPO + IcePop)
           ├─ agentic RL (async; critic PPO for long horizon in 5.2)
           ├─ general RL
           └─ on-policy cross-stage distillation ─► instruct model

GLM-5.3 (Reported): same machinery, scaled —
  more environments · more tasks · SAO with compaction · 2.3× RL throughput
```

[Module 13](../modules/13-sft.md) builds response-masked SFT and
[Beyond RL](../beyond/rl.md) builds the GRPO loop — both **built in g2c** at
toy scale. Critic-based PPO adds a value model the course deliberately avoids
(GRPO exists to not need one), and the cross-stage distillation is the same
**conceptual bridge** to [Module 16B](../modules/16b-synthetic-data.md) that
DeepSeek's OPD is: on-policy trajectories and teacher logits, not
teacher-written text.

### SAO: RL sampling redesigned for long horizons

GLM-5.3's blog credits SAO — Single-Rollout Asynchronous Optimization — for
its long-horizon RL. The SAO paper replaces GRPO's group-wise sampling (many
rollouts per prompt, advantage from the group mean) with single rollouts and
strict double-sided token-level clipping, which matters when one rollout is a
multi-hour agent trajectory and collecting a group of 32 per prompt is the
bottleneck. **G2C interpretation:** this is the group-relative baseline from
[Beyond RL](../beyond/rl.md) being *removed* under systems pressure — the toy
module's design choice, revisited because a group of long-horizon rollouts
became too expensive to sit in one batch. The mechanism is a **conceptual
bridge**; the course implements the group-wise form SAO departs from.

### Environments are synthesized, then verified by a judge

**Reported**, verbatim: "Research agents collect task patterns from real work
and turn them into runnable long-horizon environments with multi-step
dependencies and hidden state; a judge agent then attempts each task to verify
that it is actually solvable."

**G2C interpretation.** This is the Self-Instruct shape from
[Module 16B](../modules/16b-synthetic-data.md) applied to RL environments
instead of SFT pairs: a generator proposes, an independent check gates, and
only the survivors enter training. The course's design rule that the proposer
must not grade its own output reappears here as a separate judge agent whose
job is to prove solvability before a task can issue reward. What is **not
disclosed** is everything quantitative: environment counts, rejection rates,
or how much of the 5.3 delta each environment family bought.

### Thinking effort is now a product contract

Three levels (`low`, `high`, `max`; `max` recommended for coding), and
thinking can no longer be disabled — `thinking.type: enabled` is mandatory, a
breaking API change from 5.2. **G2C interpretation:** reasoning length has
become a serving-time knob trained in during post-training, the
production-scale relative of the decode-time budgets studied in
[Module 11](../modules/11-sampling.md). Removing the off switch is a
distribution statement: Z.ai no longer trains or evaluates a no-thinking mode
worth supporting.

## The cyber capability and the delayed weights

**Reported.** Z.ai frames offensive-security skill as emergent from scaled
agentic RL — capability that "continued to develop as training scaled," with
the model "reasoning across multiple stages of exploitation." Its disclosure
program reports 2,436 vulnerabilities found across 269 open-source projects,
1,097 of medium-to-high severity, 53 publicly disclosed with 2,383 under
embargo, the oldest bug roughly 40 years old and the average one undetected
for 26.6 years. CyberGym 84.5 is claimed as the best published result.
Benchmark evaluation ran behind domain whitelists, and the weight release is
explicitly held for hardening because of this capability. Third-party
reporting adds color but, at the snapshot date, no independent verification of
the ledger's contents.

**G2C interpretation.** The course's security material —
[Module 18](../modules/18-tools.md)'s prompt-injection lab and untrusted-tool-output
rule, [Beyond Harness](../beyond/harness.md)'s insistence that a permission
check is not a sandbox — treats the *model* as the thing to defend. GLM-5.3 is
the other direction: the model as the thing someone else must defend against,
with release engineering (evaluation, hardening, staged weights) as the
control surface. The mechanisms are course material; the policy layer is
**not yet covered** anywhere in g2c, and a Brief is the right container for
it — it is a property of releases, not of laptop-scale builds.

**Not disclosed.** What "hardening" changes in the weights, how the safety
evaluation is scored, and whether the published checkpoint will match the
API-served model.

## Coverage map

| GLM-5.3 mechanism | G2C connection | Coverage |
|---|---|---|
| Transformer, causal objective, MoE total/active split | Modules 07–10, Beyond MoE | **Built in g2c** at mechanism scale |
| MLA latent KV compression + DSA learned sparse selection | Modules 08, 16 + Beyond Linear Attention | **Conceptual bridge** |
| IndexShare cross-layer indexer reuse | Module 16's cache reasoning | **Conceptual bridge** |
| Staged long-context extension to 1M | Beyond Midtraining | **Conceptual bridge** |
| Response-masked SFT with thinking traces | Module 13 | **Built in g2c** |
| GRPO reasoning RL | Beyond RL | **Built in g2c** at toy scale |
| Critic-based PPO, SAO single-rollout async RL | Beyond RL's GRPO baseline | **Conceptual bridge** |
| Environment synthesis with judge-agent gating | Module 16B's propose/filter funnel | **Conceptual bridge** |
| Anti-reward-hacking reward filtering | Module 15's evaluation-integrity concerns | **Conceptual bridge** |
| On-policy cross-stage distillation | Module 16B | **Conceptual bridge** |
| MTP speculative decoding | Modules 06, 11, 16 | **Not yet covered** |
| Muon (Split) optimizer | Module 03B baseline | **Not yet covered** |
| slime RL infrastructure, 2.3× throughput systems work | no distributed-training track | **Not yet covered** |
| Emergent-capability release policy | Brief-level only | **Not yet covered** by design |

## Claims versus demonstrated evidence

### What the primary sources establish directly

- The blog and docs commit, in plain text, to the frozen-base claim, the
  two-week weight timeline, the mandatory-thinking API change, and the
  benchmark table's numbers as Z.ai's own measurements.
- The lineage architecture is genuinely well documented: the GLM-5 report plus
  the GLM-5.2 `config.json` pin every structural number this Brief cites, and
  the 5.2 weights those numbers describe are public today.
- The 5.2 → 5.3 delta column is reproducible arithmetic from one vendor's
  published table.

### What remains reported rather than independently established here

- Every GLM-5.3 quality and capability claim is vendor-evaluated against an
  API-only model. Until weights are published, no independent measurement of
  any table row is possible even in principle.
- "Same base model" is asserted, not yet verifiable; weight diffing becomes
  possible only after release.
- The vulnerability ledger's embargoed majority (2,383 of 2,436) cannot be
  inspected; the emergence narrative — capability arising from scaled general
  agentic RL rather than targeted cyber training — is not supported by any
  disclosed ablation.
- No GLM-5.3 technical report exists: post-training compute, data volumes,
  environment counts, reward configurations, and the hardening methodology are
  all **not disclosed**.
- The license for the forthcoming weights is unstated; MIT is lineage
  precedent only.

At 744B total parameters — roughly 1.5TB of BF16 weights before any serving
state — GLM-5.3 sits far outside the course's laptop boundary even after its
weights ship, and at the snapshot date it is API-only besides. This Brief asks
you to run nothing.

## A practical reading route

There is no report to read front-to-back, so read the chain in dependency
order:

1. **The [GLM-5.3 blog](https://z.ai/blog/glm-5.3), twice.** Once for claims,
   once marking each sentence as frozen-base fact, post-training change, or
   evaluation result.
2. **The delta column** of the benchmark table, before any cross-vendor row.
   Ask of each suite: how long-horizon is it, and does the gain pattern match
   "more environments"?
3. **GLM-5's report, §§ on architecture and post-training** — for what the
   frozen base actually is and the pipeline 5.3 scaled. Skim; you have built
   the load-bearing parts.
4. **The [SAO paper's](https://arxiv.org/abs/2607.07508) introduction** — just
   far enough to see why group-wise sampling breaks at long horizons.
5. **The [GLM-5.2 `config.json`](https://huggingface.co/zai-org/GLM-5.2/raw/main/config.json)** —
   every field maps to something you built or a Beyond module you read. This
   file is the ground truth the prose inherits from.
6. **The cyber section last**, with the evidence labels in hand: ledger
   numbers are Reported; emergence is a narrative; the embargo is a reason you
   cannot check.

## After this Brief, you should be able to

- explain why a frozen base makes GLM-5.3's benchmark deltas unusually
  attributable, and what confound (vendor-run evaluation) still remains;
- trace a claim about GLM-5.3's architecture back through the "same base
  model" assertion to the GLM-5/5.2 documents that actually support it;
- describe what SAO removes from GRPO and the systems pressure that motivated
  removing it;
- connect judge-gated environment synthesis to Module 16B's rule that the
  proposer must not grade its own output;
- state what the vulnerability ledger establishes, what it merely reports, and
  why the weight delay is a release-policy mechanism rather than a model
  mechanism;
- explain why "disabling thinking is no longer supported" is a training-
  distribution statement and not just an API deprecation;
- label any sentence in the release as reported, reported-by-lineage, derived,
  interpreted, or not disclosed.
