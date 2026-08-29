# Beyond — Long context: position, training, and evaluation

> **Question this module answers:** *What does a model's context-window number actually promise—and what makes that promise real?*

<!-- TODO(hero pipeline): asset not yet generated -->
![A context-window claim decomposed into positional representation, long-sequence training, attention cost, utilization tests, and short-context retention.](long-context/BeyondLongContext-Hero.png)

A context limit is an interface constraint, not a capability certificate. A model may accept a long tensor yet fail to locate distant evidence, extrapolate its position representation, afford the attention computation, or preserve its short-context quality. This module separates those promises, then proposes a laptop experiment comparing learned absolute positions, RoPE, and NoPE outside their training range.

> **Draft status.** This is a proposed Beyond module specification. Its package, tests, notebook, rubric, and hero asset do not exist yet; the API and exercise counts below are design targets rather than a runnable assignment.

> **This is a Beyond module.** Beyond modules sit outside the numbered course: nothing in Modules 00–20 depends on them, and they are not part of finishing the course.

---
## Before you start

* *Review* [09-transformer-block](../modules/09-transformer-block.md) for causal masking and the course model's learned position table
* *Review* [Beyond Midtraining](midtraining.md) for continued next-token training and retention
* *Review* [The efficient-attention map](attention-efficiency.md) for the cost of making the sequence longer
* *Finish* `g2c/attention`, `g2c/transformer`, and `g2c/training`
* *Run* `./datasets.sh --tiny` if TinyStories is missing

---
## Where this fits in

Module 09 adds a learned vector for each position up to `max_seq_len`. That is ideal for making position visible, but it couples the model to a finite table. Production decoder models more often use rotary position embeddings (RoPE), sometimes adapt their frequencies for a longer range, and occasionally omit an explicit position signal in some or all layers.

Position is only one layer of the problem. [Beyond Midtraining](midtraining.md) explains why a longer training stage is still continued pretraining; [the efficient-attention map](attention-efficiency.md) separates dense kernels, compressed caches, sparsity, and recurrence. This module owns the seam between them: how position, data, length curriculum, compute, and evaluation combine into an honest long-context claim.

## The big idea

“Supports 128K context” can mean at least five different things:

| Claim | Question to ask |
|---|---|
| **Acceptance** | Can the implementation ingest that many tokens without an index or shape error? |
| **Affordability** | Can memory, bandwidth, and arithmetic sustain the workload? |
| **Exposure** | Did training contain sequences and dependencies at comparable distances? |
| **Utilization** | Can the model actually recover and use distant evidence? |
| **Retention** | Did extension preserve ordinary short-context behavior? |

A credible window needs evidence for all five. Configuration proves only acceptance.

### Position is not the same as order

The course model learns an absolute lookup table:

```text
hidden_t = token_embedding(x_t) + position_embedding[t]
```

Positions beyond the table do not exist. Resizing it initializes new parameters that still need training, while interpolating old vectors changes what familiar indices mean.

RoPE instead rotates query and key coordinates by a position-dependent angle. In each two-dimensional pair,

```text
q_t' = R(t) q_t        k_i' = R(i) k_i

q_t' · k_i' = q_tᵀ R(i - t) k_i
```

The dot product depends on relative displacement, but the frequencies still define a trained range. Position interpolation and frequency-scaling methods such as YaRN change how longer indices map onto those rotations. They improve the representation's reach; they do not teach the model which distant facts matter.

NoPE removes an explicit position embedding. It does **not** make a causal decoder unordered. Token `t` sees a different prefix from token `t-1`, and hidden states recursively carry that asymmetric history. Research results show useful length generalization in some causal-decoder settings, alongside a finite effective context and attention distraction. NoPE is therefore a real design point, not a universal cure.

Hybrid designs can mix RoPE and NoPE layers. That makes position handling a layer-pattern choice much like hybrid attention: some layers receive a strong explicit relative signal while others are free from a fixed rotation schedule. Whether the combination helps must be measured at the target scale and task.

### A long window is trained behavior

An extension recipe changes several controls together:

```text
base checkpoint
      ↓
choose or adapt position strategy
      ↓
increase sequence lengths progressively
      ↓
train on long documents + explicit distant dependencies
      ↓
evaluate long utilization AND short retention
```

Long documents alone are insufficient when most next-token predictions remain local. Synthetic passkeys, repeated-key disambiguation, and associative-retrieval examples create gradients whose correct prediction depends on distant context. They complement rather than replace natural long documents.

A mixed-length curriculum avoids paying maximum dense-attention cost on every update and keeps short examples represented. For dense attention, doubling `T` quadruples the `T × T` pair count. The [hardware-aware architecture](hardware.md) module explains why realized time can still diverge from that arithmetic.

### Evaluate a profile, not one score

Needle-in-a-haystack is a diagnostic: it tests retrieval under controlled placement and distance. It can be gamed by simple lexical matching and says little about integrating evidence spread across a document. A useful evaluation matrix includes:

- language-model loss bucketed by token position;
- passkey or needle retrieval swept over depth and total length;
- associative retrieval with distractors and repeated keys;
- at least one natural long-document task requiring evidence use;
- the original short-context validation and task suite.

Plotting accuracy against distance exposes an **effective context** that a single advertised maximum hides.

## Concepts to internalize

- **A maximum sequence length proves acceptance, not use.** Affordability, exposure, utilization, and retention need separate evidence.
- **Position encoding and causal order are different.** NoPE removes an explicit signal; the causal computation still exposes ordered prefixes.
- **RoPE extrapolation is a frequency problem.** Scaling or interpolation changes the position map but cannot substitute for long-range training data.
- **Long context is a data problem.** Examples need dependencies at the distances the model should learn to use.
- **Length extension is a retention problem.** Always evaluate the original short regime alongside the new long one.
- **Architecture and kernels bound the feasible curriculum.** Dense, sparse, compressed-cache, and recurrent models pay different long-sequence bills.

### What we don't cover

- **Million-token reproduction.** The module studies the mechanisms at `64 → 128 → 256` tokens so the experiment fits an M-series laptop.
- **One winning position scheme.** APE, RoPE, NoPE, and hybrids have architecture- and task-dependent tradeoffs.
- **A production YaRN implementation.** The scaffold isolates position interpolation and basic RoPE; frequency schedules remain a reading bridge.
- **Sparse or recurrent attention implementations.** Those belong to [the efficient-attention map](attention-efficiency.md) and [linear attention](linear-attention.md).
- **Long-context post-training.** The required path uses raw-token causal loss; instruction data at long lengths is a separate stage.

---
## What you'll build

Planned package: `g2c/long_context/`

```python
def apply_rope(q, k, *, positions, base=10_000): ...       # scaffolded
def interpolate_positions(positions, *, train_len, target_len): ...  # scaffolded

class PositionStrategy: ...                               # boilerplate protocol
class LearnedAbsolutePosition(PositionStrategy): ...      # provided adapter
class RotaryPosition(PositionStrategy): ...               # scaffolded application
class NoPosition(PositionStrategy): ...                    # provided zero-op
```

The notebook will train matched tiny causal models at context 64, then evaluate at 64, 128, and 256. It will compare configuration-only extension, position adaptation without continued training, and a short mixed-length continuation. The goal is a distance profile, not a promised winner.

## How to run the tests

Planned commands:

```bash
source .venv/bin/activate

pytest tests/test_long_context.py
pytest tests/test_long_context.py -x
pytest tests/test_long_context.py -k rope
pytest tests/test_long_context.py -k position
```

The contract tests will pin RoPE shape and norm preservation, relative-shift behavior, interpolation bounds, finite APE range, NoPE's lack of position parameters, and causality across every strategy.

## Exercises

Open the planned working notebook with `./notebook.sh long-context` (or `./notebook.sh long-context --fresh`), write answers in the `Question:` / `Answer:` cells, and ask a coding agent for hints or grading. Partial submissions are fine because blank answers are skipped.

1. **Build the position mechanisms.** Implement RoPE and interpolation, then inspect rotations and attention logits across positions.
2. **Separate acceptance from utilization.** Extend each model's configured limit and run a passkey sweep before any continuation training.
3. **Continue at mixed lengths.** Compare a short-only control with a `64 → 128 → 256` curriculum under an equal-token budget.
4. **Measure the context profile.** Plot retrieval by distance, loss by position, and short-context validation loss for every checkpoint.
5. **Explain NoPE carefully.** Use a causal-prefix intervention to show why removing explicit position vectors does not remove ordered computation.

## Pitfalls to expect

- **Calling a successful forward pass long-context capability.** That establishes acceptance only.
- **Changing architecture, data, and token budget in one comparison.** Keep matched controls so the source of improvement remains legible.
- **Training only on padded short examples.** Tensor length is not dependency distance.
- **Treating one needle placement as an evaluation.** Sweep depth, length, distractors, and seeds.
- **Reading NoPE as permutation invariance.** Causal masking and prefix-dependent hidden states still expose order.
- **Forgetting the old regime.** A longer effective context can coexist with worse short-context loss.

## M-series notes

- The required range stops at 256 tokens; dense attention at 512 is an optional diagnostic, not part of completion.
- Equal-token comparisons matter because longer sequences otherwise receive more attention work per optimization step.
- MPS timing must synchronize before and after a measured region; timing helpers belong in `g2c/notebook_extras/`.

---
## Reading

- **Kazemnejad et al., [“The Impact of Positional Encoding on Length Generalization in Transformers”](https://arxiv.org/abs/2305.19466) (2023).** The central APE/RoPE/NoPE comparison and the argument that causal decoders can recover position information implicitly.
- **Wang et al., [“Length Generalization of Causal Transformers without Position Encoding”](https://arxiv.org/abs/2404.12224) (2024).** Evidence that NoPE has useful extrapolation and a finite effective context shaped by attention distraction.
- **Peng et al., [“YaRN: Efficient Context Window Extension of Large Language Models”](https://arxiv.org/abs/2309.00071) (2023).** RoPE frequency scaling for extending a pretrained context regime.
- **Xiong et al., [“Effective Long-Context Scaling of Foundation Models”](https://ai.meta.com/research/publications/effective-long-context-scaling-of-foundation-models/) (2023).** Continued training, long-data mixtures, and progressive lengths.
- **Qwen Team, [“Qwen2.5-1M Technical Report”](https://arxiv.org/abs/2501.15383) (2025).** A full production stack spanning long-data synthesis, progressive training, evaluation, and inference.

## Deliverable checklist

- [ ] Planned tests pass for RoPE, interpolation, NoPE, and causal behavior.
- [ ] Notebook separates configured acceptance from measured utilization.
- [ ] Distance and position profiles compare the matched position strategies.
- [ ] Short-context retention is reported beside long-context results.
- [ ] You can explain why NoPE preserves causal order and why position adaptation alone is insufficient.
