# Beyond — Midtraining: continued pretraining and data mixtures

> **Question this module answers:** *How can continued next-token training add a capability without erasing what the base model already knows?*

<!-- TODO(hero pipeline): asset not yet generated -->
![One base checkpoint forks into Python-only and Python-plus-replay midtraining branches, then both meet again at a three-domain evaluation table balancing adaptation against retention.](midtraining/BeyondMidtraining-Hero.png)

Pretraining built a general next-token predictor; post-training will teach it how to answer. Between those stages, modern model recipes often continue the same pretraining objective on a deliberately changed distribution: more code, mathematics, another language, longer sequences, or synthetic capability data. This module makes that middle stage concrete. You will fork `TinyLLM-30M-base` into equal-token Python-only and Python-plus-replay runs, then measure the bargain each branch made across code, general text, and stories.

> **This is a Beyond module.** Beyond modules sit outside the numbered course: nothing in Modules 00–20 depends on them, and they are not part of finishing the course. Come here in any order, whenever a model card or paper names the idea and you want the load-bearing version built and broken on your own machine.

---
## Before you start

* *Review* [09B-pretraining](../modules/09b-pretraining.md) for causal language-model loss and [10-tinyllm](../modules/10-tinyllm.md) for the trainer and durable checkpoint
* *Review* [13-sft](../modules/13-sft.md) for the objective boundary this module sits immediately before
* *Finish* `g2c/pretraining`, `g2c/training`, and `g2c/transformer`
* *Run* `./datasets.sh --small` for G2C Corpus v1 and `./checkpoints.sh tinyllm-30m` for the required base checkpoint
* *Run* `G2C_APPLY_SOLUTIONS=01-10 ./notebook.sh midtraining` instead of the plain launch if you are entering without your own implementations

---
## Where this fits in

The main course moves from scaling experiments in Module 12 to supervised instruction tuning in Module 13. That is a clean teaching arc, but production model development often inserts another stage. The model continues predicting the next token, now under a data distribution chosen to strengthen particular capabilities before alignment begins.

Names vary. **Continued pretraining**, **domain-adaptive pretraining**, **second-stage pretraining**, and **midtraining** overlap without being perfect synonyms. This module uses *midtraining* as the umbrella and implements one precise case: **domain-adaptive continued pretraining** on Python with general-data replay.

The distinction from SFT is objective-level, not marketing-level:

```
   pretraining / midtraining                 supervised fine-tuning
   ─────────────────────────                 ──────────────────────
   raw token stream                          prompt + response examples
   predict every next token                  mask prompt; train on response
   changes knowledge/capability mix          teaches prompted behavior
   no assistant role is required             chat template is part of data
```

If a run uses raw Python and ordinary causal loss, it is continued pretraining even if it is short. If it uses instruction/response records and response masking, it is SFT even if it updates every weight. The objective and data contract define the stage.

## The big idea

Midtraining changes a model by changing what it continues to predict. That sounds almost too simple, but it creates a three-way design problem:

```
   target-domain gain
          ▲
          │       ● Python-only
          │     ╱
          │   ● replay mixture
          │ ╱
          ●──────────────────► retained general capability
        base

   compute is fixed: moving along the frontier changes which tokens
   receive the training budget, not how much compute the run gets
```

The Python-only branch spends every update on the new domain. It should adapt fastest and has no direct pressure to preserve the old distribution. The replay branch spends 80% of the same token budget on Python and 20% on non-code examples from the base corpus. It gets fewer Python tokens, deliberately, in exchange for a continuing gradient from the old world.

### Same objective, new distribution

For a token sequence `x₁ … x_T`, nothing about the loss changes:

```
   L = − 1/T · Σ_t log pθ(x_t | x_<t)
```

Only the source distribution changes:

```
   Python-only:     x ~ D_python

   replay branch:  x ~ 0.8 D_python + 0.2 D_general
```

That makes the data mixture a model-design parameter. `TokenMixture` samples each batch row from one source, then samples a contiguous language-model window inside that source. It never concatenates Python and prose before choosing windows; doing so would manufacture fake boundary transitions and train on them.

The 80/20 ratio is a course default, not a universal recipe. Source similarity, base-model maturity, target-domain size, learning rate, and token budget all move the frontier. The optional sweep treats the ratio as the empirical variable it is.

### Equal tokens make the comparison honest

Both required branches receive exactly:

```
   300 steps × batch 4 × context 256 = 307,200 training tokens
```

This is an **equal-total-token** comparison. The branches use approximately equal compute, but the replay branch sees only about 80% as many Python tokens. That sacrifice is the intervention we want to measure.

An equal-Python-token comparison would answer a different question. It would give the replay branch additional general tokens and therefore additional compute. Useful for deployment planning, perhaps, but not a clean test of what replay buys under a fixed budget.

`TokenMixture` implements 80/20 in expectation by choosing a source independently for each batch row. A batch of four cannot literally contain 3.2 Python examples. The notebook records realized fractions over the run; enough batches converge toward the configured ratio. Reproducibility comes from the trainer's seeded generator, including across checkpoint resume.

### Adaptation and forgetting are both gradients

Catastrophic forgetting sounds like a special failure mechanism. Here it is the ordinary optimizer doing exactly what it was asked to do. Every Python-only gradient improves predictions under `D_python`; nothing in that objective rewards preserving predictions on stories or general prose. Shared parameters move toward the new distribution, and old-domain loss can rise.

Replay adds old-domain gradients back into the update. It does not freeze general capability or guarantee preservation; it changes the expected gradient:

```
   g_python-only = E_python[∇L]

   g_replay      = 0.8 E_python[∇L] + 0.2 E_general[∇L]
```

That equation is the module. At large scale, teams spend enormous effort choosing the distributions represented by those expectation signs.

### Restart the optimizer gently

The base checkpoint finished a long cosine decay near a small learning rate. A new stage needs a new schedule horizon, but restarting at the original pretraining peak can damage useful weights before the new distribution has supplied a stable direction.

The required run uses AdamW, a short 20-step rewarm, `2e-5` peak learning rate, and cosine decay to `2e-6`—one tenth of the base run's `2e-4` peak. This is a calibrated laptop recipe, not a scaling law. The principles are more durable:

- use a fresh optimizer state unless the continuation is truly one uninterrupted run;
- rewarm briefly rather than applying the peak update immediately;
- use a lower peak for adaptation than for training the same model from random initialization;
- compare runs at the same token budget and schedule.

### Evaluate a matrix, not the training loss

Training loss answers “how well did this branch predict the distribution it sampled?” It cannot measure a domain the branch never sees. The notebook freezes three validation streams before either branch trains:

| Domain | What it measures |
|---|---|
| held-out CodeSearchNet Python | target-domain adaptation |
| held-out FineWeb-Edu + Cosmopedia | general-text retention |
| held-out TinyStories | retention on a visibly different style |

Each checkpoint—base, Python-only, replay—runs on the exact same seeded windows. Report absolute loss and change from base. Lower Python loss is adaptation; positive general/story deltas are forgetting; replay succeeds when it moves those retention deltas toward zero without surrendering all of the target gain.

Reference calibration at 307,200 tokens produced the intended small but measurable pattern: Python-only gained slightly more on Python while general and story loss rose; 80/20 replay preserved those domains while retaining most of the Python gain. Your exact values may differ with MPS kernels and checkpoint provenance. The direction and the evidence matter more than matching decimals.

Samples are secondary evidence. A 30M base model does not become a capable coding assistant after 300 steps. Fixed code and prose prompts can reveal local syntax or style shifts, while a Python `ast.parse` rate may be reported as **experimental** if it has enough non-degenerate signal. Do not replace held-out loss with a handful of attractive generations.

### Long-context extension is also midtraining

Domain adaptation is only one use of this stage. Modern recipes also continue training to change the model's context regime. A useful long-context model needs more than a larger `max_seq_len` value:

```
   base checkpoint
         ↓
   adapt positional representation
         ↓
   progressively increase sequence length
         ↓
   train on long documents + synthetic long-range dependencies
         ↓
   evaluate long-context use AND short-context retention
```

Four pieces interact:

1. **Position representation.** RoPE models often use interpolation or frequency-scaling methods such as YaRN. The course `TransformerLM` uses learned absolute positional embeddings; extending it would require resizing and training that table, making the exercise architecture surgery as well as midtraining.
2. **Long-sequence exposure.** Accepting a long tensor does not mean the model learned to use distant evidence. Training needs examples whose prediction genuinely depends on far-away tokens.
3. **A length curriculum.** Recipes commonly increase sequence length progressively. This controls optimization shock and avoids paying maximum attention cost for every early update.
4. **Two-regime evaluation.** Needle retrieval is a useful diagnostic, not a complete long-context evaluation. Test retrieval, modeling of long documents, and ordinary short-context quality.

For dense attention, the matrix cost grows quadratically:

| Context | Attention cells | Relative to 256 |
|---:|---:|---:|
| 256 | 65,536 | 1× |
| 512 | 262,144 | 4× |
| 1,024 | 1,048,576 | 16× |
| 2,048 | 4,194,304 | 64× |

That is why long-context extension remains lecture-only here. On this course model it would combine positional-embedding changes, a new data pipeline, expensive training, and specialized evaluation. The [linear-attention](linear-attention.md) Beyond module attacks the inference/training cost from another direction; this section explains why that cost matters.

## Concepts to internalize

- **Midtraining keeps the base objective and changes its distribution.** Raw-token causal loss distinguishes it from response-masked SFT.
- **Mixture weights are model-design parameters.** They trade target exposure for retention gradients.
- **Equal total tokens are the fairness control.** Equal target tokens would grant replay extra compute.
- **Forgetting is an expected consequence, not a mysterious bug.** The new objective contains no old-domain preservation term unless replay supplies one.
- **Evaluate every affected domain.** In-domain loss alone can make destructive adaptation look unambiguously successful.
- **Long context is trained behavior, not a configuration claim.** Position handling, long data, curriculum, and evaluation must agree.

### What we don't cover

- **Full long-context training.** It requires positional changes and a substantially larger evaluation surface; the lecture notes provide the map.
- **Tokenizer adaptation.** All branches retain `G2CTokenizer`, isolating weight adaptation. Adding vocabulary changes embeddings and creates a different experiment.
- **Instruction or preference tuning.** Modules 13–14 cover those objectives after the base-capability stage.
- **Synthetic capability curricula.** Mathematics and reasoning midtraining often use generated data; [synthetic data](../modules/16b-synthetic-data.md) teaches the quality-control side separately.
- **A universal replay ratio.** The optional sweep demonstrates why none exists.

---
## What you'll build

Package: `g2c/midtraining/`

```python
class TokenMixture:                                      # boilerplate provided
    sources: dict[str, TokenSource]
    weights: Tensor
    example_counts: dict[str, int]

    def get_lm_batch(                                    # scaffolded
        self, batch_size, context_length, *, generator=None
    ) -> tuple[Tensor, Tensor]: ...

def evaluate_domain_losses(                              # scaffolded
    model, domains, *, batch_size, context_length,
    eval_iters=20, device="auto", seed=0,
) -> dict[str, float]: ...
```

The notebook reuses Module 10's `Trainer`, checkpoint-resume plumbing, `G2CTokenizer`, and `TinyLLM-30M-base`. Source-specific decompression/tokenization and plotting live in `g2c/notebook_extras/midtraining.py`; they are plumbing, not student work.

The durable outputs are:

```text
artifacts/models/TinyLLM-30M-mid-python/
artifacts/models/TinyLLM-30M-mid-python-replay/
```

Both manifests name `TinyLLM-30M-base` as their parent and record source mixture, token budget, schedule, seed, and realized branch configuration.

## How to run the tests

```bash
source .venv/bin/activate

pytest tests/test_midtraining.py
pytest tests/test_midtraining.py -x
pytest tests/test_midtraining.py -k mixture
pytest tests/test_midtraining.py -k evaluate
```

The tests use tiny in-memory token streams and a deterministic stub model. No corpus, checkpoint, MPS device, or network is needed.

Initial scaffold state: **5 passed, 10 failed**. Constructor and validation
plumbing starts green; mixture sampling and per-domain evaluation turn green in
the suggested order from the test-file header.

## Exercises

Open the working notebook with `./notebook.sh midtraining` (or `./notebook.sh midtraining --fresh` to reset from the clean scaffold), write answers in the `Question:` / `Answer:` cells, and ask a coding agent for hints or grading. Partial submissions are fine because blank answers are skipped.

1. **Establish the base.** Load `TinyLLM-30M-base`, build frozen Python/general/story validation streams, and record its three-domain loss row.
2. **Build the mixture.** Implement `TokenMixture.get_lm_batch`, inspect source-pure rows, and verify that realized 80/20 fractions converge over many batches.
3. **Run equal-token branches.** Fork the same base checkpoint into 100/0 and 80/20 runs. Train each for 300 × 4 × 256 = 307,200 tokens with the same schedule and seed.
4. **Measure adaptation versus retention.** Implement `evaluate_domain_losses`; build the three-checkpoint × three-domain table and delta plot.
5. **Inspect behavior carefully.** Compare fixed Python and prose continuations. Optionally report syntax-validity as an experimental metric, but interpret it alongside loss and raw samples.
6. **Sweep replay ratio (optional).** Compare 100/0, 90/10, 80/20, and 50/50 under a shorter equal-token budget. Plot the adaptation-retention frontier rather than choosing a winner by one domain.

## Pitfalls to expect

- **Starting one branch from the other.** Both must fork the identical base checkpoint or the comparison is sequential training, not a controlled branch.
- **Equalizing Python tokens instead of total tokens.** That gives replay extra compute and answers a different question.
- **Building windows across corpus boundaries.** Sample a source first, then a window inside it.
- **Reporting configured rather than realized mixture.** Small batches fluctuate; record the observed allocation.
- **Evaluating on training slices.** Improvement there can be memorization. Validation shards are fixed before training.
- **Calling lower Python loss an unconditional win.** The retention columns are part of the result.
- **Restarting at the original peak learning rate.** A mature checkpoint needs a gentler continuation schedule.
- **Calling a larger context limit long-context capability.** Utilization must be trained and evaluated.

## M-series notes

- **Required training is short but real.** On the calibration machine, the two 307,200-token branches took roughly five minutes combined on MPS, plus tokenization and evaluation. Hardware and background load vary.
- **Memory is dominated by the 30M model, optimizer state, and activations.** The required `B=4, T=256` configuration is intentionally conservative for 16GB machines.
- **Source tokenization is bounded.** The notebook reads 8MB per training domain and 1MB per validation domain rather than materializing G2C Corpus v1.
- **Checkpoints are resumable.** Rolling state lives under `data/work/beyond-midtraining/`; durable completed models live under `artifacts/models/`.

---
## Reading

Primary:

- **Gururangan, Marasović, Swayamdipta et al., [“Don't Stop Pretraining: Adapt Language Models to Domains and Tasks”](https://arxiv.org/abs/2004.10964) (2020).** Domain- and task-adaptive pretraining, and the cleanest historical framing for this module.
- **Parmar, Satheesh, Patwary et al., [“Reuse, Don't Retrain: A Recipe for Continued Pretraining of Language Models”](https://arxiv.org/abs/2407.07263) (2024).** Data-distribution and learning-rate guidance for continuing mature checkpoints.
- **Elhady, Agirre, Artetxe, [“Emergent Abilities of Large Language Models under Continued Pretraining for Language Adaptation”](https://arxiv.org/abs/2506.00288) (2025).** Why target-language perplexity can conceal broader capability loss and why replay matters.

Long-context extension:

- **Xiong et al., [“Effective Long-Context Scaling of Foundation Models”](https://ai.meta.com/research/publications/effective-long-context-scaling-of-foundation-models/) (2023).** Continual pretraining with longer sequences, long-data upsampling, and length curricula.
- **Peng et al., [“YaRN: Efficient Context Window Extension of Large Language Models”](https://arxiv.org/abs/2309.00071) (2023).** Position-frequency adaptation for RoPE models.
- **Qwen Team, [“Qwen2.5-1M Technical Report”](https://arxiv.org/abs/2501.15383) (2025).** Progressive long-context pretraining, data synthesis, post-training, evaluation, and inference as one system.

## Deliverable checklist

- [ ] All tests in `tests/test_midtraining.py` pass.
- [ ] Both branches start from the identical `TinyLLM-30M-base` artifact and consume 307,200 tokens.
- [ ] Both derived artifacts are saved with parent lineage and mixture metadata.
- [ ] Notebook contains the three-checkpoint × three-domain loss table and delta plot.
- [ ] Fixed Python and prose samples are shown without treating isolated attractive output as the primary metric.
- [ ] You can explain why equal total tokens, not equal Python tokens, is the fairness control used here.
- [ ] You can explain why replay changes the expected gradient and why it cannot guarantee zero forgetting.
- [ ] You can explain why long-context extension requires position adaptation, long-sequence training, a curriculum, and two-regime evaluation.
