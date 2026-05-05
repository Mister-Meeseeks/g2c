# Model Artifacts and Course Tracks

This note defines the reusable model artifacts for the second half of the
course. It is meant to align students, maintainers, and coding agents when
Modules 10-20 refer to saved models, tokenizers, datasets, evals, and runtime
backends.

The main idea:

> Expensive or reused outputs become named course artifacts. Tiny models teach
> the mechanism; production local models show what scale unlocks through the
> same harness.

## Model Families

The course uses three model families.

### StoryLM

StoryLM is the TinyStories/narrative track. It is the most accessible path for
students with smaller machines because TinyStories is compact, coherent, and
easy to evaluate.

StoryLM should demonstrate:

- next-token language modeling from scratch;
- visible improvement from tokenizer quality, data quality, and scale;
- sampling behavior;
- story-instruction SFT;
- story-focused DPO;
- story-domain evaluation and targeted data planning.

StoryLM is not expected to be a general assistant. If it is used in Modules
13-15, the SFT, DPO, and eval tasks should stay in the story domain.

### TinyLLM

TinyLLM is the broader from-scratch model track. It is the best target for
assistant-shaped SFT, DPO, and eval before the course pivots to a production
model.

TinyLLM should demonstrate:

- broader language modeling on a compact course corpus;
- simple instruction following after SFT;
- toy structured tool-call formatting;
- meaningful failure modes in eval;
- comparison against ProdLLM through the same runtime harness.

TinyLLM is still not a useful general assistant. It teaches mechanisms and
failure modes; it does not replace a pretrained instruct model.

### ProdLLM

ProdLLM is the local pretrained instruct backend used in Modules 16-20. It is a
conceptual role, not a fixed model name. Students should choose a local model
that fits their machine through Ollama, MLX, llama.cpp, or a similar local
runtime.

Examples by hardware tier:

- smaller fallback: a 1.5B-3B instruct model;
- recommended: a 7B-8B instruct model;
- stretch: a 14B-class instruct model if local memory and speed allow.

The course may use one reference model for tested examples, but module design
should avoid hardcoding that choice. Later assistant-system code should talk to
a `ProdLLM` backend interface.

## Hardware Tracks

Modules 10-15 should support multiple local-compute tracks.

### Small Track

The small track is the required-compatible path for weaker machines. It should
run on modest M-series hardware and teach every mechanism, even if the samples
are weak.

Typical artifacts:

- `ShakespeareLM-1M`
- `StoryLM-Small`
- `StoryLM-Instruct`
- `StoryLM-DPO`

### Default Track

The default track is the main local experience. It should produce a satisfying
from-scratch language model artifact without assuming cloud GPUs.

Typical artifacts:

- `StoryTokenizer`
- `StoryLM`
- `TinyLLM` when the broader compact corpus path is available
- `StoryLM-Instruct` or `TinyLLM-Instruct`
- `StoryLM-DPO` or `TinyLLM-DPO`

### Stretch Track

The stretch track is optional. It is for students with more unified memory,
more time, or a willingness to let local runs continue for hours.

Typical artifacts:

- `TinyLLM-Large`
- larger StoryLM/TinyLLM comparison checkpoints
- longer eval and sampling runs

Stretch artifacts should improve the qualitative experience, but the required
course path should not depend on them.

## Named Artifacts

The durable model artifacts are:

- `ShakespeareLM-1M`: the first real Transformer language-model milestone.
- `StoryTokenizer`: TinyStories BPE tokenizer, saved because it is slow enough
  to regenerate and needed for StoryLM checkpoints.
- `StoryLM-Small`: smaller TinyStories model for weak hardware and fast loops.
- `StoryLM`: default TinyStories model, intended to feel meaningfully
  language-like.
- `TinyLLM-Small`: smaller broad-corpus model when the TinyLLM path exists.
- `TinyLLM`: broader from-scratch model used for assistant-shaped tracks.
- `TinyLLM-Large`: optional larger local stretch model.
- `StoryLM-Instruct`: StoryLM after story-instruction SFT.
- `TinyLLM-Instruct`: TinyLLM after assistant-style SFT.
- `StoryLM-DPO`: StoryLM after story-domain preference tuning.
- `TinyLLM-DPO`: TinyLLM after assistant/tool-format preference tuning.

The durable non-model artifacts are:

- tokenizers;
- SFT datasets;
- DPO preference datasets;
- eval reports;
- targeted improvement datasets;
- sampling traces;
- tool-call traces;
- RAG indexes;
- agent transcripts.

MNIST models and small one-off training runs are not durable artifacts. They
are cheap to regenerate and do not carry forward into the language or assistant
stack.

## Artifact Contract

Model artifacts should be reloadable without notebook state. Use this shape
unless a module has a documented reason to differ:

```text
artifacts/models/<artifact-name>/
  model.pt
  config.json
  tokenizer.pkl
  manifest.json
```

Tokenizer artifacts should be saved separately when they are reused by multiple
models:

```text
artifacts/tokenizers/<tokenizer-name>/
  tokenizer.pkl
  config.json
  manifest.json
```

Training and eval data should be saved as data artifacts rather than hidden in
notebook outputs:

```text
artifacts/datasets/<dataset-name>/
  train.jsonl
  val.jsonl
  manifest.json

artifacts/evals/<eval-name>/
  runs.jsonl
  summary.md
  failure_cases.md
```

`manifest.json` should record enough information to understand provenance:

- artifact name;
- module that produced it;
- source dataset or corpus slice;
- tokenizer artifact;
- model config;
- training config;
- seed;
- git commit if available;
- creation date;
- notes on intended use.

## Module Map

### Module 10 - Milestone: Your First LLM

Module 10 is the first durable model-artifact boundary.

It should produce some subset of:

- `ShakespeareLM-1M`;
- `StoryTokenizer`;
- `StoryLM-Small`;
- `StoryLM`;
- optional `TinyLLM` if the broader corpus path is available.

### Module 11 - Sampling and Decoding

Module 11 should not change weights. It should reuse saved base models and save
sampling traces that show how decoding changes behavior.

### Module 12 - Scaling Experiments

Module 12 should compare artifacts across size, data, tokenizer, and compute
budgets. It may create additional checkpoints, but the core output is a scaling
comparison report.

### Module 13 - Instruction Tuning

Module 13 updates student-trained model weights.

Supported tracks:

- StoryLM track: story-writing instructions and story-editing prompts.
- TinyLLM track: assistant-style instructions, simple format following, and
  toy tool-call syntax.

It should produce:

- `StoryLM-Instruct`; or
- `TinyLLM-Instruct`.

ProdLLM should not be the default training target for Module 13. Fine-tuning a
production model can be optional later, but the core lesson is weight updates on
the student's own small model.

### Module 14 - Preference Tuning

Module 14 updates student-trained model weights with preference data.

Supported tracks:

- StoryLM track: coherent vs incoherent story continuations, tone, repetition,
  continuity, and prompt adherence.
- TinyLLM track: concise vs rambling answers, valid vs malformed chat format,
  valid vs invalid toy tool calls, and task adherence.

It should produce:

- `StoryLM-DPO`; or
- `TinyLLM-DPO`.

### Module 15 - Evaluation

Module 15 should not update weights by default. It should evaluate base, SFT,
and DPO artifacts, then turn failures into a targeted improvement dataset.

Supported tracks:

- StoryLM track: story-domain evals and story-targeted data.
- TinyLLM track: assistant/task/tool-format evals and targeted SFT or DPO data.

The deliverable should include:

- eval report;
- failure-case analysis;
- targeted improvement dataset;
- optional rerun instructions for students who want to close the loop by going
  back to Module 13 or 14.

Qwen-class or other ProdLLM models may appear as a preview comparison, but the
main failure-analysis target should be the student's tiny model artifacts.

### Modules 16-20 - Assistant Systems

Modules 16-20 pivot to ProdLLM for usable assistant behavior.

ProdLLM is not a trained artifact from this repo. It is a backend selection and
runtime configuration. These modules should save system artifacts instead:

- model backend config;
- prompts;
- RAG indexes;
- tool schemas;
- tool-call traces;
- agent transcripts;
- eval reports.

The runtime harness should make the contrast visible: the same messages, tools,
and evals can run against TinyLLM/StoryLM when useful, but ProdLLM is the
default backend for actually usable assistant behavior.

## Pedagogical Rules

- Save artifacts when they are expensive to regenerate, reused later, needed
  for comparison, or emotionally meaningful as milestones.
- Do not save every notebook experiment.
- Keep StoryLM tasks story-domain unless the module is explicitly demonstrating
  failure outside the model's domain.
- Keep TinyLLM tasks assistant-shaped but honest about capability.
- Use ProdLLM as a generic role, not as a fixed model name.
- Make larger local runs optional; never require cloud compute for the core
  path.
