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

The course uses four model-family roles.

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
- comparison against ProdLM through the same runtime harness.

TinyLLM is still not a useful general assistant. It teaches mechanisms and
failure modes; it does not replace a pretrained instruct model.

### ProdLM

ProdLM is the local pretrained instruct backend used in Modules 16-20. It is a
conceptual role, not a fixed model name. Students should choose a local model
that fits their machine through Ollama, MLX, llama.cpp, or a similar local
runtime.

Examples by hardware tier:

- smaller fallback: a 1.5B-3B instruct model;
- recommended: a 7B-8B instruct model;
- stretch: a 14B-class instruct model if local memory and speed allow.

The course may use one reference model for tested examples, but module design
should avoid hardcoding that choice. Later assistant-system code should talk to
a `ProdLM` backend interface.

### BaseLM

BaseLM is the small external pretrained base-model fallback for Modules 13-16.
It is not the main assistant-system backend and it is not trained from scratch
inside the course. Its role is to make SFT, DPO, and eval pedagogically useful
when a student's StoryLM/TinyLLM artifact is too weak to show the behavior.

BaseLM is a role, not a fixed model. Pick any small open-weight base model
that runs comfortably on your machine and serves the pedagogy well. Bind it
with `./baselm.sh <hf-model-id>`; the chosen backend is recorded
in `artifacts/models/BaseLM-base/manifest.json` and the model weights live in the
local HF hub cache under `data/cache/baselm/huggingface/`. Notebooks reference the
role `BaseLM`, not the exact artifact directory, so swapping the backend does not
require notebook edits.

BaseLM setup is separate from `datasets.sh` because it is a model artifact:

```bash
./baselm.sh
```

## Hardware Tracks

Modules 10-15 should support multiple local-compute tracks.

### Tiny Track

The tiny track is the required-compatible path for weaker machines. It should
run on modest M-series hardware and teach every mechanism, even if the samples
are weak.

Typical artifacts:

- `ShakespeareLM-1M-base`
- `StoryLM-1M-base`
- `StoryLM-5M-base`
- `StoryLM-5M-SFT`
- `StoryLM-5M-DPO`

### Standard Track

The standard track is the main local experience. It should produce a satisfying
from-scratch language model artifact without assuming cloud GPUs.

Typical artifacts:

- `StoryTokenizer`
- `StoryLM-30M-base`
- `TinyLLM-30M-base` when the broader compact corpus path is available
- `StoryLM-30M-SFT` or `TinyLLM-30M-SFT`
- `StoryLM-30M-DPO` or `TinyLLM-30M-DPO`

### Full / Stretch Track

The full/stretch track is optional. It is for students with more unified memory,
more time, or a willingness to let local runs continue for hours.

Typical artifacts:

- `TinyLLM-100M-base`
- larger StoryLM/TinyLLM comparison checkpoints
- longer eval and sampling runs

Stretch artifacts should improve the qualitative experience, but the required
course path should not depend on them.

## Named Artifacts

### Naming grammar

Every model artifact name is built from up to three components:

```
<family>[-<size>][-<stage>]

family ∈ {ShakespeareLM, StoryLM, TinyLLM, BaseLM, ProdLM}
size   = approximate parameter count, written as <N>M or <N>B (e.g. -1M, -5M, -10M, -30M, -100M, -1B)
stage  ∈ {base, SFT, DPO}
```

Component rules by family:

| Family | Size component | Stage component |
|---|---|---|
| `ShakespeareLM`, `StoryLM`, `TinyLLM` | **required** (we train them at known sizes) | **required on disk** |
| `BaseLM` | **forbidden** (we don't train it; one pretrained checkpoint) | **required on disk** |
| `ProdLM` | **forbidden** (backend selection, not a checkpoint) | **forbidden** |

The size is the trained parameter count, bucketed onto the family's known
size tiers (`-1M`, `-5M`, `-10M`, `-30M`, `-100M`) rather than reported
precisely. It follows industry convention (Qwen-9B, Llama-3-8B) of putting a
parameter count in the name rather than counting training tokens. SFT/DPO
derivatives keep the base's size since fine-tuning does not change parameter
count.

**On-disk artifact names always include every applicable component.** The base
stage is explicit: `StoryLM-30M-base`, `TinyLLM-30M-base`, `BaseLM-base`.
Saving `StoryLM-30M-SFT` writes to `artifacts/models/StoryLM-30M-SFT/`, never
to `artifacts/models/StoryLM-SFT/`. The directory name *is* the artifact key,
so two runs at the same family/size/stage collide; the second overwrites the
first.

### Load-time aliases

A name with the size component omitted is an **alias**, resolved at load time
to the largest available artifact in that family at that stage. Aliases are
strictly within family and within stage:

- `StoryLM` → largest `StoryLM-<N>-base` on disk
- `StoryLM-30M` → `StoryLM-30M-DPO`, then `StoryLM-30M-SFT`, then
  `StoryLM-30M-base` when a notebook wants the strongest available stage
- `StoryLM-SFT` → largest `StoryLM-<N>-SFT` on disk
- `TinyLLM-DPO` → largest `TinyLLM-<N>-DPO` on disk

Aliases are never written to disk. `BaseLM` is a role/selector for
`BaseLM-DPO`, then `BaseLM-SFT`, then `BaseLM-base` depending on context.
`ProdLM` is a backend role, not a model artifact. Cross-family fallback (a
notebook walking `TinyLLM → StoryLM → ShakespeareLM → BaseLM` when nothing
better exists) is **not** part of the alias rule; that lives at the notebook
orchestration layer.

### Derived-artifact lineage

Derived artifacts (`-SFT`, `-DPO`) record their parent in the manifest's
`base_artifact` field. The chain is followable: `BaseLM-DPO` →
`BaseLM-SFT` → `BaseLM-base`; `TinyLLM-30M-DPO` → `TinyLLM-30M-SFT` →
`TinyLLM-30M-base`. A reader can walk back to the source by following that
field rather than parsing the name.

### Canonical artifact list

Base models (self-trained):

- `ShakespeareLM-1M-base`: the first real Transformer language-model milestone.
- `StoryLM-1M-base`: the small TinyStories scaling anchor from Module 12.
- `StoryLM-5M-base`: smaller TinyStories model for weak hardware and fast loops.
- `StoryLM-30M-base`: default TinyStories model, intended to feel meaningfully
  language-like.
- `TinyLLM-30M-base`: broader from-scratch model used for assistant-shaped tracks.
- `TinyLLM-100M-base`: optional larger local stretch model.

External base model:

- `BaseLM-base`: small pretrained base used as a fallback for Modules 13-16 when
  self-trained models are too weak.

Backend role (no checkpoint):

- `ProdLM`: local pretrained instruct backend used in Modules 16-20.

Derived (SFT):

- `StoryLM-<N>-SFT`: StoryLM after story-instruction SFT.
- `TinyLLM-<N>-SFT`: TinyLLM after assistant-style SFT.
- `BaseLM-SFT`: BaseLM after the same SFT pass.

Derived (DPO):

- `StoryLM-<N>-DPO`: StoryLM after story-domain preference tuning.
- `TinyLLM-<N>-DPO`: TinyLLM after assistant/tool-format preference tuning.
- `BaseLM-DPO`: BaseLM after preference tuning, layered on top of `BaseLM-SFT`.

The companion tokenizer artifacts (`ShakespeareTokenizer`, `StoryTokenizer`,
`G2CTokenizer`) live under `artifacts/tokenizers/` and follow a separate
naming scheme (no size, no stage); see the Artifact Contract section.

The durable non-model artifacts are:

- tokenizers;
- raw corpora;
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
  manifest.json
```

The tokenizer is referenced by name in `manifest.tokenizer_artifact` rather
than duplicated inside the model directory. This lets several models share one
tokenizer artifact without copying it.

External Hugging Face model artifacts, such as `BaseLM-base`, keep the HF-native
layout instead of `model.pt`:

```text
artifacts/models/BaseLM-base/
  config.json
  manifest.json
  hf_tokenizer/
```

Fine-tuned external artifacts add `hf_model/` next to `hf_tokenizer/`.

Tokenizer artifacts are saved once and shared across models that use them:

```text
artifacts/tokenizers/<tokenizer-name>/
  tokenizer.json
  ids.uint32      # small encoded inspection sample, not the full corpus
  manifest.json
```

A tokenizer artifact stores its trained vocab as the maximum usable size.
Downstream consumers (tokenized corpora, model checkpoints) record the slice
they actually use in their own manifests via the `vNNNN` name suffix and the
`vocab_size` / `effective_vocab_size` fields. Truncating a BPE tokenizer to its
first N merges yields a valid smaller tokenizer, so the larger artifact is a
superset of every slice taken from it. This is why the three numbers along a
chain need not match: `StoryTokenizer` may hold a trained vocab of 8192 while
`StoryLM-tinystories-full-v4096` and `StoryLM` both pin a working vocab of 4096.

Training and eval data should be saved as data artifacts rather than hidden in
notebook outputs:

```text
data/datasets/g2c-corpus-v1/
  manifest.json
  raw/
    fineweb-edu-dedup/
      train-00000.txt.gz
      val-00000.txt.gz
    cosmopedia-v2/
    tinystories/
    codesearchnet-python/

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
- final train/val losses (the golden calibration numbers for that run);
- notes on intended use;
- a `distribution` block when the artifact was fetched from a published
  release rather than trained locally (see below).

## Published Reference Checkpoints

Module 10's scale-up runs cost hours-to-overnight, and Module 12's scaling
comparison is inert without them. The reference checkpoints exist so Modules
12+ (and Part II direct entry) are never hostage to that wall-clock. Training
your own ladder remains Module 10's deliverable; the download is the honest
escape hatch, never the default path, and nothing in the course auto-fetches
it.

**What ships.** `StoryLM-1M-base`, `StoryLM-5M-base`, `StoryLM-30M-base`,
`TinyLLM-30M-base`, plus the tokenizer artifacts they reference by name
(`StoryTokenizer`, `G2CTokenizer`). Each asset is a tarball of the artifact
directory, unpacking directly into `artifacts/models/` or
`artifacts/tokenizers/`.

**Hosting.** GitHub release assets on the `checkpoints-v1` tag, decoupled from
course version tags so checkpoints can be re-published without cutting a
course release. Retrains that keep the same tokenizer/corpus generation
replace assets in place (URLs stay stable; checksums update); breaking changes
bump the tag. The release URLs live as `*_URL` variables in `checkpoints.sh`,
so the weekly dataset-urls canary covers them automatically.

**Provenance.** Published checkpoints must be trained with the reference
solutions at a recorded commit. `scripts/package_checkpoints.py` refuses to
package a manifest missing `git_commit`, `seed`, or the final train/val
losses, and injects a `distribution` block (release tag, asset name, packaging
timestamp) into the packaged copy of the manifest — that block is how
notebooks distinguish a downloaded reference artifact from a self-trained one.
The packager also verifies each model's tokenizer covers its vocab: a
tokenizer trained *larger* than the model is fine by design (consumers encode
via `encode_with_vocab_size(text, model.vocab_size)`, truncating BPE merges to
the model's prefix vocab), but a tokenizer smaller than the model's vocab is a
broken pair and fails packaging.

**Student contract.** `./checkpoints.sh` never overwrites an existing
artifact — a student's own overnight run is unclobberable — and verifies
sha256 checksums that `package_checkpoints.py --update-script` writes into the
fetch script at packaging time.

**Re-publish flow.**

```
.venv/bin/python scripts/package_checkpoints.py --update-script
gh release upload checkpoints-v1 data/cache/checkpoint-dist/*.tar.gz --clobber
```

## G2C Corpus v1

G2C Corpus v1 is the broader raw-text corpus for the TinyLLM track. The corpus
builder keeps subcorpora separate so later tokenizers and training scripts can
sample different mixtures without rebuilding the raw text.

The full target mix is:

- FineWeb-Edu-Dedup: ~5GB
- Cosmopedia v2: ~2.5GB
- TinyStories: ~1GB, stored as gzip-compressed 100MB-uncompressed text shards
- CodeSearchNet: ~1GB

The small target uses the same ratios at about 1GB total. CodeSearchNet defaults
to Python-only because the course code is Python and small models have limited
capacity. The builder accepts `--codesearchnet-js-ratio` for experiments that
shift part of the code quota to JavaScript while keeping the total code budget
fixed.

Build commands:

```bash
./datasets.sh              # full preload, including data/datasets/g2c-corpus-v1/
./datasets.sh --small      # preload with data/datasets/g2c-corpus-v1-small/
./datasets.sh --tiny       # only a 100MB TinyStories sample
./datasets.sh g2c-corpus-small
./datasets.sh g2c-corpus-full
./datasets.sh g2c-corpus-full --codesearchnet-js-ratio 0.2
```

The small corpus is stored under `data/datasets/g2c-corpus-v1-small/`; the full corpus is
stored under `data/datasets/g2c-corpus-v1/`. Both use compressed raw text shards. Quotas
are measured in uncompressed normalized UTF-8 bytes, not compressed file size.
Later tokenizer/training scripts should read `manifest.json` instead of assuming
a fixed file list.

Raw corpora are `data/` assets, not `artifacts/` outputs. Tokenizers, trained
models, checkpoints, eval reports, and post-training datasets produced from the
raw corpus should be saved under `artifacts/`.

## Module Map

### Module 10 - Milestone: TinyLLM

Module 10 is the first durable model-artifact boundary.

It should produce some subset of:

- `ShakespeareLM-1M-base`;
- `StoryTokenizer`;
- `StoryLM-5M-base`;
- `StoryLM-30M-base`;
- optional `TinyLLM-30M-base` if the broader corpus path is available.

Module 12 can add `StoryLM-1M-base` as the low-end TinyStories scaling point.

### Module 11 - Sampling and Decoding

Module 11 should not change weights. It should reuse saved base models and save
sampling traces that show how decoding changes behavior.

### Module 12 - Scaling Experiments

Module 12 should make the clean scaling comparison inside one corpus family:
`StoryLM-1M-base`, `StoryLM-5M-base`, and `StoryLM-30M-base` on TinyStories
with the StoryTokenizer. It may create the `StoryLM-1M-base` checkpoint if
missing, but should reuse the longer Module 10 runs for the 5M and 30M points. Cross-corpus
comparisons with ShakespeareLM or TinyLLM are useful context, not the headline
scaling curve.

### Module 13 - Instruction Tuning

Module 13 updates student-trained model weights.

Supported tracks:

- StoryLM track: story-writing instructions and story-editing prompts.
- TinyLLM track: assistant-style instructions, simple format following, and
  toy tool-call syntax.

It should produce:

- `StoryLM-<N>-SFT`; or
- `TinyLLM-<N>-SFT`; or
- `BaseLM-SFT` when the self-trained models are too weak to show SFT
  behavior clearly.

(`<N>` carries through whichever size the base model used; e.g.
`StoryLM-30M-SFT` from `StoryLM-30M-base`.)

ProdLM should not be the default training target for Module 13. Fine-tuning a
production model can be optional later, but the core lesson is weight updates on
the student's own small model (or, as a fallback, on BaseLM). In practice that
fallback is the common case: the notebook defaults to BaseLM because a 1M- or
5M-class model rarely shows the behavioral shift clearly, so switch to your own
artifact once it is 30M or larger.

### Module 14 - Preference Tuning

Module 14 updates student-trained model weights with preference data.

Supported tracks:

- StoryLM track: coherent vs incoherent story continuations, tone, repetition,
  continuity, and prompt adherence.
- TinyLLM track: concise vs rambling answers, valid vs malformed chat format,
  valid vs invalid toy tool calls, and task adherence.

It should produce:

- `StoryLM-<N>-DPO`; or
- `TinyLLM-<N>-DPO`; or
- `BaseLM-DPO` layered on top of `BaseLM-SFT` when the BaseLM track is in use.

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

A ProdLM backend may appear as a preview comparison, but the main
failure-analysis target should be the student's tiny model artifacts (or
BaseLM-SFT/BaseLM-DPO when those stand in for them).

### Modules 16-20 - Assistant Systems

Modules 16-20 pivot to ProdLM for usable assistant behavior.

ProdLM is not a trained artifact from this repo. It is a backend selection and
runtime configuration. These modules should save system artifacts instead:

- model backend config;
- prompts;
- RAG indexes;
- tool schemas;
- tool-call traces;
- agent transcripts;
- eval reports.

Modules 16-18 (inference, RAG, tools) keep the runtime harness backend-agnostic:
the same messages, tools, and evals can run against TinyLLM/StoryLM when useful,
but ProdLM is the default backend for actually usable assistant behavior.

Modules 19 and 20 (agent loops, capstone) default to **ProdLM**. Self-trained
models are exposed as a comparison backend via `MODEL_SELECTION` — the loader
prefers `-DPO`, then `-SFT`, then the base artifact — but they are not a viable
main backend, because tiny from-scratch models do not reliably follow the ReAct
or multi-turn assistant formats the loops depend on. Characterizing that failure
boundary is the point of the capstone's comparison exercise. The deterministic
cells in both notebooks use a `FakeBackend` for architecture testing.

## Pedagogical Rules

- Save artifacts when they are expensive to regenerate, reused later, needed
  for comparison, or emotionally meaningful as milestones.
- Do not save every notebook experiment.
- Keep StoryLM tasks story-domain unless the module is explicitly demonstrating
  failure outside the model's domain.
- Keep TinyLLM tasks assistant-shaped but honest about capability.
- Use ProdLM as a generic role, not as a fixed model name.
- Make larger local runs optional; never require cloud compute for the core
  path.
