# From Gradients to ChatGPT

![Cover illustration: a circus-themed map of the course. Numbered booths under a big top represent each of the twenty modules — pretraining, tokenization, embeddings, gradient descent, self-attention, multi-head attention, the transformer (a central tower of "Add & Norm / Feed Forward / Multi-Head Attention" floors), sampling, SFT, DPO, RAG, tools, the agent, the eval inspector, and an "inference booth" robot at the bottom. Banners read "A tiny LLM stack from first principles," "Data. Compute. Curiosity. That's all you need," and "Built step by step under the big top."](docs/CourseCover.png)

A self-study course on the core building blocks of LLMs, modeled after *From NAND to Tetris*.

This repo contains both the instructional material and the student work product. The codebase grows layer-by-layer from scalar autodiff up through a working chat assistant — each module's deliverable becomes a building block for the next.

The hard constraint: **all tasks must be runnable on an M-series MacBook**, without access to expensive cloud environments or GPUs.

## Contents

Start with the [syllabus](docs/syllabus.md) for the full 20-week arc plus the fast Module 00 review. Modules 03B and 09B are draft insertions being tested before a possible full renumber after the full course is drafted. Lessons published so far:

| #  | Module                                                          | Phase                |
| -- | --------------------------------------------------------------- | -------------------- |
| 00 | [Prerequisite review](docs/modules/00-prerequisite-review.md)   | 0 — Review           |
| 01 | [Scalar autodiff](docs/modules/01-autodiff.md)                  | I — Foundations      |
| 02 | [Tensors and matmul](docs/modules/02-tensors.md)                | I — Foundations      |
| 03 | [A first neural network](docs/modules/03-nn.md)                 | I — Foundations      |
| 03B | [Training](docs/modules/03b-training.md) | I — Foundations |
| 04 | [Tokenization](docs/modules/04-tokenizer.md)                    | II — Language        |
| 05 | [Embeddings and positions](docs/modules/05-embeddings.md)       | II — Language        |
| 06 | [Next-token prediction](docs/modules/06-language-models.md)     | II — Language        |
| 07 | [Self-attention](docs/modules/07-attention.md)                  | III — The transformer |
| 08 | [Multi-head attention](docs/modules/08-multi-head-attention.md) | III — The transformer |
| 09 | [The transformer block](docs/modules/09-transformer-block.md)   | III — The transformer |
| 09B | [Pretraining](docs/modules/09b-pretraining.md)              | III — The transformer |
| 10 | [Milestone: TinyLLM](docs/modules/10-tinyllm.md) | III — The transformer |
| 11 | [Sampling and decoding](docs/modules/11-sampling.md)            | IV — Behavior shaping |
| 12 | [Scaling experiments](docs/modules/12-scaling.md)               | IV — Behavior shaping |
| 13 | [Instruction tuning (SFT)](docs/modules/13-sft.md)              | IV — Behavior shaping |
| 14 | [Preference tuning (DPO)](docs/modules/14-dpo.md)               | IV — Behavior shaping |
| 15 | [Hallucination and evaluation](docs/modules/15-evaluation.md)   | IV — Behavior shaping |
| 16 | [Local pretrained models and inference](docs/modules/16-inference.md) | V — Assistant systems |
| 17 | [Retrieval-augmented generation](docs/modules/17-rag.md)         | V — Assistant systems |
| 18 | [Tool use](docs/modules/18-tools.md)                            | V — Assistant systems |
| 19 | [Agent loops](docs/modules/19-agent.md)                         | V — Assistant systems |
| 20 | [Capstone: a tiny ChatGPT](docs/modules/20-capstone.md)         | V — Assistant systems |

## Foundational choices

- **Pacing unit.** Each module is one "week" of effort at the level of a rigorous elite-college course. Calendar pace is whatever it ends up being.
- **Framework.** PyTorch with the MPS backend (Apple Silicon GPU) is the primary framework throughout. MLX is used selectively for inference-heavy stages where Apple-native performance matters. Rationale: PyTorch has the deepest ecosystem and the broadest reference material; MLX is faster on M-series but is an optimization tool, not a pedagogical one.
- **From-scratch boundary.** Weeks 1–11 (autodiff through pretraining + sampling) build the concept under study from scratch — no `torch.nn.MultiheadAttention` in the attention module, etc. Using PyTorch tensor primitives and autograd as substrate is fine once those layers are themselves established. Weeks 12–15 (scaling, SFT, DPO, eval) keep working with the model you trained yourself; quality is visibly toy and that's the point.
- **Capstone pivot.** Weeks 16–20 (inference, RAG, tools, agents, capstone) pivot to a local pretrained instruct model sized to your machine. The course calls this role `ProdLM`: the point is the backend contract, not one fixed model name.
- **Repo style.** One monorepo. The `g2c/` Python package grows over the course; each module's submodule is consumed by later modules.

## Hardware tracks and saved artifacts

The course is designed for M-series Macs, but not every student needs the same
download size, model size, or training time. The track system is artifact-based:
prepare as much data as you want, and notebooks should load the strongest local
artifact they can find.

- **Tiny:** `./datasets.sh --tiny` prepares a 100MB TinyStories sample path.
- **Standard:** `./datasets.sh --small` prepares full TinyStories plus the small
  G2C corpus. This is the recommended local course experience.
- **Full:** `./datasets.sh` prepares the full G2C corpus path for stretch runs.
- **BaseLM:** Modules 13-16 can use a small external pretrained base model when
  your self-trained model is too weak.
- **ProdLM:** Modules 16-20 use a local pretrained instruct model sized to your
  machine. Configure it with `./prodlm.sh --model-id <ollama-tag>`; this also
  pulls `nomic-embed-text` for RAG.

See [Course Tracks and Artifacts](docs/tracks.md) for download/storage/time
estimates and artifact names. The deeper maintainer design note lives at
[Model Artifacts and Course Tracks](docs/design/model-artifacts-and-tracks.md).

## Repository layout

```
docs/
  CourseCover.png       # cover illustration (used in README)
  syllabus.md           # the detailed syllabus
  modules/
    00-prerequisite-review.md
    01-autodiff.md      # lesson, motivation, exercises, deliverable spec
    01-autodiff/        # assets for module 01 (diagrams, supplementary files)
    02-tensors.md
    02-tensors/
    ...
g2c/                    # the work-product Python package — grows over the course
  autodiff/             # module 1
  tensors/              # module 2
  nn/                   # module 3
  tokenizer/            # module 4
  ...
notebooks/
  clean/                # canonical pristine notebooks
  solutions/            # working notebook copies and solved notebooks
data/                   # corpora and datasets (large files gitignored)
prompts/                # course brainstorms, prompt drafts
tests/                  # tests across modules
README.md
AGENTS.md
```

The split is intentional: `docs/` is what a student reads, `g2c/` is what the student builds. They evolve together.

## Getting started

Prerequisites (system-level — checked, not installed, by `setup.sh`):

- **Python 3.11+** — `brew install python@3.11` (or via pyenv)
- **uv** — `brew install uv` (or `curl -LsSf https://astral.sh/uv/install.sh | sh`)

Then bootstrap the project environment:

```bash
./setup.sh
```

The script is idempotent. It creates a project-local venv at `./.venv`, installs `g2c` (editable) plus dev dependencies, prepares the small TinyShakespeare corpus, and runs a smoke test that verifies PyTorch's MPS backend works on your machine. Re-run it any time to re-verify.

To bootstrap everything in one shot — venv + all optional datasets, BaseLM, and ProdLM — run `./setup.sh --full`. It chains `./datasets.sh`, `./baselm.sh`, and `./prodlm.sh` after the normal setup. See `./setup.sh --help` for details.

The first run may download `data/tinyshakespeare.txt` for language-model training. Larger optional datasets live behind `datasets.sh` so the normal setup stays fast. You can wait until a module asks for one, or preload all optional course data up front:

```bash
./datasets.sh --tiny       # Tiny track: 100MB TinyStories + tokenizer/tokenized artifact
./datasets.sh --small      # Standard track: GloVe, TinyStories, small G2C corpus + artifacts
./datasets.sh              # Full track: GloVe, TinyStories, full G2C corpus + artifacts
./datasets.sh glove        # Module 05 pretrained vectors (~822MB download)
./datasets.sh tinystories  # Module 10 scale-up corpus as compressed 100MB text shards
./datasets.sh all          # same as ./datasets.sh
```

Modules 13-16 can also use a small external pretrained base model when your
self-trained model is too weak to make SFT/DPO behavior visible:

```bash
./baselm.sh                # registers the default BaseLM artifact
```

Modules 16-20 use a capable local instruction model through the ProdLM backend:

```bash
./prodlm.sh --model-id llama3.2:3b
```

That command also pulls the default RAG embedding model, `nomic-embed-text`.

The broader TinyLLM track uses **G2C Corpus v1**, a generated local raw-text
corpus built from streamed upstream datasets:

```bash
./datasets.sh g2c-corpus-small  # ~1GB raw text under data/g2c-corpus-v1-small/
./datasets.sh g2c-corpus-full   # ~9.5GB raw text under data/g2c-corpus-v1/
./datasets.sh g2c-corpus-full --codesearchnet-js-ratio 0.2
```

By default, the CodeSearchNet slice is Python-only. The JavaScript ratio flag
moves that fraction of the code quota to JavaScript while keeping the total code
budget fixed.

Raw corpora and downloaded datasets live under `data/`. Durable outputs produced
from them, such as trained tokenizers, model checkpoints, eval reports, and
post-training datasets, belong under `artifacts/`.

The download targets are idempotent: later runs skip files that already exist
and resume partial downloads when possible. The G2C corpus builder refuses to
overwrite an existing non-empty output directory unless you pass `--force`.
`datasets.sh` also builds the matching tokenizer and disk-backed tokenized
corpus artifacts, so later notebooks do not need to repeat those expensive
steps in the kernel.

After setup:

```bash
source .venv/bin/activate
python -m pytest                # full suite; many tests intentionally fail until implemented
python scripts/smoke_test.py    # re-run env health check
python scripts/artifact_status.py # inspect local datasets, tokenizers, and model artifacts
./sysprobe.sh                   # probe what your machine can handle (training sizes,
                                # inference throughput, ProdLM fit before downloading)
```

For notebook exercises, open the working copy through the launcher:

```bash
.venv/bin/python scripts/open_notebook.py 01           # create/resume notebooks/solutions/01-*.ipynb
.venv/bin/python scripts/open_notebook.py 01 --fresh   # archive the old working copy, then reset from notebooks/clean/
```

Or, if you'd rather not think about per-module setup, use the one-stop wrapper:

```bash
./notebook.sh 01           # runs setup.sh, runs the module's tests, then opens the notebook
./notebook.sh 13 --fresh   # also runs ./baselm.sh before opening (other modules pull in datasets.sh / prodlm.sh as needed)
```

See `./notebook.sh --help` for the full module → extras mapping.

Each module's notebook holds both the runnable cells and the written exercises. Written prompts appear inline as `"Question: ..."` / `"Answer: "` string-literal code cells; you fill in the answer string. When you're ready, ask a coding agent to grade the notebook — partial work is fine, blank answers are skipped.

Begin with `docs/syllabus.md`, then `docs/modules/00-prerequisite-review.md`, then `docs/modules/01-autodiff.md`.

## Pristine scaffolds and worked implementations

The repo is structured for two audiences at once: anyone working through the course, and the author maintaining a worked reference. There is one canonical branch, `main`. Pedagogical functions under `g2c/<topic>/` are left as `# TODO` + `raise NotImplementedError` scaffolds — that's the experience a course-taker sees on clone.

Canonical worked implementations live alongside the scaffolds in `g2c/solutions/`. Each file there mirrors the path of the file it implements: methods of class `Foo` in `g2c/<topic>/<file>.py` are held by `_FooImpl` inside `g2c/solutions/<topic>/<file>.py`, and module-level functions are mirrored at the module level.

By default, notebooks launch against scaffolds — the student experience. To launch with implementations live, pass `--solutions` to the launcher (or, if you're invoking pytest directly, set the env var):

```bash
./notebook.sh 01 --solutions        # launch module 01 with solutions applied
G2C_APPLY_SOLUTIONS=1 pytest        # full pytest suite with solutions applied
```

Under the hood, `g2c/__init__.py` checks `G2C_APPLY_SOLUTIONS` at import time and, if set, calls `g2c.solutions.apply()` once. That rebinds every mirror impl onto its scaffold target, so `g2c.autodiff.value.Value.__add__` and friends become live without changing any file a student reads.

The invariant is enforced as a parametrized pytest at `tests/test_scaffold_invariant.py`: if a worked implementation ever leaks into `g2c/<topic>/`, that test fails by qualified function name (e.g. `g2c.autodiff.value.Value.__add__`), so the regression is immediately visible.

If you fork the course to do your own work, branch off `main` and use whatever name you like (`student/<name>` is a reasonable convention).
