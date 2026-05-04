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
| 10 | [Milestone: Your First LLM](docs/modules/10-your-first-llm.md) | III — The transformer |
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
- **Capstone pivot.** Weeks 16–20 (inference, RAG, tools, agents, capstone) pivot to a small pretrained open model (e.g. Llama 3 8B, Qwen 2.5) via Ollama or MLX, so the resulting assistant is actually usable.
- **Repo style.** One monorepo. The `g2c/` Python package grows over the course; each module's submodule is consumed by later modules.

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

The first run may download `data/tinyshakespeare.txt` for language-model training. Larger optional datasets live behind `datasets.sh` so the normal setup stays fast. You can wait until a module asks for one, or preload all optional course data up front:

```bash
./datasets.sh              # preload all optional course datasets
./datasets.sh glove        # Module 05 pretrained vectors (~822MB download)
./datasets.sh tinystories  # Module 10 scale-up corpus (~1.94GB text)
./datasets.sh all          # both large datasets
```

The script is idempotent: later runs skip files that already exist and resume partial downloads when possible.

After setup:

```bash
source .venv/bin/activate
python scripts/test_clean.py    # tests that should pass on the pristine scaffold
python -m pytest                # full suite; many tests intentionally fail until implemented
python scripts/smoke_test.py    # re-run env health check
```

For notebook exercises, open the working copy through the launcher:

```bash
.venv/bin/python scripts/open_notebook.py 01           # create/resume notebooks/solutions/01-*.ipynb
.venv/bin/python scripts/open_notebook.py 01 --fresh   # archive the old working copy, then reset from notebooks/clean/
```

Begin with `docs/syllabus.md`, then `docs/modules/00-prerequisite-review.md`, then `docs/modules/01-autodiff.md`.

## Branching model

The repo is structured for two audiences at once: anyone working through the course, and the author maintaining a worked-out reference. Two long-running branches:

- **`main`** — pristine course material. Lesson pages, scaffolded `Value` classes with `# TODO` markers, tests that fail until the student fills things in. Clone this branch to take the course.
- **`solutions`** — the author's working branch with everything filled in. Doubles as a reference answer key.

Updates flow `main → solutions` only. When course material improves (a clearer lesson, a better scaffold), commits land on `main`, then `git merge main` into `solutions` brings the improvement forward without leaking solutions back to `main`. If you fork the course to do your own work, branch off `main` and use whatever name you like (`student/<name>` is a reasonable convention).
