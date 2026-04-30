# From Gradients to ChatGPT

![Cover illustration: a circus-themed map of the course. Numbered booths under a big top represent each of the twenty modules — pretraining, tokenization, embeddings, gradient descent, self-attention, multi-head attention, the transformer (a central tower of "Add & Norm / Feed Forward / Multi-Head Attention" floors), sampling, SFT, DPO, RAG, tools, the agent, the eval inspector, and an "inference booth" robot at the bottom. Banners read "A tiny LLM stack from first principles," "Data. Compute. Curiosity. That's all you need," and "Built step by step under the big top."](docs/CourseCover.png)

A self-study course on the core building blocks of LLMs, modeled after *From NAND to Tetris*.

This repo contains both the instructional material and the student work product. The codebase grows layer-by-layer from scalar autodiff up through a working chat assistant — each module's deliverable becomes a building block for the next.

The hard constraint: **all tasks must be runnable on an M-series MacBook**, without access to expensive cloud environments or GPUs.

(The repository directory is historically named `scalarToLLM`; the formal course title is *From Gradients to ChatGPT*.)

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
notebooks/              # exploratory work, runs, attention visualizations
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

The script is idempotent. It creates a project-local venv at `./.venv`, installs `g2c` (editable) plus dev dependencies, and runs a smoke test that verifies PyTorch's MPS backend works on your machine. Re-run it any time to re-verify.

After setup:

```bash
source .venv/bin/activate
pytest                          # run tests
python scripts/smoke_test.py    # re-run env health check
```

Begin with `docs/syllabus.md`, then `docs/modules/01-autodiff.md`.

## Branching model

The repo is structured for two audiences at once: anyone working through the course, and the author maintaining a worked-out reference. Two long-running branches:

- **`main`** — pristine course material. Lesson pages, scaffolded `Value` classes with `# TODO` markers, tests that fail until the student fills things in. Clone this branch to take the course.
- **`solutions`** — the author's working branch with everything filled in. Doubles as a reference answer key.

Updates flow `main → solutions` only. When course material improves (a clearer lesson, a better scaffold), commits land on `main`, then `git merge main` into `solutions` brings the improvement forward without leaking solutions back to `main`. If you fork the course to do your own work, branch off `main` and use whatever name you like (`student/<name>` is a reasonable convention).
