# AGENTS.md

Guidance for AI coding agents working in this repo.

## What this repo is

*From Gradients to ChatGPT* — a self-study course modeled after *From NAND to Tetris*, building a tiny LLM stack from scalar autodiff up through a working chat assistant. The repo serves dual purposes:

1. **Course material** — syllabus and per-module lessons (`docs/`)
2. **Work product** — the student's evolving implementation (`g2c/`)

The student is the repo author. Both roles are live.

## Hard constraints

- **Runs on an M-series MacBook.** No cloud GPUs, no paid compute. Code, datasets, and model sizes must stay within what an M1/M2/M3/M4 with 16–64GB unified memory can execute.
- **From-scratch through the architecture.** Weeks 1–11 must not import a high-level abstraction for the concept under study. Don't use `torch.nn.MultiheadAttention` inside the attention module — the point is to build it. Using PyTorch tensor primitives, autograd (after week 1), and standard optimizers is fine when the concept under study is something else. Weeks 12–15 keep working with the self-trained model. Weeks 16–20 pivot to a pretrained open model.
- **Pedagogy over performance.** Code should be legible. Optimize for "every internal piece is understandable" over "this runs fastest." Performance work is its own later concern.

## Stack

- **Python 3.11+**
- **PyTorch with MPS backend** as primary
- **MLX** for inference-heavy stages where Apple-native performance matters
- **Jupyter notebooks** for exploration and visualization (in `notebooks/`)
- **Ollama / llama.cpp** for running pretrained open models in the capstone

## Layout conventions

- `docs/modules/NN-name.md` — lesson + motivation + exercises + deliverable spec for module NN
- `g2c/<topic>/` — Python subpackage for that module's deliverable
- `notebooks/NN-*.ipynb` — exploratory notebooks tied to module NN
- `data/` — datasets; anything large is gitignored
- `tests/test_<topic>.py` — tests for each module's public API

## When working on a module

- Read `docs/modules/NN-name.md` first to understand intent.
- The deliverable goes in `g2c/<topic>/`.
- Add tests under `tests/test_<topic>.py`.
- Each module's public API should be minimal and stable — later modules will import it.

## What not to do

- Don't paper over a missing from-scratch implementation by reaching for a high-level library. If `g2c.attention.MultiHeadAttention` doesn't exist yet, build it; don't import `torch.nn.MultiheadAttention` as a substitute.
- Don't add cloud-dependent code paths (paid APIs, hosted training).
- Don't preemptively scale up. Tiny corpora, tiny models. The course's whole point is that the tiny version teaches the idea.
- Don't write speculative scaffolding for modules that haven't been started yet.
