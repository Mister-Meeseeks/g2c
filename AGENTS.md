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
- `docs/modules/NN-name/` — assets for that module (images, diagrams, supplementary files). Reference from the lesson with relative paths, e.g. `![](NN-name/summary.png)`.
- `g2c/<topic>/` — Python subpackage for that module's deliverable
- `notebooks/NN-*.ipynb` — exploratory notebooks tied to module NN
- `data/` — datasets; anything large is gitignored
- `tests/test_<topic>.py` — tests for each module's public API

## When working on a module

- Read `docs/modules/NN-name.md` first to understand intent.
- The deliverable goes in `g2c/<topic>/`.
- Add tests under `tests/test_<topic>.py`.
- Each module's public API should be minimal and stable — later modules will import it.

## When authoring a new module

When building scaffolding for any module that has a coding exercise, the goal is for the student to focus on the conceptual core, not on Python plumbing. Always provide:

- **The boilerplate, fully implemented.** Class skeletons, constructors, `__repr__`, and any convenience operators that aren't the point of the lesson. The student's attention should land on the math/CS being taught, not on dunder-method ergonomics or argument validation.
- **Scaffolds for the methods that ARE the point.** Empty bodies that `raise NotImplementedError`, with docstrings that name the contract — including, where helpful, the local rule (e.g., the gradient formula for an autodiff op, the recurrence for an attention computation). This gives the student the API surface and the math, but not the implementation.
- **A test suite that pins the contract.** Comprehensive enough that "all tests pass" is a strong signal of correctness. Tests should fail informatively against the empty scaffolding so the failing test names tell the student what to implement next.
- **A suggested implementation order.** A docstring at the top of the test file (or a checklist in the lesson page) describing which TODOs to tackle first. Each step should turn a coherent batch of tests green so progress is visible.

The lesson page (`docs/modules/NN-name.md`) should have a dedicated **Scaffolding and how to run the tests** section pointing the student at the `# TODO` markers and the relevant `pytest` invocations.

The principle: a student should be able to type `pytest -x`, read the failing test name, and use it as their next directive — with no software-engineering overhead between them and the concept under study.

### Lesson page structure

Use this section ordering as the canonical template (established in Modules 01 and 02):

```
# Module NN — <topic>
> Question this module answers
[hero image + italic caption]

## Prerequisites
  ### Math
  ### Computer science
  ### Programming
  ### What you can skip

## Why we start here
## The big idea
## Concepts to internalize
## Scaffolding and how to run the tests
## What you'll build
## Exercises
## Pitfalls to expect
## Reading
## Deliverable checklist
## M-series notes
```

The order is stable; not every module needs every section. Skip what doesn't apply (e.g., M-series notes for a pure-CPU module). When skipping, drop the section heading entirely rather than leaving it empty.

### Image assets and captions

- **Filename convention:** `ModuleNN-<Descriptor>.png` inside `docs/modules/NN-name/`. The headline summary image is `ModuleNN-Hero.png`; specific diagrams use descriptive PascalCase names (`Module02-MatMul.png`, `Module02-Ladder.png`).
- **Reference from the lesson** with relative paths: `![alt text](NN-name/ModuleNN-Foo.png)`. Always include real alt text — it's the fallback when the image fails to render and matters for accessibility.
- **Every image gets a caption** underneath in *italics*. The caption explains both what the image shows AND why it matters here, ideally tying back to a specific exercise, concept, or upcoming module. Captions are signal, not decoration.
- **Hero image placement:** immediately after the question pull-quote, before the Prerequisites section.

### Test file conventions

The top of each test file should have a docstring with a numbered "Suggested order to implement & turn green" — mapping each implementation step to the tests it unblocks. The student should be able to read this once and know exactly where to start. Construction / repr / boilerplate tests pass from the start (since the boilerplate is implemented), serving as a sanity check on the test file itself; the rest fail with `NotImplementedError` until the student implements.

### Visual aids in lesson pages

Use small ASCII diagrams to crack genuinely dense conceptual sections — particularly anything involving graph topology, shape arithmetic, alignment rules, or memory layout — where prose alone makes the structural relationship hard to see. The bar is "would a reader struggle to picture this without it?" Don't add diagrams for visual flair or because diagrams seem like a nice idea. Roughly a handful per module is the ceiling; some modules may have zero, which is fine. Reserve image assets in `docs/modules/NN-name/` for content that genuinely needs full graphics; for everything else, ASCII inside a fenced code block reads cleanly in every markdown viewer the student is likely to use.

## What not to do

- Don't paper over a missing from-scratch implementation by reaching for a high-level library. If `g2c.attention.MultiHeadAttention` doesn't exist yet, build it; don't import `torch.nn.MultiheadAttention` as a substitute.
- Don't add cloud-dependent code paths (paid APIs, hosted training).
- Don't preemptively scale up. Tiny corpora, tiny models. The course's whole point is that the tiny version teaches the idea.
- Don't write speculative scaffolding for modules that haven't been started yet.
