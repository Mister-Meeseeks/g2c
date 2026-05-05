# AGENTS.md

Guidance for AI coding agents working in this repo.

## What this repo is

*From Gradients to ChatGPT* — a self-study course modeled after *From NAND to Tetris*, building a tiny LLM stack from scalar autodiff up through a working chat assistant. The repo serves dual purposes:

1. **Course material** — syllabus and per-module lessons (`docs/`)
2. **Work product** — the student's evolving implementation (`g2c/`)

The student is the repo author. Both roles are live.

## Hard constraints

- **Runs on an M-series MacBook.** No cloud GPUs, no paid compute. Code, datasets, and model sizes must stay within what an M1/M2/M3/M4 with 16–64GB unified memory can execute.
- **From-scratch through the architecture.** Weeks 1–11 must not import a high-level abstraction for the concept under study. Don't use `torch.nn.MultiheadAttention` inside the attention module — the point is to build it. Using PyTorch tensor primitives, autograd (after week 1), and standard optimizers is fine when the concept under study is something else. Weeks 12–15 keep working with the self-trained StoryLM/TinyLLM model tracks. Weeks 16–20 pivot to `ProdLLM`: a local pretrained instruct backend sized to the student's machine.
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
- `notebooks/clean/NN-*.ipynb` — canonical pristine notebooks tied to module NN
- `notebooks/solutions/NN-*.ipynb` — working or solved notebook copies; use `.venv/bin/python scripts/open_notebook.py NN` to create or resume, and `--fresh` to archive the existing copy before resetting from clean
- `answers/module-NN.md` — student-owned written answers for that module's exercises
- `docs/rubrics/module-NN.md` — course-owned grading rubric for written answers; use for review, not as a replacement for the student's work
- `docs/design/model-artifacts-and-tracks.md` — durable model artifact, hardware track, and Modules 10-20 backend plan
- `data/` — datasets; anything large is gitignored
- `tests/test_<topic>.py` — tests for each module's public API

## When working on a module

- Read `docs/modules/NN-name.md` first to understand intent.
- For Modules 10-20, also read `docs/design/model-artifacts-and-tracks.md` before changing model artifacts, saved outputs, SFT/DPO/eval tracks, or production-model backend assumptions.
- The deliverable goes in `g2c/<topic>/`.
- Add tests under `tests/test_<topic>.py`.
- Each module's public API should be minimal and stable — later modules will import it.

## When checking student answers

When the user asks to grade, check, review, or give feedback on answers for a module:

1. Read `docs/modules/NN-name.md` first to understand the exercises and teaching intent.
2. Read `docs/rubrics/module-NN.md` if it exists. Treat it as the grading contract.
3. Read `answers/module-NN.md`. Do not edit `answers/` files unless explicitly asked.
4. Check conceptual correctness first, then math, then code or environment claims.
5. If the module has relevant tests or smoke checks, run them when they materially affect the review, and include the command result in the feedback.
6. Do not paste a full worked solution by default. Give the smallest correction or hint that would let the student repair the answer.
7. Treat each exercise independently:
   - If `Help request / hint request` is non-empty and `Student answer` is blank, tutor instead of grading. Give the smallest useful hint, explanation, or next step.
   - If `Student answer` is non-empty, grade it. If `Help request / hint request` is also non-empty, answer the question first, then grade the submitted answer.
   - If both sections are blank, skip the item unless the user explicitly asks for a completeness check.
8. Do not treat blank answers as wrong. If you mention them at all, group them briefly as `not submitted` with no correctness judgment.
9. Report feedback for submitted answers with one of these statuses: `correct`, `mostly correct`, `partially correct`, or `needs revision`.
10. Do not paste a full worked solution in response to a hint request unless the student explicitly asks for the solution. Prefer progressive hints that leave the next reasoning step to the student.
11. If an answer is ambiguous, say what assumption you made and what the student should clarify.
12. If one or more submitted answers are `partially correct` or `needs revision`, gently offer 2-3 focused inline practice problems for the weakest concept.

## When Giving Additional Practice

Prefer inline chat practice over creating files. When the student asks for more problems, drills, practice, remediation, or another round on a weak area:

1. Read `docs/modules/NN-name.md`, `docs/rubrics/module-NN.md`, and any relevant previous answers or practice feedback.
2. Generate 2-3 focused problems directly in chat, unless the user asks for a larger set.
3. Do not include solutions up front.
4. Tell the student they can answer one, some, or all of the problems.
5. Grade only the problems the student attempts.
6. Calibrate difficulty from the student's latest mistakes: start near the missed concept, then add one small variation per problem.
7. Offer another short round if it would help, but do not force it.
8. If the user directly asks for practice without first using the answer file, generate inline problems immediately.
9. Create practice files only if the user explicitly asks to save a drill set.

## When authoring a new module

When building scaffolding for any module that has a coding exercise, the goal is for the student to focus on the conceptual core, not on Python plumbing. Always provide:

- **The boilerplate, fully implemented.** Class skeletons, constructors, `__repr__`, and any convenience operators that aren't the point of the lesson. The student's attention should land on the math/CS being taught, not on dunder-method ergonomics or argument validation.
- **Scaffolds for the methods that ARE the point.** Empty bodies that `raise NotImplementedError`, with docstrings that name the contract — including, where helpful, the local rule (e.g., the gradient formula for an autodiff op, the recurrence for an attention computation). This gives the student the API surface and the math, but not the implementation.
- **A test suite that pins the contract.** Comprehensive enough that "all tests pass" is a strong signal of correctness. Tests should fail informatively against the empty scaffolding so the failing test names tell the student what to implement next.
- **A suggested implementation order.** A docstring at the top of the test file (or a checklist in the lesson page) describing which TODOs to tackle first. Each step should turn a coherent batch of tests green so progress is visible.
- **An answer workspace and rubric for written exercises.** Add `answers/module-NN.md` with blank student-owned `Help request / hint request`, `Student answer`, and `Notes / uncertainty` slots, add `docs/rubrics/module-NN.md` with grading criteria, and point the lesson's scaffolding section at these files.
- **An answer link at the start of exercises.** The first paragraph after every `## Exercises` heading should link to `answers/module-NN.md`, say that students can ask for hints or submit answers for agent grading, and state that partial submissions are fine because blank answer sections are skipped.

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
