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
- `g2c/notebook_extras/<topic>.py` — non-pedagogical notebook helpers (progress bars, matplotlib glue, run-orchestration wrappers) used by notebooks but not implemented by students
- `g2c/artifacts/models.py` — `save_model_artifact` / `load_model_artifact` implementing the durable model artifact convention from `docs/design/model-artifacts-and-tracks.md`. Use these rather than ad-hoc `torch.save` calls when persisting a trained model that downstream modules consume.
- `artifacts/models/<name>/` — saved model artifacts (`model.pt`, `config.json`, `manifest.json`); tokenizer is referenced by name in the manifest, not duplicated.
- `data/module*-*.ckpt` — rolling training checkpoints (model + optimizer + step + history). These are caches, not artifacts; safe to wipe to retrain.
- `notebooks/clean/NN-*.ipynb` — canonical pristine notebooks tied to module NN. Written exercises live here as `Question:` / `Answer:` string-literal code cells alongside the runnable cells; this is also where students write their answers (in `notebooks/solutions/`).
- `notebooks/solutions/NN-*.ipynb` — working or solved notebook copies; use `.venv/bin/python scripts/open_notebook.py NN` to create or resume, and `--fresh` to archive the existing copy before resetting from clean
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
3. Read the student's working notebook at `notebooks/solutions/NN-*.ipynb`. If no solutions copy exists yet, fall back to `notebooks/clean/NN-*.ipynb`. Do not edit the student's notebook unless explicitly asked.
4. Check conceptual correctness first, then math, then code or environment claims.
5. If the module has relevant tests or smoke checks, run them when they materially affect the review, and include the command result in the feedback.
6. Do not paste a full worked solution by default. Give the smallest correction or hint that would let the student repair the answer.
7. Each written exercise lives as `Question:` / `Answer:` string-literal cells in the notebook. Treat each `Question:` independently:
   - If `"Answer: "` is empty, treat the item as not submitted and skip it unless the user asks for a completeness check.
   - If the answer string contains a hint or help request (the student wrote text like "stuck — what's the chain rule again?"), tutor instead of grading: give the smallest useful hint, explanation, or next step.
   - If the answer is a real attempt, grade it. If it also contains a help request, answer the question first, then grade the attempt.
8. Do not treat blank answers as wrong. If you mention them at all, group them briefly as `not submitted` with no correctness judgment.
9. Report feedback for submitted answers with one of these statuses: `correct`, `mostly correct`, `partially correct`, or `needs revision`.
10. Do not paste a full worked solution in response to a hint request unless the student explicitly asks for the solution. Prefer progressive hints that leave the next reasoning step to the student.
11. If an answer is ambiguous, say what assumption you made and what the student should clarify.
12. If one or more submitted answers are `partially correct` or `needs revision`, gently offer 2-3 focused inline practice problems for the weakest concept.

## When Giving Additional Practice

Prefer inline chat practice over creating files. When the student asks for more problems, drills, practice, remediation, or another round on a weak area:

1. Read `docs/modules/NN-name.md`, `docs/rubrics/module-NN.md`, and any relevant previous answers (from `notebooks/solutions/NN-*.ipynb`) or practice feedback.
2. Generate 2-3 focused problems directly in chat, unless the user asks for a larger set.
3. Do not include solutions up front.
4. Tell the student they can answer one, some, or all of the problems.
5. Grade only the problems the student attempts.
6. Calibrate difficulty from the student's latest mistakes: start near the missed concept, then add one small variation per problem.
7. Offer another short round if it would help, but do not force it.
8. If the user directly asks for practice without first answering anything in the notebook, generate inline problems immediately.
9. Create practice files only if the user explicitly asks to save a drill set.

## When authoring a new module

When building scaffolding for any module that has a coding exercise, the goal is for the student to focus on the conceptual core, not on Python plumbing. Always provide:

- **The boilerplate, fully implemented.** Class skeletons, constructors, `__repr__`, and any convenience operators that aren't the point of the lesson. The student's attention should land on the math/CS being taught, not on dunder-method ergonomics or argument validation.
- **Scaffolds for the methods that ARE the point.** Empty bodies that `raise NotImplementedError`, with docstrings that name the contract — including, where helpful, the local rule (e.g., the gradient formula for an autodiff op, the recurrence for an attention computation). This gives the student the API surface and the math, but not the implementation.
- **A test suite that pins the contract.** Comprehensive enough that "all tests pass" is a strong signal of correctness. Tests should fail informatively against the empty scaffolding so the failing test names tell the student what to implement next.
- **A suggested implementation order.** A docstring at the top of the test file (or a checklist in the lesson page) describing which TODOs to tackle first. Each step should turn a coherent batch of tests green so progress is visible.
- **Embedded answer slots in the clean notebook.** For each written exercise, add a code cell of `"Question: ..."` / `"Answer: "` string literals immediately after the exercise's prompt or run cells. The closing cell of the notebook should remind the student that a coding agent can grade the notebook and that partial work is fine.
- **A rubric for written exercises.** Add `docs/rubrics/module-NN.md` with grading criteria. Its preamble should point at `notebooks/solutions/NN-*.ipynb` (with `notebooks/clean/NN-*.ipynb` as the fallback) as the source of student answers.
- **A workflow note at the start of exercises.** The first paragraph after every `## Exercises` heading should tell students to open the working notebook with `.venv/bin/python scripts/open_notebook.py NN`, write their answers in the `Question:` / `Answer:` cells, and ask a coding agent for hints or grading. State that partial submissions are fine because blank answers are skipped.

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

## Notebook style

Student-facing notebooks should foreground the conceptual flow, not the plumbing. When authoring or cleaning a notebook:

- **Configs live near consumption.** No global "Run Configuration" wall at the top. Corpus knobs go in the prep cell, model and trainer config dicts at the top of the cell that uses them, sample prompts inline with the sample cell. A small shared base is fine only if it materially cuts duplication.
- **No user-controlled run gates.** Don't add `run_X = True/False` toggles to skip long-running cells. If a student doesn't want to run a cell, they don't run it; Jupyter handles "stop execution." Environment checks (e.g. "TinyStories not downloaded → skip with friendly message") are different and should stay.
- **Extract notebook plumbing, not concepts.** IPython display helpers, matplotlib chart glue, and `Trainer` progress wrappers belong in `g2c/notebook_extras/<topic>.py`, not inline. Code that IS the experiment the student is meant to read (an LR sweep loop, an ablation set) stays inline. Code the lesson explicitly frames as transitional (e.g. "Module 11 will build the real generation utilities, here's a stand-in") also stays inline so the framing is visible.
- **Never put student-implemented code in `g2c/notebook_extras/`.** That directory is the escape hatch for non-pedagogical helpers; pedagogical deliverables go in `g2c/<topic>/`.

## What not to do

- Don't paper over a missing from-scratch implementation by reaching for a high-level library. If `g2c.attention.MultiHeadAttention` doesn't exist yet, build it; don't import `torch.nn.MultiheadAttention` as a substitute.
- Don't add cloud-dependent code paths (paid APIs, hosted training).
- Don't preemptively scale up. Tiny corpora, tiny models. The course's whole point is that the tiny version teaches the idea.
- Don't write speculative scaffolding for modules that haven't been started yet.
