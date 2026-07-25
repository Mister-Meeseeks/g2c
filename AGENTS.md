# AGENTS.md

Guidance for AI coding agents working in this repo.

## What this repo is

*From Gradients to ChatGPT* — a self-study course modeled after *From NAND to Tetris*, building a tiny LLM stack from scalar autodiff up through a working chat assistant. The repo serves dual purposes:

1. **Course material** — syllabus and per-module lessons (`docs/`)
2. **Work product** — the student's evolving implementation (`g2c/`)

Users of the repo may either be students working on the course, or project maintainers developing or updating the course.  Unit indicated otherwise, assume you're interacting with a course student.

## Course structure

The course is presented in **two parts**, split at the Module 11/12 boundary. The parts are a presentation layer only — module numbers are load-bearing in `notebook.sh`, `data/work/moduleNN/`, test names, and doc cross-links, so **never renumber modules**.

| Part | Group | Modules |
| ---- | ----- | ------- |
| I — From gradients to a language model | Prerequisite review | 00 |
| I | Foundations | 01, 02, 03, 03B |
| I | Language | 04, 05, 06 |
| I | The transformer | 07, 08, 09, 09B, 10, 11 |
| II — From a language model to ChatGPT | Behavior shaping | 12, 13, 14, 15 |
| II | Assistant systems | 16, 17, 18, 19, 20 |

Part I ends at Module 11 rather than Module 10 because a model without decoding loops on greedy sampling; Module 11 is what makes the artifact something a student can show someone. Part I is framed as a genuine finish line.

Two lesson pages carry **deliberate exceptions to the lesson-page template** below, both marking the part boundary. Don't "fix" them back:

- `11-sampling.md` ends with a `## You've finished Part I` section *after* `## Deliverable checklist` — the last thing on the page, where a reader lands when the module is done.
- `12-scaling.md` has an `## Entering at Part II` section between the intro and `## Before you start`, documenting direct entry.

Part II is close to self-contained: it imports only `g2c/nn/`, `g2c/training/` (Modules 03/03B) and `g2c/sampling/` (Module 11), and its BaseLM/ProdLM paths don't need a student-trained model. Module 12 documents direct entry via `G2C_APPLY_SOLUTIONS=01-11`. If you change what Part II imports from Part I, update that section.

The groups are descriptive labels, not a numbered taxonomy — there is no "Phase I/II/III" scheme any more. **The part/group tables in `README.md`, `docs/index.md`, and `docs/syllabus.md` must stay identical**; the previous phase names silently drifted apart across those three files, which is how they became vestigial.

## Hard constraints

- **Runs on an M-series MacBook.** No cloud GPUs, no paid compute. Code, datasets, and model sizes must stay within what an M1/M2/M3/M4 with 16–64GB unified memory can execute.
- **From-scratch through the architecture.** Weeks 1–11 must not import a high-level abstraction for the concept under study. Don't use `torch.nn.MultiheadAttention` inside the attention module — the point is to build it. Using PyTorch tensor primitives, autograd (after week 1), and standard optimizers is fine when the concept under study is something else. Weeks 12–15 pivot to using the weights from a larger pretrained base model `BaseLM` . Weeks 16–20 pivot to `ProdLLM`: a local pretrained instruct Ollama backend sized to the student's machine.
- **Pedagogy over performance.** Code should be legible. Optimize for "every internal piece is understandable" over "this runs fastest." Performance work is its own later concern.

## Stack

- **Python 3.11+**
- **PyTorch with MPS backend** as primary
- **MLX** for inference-heavy stages where Apple-native performance matters
- **Jupyter notebooks** for exploration and visualization (in `notebooks/`)
- **Ollama / llama.cpp** for running pretrained open models in the capstone

Dependencies and build config live in `pyproject.toml`.

## Layout conventions

- `docs/modules/NN-name.md` — lesson + motivation + exercises + deliverable spec for module NN
- `docs/modules/NN-name/` — assets for that module (images, diagrams, supplementary files). Reference from the lesson with relative paths, e.g. `![](NN-name/summary.png)`.
- `g2c/<topic>/` — Python subpackage for that module's deliverable
- `g2c/notebook_extras/<topic>.py` — non-pedagogical notebook helpers (progress bars, matplotlib glue, run-orchestration wrappers) used by notebooks but not implemented by students
- `g2c/artifacts/models.py` — `save_model_artifact` / `load_model_artifact` implementing the durable model artifact convention from `docs/design/model-artifacts-and-tracks.md`. Use these rather than ad-hoc `torch.save` calls when persisting a trained model that downstream modules consume.
- `artifacts/models/<name>/` — saved model artifacts (`model.pt`, `config.json`, `manifest.json`); tokenizer is referenced by name in the manifest, not duplicated.
- `data/work/moduleNN/` — module-specific working files (rolling training checkpoints, hand-authored SFT/DPO datasets, sandbox directories for the tool/agent/capstone exercises). These are caches, not artifacts; safe to wipe to retrain.
- `notebooks/clean/NN-*.ipynb` — canonical pristine notebooks tied to module NN. Written exercises live here as `Question:` / `Answer:` string-literal code cells alongside the runnable cells; this is also where students write their answers (in `notebooks/solutions/`).
- `notebooks/solutions/NN-*.ipynb` — working or solved notebook copies; use `./notebook.sh NN` to create or resume, and `./notebook.sh NN --fresh` to archive the existing copy before resetting from clean
- `docs/rubrics/module-NN.md` — course-owned grading rubric for written answers; use for review, not as a replacement for the student's work
- `docs/design/model-artifacts-and-tracks.md` — durable model artifact, hardware track, and Modules 10-20 backend plan
- `mkdocs.yml` + `.github/workflows/docs.yml` — lesson pages build to <https://mister-meeseeks.github.io/g2c/> on every push to `main`. Edits under `docs/` are publicly visible within a few minutes; preview locally with `pip install -r docs/requirements.txt && mkdocs serve`.
- `.github/workflows/bootstrap.yml` — runs `./setup.sh` on a clean checkout and asserts it produces the TinyShakespeare corpus, the `ShakespeareTokenizer` artifact, and the `.venv/.g2c-setup-complete` sentinel Module 00 checks. `tests.yml` installs deps directly and so never exercises the setup path; this job is what proves the README quickstart still works for a new student.
- `.github/workflows/dataset-urls.yml` + `scripts/check_dataset_urls.py` — weekly link-rot check for the corpora `setup.sh`/`datasets.sh` download. The checker extracts URLs from those scripts (any `*_URL`/`*_url` shell variable) rather than restating them, so it cannot drift from what is actually downloaded. **When adding a new download, assign it to a `*_URL` variable** and it is covered automatically. Run locally with `python3 scripts/check_dataset_urls.py` (stdlib only, no venv needed).
- `data/` — local, mostly-gitignored working tree:
  - `data/datasets/` — raw corpora and benchmark datasets (TinyShakespeare, TinyStories, G2C Corpus v1, MNIST).
  - `data/embeddings/` — pretrained embedding artifacts (GloVe).
  - `data/cache/` — reproducible caches (safe to wipe; rebuilt by setup/build scripts). Includes the HF hub cache `data/cache/baselm/` backing the `BaseLM` artifact's weights and tokenized-corpus caches like `data/cache/token-corpus/StoryLM-tinystories-full-v4096/`.
  - `data/work/moduleNN/` — module-specific working files (see entry above).
- `tests/test_<topic>.py` — tests for each module's public API
- `.github/ISSUE_TEMPLATE/` — issue forms for students. `01-stuck.yml` is the important one: it captures **which module** someone stalled on as a structured dropdown, plus how long they'd been working. The repo has no telemetry by design, so these reports are the only signal about where the course loses people. Keep the module dropdown in sync when modules change.
- `CONTRIBUTING.md` — student-facing contribution guide. Leads with "tell us where you got stuck" rather than with patches, and warns that a red `pytest` on a fresh clone is expected.
- `scripts/module_word_counts.py` — lesson word counts by section; `--reading-time` converts them to per-module reading estimates. Reading time is *derived on demand*, never pasted into a doc as a static table — that is how the old phase names rotted.

**Do not publish per-module effort-hour estimates.** The measurable signals (scaffold count, exercise count, word count) track volume, not difficulty, and they invert on the hardest modules — Module 07 has two scaffolded functions and is among the toughest weeks in the course. The syllabus says plainly that this number is deliberately absent and explains why; replace it only with real data from stuck reports, not with a derived guess.

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
- **A workflow note at the start of exercises.** The first paragraph after every `## Exercises` heading should tell students to open the working notebook with `./notebook.sh NN` (or `./notebook.sh NN --fresh` to reset from the clean scaffold), write their answers in the `Question:` / `Answer:` cells, and ask a coding agent for hints or grading. State that partial submissions are fine because blank answers are skipped.

The lesson page (`docs/modules/NN-name.md`) should have a dedicated **Scaffolding and how to run the tests** section pointing the student at the `# TODO` markers and the relevant `pytest` invocations.

The principle: a student should be able to type `pytest -x`, read the failing test name, and use it as their next directive — with no software-engineering overhead between them and the concept under study.

### Lesson page structure

Every lesson page is divided into five top-level sections, separated by `---` horizontal rules. The five sections, in order:

1. **Intro** — title, question pull-quote, hero image, and a short non-italic orientation paragraph.
2. **Before you start** — a small bulleted list of prerequisites the student should resolve before opening the notebook.
3. **Lecture notes** — `Where this fits in`, `The big idea`, `Concepts to internalize`.
4. **Homework** — `What you'll build`, `How to run the tests`, `Exercises`, `Pitfalls to expect`, `M-series notes`. (Pitfalls and M-series notes belong here because they are about doing the assignment, not about background.)
5. **Closing notes** — `Reading`, `Deliverable checklist`.

The full template, with the four divider positions:

```markdown
# Module NN — <topic>
> **Question this module answers:** *<one-line question>*

![<one-sentence alt text>](<NN-name>/Module<NN>-Hero.png)

<short non-italic orientation paragraph — intro text, not a caption.>

---
## Before you start

* *Review* [[<previous module or primer>]] for <what>
* *Finish* <prior g2c deliverable or artifact> if not already done
* *Run* `<setup or dataset script>`
* <short setup directive — e.g. "Set up your editor for Python">

---
## Where this fits in
## The big idea
## Concepts to internalize

---
## What you'll build
## How to run the tests
## Exercises
## Pitfalls to expect
## M-series notes

---
## Reading
## Deliverable checklist
```

The order is stable; not every module needs every section. Skip what doesn't apply (e.g., M-series notes for a pure-CPU module). When skipping, drop the heading entirely rather than leaving it empty. Modules 01–03 are the reference implementations.

**Section conventions:**

- **Before you start.** Keep it short. Don't enumerate every dependency — only the actions the student should take before opening the notebook. Use a small vocabulary of bullet leads:
  - `*Review*` — a previous module, primer, or short concept page
  - `*Finish*` — a prior `g2c` deliverable or saved artifact this module needs
  - `*Run*` — a setup or data-prep script
  - or a short imperative ("Set up your editor for Python")
- **Where this fits in.** Conversational framing of how this module connects to the rest of the course — what came before, what builds on it. Not a formal "why we start here" justification.
- **What you'll build.** The orientation a student needs to start the work: what package, the public API surface, and an end-to-end usage sketch. It is not a maintainer-level walkthrough of every scaffolded file. The focus is the student doing the pedagogical parts of the module, not maintaining the scaffold.
- **How to run the tests.** Open the block with `source .venv/bin/activate`, then list the relevant `pytest` commands (the bare `pytest` form, not `python -m pytest`), including the initial pass/fail count. Skip the implementation-order checklist unless the scaffolds have non-obvious dependencies that genuinely guide the student.
- **What we don't cover.** Optional H3 at the very end of `## Concepts to internalize` — the closing item of the lecture-notes block, immediately before the lecture→homework divider. List out-of-scope topics with a brief rationale each. Use present tense (it's a scope declaration about the module, not a retrospective on the lesson the reader just finished).
- **M-series notes.** Sits between `Pitfalls to expect` and the closing-notes divider — last thing in the Homework block, immediately before `Reading`.

### Image assets and captions

- **Filename convention:** `ModuleNN-<Descriptor>.png` inside `docs/modules/NN-name/`. The headline summary image is `ModuleNN-Hero.png`; specific diagrams use descriptive PascalCase names (`Module02-MatMul.png`, `Module02-Ladder.png`).
- **Reference from the lesson** with relative paths: `![alt text](NN-name/ModuleNN-Foo.png)`. Always include real alt text — it's the fallback when the image fails to render and matters for accessibility.
- **Hero image** is followed by a blank line and then a short non-italic orientation paragraph. The paragraph is intro text, not a caption — it sets up the module rather than narrating the image, so the looser binding (blank line, no italics) is correct.
- **All other figures** sit inside Lecture notes / Homework and use an *italic* caption directly under the image, with **no blank line** between the image and its caption. The tight binding signals that the italic text is reading the image. The caption explains both what the image shows AND why it matters here, ideally tying back to a specific exercise, concept, or upcoming module. Captions are signal, not decoration.
- **Hero image placement:** immediately after the question pull-quote, before the first `---` divider that opens `Before you start`.

### Figure placement philosophy

Place diagrams according to the role they play in the reader's understanding, not by a fixed "before" or "after" rule.

- **Orientation diagrams** belong near the top of a section, usually after one short setup paragraph. Use this when the visual gives the student a mental model to hold while reading: architecture overviews, training-loop maps, data-flow diagrams, tensor-shape overviews.
- **Dense synthesis diagrams** belong after the explanation. Use this when the visual collects many ideas into one poster-like summary. Shown too early, these can overload the student; shown after the pieces are introduced, they become a useful review surface.
- **Mechanical diagrams** belong exactly where the mechanism is first needed. Use this for shape arithmetic, broadcasting, mask alignment, memory layout, indexing, or other rules where the visual is a working aid. Give the student the minimal vocabulary first, then place the diagram before the worked examples it helps decode.
- **Captions are the default for instructional figures.** If the visual is not a hero image, include a short italic caption that says what the reader is looking at and why it matters at this point in the module. If a figure feels too minor to deserve a caption, consider using an inline ASCII sketch or removing it.

### Test file conventions

The top of each test file should have a docstring with a numbered "Suggested order to implement & turn green" — mapping each implementation step to the tests it unblocks. The student should be able to read this once and know exactly where to start. Construction / repr / boilerplate tests pass from the start (since the boilerplate is implemented), serving as a sanity check on the test file itself; the rest fail with `NotImplementedError` until the student implements.

## Pedagogical scaffolds and worked implementations

`g2c/<topic>/` is the student's working surface — every pedagogical function is left as `# TODO` + `raise NotImplementedError`. Canonical worked implementations live in a parallel mirror under `g2c/solutions/`:

- Each pedagogical class `Foo` in `g2c/<topic>/<file>.py` gets a `_FooImpl` holder class in `g2c/solutions/<topic>/<file>.py`. The holder's methods are bound onto the real class by `g2c.solutions.apply()`.
- Module-level pedagogical functions are mirrored at the module level and bound the same way.

Notebooks do not need a per-notebook `apply()` cell. `g2c/__init__.py` checks `G2C_APPLY_SOLUTIONS` at import time and, if set, calls `g2c.solutions.apply()` before any other code touches the package. To launch a worked notebook with implementations live:

```bash
./notebook.sh 01 --solutions
```

`--solutions` exports `G2C_APPLY_SOLUTIONS=1` so the pre-launch test run and the Jupyter kernel both see the patched package. Without the flag (the default), notebooks launch against scaffolds — that's the student experience. Same lever for pytest:

```bash
G2C_APPLY_SOLUTIONS=1 pytest        # full suite with solutions
pytest                              # scaffolds; many tests intentionally fail
```

`apply()` is idempotent — re-running it just re-binds the same impls.

### Per-module selection

The course is a linear chain, so a bug in an early module surfaces much later. Handing back *everything* to get unstuck would also discard the modules the student got right, so the selection is scoped:

```bash
./notebook.sh 12 --solutions=01-07   # reference impls for 01-07, student's own from 08
G2C_APPLY_SOLUTIONS=01-07 pytest     # same lever for pytest
G2C_APPLY_SOLUTIONS=07,09b pytest    # specific modules
G2C_APPLY_SOLUTIONS=attention pytest # a whole package topic
G2C_APPLY_SOLUTIONS=0 pytest         # explicitly off
```

In Python: `g2c.solutions.apply(["01-07"])`. Bare `apply()` still binds everything.

The map lives in `g2c/solutions/_selection.py`. Two things it exists to get right, both of which a topic-level filter would break:

- **Modules 07 and 08 share `g2c/attention/`.** Selection is per *mirror file*, so `07` yields `self_attention` only and never hands over Module 08's multi-head answer.
- **`TransformerLM.forward` (Module 09) and `TransformerLM.forward_cached` (Module 16) share a class.** Those two entries carry member-level `include`/`exclude` sets, so selecting 16 does not leak Module 09's transformer forward pass.

**When adding or renaming a mirror file, update `MODULE_TARGETS`.** `tests/test_solutions_selection.py` asserts the map and the mirror tree match exactly in both directions, so drift fails by name rather than silently under-applying.

`tests/test_scaffold_invariant.py` is a parametrized pytest that asserts every function targeted by the mirror is still a direct `# TODO` + `raise NotImplementedError` scaffold. If a worked implementation ever leaks back into `g2c/`, that test fails by qualified function name (e.g. `g2c.autodiff.value.Value.__add__`).

When adding a new worked implementation:

1. Implement it under `g2c/solutions/<topic>/<file>.py` following the holder convention above. Use absolute imports inside the mirror file (`from g2c.<topic>.<file> import Foo`), not relative ones.
2. Run `pytest tests/test_scaffold_invariant.py` — the target function in `g2c/<topic>/` must still be scaffolded.
3. Verify the impl works end-to-end with `G2C_APPLY_SOLUTIONS=1 pytest tests/test_<topic>.py`.

The `g2c/solutions/` mirror is canonical and lives in `main` — it is the source of truth, maintained by editing the holder files in place (the three steps above). There is no regeneration script; the `solutions` git branch the mirror was once generated from is defunct.

### Visual aids in lesson pages

Use small ASCII diagrams to crack genuinely dense conceptual sections — particularly anything involving graph topology, shape arithmetic, alignment rules, or memory layout — where prose alone makes the structural relationship hard to see. The bar is "would a reader struggle to picture this without it?" Don't add diagrams for visual flair or because diagrams seem like a nice idea. Roughly a handful per module is the ceiling; some modules may have zero, which is fine. Reserve image assets in `docs/modules/NN-name/` for content that genuinely needs full graphics; for everything else, ASCII inside a fenced code block reads cleanly in every markdown viewer the student is likely to use.

## Notebook style

Student-facing notebooks should foreground the conceptual flow, not the plumbing. When authoring or cleaning a notebook:

- **Open with the canonical intro cell.** The first cell is markdown and follows this template (see `notebooks/clean/00-prerequisite-review.ipynb` for the reference):

  ```markdown
  # Module NN - <Title>

  <one short paragraph orienting the student to what this notebook is for>

  1. Read the lesson page (`docs/modules/NN-name.md`).
  2. Open this notebook with `./notebook.sh NN`.
  3. Answer the `Question:` / `Answer:` cells below.
  4. When you're ready, ask a coding agent to grade your notebook.

  Partial work is fine. Blank `Answer: ""` strings are skipped, not counted wrong. If you'd like a hint instead of a grade, write the request inline in the answer string and the agent will tutor first.
  ```

  Keep the workflow list and the trailing "Partial work is fine..." paragraph verbatim across modules. The only per-module variation is the title, the orientation paragraph, the lesson page filename, and the `./notebook.sh` argument (e.g. `03b`, `09b`).
- **Configs live near consumption.** No global "Run Configuration" wall at the top. Corpus knobs go in the prep cell, model and trainer config dicts at the top of the cell that uses them, sample prompts inline with the sample cell. A small shared base is fine only if it materially cuts duplication.
- **No user-controlled run gates.** Don't add `run_X = True/False` toggles to skip long-running cells. If a student doesn't want to run a cell, they don't run it; Jupyter handles "stop execution." Environment checks (e.g. "TinyStories not downloaded → skip with friendly message") are different and should stay.
- **Extract notebook plumbing, not concepts.** IPython display helpers, matplotlib chart glue, and `Trainer` progress wrappers belong in `g2c/notebook_extras/<topic>.py`, not inline. Code that IS the experiment the student is meant to read (an LR sweep loop, an ablation set) stays inline. Code the lesson explicitly frames as transitional (e.g. "Module 11 will build the real generation utilities, here's a stand-in") also stays inline so the framing is visible.
- **Never put student-implemented code in `g2c/notebook_extras/`.** That directory is the escape hatch for non-pedagogical helpers; pedagogical deliverables go in `g2c/<topic>/`.

## What not to do

- Don't paper over a missing from-scratch implementation by reaching for a high-level library. If `g2c.attention.MultiHeadAttention` doesn't exist yet, build it; don't import `torch.nn.MultiheadAttention` as a substitute.
- Don't add cloud-dependent code paths (paid APIs, hosted training).
- Don't preemptively scale up. Tiny corpora, tiny models. The course's whole point is that the tiny version teaches the idea.
- Don't write speculative scaffolding for modules that haven't been started yet.

