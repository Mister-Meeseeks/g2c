# Contributing

This is a self-study course, so "contributing" means something a bit different here than
in a normal library. The most valuable thing you can send is **information about where
the course failed you** — not a patch.

## The most useful thing you can do

**Tell us where you got stuck.** Open a [stuck issue](https://github.com/Mister-Meeseeks/g2c/issues/new?template=01-stuck.yml)
naming the module. You don't need to have solved it, and you don't need a
well-formed question. "I lost the thread somewhere in the backward pass" is a real
data point.

There is no telemetry in this repo and never will be — everything runs locally and
nothing phones home. That means stuck issues are the *only* way we learn which module
is the wall. A module that quietly loses half the people who reach it looks exactly
like a module nobody had trouble with.

## Reporting problems

| What | Template |
| ---- | -------- |
| Stuck on the material | [I got stuck on a module](https://github.com/Mister-Meeseeks/g2c/issues/new?template=01-stuck.yml) |
| Setup, scripts, downloads, or a suspect test | [Something is broken](https://github.com/Mister-Meeseeks/g2c/issues/new?template=02-broken.yml) |
| Wrong claim, unclear passage, typo | [Lesson feedback](https://github.com/Mister-Meeseeks/g2c/issues/new?template=03-lesson-feedback.yml) |

Before filing a broken-test issue, check whether it's a scaffold rather than a bug —
most tests fail on a fresh clone by design:

```bash
G2C_APPLY_SOLUTIONS=1 pytest    # green here means the scaffolding is fine
pytest                          # red here is expected; it's the work you haven't done
```

## Questions and study groups

Anything conversational — "am I understanding attention right?", "is my loss
curve normal?", "anyone else working through Part II?" — belongs in
[GitHub Discussions](https://github.com/Mister-Meeseeks/g2c/discussions).
Issues are for the templates above; Discussions is for everything else,
including organizing a study group (the course works well with a small weekly
cohort).

One honest expectation: this course is a maintained side project, not a
staffed product. Expect replies to issues and discussions within **days to a
week**, not hours. For getting unstuck, a coding agent pointed at your
notebook (the workflow every lesson's Exercises section describes) will almost
always be faster than waiting on a human reply — that's by design.

## Pull requests

Corrections to lessons — wrong math, stale claims, broken links, typos — are welcome
directly as PRs. So are fixes to setup scripts and tests.

Three things to know before opening a larger PR:

**The scaffold surface is frozen after release.** Students edit the files under
`g2c/` (outside `g2c/solutions/` and `g2c/notebook_extras/`) in place, so an
upstream change to them lands as a merge conflict inside somebody's homework.
Post-release, fixes route to the solutions mirror, tests, docs, and rubrics —
all of which merge cleanly into any student's copy. Adding new files is fine.
CI enforces this against the release tag recorded in `.github/scaffold-freeze`;
a genuinely unavoidable scaffold change goes in that file's exception ledger
with a note telling students how to patch an already-edited copy.

**Don't put worked implementations in `g2c/`.** Every pedagogical function there is
deliberately a `# TODO` + `raise NotImplementedError` scaffold. Canonical implementations
live in the parallel mirror at `g2c/solutions/`, and `tests/test_scaffold_invariant.py`
fails by qualified function name if one leaks back. If you're fixing a reference
implementation, edit the mirror.

**New content should follow the existing shape.** Lesson pages have a fixed five-section
structure, notebooks have a canonical intro cell and `Question:` / `Answer:` slots, and
each module gets a rubric. `AGENTS.md` documents all of it — it's written for coding
agents but it's the accurate description of the conventions.

## Running the checks

```bash
source .venv/bin/activate

ruff check .                        # lint
G2C_APPLY_SOLUTIONS=1 pytest        # full suite against reference implementations
pytest tests/test_scaffold_invariant.py   # no solutions leaked into g2c/
bash scripts/check_scaffold_freeze.sh     # released scaffold files unchanged
python3 scripts/check_dataset_urls.py     # course downloads still resolve
python3 scripts/check_model_ids.py        # BaseLM/Ollama model ids still resolve
mkdocs build --strict               # docs build clean (pip install -r docs/requirements.txt)
```

CI runs the same things, plus a job that clones fresh and runs `./setup.sh` end to end.

## Scope

The course has hard constraints that shape what can be accepted:

- **It runs on an M-series MacBook.** No cloud GPUs, no paid APIs, no hosted training.
- **From-scratch through the architecture.** Part I must not import a high-level
  abstraction for the concept under study — the point of the attention module is to
  build attention, not to call `torch.nn.MultiheadAttention`.
- **Pedagogy over performance.** Legible beats fast. A clever optimization that makes
  a lesson harder to read is a regression.
- **Tiny everything.** Tiny corpora, tiny models. The thesis is that the tiny version
  teaches the idea.

Changes that need more compute, more money, or a bigger model than the course's
constraints allow will get turned down even when they're good ideas — not because
they're wrong, but because they break the promise that any of this runs on a laptop
someone already owns.
