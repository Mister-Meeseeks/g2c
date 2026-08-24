# Roadmap

What the course doesn't teach yet, and what's planned. Every module already declares its own deliberate scope cuts in a "What we don't cover" section — those are decisions, not debts, and they aren't repeated here. This page is the shorter list: the things we think the course *should* eventually cover, roughly ordered by conviction.

Two honesty notes, in the house style. There are no dates on anything below — this is a maintained side project, and a date would be a guess wearing a suit. And the ordering is provisional on purpose: the repo has no telemetry, so [stuck reports](https://github.com/Mister-Meeseeks/g2c/issues/new?template=01-stuck.yml) and [Discussions](https://github.com/Mister-Meeseeks/g2c/discussions) are what actually move items up this list. If one of these gaps is the thing you came for, say so — that's a data point we can't get any other way.

---

## Planned additions

### A broken-trainers debugging lab

The course teaches diagnosis (curve reading in 03B, the `log V` baseline in 09B/10, "Diagnose the run") but never adversarially: N trainers with planted bugs — missing √D, mask after softmax, step before clip, bad init — diagnosed from loss curves and samples alone. The NAND-to-Tetris-flavored version of debugging. The likely shape is a lab, not a module: sealed evidence packs (curves, samples, configs — everything except the bug) generated at maintainer time, plus an optional lever to reproduce any diagnosed bug in your own live trainer and confirm the fix.

---

## Beyond modules

Standalone modules outside the numbered course, for the mechanisms behind the model cards you read. The contract: nothing in Modules 00–20 depends on them, they never renumber anything, and a topic earns one only after it shows up centrally across multiple independent model families *and* admits a laptop-scale build — not because one release made it famous.

In progress — drafts live in the repo, not yet on the site:

- Mixture of experts
- Linear attention
- RLVR with GRPO
- Multimodal language models
- Agent harness engineering
- Midtraining
- Speculative decoding and multi-token prediction
- Muon and orthogonalized updates

Watchlist — evidence accumulating, listed as observations rather than promises: diffusion and other non-autoregressive generation; low-precision training.

---

## G2C Briefs

Briefs are dated field guides to individual releases. They map a current
technical report onto the durable course and Beyond mechanisms, label reported
claims separately from derived arithmetic and course interpretation, and state
what the available sources do not disclose. They carry no notebook, scaffold,
test suite, or downstream dependency.

In progress — drafts live in the repo, not yet on the site:

- DeepSeek V4 — reading a frontier model stack

An uncovered mechanism in a Brief does not automatically become roadmap work.
It still has to recur across independent model families and admit a useful
laptop-scale build before it earns a Beyond module.

---

## Future arcs

Bigger than additions — candidate themes for a second course, listed as directions rather than promises:

- **Interpretability: you built it — now open it up.** Logit lens, activation patching, ablations, an induction-head hunt — on the model *you trained*. Module 08 already concedes its attention visualizations underwhelm at tiny scale; a dedicated arc is where that debt pays off, because tiny models aren't a limitation for interpretability — they're the superpower. Fully inspectable, fully yours.
- **Performance and serving.** Quantization internals, attention tiling, batching, speculative decoding — the "make it fast" half that this course explicitly trades away for legibility.

---

## What won't be added

The constraints are the course's identity, not its current limitations:

- **No cloud paths.** No paid APIs, no hosted training, no GPU rental. If it doesn't run on an M-series laptop, it doesn't go in.
- **No scale for scale's sake.** Tiny corpora and tiny models are the pedagogy, not a budget compromise.
- **No per-module time estimates.** The measurable proxies track volume, not difficulty, and they invert on the hardest modules. This number stays absent until real stuck-report data can replace it.
