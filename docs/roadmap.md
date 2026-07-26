# Roadmap

What the course doesn't teach yet, and what's planned. Every module already declares its own deliberate scope cuts in a "What we don't cover" section — those are decisions, not debts, and they aren't repeated here. This page is the shorter list: the things we think the course *should* eventually cover, roughly ordered by conviction.

Two honesty notes, in the house style. There are no dates on anything below — this is a maintained side project, and a date would be a guess wearing a suit. And the ordering is provisional on purpose: the repo has no telemetry, so [stuck reports](https://github.com/Mister-Meeseeks/g2c/issues/new?template=01-stuck.yml) and [Discussions](https://github.com/Mister-Meeseeks/g2c/discussions) are what actually move items up this list. If one of these gaps is the thing you came for, say so — that's a data point we can't get any other way.

---

## Planned additions

### Module 13B — LoRA

The headliner, and the omission we'd defend least. LoRA is the laptop-native fine-tuning method — for a course whose identity is "everything on your Mac," teaching only full fine-tuning is a gap, not a scope choice. It's also mechanically ideal for this course: two low-rank matrices, an adapter that starts as an exact no-op, merge/unmerge semantics, and a parameter count you can derive on paper (rank 8 on the 360M `BaseLM` trains roughly 0.1% of the weights). The plan is a full B-module in the 03B/09B pattern: new `g2c/lora/` package, scaffolds, tests, notebook, rubric — same SFT objective as Module 13, new parameterization, and a door opened to fine-tuning 1–3B local models that full fine-tuning can't touch on a 16GB machine.

### Constrained decoding

You own the sampler — that's the whole point of Module 11 — which makes this buildable here when every other course has to hand-wave it. The plan: a JSON-subset automaton that masks logits token by token, so your Module 18 tool calls become *unparseable-by-construction* rather than parsed-hopefully. The mechanism lands in `g2c/sampling/`; the teaching is staged in Module 18, where the malformed-JSON pain is fresh. The payoff is the contrast — a constrained weak model emits 100% parseable calls with weak content, because the grammar carries the syntax so the model only has to carry the semantics. `ProdLM`'s `format: json` flag then becomes recognizable as the production packaging of the same idea, and the reason it's a server-side feature: constraining needs the logits.

### A broken-trainers debugging lab

The course teaches diagnosis (curve reading in 03B, the `log V` baseline in 09B/10, "Diagnose the run") but never adversarially: N trainers with planted bugs — missing √D, mask after softmax, step before clip, bad init — diagnosed from loss curves and samples alone. The NAND-to-Tetris-flavored version of debugging.

### Synthetic SFT data

Self-Instruct is in Module 13's reading list but never exercised. The plan: generate instruction data with ProdLM, fine-tune TinyLLM on it, and compare against your 50 hand-authored pairs — the modern data recipe, run entirely locally, with a pleasing symmetry: TinyStories, the corpus you pretrained on, is itself synthetic data. The largest item here, because the generator (ProdLM) doesn't arrive until Module 16 and the design has to respect the module order.

---

## Future arcs

Bigger than additions — candidate themes for a second course, listed as directions rather than promises:

- **Interpretability: you built it — now open it up.** Logit lens, activation patching, ablations, an induction-head hunt — on the model *you trained*. Module 08 already concedes its attention visualizations underwhelm at tiny scale; a dedicated arc is where that debt pays off, because tiny models aren't a limitation for interpretability — they're the superpower. Fully inspectable, fully yours.
- **Reasoning and RL post-training.** Reward models, PPO→GRPO, RL on verifiable toy tasks. Fills the biggest modern-relevance gap (Module 14 stops at DPO by design).
- **Performance and serving.** Quantization internals, attention tiling, batching, speculative decoding — the "make it fast" half that this course explicitly trades away for legibility.
- **Architecture extensions.** MoE, linear attention, multimodality — the mechanisms behind the model cards you read.

---

## What won't be added

The constraints are the course's identity, not its current limitations:

- **No cloud paths.** No paid APIs, no hosted training, no GPU rental. If it doesn't run on an M-series laptop, it doesn't go in.
- **No scale for scale's sake.** Tiny corpora and tiny models are the pedagogy, not a budget compromise.
- **No per-module time estimates.** The measurable proxies track volume, not difficulty, and they invert on the hardest modules. This number stays absent until real stuck-report data can replace it.
