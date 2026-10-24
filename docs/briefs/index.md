# G2C Briefs

G2C Briefs are dated field reports on new developments in AI: model
releases, research results, new training and serving techniques, and the
tools built around them. The numbered course builds the stack; Beyond modules
isolate durable mechanisms; Briefs ask whether those pieces are enough to read
a real technical report, paper, or announcement without it collapsing into a
wall of product names and benchmark tables.

A Brief is deliberately not a course module. It has no notebook, scaffold,
test suite, rubric, or deliverable. It does not ask a laptop to reproduce a
frontier training run. Instead, it takes one development, maps each claim back
to something in g2c, and names the remaining gaps plainly. For a model release
that means tracing the stack from architecture through training and serving;
for research it means what is claimed, what evidence supports it, and where it
is still open.

## How to read the evidence labels

Every Brief separates four kinds of statements:

- **Reported** — a number, claim, or observation made by the source (the
  releasing organization, or the authors of the research). The Brief links
  the primary source but does not present the claim as independent
  verification.
- **Derived** — arithmetic computed directly from disclosed values. The inputs
  and calculation are shown.
- **G2C interpretation** — an explanatory connection or inference made by the
  course. It should make the source easier to reason about without pretending
  to be a new experimental result.
- **Not disclosed** — information needed for a stronger conclusion that the
  available primary sources do not provide.

Briefs pin a source snapshot and a verification date. If a release changes,
the original reading remains legible as a historical snapshot; substantive
updates receive an explicit note rather than a silent rewrite.

## Coverage labels

The mechanism map in each Brief uses three labels:

| Label | Meaning |
|---|---|
| **Built in g2c** | You have implemented the load-bearing mechanism at laptop scale. |
| **Conceptual bridge** | Existing course work supplies the right mental model, but not this exact production design. |
| **Not yet covered** | The source introduces a mechanism for which g2c does not yet provide enough machinery. |

`Not yet covered` is not an automatic promise of a new Beyond module. A topic
earns one only after it recurs across independent model families and admits a
useful laptop-scale build. Briefs are evidence for that decision, not a way
around it.

## Briefs

- [DeepSeek V4 — Reading a frontier model stack](deepseek-v4.md) — MoE,
  compressed sparse attention, million-token training, specialist RL,
  on-policy distillation, and the infrastructure that makes the efficiency
  claims operational.
- [DeepSeek V4.1-Flash — A model designed around its KV cache](deepseek-v4.1-flash.md)
  — the causal encoder-decoder, cross-layer cache sharing (CSA2), FP4 cache
  entries, bounded replay, data-driven post-training, and the October 17
  (1017) checkpoint. Assumes no V4 background.
- [Neural Networks Are Already Clean — An argument about what neurons are](neurons-already-clean.md)
  — a research Brief on Goodfire's October 2026 essay arguing that apparent
  polysemanticity comes from bad tools, bad learning, and recursive
  generalization rather than superposition, with the evidence, its limits,
  and what you can check on your own BaseLM.
- [System One models — Decisions without generation](system-one-models.md)
  — a category Brief on models that answer typed questions in one forward
  pass instead of generating text: the shared API, Cloudflare's Clef head read
  from its released code, training for calibration, Clairic's extraction
  model, and what "can't hallucinate" does and doesn't mean.
- [DeepInfer — Reading a production inference engine](deepinfer.md) — DeepSeek's
  open-sourced serving engine, read for the places where course pieces return
  at scale: prefill vs decode, paged and persistent KV caches, reproducible
  sampling and batch invariance, budgeted speculative verification, and
  special-token safety at the tokenizer boundary.
- [MiniMax M3.1 Flash — Three changes, read from the config](minimax-m3.1-flash.md)
  — a release with no technical report, read by diffing configs and counting
  tensors: MiniMax Sparse Attention in every layer, quantization-aware
  training to 4-bit experts, and DeepSeek's DSpark drafter with a Markov head
  that is a low-rank bigram model. Assumes no M3 background.
