# G2C Briefs

G2C Briefs are dated field guides to current model releases. The numbered
course builds the stack; Beyond modules isolate durable mechanisms; Briefs ask
whether those pieces are enough to read a real technical report without the
report collapsing into a wall of product names and benchmark tables.

A Brief is deliberately not a course module. It has no notebook, scaffold,
test suite, rubric, or deliverable. It does not ask a laptop to reproduce a
frontier training run. Instead, it traces one release from architecture through
training and serving, maps each claim back to something in g2c, and names the
remaining gaps plainly.

## How to read the evidence labels

Every Brief separates four kinds of statements:

- **Reported** — a number or claim made by the releasing organization. The
  Brief links the primary source but does not present the claim as independent
  verification.
- **Derived** — arithmetic computed directly from disclosed values. The inputs
  and calculation are shown.
- **G2C interpretation** — an explanatory connection or inference made by the
  course. It should make the report easier to reason about without pretending
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
| **Not yet covered** | The report introduces a mechanism for which g2c does not yet provide enough machinery. |

`Not yet covered` is not an automatic promise of a new Beyond module. A topic
earns one only after it recurs across independent model families and admits a
useful laptop-scale build. Briefs are evidence for that decision, not a way
around it.

## Briefs

- [DeepSeek V4 — Reading a frontier model stack](deepseek-v4.md) — MoE,
  compressed sparse attention, million-token training, specialist RL,
  on-policy distillation, and the infrastructure that makes the efficiency
  claims operational.

