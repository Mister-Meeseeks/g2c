# Rubric — Beyond: Muon

Student answers live in `notebooks/solutions/muon-orthogonalized-updates.ipynb`
(fall back to `notebooks/clean/muon-orthogonalized-updates.ipynb`). Grade each
submitted `Question:` / `Answer:` independently; skip blank answers.

## Q1 — Reading the spectrum plots

- **Correct**: describes the measured progression — the normalized input's
  singular values sit well below 1, early passes amplify the small values
  steeply (the quintic is steep near zero) while values near 1 stay parked
  (the fixed point), and by five passes the spectrum sits in a loose band
  around 1, not exactly at 1. Explains that the Frobenius normalization is
  what puts the spectrum inside the polynomial's basin, and that singular
  *vectors* are untouched throughout.
- **Mostly correct**: right mechanism but claims the iteration converges the
  spectrum exactly to 1, or omits why normalization must come first.
- **Partially correct**: describes the plots without connecting them to the
  polynomial's shape, or attributes the flattening to the normalization
  alone.

## Q2 — Where each optimizer spends its step

- **Correct**: reports the measured spectra — the near-rank-1 momentum's
  AdamW-style update remains dominated by the same one direction, while
  Muon's update spreads energy across directions at comparable gains.
  Connects this to the module's framing: AdamW adapts per coordinate and
  ignores matrix structure; Muon equalizes per direction. A strong answer
  also notes the limit of the demo (one synthetic matrix, not a claim about
  training outcomes).
- **Partially correct**: states the AdamW/Muon contrast from the lesson
  without engaging with the computed spectra.
- **Needs revision**: claims Muon's update is "bigger" or "better" rather
  than differently shaped, or reads orthogonalization as changing the
  update's directions.

## Q3 — The race and the learning-rate landscape

- **Correct**: reports best-of-sweep versus best-of-sweep curves and names
  the winner *at this scale and budget* without overclaiming; gives the
  approximate ratio between the two optimizers' best learning rates (they
  should be far apart — order tens of times) and says which degraded more
  gracefully off its optimum. Treats a surprising result (AdamW winning,
  a tie) as data, ideally with a second seed as the control.
- **Mostly correct**: fair sweep and honest reading, but no comment on lr
  sensitivity, or vice versa.
- **Partially correct**: compares the two optimizers at a single shared
  learning rate, or generalizes the toy result to "Muon is better" without
  qualification.

## Q4 — Violating the partition

- **Correct**: reports what actually happened when embeddings ran through
  Muon — degradation, no visible change, or something odd — and explains
  the *argument* either way: an embedding table's rows are unrelated
  per-token objects, so orthogonalization mixes updates across tokens that
  share no batches; at toy scale with a tiny vocabulary the damage may be
  invisible, and saying so is a correct observation, not a failure.
- **Partially correct**: recites the rows-are-unrelated argument without
  reporting the run, or reports the run with no mechanism.
- **Needs revision**: concludes from a null toy result that the partition
  rule is unnecessary in general.

## Q5 — The optimizer-state bill

- **Correct**: counts parameters on each side of the partition for the race
  model, computes AdamW-everything state (two buffers per parameter) versus
  the hybrid (one buffer for Muon-side matrices, two for the AdamW-side
  remainder), gets the arithmetic right to within rounding, and connects it
  to Module 13B's memory-tenant table — the `m`/`v` lines shrink for every
  parameter Muon owns. Concludes only about memory, not speed or quality.
- **Mostly correct**: right formulas with a minor count slip (e.g. forgets
  the positional table or `head_bias` on the AdamW side).
- **Partially correct**: computes total parameters rather than optimizer
  state, or turns the memory arithmetic into an unsupported quality claim.
