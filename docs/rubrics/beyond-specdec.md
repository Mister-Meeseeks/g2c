# Rubric — Beyond: Speculative decoding and MTP

Student answers live in `notebooks/solutions/specdec-draft-and-verify.ipynb`
(fall back to `notebooks/clean/specdec-draft-and-verify.ipynb`). Grade each
submitted `Question:` / `Answer:` independently; skip blank answers.

## Q1 — Reading the losslessness check

- **Correct**: confirms the speculative and plain-greedy outputs were
  byte-identical and explains *why* the guarantee holds: greedy verification
  accepts a draft token only where it equals the target's own argmax, and on
  the first disagreement the emitted token IS the target's argmax, so the
  emitted sequence is the target's greedy chain by induction. Reads the
  `SpecStats` correctly (tokens per pass above 1.0 means serial target
  passes were removed).
- **Mostly correct**: confirms equality and reads the stats but explains the
  guarantee only as "the target checks the drafts" without the
  correction-token half of the argument.
- **Partially correct**: treats the identical outputs as a lucky empirical
  result rather than a structural property, or reads acceptance rate as
  output quality.

## Q2 — Reading the acceptance sweep

- **Correct**: reports tokens-per-pass across `k` and both drafters, and
  explains the diminishing-returns shape: each extra draft position only
  pays off with probability that the whole prefix before it survives, so
  the marginal value of position `k` falls as `k` grows. Compares the 1M
  and 5M drafters in terms of accepted-prefix progress or tokens-per-pass
  bought per unit of drafting cost, using their own measurements.
- **Mostly correct**: right readings but no mechanism for why large `k`
  flattens, or compares drafters on acceptance alone ignoring their cost.
- **Partially correct**: reports numbers without interpretation, or claims a
  bigger `k` is always better.

## Q3 — The wall-clock honesty benchmark

- **Correct**: reports both tokens-per-pass and measured tokens/sec, with
  MPS synchronization noted, and explains any gap between the two: at this
  scale each verification pass recomputes the full prefix and the drafter
  loops in Python, so removed target passes need not become removed
  wall-clock. "The accounting improved and the clock didn't" (or barely
  did) is a fully correct result when the measurement supports it.
- **Mostly correct**: honest measurement but attributes a gap vaguely to
  "overhead" without naming prefix recomputation or the drafter loop.
- **Partially correct**: reports only the favorable metric, times without
  synchronization, or predicts production speedups from the toy timing.

## Q4 — The MTP head and self-speculation

- **Correct**: reports the trained head's two-ahead loss/accuracy against
  the base's one-ahead performance without assuming which score must be
  higher. Explains that the head receives `h_t` and the actual `x_{t+1}`
  embedding but not the trunk's fully updated `h_{t+1}`, and uses that
  information difference as a possible explanation for the measured result.
  Compares self-speculation's acceptance and tokens-per-pass against the
  separate-drafter runs and notes both the head's limited reach and the
  notebook's proposal-pass caveat: this toy loop does not obtain a production
  MTP pipeline's near-zero incremental proposal cost.
- **Mostly correct**: reports both comparisons and identifies the missing
  updated trunk state, but overstates the result as a universal ranking or
  omits the proposal-pass caveat.
- **Partially correct**: trains the head but evaluates it as if it predicted
  `x_{t+1}`, or compares self-speculation and separate-drafter runs on
  different prompts/budgets without flagging it.

## Q5 — The cost model

- **Correct**: writes tokens per unit of target-compute as
  `(E[accepted drafts] + 1) / (1 + k·c)` (or an equivalent formulation),
  with `c` the drafter/target cost ratio; plugs in their measured accepted
  drafts and a defensible `c` (parameter-ratio estimates are acceptable if
  stated); and derives the break-even acceptance for `k = 4`. Treats GLM-5's
  2.76 as a reported accept length from a private prompt set with four
  speculative steps, not automatically as this notebook's tokens per target
  pass. States that the source's counter convention is needed for a numeric
  substitution, while the qualitative opportunity remains: a shared MTP
  drafter can make proposal cost small relative to a separate model. Notes
  that wall-clock still needs the systems caveats from Q3.
- **Mostly correct**: correct formula and substitution with a slip in the
  break-even algebra, or gives the careful GLM comparison without naming the
  missing metric convention.
- **Partially correct**: formula ignores drafting cost entirely, or treats
  accepted drafts, emitted tokens, and acceptance *rate* as interchangeable.
