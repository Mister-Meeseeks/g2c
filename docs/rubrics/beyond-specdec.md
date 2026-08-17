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
  and 5M drafters in terms of acceptance bought per unit of drafting cost,
  using their own measurements.
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
  the base's one-ahead performance and explains why predicting `x_{t+2}` is
  strictly harder (the head must marginalize over the unseen `x_{t+1}`'s
  consequences — one extra step of uncertainty). Compares self-speculation's
  acceptance and tokens-per-pass against the separate-drafter runs and notes
  the structural trade honestly: near-free drafting per token versus `k`
  limited to the head's reach.
- **Mostly correct**: reports both measurements but explains the two-ahead
  gap only as "harder" without the extra-uncertainty mechanism.
- **Partially correct**: trains the head but evaluates it as if it predicted
  `x_{t+1}`, or compares self-speculation and separate-drafter runs on
  different prompts/budgets without flagging it.

## Q5 — The cost model

- **Correct**: writes tokens per unit of target-compute as
  `(E[accepted] + 1) / (1 + k·c)` (or an equivalent formulation), with `c`
  the drafter/target cost ratio; plugs in their measured acceptance and a
  defensible `c` (parameter-ratio estimates are acceptable if stated);
  derives the break-even acceptance for `k = 4`; and interprets GLM-5's
  reported 2.76 acceptance length as roughly 2.76 tokens per target pass
  with a near-free drafter (`c ≈ 0` for a parameter-shared MTP layer).
  Concludes about target-compute, noting wall-clock needs the systems
  machinery from Q3's caveat.
- **Mostly correct**: correct formula and substitution with a slip in the
  break-even algebra, or omits the GLM interpretation.
- **Partially correct**: formula ignores drafting cost entirely, or treats
  acceptance length and acceptance *rate* as interchangeable.
