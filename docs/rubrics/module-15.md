# Module 15 Rubric

Use this rubric to grade submitted answers in the student's working notebook (`notebooks/solutions/15-evaluation.ipynb`, falling back to `notebooks/clean/15-evaluation.ipynb` if no solutions copy exists yet). Each exercise has `Question:` / `Answer:` string cells; treat a blank `"Answer: "` as not submitted (skip it). If the student wrote a hint or help request inside the answer string, tutor first before grading and avoid giving away the full solution unless they explicitly ask. Give feedback exercise by exercise and avoid replacing the student's work with a full solution unless explicitly asked.

Most answers in this module report on the student's own eval runs, whose numbers vary by artifact and machine. Grade the reasoning shape — cited numbers from their reports, correct causal attribution — not specific values.

## Exercise 15.01 — Raw vs length-normalized MC reports

A correct answer should include:

- Whether the two reports agree, with accuracy numbers (overall or per category) cited from both.
- The mechanism: raw scoring sums token log-probs, so longer options accumulate more negative mass and are penalized; length normalization divides by token count and removes that bias.
- The inference: agreement means option token-lengths are roughly balanced within each item (the starter set was authored that way); disagreement in a category means options there differ in length and the raw score reflects length bias, not model knowledge.

Common issues:

- Attributing raw/normalized disagreement to run-to-run noise — MC scoring is deterministic; the delta is structural.
- Declaring one report "more correct" in general without tying the claim to option lengths.
- Comparing only overall accuracy and skipping the per-category breakdown the cell prints.

## Exercise 15.02 — Read the reliability plot

A correct answer should include:

- A verdict (overconfident / underconfident / well calibrated) read correctly off the plot: bars below the diagonal mean overconfident, above mean underconfident. Small over-trained models typically land overconfident, but grade the read against their plot, not a prescribed direction.
- The confidence range where the confidence-vs-accuracy gap is largest, cited from their bins.

Common issues:

- Reading the axes backwards (confusing which axis is confidence and which is empirical accuracy).
- Conflating accuracy with calibration — "only 60% accurate so it's miscalibrated" misses that a 60%-confident 60%-accurate model is perfectly calibrated.
- Over-reading sparse bins: at 60 questions and 5 bins, a bin with a handful of examples is noise.

## Exercise 15.03 — Labeling the hallucination probes

A correct answer should include:

- The dominant failing category from their labeled misses, backed by the per-category accuracy or the `failure_notes` table.
- A remediation that matches the category: unanswerable/refusal failures → add refusal ("I don't know") examples to SFT or refusal-preferring DPO pairs; contextual failures → add examples requiring facts copied from the prompt; format failures → more formatting examples.
- Recognition that factual failures are mostly a pretraining-capacity limit — SFT/DPO data shapes behavior, it does not add knowledge.

Common issues:

- Proposing more SFT data as the fix for factual-knowledge gaps.
- Labeling by the probe's expected kind instead of the observed failure (a "factual" probe can fail by refusing; the observed behavior is what gets labeled).
- Counting matcher artifacts as hallucinations — `contains_match` misses a correct answer phrased without any reference keyword.

## Exercise 15.04 — The arithmetic cliff

A correct answer should include:

- The digit-count boundary from the per-bucket accuracy (typically 1-digit mostly right, collapse by 2- or 3-digit).
- A characterization of the wrong answers — near-misses (right digit count, carry/off-by-one errors) vs unrelated numbers — with at least one cited example. Near-misses mean the model learned the answer *shape* without the algorithm.

Common issues:

- Judging the cliff from one prompt instead of the bucket breakdown.
- Not distinguishing matcher artifacts from genuine misses — `numeric_match` grabs the first number in the prediction, so an echoed operand scores as a wrong answer.
- Expecting a smooth curve and describing away a genuine collapse.

## Exercise 15.05 — Generated-answer confidence

A correct answer should include:

- A comparison of confidence on wrong vs right arithmetic answers from the rescored report; the typical finding is the model is about as confident when wrong as when right, so its confidence cannot be trusted as a correctness signal at this scale.
- The caveat that this per-token `exp(mean logp)` rescore is not a closed-set probability — there is no candidate set to normalize against, so it is a weaker signal than MC confidence/ECE.

Common issues:

- Treating the rescored value as a calibrated probability comparable to closed-set confidence.
- Carrying the closed-set calibration verdict over to generation without evidence.
- No cited numbers from the rescored report or reliability plot.

## Exercise 15.06 — DPO vs SFT parent

A correct answer should include:

- What actually moved between the two printed reports — accuracy, mean confidence/ECE, both, or neither — with numbers from each.
- The interpretation: DPO shapes preferences/behavior, so at this scale MC accuracy often barely moves while confidence and calibration shift; "improved the behavior you care about" vs "merely moved the model" answered from that evidence.

Common issues:

- Declaring DPO a failure because factual MC accuracy did not rise — that was never DPO's training signal.
- Comparing a raw report from one model against a length-normalized report from the other.
- Reading small deltas on a 60-question eval as significant.

## Exercise 15.07 — The trust sentence

A correct answer should include:

- All three slots filled, each grounded in a report from the notebook (e.g., trust for simple factual/formatting per MC category accuracy; not for multi-digit arithmetic or unanswerable questions per the probes).
- A "because" that cites both accuracy and calibration evidence — a category can be accurate yet miscalibrated, and the sentence should reflect which failure mode drives the distrust.

Common issues:

- Generic slots ("simple things", "hard things") with no numbers behind them.
- Trusting a category on accuracy alone while ignoring its calibration.
- Evidence drawn from one memorable sample instead of the aggregated reports.
