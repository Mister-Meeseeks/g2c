"""Calibration metrics — does the model know how confident to be?

A model is **calibrated** if, on the subset of examples where it says
"70% confident," it is correct 70% of the time. Across all confidence
levels. Calibration is independent of accuracy: a model can be highly
accurate but wildly over-confident, or barely accurate but
well-calibrated.

We compute two diagnostics:

  * **Expected calibration error (ECE)** (Naeini et al., 2015). Bin
    predictions by confidence; in each bin, compare the bin's mean
    confidence to the bin's empirical accuracy; weight the absolute
    differences by bin frequency. Returns a single scalar on `[0, 1]`,
    lower is better. This is the canonical scalar calibration metric.

  * **Reliability curve.** The same per-bin (mean_confidence,
    accuracy, count) triples, returned as parallel arrays. Used for
    plotting the diagonal-vs-actual diagram. The curve is what ECE
    summarizes; if you only ever plot one calibration thing, plot
    this.

Both are scaffolded — the math is one careful loop over confidence
bins.

A pedagogical note on calibration. ECE is far from the only
calibration metric (also: Brier score, log loss, maximum calibration
error, reliability diagrams). ECE is widely used because it returns
a single comparable number, but it's not without flaws — it's
sensitive to bin count, it doesn't penalize asymmetric mis-calibration
(over- vs under-confident the same way), and it can be optimized
without improving the ranking. Read Guo et al. 2017 for context.
"""
from __future__ import annotations

import math  # noqa: F401 (for the student implementation)


def expected_calibration_error(
    confidences: list[float],
    correct: list[bool],
    *,
    n_bins: int = 10,
) -> float:
    """Bin-weighted expected calibration error.

    Definition (Naeini et al., 2015, Eq. 1; standard form):

        ECE = Σ_b  (|B_b| / N) · | acc(B_b) − conf(B_b) |

    where:
      * `B_b` is the b-th bin (predictions whose confidence falls in
        the b-th bucket of [0, 1]).
      * `acc(B_b)` is the fraction of correct predictions in `B_b`.
      * `conf(B_b)` is the mean confidence in `B_b`.
      * `N` is the total number of predictions.

    Args:
        confidences: per-example confidence on `[0, 1]`. Length N.
        correct: per-example correctness (bool). Length N.
        n_bins: number of equal-width bins on `[0, 1]`. Default 10.

    Returns:
        ECE on `[0, 1]`. Lower is better-calibrated; 0 is perfectly
        calibrated; 1 is maximally mis-calibrated.

    Recipe:
        1. # Validate.
           if len(confidences) != len(correct):
               raise ValueError(...)
           if len(confidences) == 0:
               raise ValueError("ECE undefined for empty input")
           if n_bins <= 0:
               raise ValueError(...)
           # Optionally check confidences are on [0, 1].

        2. # Bin edges. Bin b covers [b/n_bins, (b+1)/n_bins].
           # We want left-closed, right-closed for the LAST bin
           # (so a confidence of exactly 1.0 lands in the last bin).
           # The simplest correct binning:
           #     bin = min(int(c * n_bins), n_bins - 1)

        3. # Accumulate per-bin sums.
           N = len(confidences)
           ece = 0.0
           for b in range(n_bins):
               # Indices of examples in bin b.
               idx = [i for i, c in enumerate(confidences)
                      if min(int(c * n_bins), n_bins - 1) == b]
               if len(idx) == 0:
                   continue
               bin_acc  = sum(correct[i] for i in idx) / len(idx)
               bin_conf = sum(confidences[i] for i in idx) / len(idx)
               ece += (len(idx) / N) * abs(bin_acc - bin_conf)
           return ece

    Sanity values:
      * Perfectly calibrated: every bin's mean_confidence == its
        empirical accuracy → ECE = 0.
      * Always-1.0-confident, 50% accurate: one bin, conf=1.0,
        acc=0.5, |1.0 - 0.5| · 1.0 = 0.5 → ECE = 0.5.
      * Always-correct, always-confidence-0.0: one bin, conf=0.0,
        acc=1.0, |0.0 - 1.0| · 1.0 = 1.0 → ECE = 1.0. (Maximally
        miscalibrated — the model says "no chance" and is right
        every time.)
      * Single example: one bin contains it, |c - acc| = |c - {0,1}|
        weighted by 1.0. ECE is well-defined but noisy.

    A subtle point: ECE is NOT a smooth function of the bin count.
    Increasing `n_bins` increases granularity but also variance per
    bin. The default `n_bins=10` matches the original Naeini paper.
    For small N (< 100), consider fewer bins (5) to reduce
    per-bin noise.
    """
    # TODO
    raise NotImplementedError


def reliability_curve(
    confidences: list[float],
    correct: list[bool],
    *,
    n_bins: int = 10,
) -> tuple[list[float], list[float], list[int]]:
    """Per-bin (mean_confidence, accuracy, count) — the data behind a
    reliability diagram.

    Args:
        confidences: per-example confidence on `[0, 1]`. Length N.
        correct: per-example correctness. Length N.
        n_bins: number of equal-width bins.

    Returns:
        (bin_confidences, bin_accuracies, bin_counts):
          * bin_confidences: list of length `n_bins`. Element b is the
                             mean confidence among predictions in bin b,
                             or `nan` if the bin is empty.
          * bin_accuracies:  list of length `n_bins`. Element b is the
                             fraction correct in bin b, or `nan` if empty.
          * bin_counts:      list of length `n_bins`. Element b is the
                             number of predictions in bin b.

    Recipe:
        Same binning as `expected_calibration_error` (use
        `min(int(c * n_bins), n_bins - 1)`). For each bin, compute
        the mean confidence and accuracy of its examples; if the bin
        is empty, return `float('nan')` for both.

    Plotting (in a notebook, not part of this function):

        import matplotlib.pyplot as plt
        bc, ba, bn = reliability_curve(confs, corr)
        plt.plot([0, 1], [0, 1], 'k--', label='perfect')
        plt.plot(bc, ba, 'o-',  label='model')
        plt.bar(bc, bn, width=0.05, alpha=0.3, label='count')
        plt.xlabel('confidence'); plt.ylabel('accuracy'); plt.legend()

    The diagonal is "perfectly calibrated"; the model's curve sitting
    above means under-confident, below means over-confident. Tiny LMs
    almost always sit below the diagonal — over-confident is the
    default.

    Sanity values:
      * Empty bins return `(nan, nan, 0)` for that bin's slot.
      * Bin counts sum to N (the input length).
      * If you plot bin_confidences against bin_accuracies and the
        model is perfectly calibrated, the line is the diagonal.
    """
    # TODO
    raise NotImplementedError


def _validate_inputs(
    confidences: list[float],
    correct: list[bool],
    n_bins: int,
) -> None:
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must have the same length")
    if len(confidences) == 0:
        raise ValueError("calibration is undefined for empty input")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    for confidence in confidences:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidences must be on [0, 1]")


def _bin_index(confidence: float, n_bins: int) -> int:
    return min(int(confidence * n_bins), n_bins - 1)
