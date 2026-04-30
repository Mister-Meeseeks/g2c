"""Multiple-choice scoring and the multiple-choice eval harness.

The closed-set evaluation primitive: given a prompt and N candidate
continuations, score each by sequence-log-probability and pick the
highest. The "answer" is the argmax; the "confidence" is the softmax
over option log-probs.

Two pieces:

  * `score_multiple_choice(model, tokenizer, example, *,
    length_normalize=False) -> EvalResult`. Scores ONE example. The
    pedagogical core — the math that turns "the model assigns logp
    L_i to option i" into "the model picks option k with probability
    p_k". Scaffolded.

  * `run_multiple_choice_eval(model, tokenizer, examples, *,
    length_normalize=False, task_name="multiple_choice", compute_ece=True,
    ece_n_bins=10) -> EvalReport`. Iterates `score_multiple_choice`
    over a dataset, aggregates correctness + confidence + ECE into
    an `EvalReport`. Implemented — it's pure plumbing once
    `score_multiple_choice` and `expected_calibration_error` are.

Two design points worth flagging:

  1. **Length normalization is OFF by default.** Raw sequence log-prob
     advantages SHORTER continuations (more positive — fewer tokens
     to multiply tiny probabilities). For options of similar token-
     length this is fine. For options that differ wildly in length
     ("Yes." vs "Yes, that is a reasonable interpretation."), the raw
     score is misleading. Pass `length_normalize=True` to divide the
     sum by token count — then the score is the per-token mean log-
     prob, which is roughly length-invariant.

     The default is OFF because: at this scale, hand-authored MC
     options are usually similar-length, and the raw score is more
     interpretable as "joint log-probability of this completion." If
     you're using HuggingFace-style benchmarks (MMLU, ARC) where
     options vary in length, turn it on.

  2. **The harness re-tokenizes the prompt for every option.** No
     KV-cache, no shared-prefix optimization. At our scale this is
     fine — a 4-option example with a 50-token prompt and 5-token
     continuations is ~220 tokens of forward work, which is
     milliseconds on MPS for a 20M-param model. At production scale
     (MMLU on a 70B model), the cache-and-share optimization is
     essential — but it's an optimization, not the lesson.
"""
from __future__ import annotations

from .calibration import expected_calibration_error
from .data import EvalReport, EvalResult, MultipleChoiceExample
from .logprob import continuation_logprob


def score_multiple_choice(
    model,
    tokenizer,
    example: MultipleChoiceExample,
    *,
    length_normalize: bool = False,
) -> EvalResult:
    """Score one multiple-choice example.

    Args:
        model: a model with `forward(x: (1, T)) -> (1, T, V)`. Will be
            forwarded once per choice (so N forwards for N choices).
        tokenizer: a tokenizer with `encode(s: str) -> list[int]`.
        example: the MultipleChoiceExample.
        length_normalize: if True, divide each option's sum_logp by
            its token count before comparing. Default False.

    Returns:
        EvalResult with:
          * `correct`:   True iff predicted index == example.answer_idx.
          * `prediction`: predicted option index (int).
          * `confidence`: softmax probability assigned to the predicted
                          option, computed over the option scores
                          (per-option log-prob if not length-normalized;
                          per-token mean log-prob if length-normalized).
          * `metadata`:  dict with:
                'option_logps':   list[float] — raw sum_logp per option
                'option_lengths': list[int]   — token count per option
                'option_scores':  list[float] — the values used for
                                                argmax (per-token if
                                                length_normalize=True,
                                                else same as option_logps)
                'gold_idx':       int — the gold answer index (so the
                                        result is self-contained)

    Recipe:
        1. # For each choice, compute sum_logp and token count.
           logps:   list[float] = []
           lengths: list[int]   = []
           for choice in example.choices:
               sum_lp, n = continuation_logprob(
                   model, tokenizer, example.prompt, choice
               )
               logps.append(sum_lp)
               lengths.append(n)

        2. # Score each option: per-token mean if length_normalize,
           # else raw sum.
           if length_normalize:
               scores = [lp / max(n, 1) for lp, n in zip(logps, lengths)]
           else:
               scores = list(logps)

        3. # Predicted index = argmax of scores.
           pred_idx = max(range(len(scores)), key=lambda i: scores[i])

        4. # Confidence = softmax(scores)[pred_idx].
           # Subtract max for numerical stability.
           max_s = max(scores)
           import math
           exps = [math.exp(s - max_s) for s in scores]
           total = sum(exps)
           probs = [e / total for e in exps]
           confidence = probs[pred_idx]

        5. # Build EvalResult.
           return EvalResult(
               correct=(pred_idx == example.answer_idx),
               prediction=pred_idx,
               confidence=confidence,
               metadata={
                   'option_logps':   logps,
                   'option_lengths': lengths,
                   'option_scores':  scores,
                   'gold_idx':       example.answer_idx,
               },
           )

    Sanity values:
      * Single-choice example (only one option): the argmax is trivially
        0, confidence is 1.0 (softmax over a singleton is 1). The
        `MultipleChoiceExample` constructor rejects `len(choices) < 2`,
        so this case shouldn't arise in practice — but the function's
        math is well-defined for it.

      * Two identical options: both score the same, argmax is 0
        (Python `max` returns the FIRST max), confidence is 0.5.

      * Uniform-logits model: every option scores
        `-len(choice_tokens) * log(V)`. With `length_normalize=True`,
        every option scores `-log(V)` regardless of length — the model
        is indifferent, and confidence collapses toward `1 / N`.
    """
    # TODO
    raise NotImplementedError


def run_multiple_choice_eval(
    model,
    tokenizer,
    examples: list[MultipleChoiceExample],
    *,
    length_normalize: bool = False,
    task_name: str = "multiple_choice",
    compute_ece: bool = True,
    ece_n_bins: int = 10,
) -> EvalReport:
    """Score every example in `examples` and aggregate into an EvalReport.

    Args:
        model: as in `score_multiple_choice`.
        tokenizer: as in `score_multiple_choice`.
        examples: list of MultipleChoiceExample. Must be non-empty.
        length_normalize: as in `score_multiple_choice`.
        task_name: label for the EvalReport.
        compute_ece: if True, compute expected calibration error
            from the per-example confidences. Set False to bypass
            ECE if `expected_calibration_error` isn't yet implemented.
        ece_n_bins: number of bins for ECE.

    Returns:
        EvalReport with `n`, `accuracy`, `mean_confidence`, `ece`, and
        the full per-example results list.

    This function is implemented — it iterates `score_multiple_choice`
    and aggregates. The pedagogical content is in the per-example
    scorer; aggregation is bookkeeping.
    """
    if len(examples) == 0:
        raise ValueError(
            "run_multiple_choice_eval needs at least one example"
        )

    results: list[EvalResult] = []
    for ex in examples:
        results.append(
            score_multiple_choice(
                model, tokenizer, ex, length_normalize=length_normalize
            )
        )

    n = len(results)
    accuracy = sum(1 for r in results if r.correct) / n

    confidences_with = [r.confidence for r in results if r.confidence is not None]
    if len(confidences_with) > 0:
        mean_confidence = sum(confidences_with) / len(confidences_with)
    else:
        mean_confidence = None

    if compute_ece and len(confidences_with) == n:
        # All results have a confidence — compute ECE.
        ece: float | None = expected_calibration_error(
            [r.confidence for r in results],
            [r.correct for r in results],
            n_bins=ece_n_bins,
        )
    else:
        ece = None

    return EvalReport(
        task_name=task_name,
        n=n,
        accuracy=accuracy,
        mean_confidence=mean_confidence,
        ece=ece,
        results=results,
    )
