# Hand-maintained solutions mirror — canonical, lives in `main` (see AGENTS.md).
"""Solutions for g2c.eval.match pedagogical scaffolds.

Patched onto the scaffold targets by g2c.solutions.apply().
"""
from __future__ import annotations

import re
import string

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def exact_match(prediction: str, references: list[str]) -> bool:
    """True iff `prediction` equals any element of `references` byte-for-byte.

    Args:
        prediction: model output string.
        references: list of acceptable answers. Must be non-empty.

    Returns:
        True if `prediction == ref` for at least one ref in references.

    Recipe:
        Validate `references` is non-empty (else raise ValueError —
        an empty reference list is a programming bug, not a "no
        match" answer). Then iterate and return on first hit.

    Notes:
      * No normalization at all — "Madrid" ≠ "madrid" ≠ "Madrid.".
        Use this when YOU control both sides of the comparison
        (e.g. the references are tokens from a closed vocabulary).
      * For factual QA over hand-authored references, `normalized_match`
        is almost always more useful — `exact_match` is too brittle
        when small models produce trailing whitespace, period, or
        capitalization variations.
    """
    if len(references) == 0:
        raise ValueError("references must be non-empty")
    for ref in references:
        if prediction == ref:
            return True
    return False


def normalized_match(prediction: str, references: list[str]) -> bool:
    """True iff `prediction` matches any reference after normalization.

    Normalization:
      1. Lowercase.
      2. Strip leading/trailing whitespace.
      3. Strip ALL punctuation (everything in `string.punctuation`).
      4. Collapse internal whitespace to single spaces.

    Args:
        prediction: model output string.
        references: list of acceptable answers. Must be non-empty.

    Returns:
        True iff normalize(prediction) == normalize(ref) for at least
        one reference.

    Recipe:
        Define a small `_normalize(s: str) -> str` that does the four
        steps above, then return True if `_normalize(prediction)`
        equals any `_normalize(ref)`.

        For step 3, the standard idiom is:

            s = s.translate(str.maketrans("", "", string.punctuation))

        For step 4:

            s = " ".join(s.split())

    Notes:
      * This is a deliberately narrow normalization. It will NOT do
        article stripping ("the Madrid" vs "Madrid") or stemming
        ("walked" vs "walks") — those require language-specific logic
        and aren't portable. SQuAD's official `evaluate_v2.py` does
        article stripping; we don't, to keep the matcher portable.
      * Apply this same normalize to BOTH prediction and reference. A
        common bug: normalize prediction but not the reference, then
        compare — never matches because the references have
        capitalization the prediction does not.

    Worked example:
        normalize("Madrid.")        == "madrid"
        normalize("  MADRID,  ")    == "madrid"
        normalize("Mad rid")        == "mad rid"   (NOT a match)
    """
    if len(references) == 0:
        raise ValueError("references must be non-empty")
    translator = str.maketrans("", "", string.punctuation)

    def normalize(text: str) -> str:
        text = text.lower().strip()
        text = text.translate(translator)
        return " ".join(text.split())

    normalized_prediction = normalize(prediction)
    for ref in references:
        if normalized_prediction == normalize(ref):
            return True
    return False


def contains_match(prediction: str, references: list[str]) -> bool:
    """True iff any reference appears as a (case-insensitive) substring.

    Args:
        prediction: model output string.
        references: list of phrases to search for. Must be non-empty.

    Returns:
        True iff `ref.lower() in prediction.lower()` for at least one
        ref.

    Recipe:
        Lowercase prediction once; check each lowercased reference
        against it via `in`.

    Notes:
      * The most permissive of the four matchers. Use when the model
        has been asked to produce a long-form response and you want
        to verify a specific fact or keyword is mentioned anywhere.
      * Beware of accidental substrings: searching for "no" matches
        any string containing "Norway", "snow", "annoy", etc. If
        you're checking refusals, anchor the reference: " no, " or
        "I cannot" rather than just "no".
      * Substring matching does NOT respect word boundaries. If you
        need word-level matching, build a small regex matcher around
        `r"\\b" + re.escape(ref) + r"\\b"` instead.
    """
    if len(references) == 0:
        raise ValueError("references must be non-empty")
    for ref in references:
        if ref.lower() in prediction.lower():
            return True
    return False


def numeric_match(
    prediction: str,
    references: list[str],
    *,
    tolerance: float = 0.0,
) -> bool:
    """True iff the first number in `prediction` matches a number in any
    reference, within `tolerance`.

    Args:
        prediction: model output string. The first match of
            `-?\\d+(\\.\\d+)?` is taken as "the model's answer".
        references: list of acceptable answers. Each is searched for
            its first number; that number is what we compare against.
        tolerance: absolute tolerance — `abs(pred_num - ref_num) <= tolerance`.
            Use `0.0` for exact integer match; use a small positive
            number for floating-point answers ("3.14" vs "3.1416").

    Returns:
        True iff a numeric value can be extracted from `prediction`
        AND from at least one reference, AND the values are within
        `tolerance`. False otherwise — including when no number can
        be extracted from `prediction` at all (the model didn't
        produce a number, so by definition it doesn't match).

    Recipe:
        1. pred_num = first match of _NUMBER_RE in prediction
           (parsed as float). If no match: return False.

        2. For each reference:
              ref_num = first match of _NUMBER_RE in ref (as float).
              if no match: skip this reference.
              if abs(pred_num - ref_num) <= tolerance: return True.

        3. Return False.

    Why "first number, not all numbers": small LMs often emit explanations
    like "Half of 100 is 50, so the answer is 50." If we compared all
    numbers, we'd accidentally match the irrelevant "100". Taking the
    FIRST number is a reasonable heuristic; for arithmetic prompts
    where the model says only "50<|end|>" it's exact.

    Notes:
      * Comma-separated thousands ("1,000") are NOT handled — the regex
        stops at the comma. If your references include "1,000", strip
        commas first (or use a more elaborate regex). Production
        eval frameworks like lm-eval-harness handle this; we don't.
      * Scientific notation ("1e5") is NOT handled by the regex. Same
        comment.
      * Negative numbers ARE handled — the leading `-` is part of the
        regex. So "-273" matches "-273" but not "273".

    Worked examples:
        numeric_match("50",          ["50"])         == True
        numeric_match("50<|end|>",   ["50."])        == True
        numeric_match("3.14",        ["3.1416"], tolerance=0.01) == True
        numeric_match("3.14",        ["3.1416"], tolerance=0.0)  == False
        numeric_match("the answer is 50", ["50"])    == True
        numeric_match("Half of 100 is 50", ["50"])   == False  # first num is 100
        numeric_match("no number here",   ["50"])    == False
    """
    if len(references) == 0:
        raise ValueError("references must be non-empty")
    pred_match = _NUMBER_RE.search(prediction)
    if not pred_match:
        return False
    pred_num = float(pred_match.group())
    for ref in references:
        ref_match = _NUMBER_RE.search(ref)
        if not ref_match:
            continue
        ref_num = float(ref_match.group())
        if abs(pred_num - ref_num) <= tolerance:
            return True
    return False
