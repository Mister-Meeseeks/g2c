"""Quality gates for synthetic instruction data (Module 16B).

Generation is the easy half of the synthetic-data recipe; every gate
in this file exists because of something a teacher model actually does
when you ask it for training data:

  * It repeats itself. Ask for 500 instructions and you will not get
    500 ideas — you'll get a smaller set of ideas, rephrased. The
    dedup gate (`ngram_overlap` + `dedupe_pairs`) measures and removes
    this, and the *rate* at which it fires is your first empirical
    look at mode collapse.
  * It drifts off-format. Preambles, follow-up questions, answers
    embedded in the instruction list. The shape gates in
    `validate_pair` catch the mechanical failures.
  * It occasionally degenerates. Stuttered words, echoed
    instructions, blank answers. Cheap gates, real failures.

Self-Instruct (Wang et al., 2022) — the recipe this module
miniaturizes — spent most of its pipeline on exactly these filters,
using ROUGE-L overlap against the growing pool. Our `ngram_overlap`
is the same idea in a form you can compute by hand.

`validate_pair` is implemented — its gates are mechanical checks.
The dedup pieces are scaffolded: they ARE the lesson. Search `# TODO`.
"""
from __future__ import annotations

from collections.abc import Iterable

# A single hand-authored pair from Module 13 looks like:
#     {"user": "What device did the model use?", "assistant": "It used MPS."}
# Synthetic pairs use the same two keys, so Module 13's SFT pipeline
# consumes them unchanged.

MAX_PAIR_CHARS = 400


def validate_pair(pair: object, *, max_chars: int = MAX_PAIR_CHARS) -> list[str]:
    """Mechanical shape gates. Returns the list of failure reasons —
    empty means the pair passed. Implemented for you.

    Gates, in order:
      * must be a dict with non-empty string "user" and "assistant"
      * neither side longer than `max_chars` (a runaway teacher answer
        is a training-data bug, not a bonus)
      * the assistant must not simply echo the instruction
      * no word stuttered three or more times in a row (the cheapest
        reliable degeneracy tell)
    """
    reasons: list[str] = []
    if not isinstance(pair, dict):
        return ["not a dict"]
    for key in ("user", "assistant"):
        value = pair.get(key)
        if not isinstance(value, str) or not value.strip():
            reasons.append(f"missing or empty {key!r}")
    if reasons:
        return reasons

    user = pair["user"].strip()
    assistant = pair["assistant"].strip()
    for key, text in (("user", user), ("assistant", assistant)):
        if len(text) > max_chars:
            reasons.append(f"{key!r} longer than {max_chars} chars")
    if user.casefold() == assistant.casefold():
        reasons.append("assistant echoes the instruction verbatim")
    for key, text in (("user", user), ("assistant", assistant)):
        words = text.casefold().split()
        for i in range(len(words) - 2):
            if words[i] == words[i + 1] == words[i + 2]:
                reasons.append(f"{key!r} stutters {words[i]!r}")
                break
    return reasons


def ngram_overlap(a: str, b: str, *, n: int = 3) -> float:
    """Jaccard similarity of word n-gram sets — the dedup yardstick.

    "How many phrase fragments do these two texts share?" Two
    rephrasings of the same instruction share most of their word
    trigrams even when no full sentence matches; two genuinely
    different instructions share almost none.

    Args:
        a, b: the texts to compare. Case-insensitive.
        n: n-gram width in words. 3 is the classic dedup setting.

    Returns:
        `|grams(a) ∩ grams(b)| / |grams(a) ∪ grams(b)|` in [0, 1].

    Recipe:

        1. Define grams(text):
               words = text.casefold().split()
               if len(words) < n:
                   # Too short for a full n-gram: the whole text is
                   # its one gram, so short texts still compare.
                   return {tuple(words)}
               return {tuple(words[i:i+n]) for i in range(len(words) - n + 1)}

        2. A, B = grams(a), grams(b)

        3. Edge cases first: if both texts are empty, they are
           identical — return 1.0. If exactly one is empty, they
           share nothing — return 0.0.
           (With the definition above, an empty text's gram set is
           {()} — handle the edge on the word lists, not the sets.)

        4. return len(A & B) / len(A | B)

    Sanity values:
        * identical texts -> 1.0 (any n)
        * no shared words -> 0.0
        * "the cat sat on the mat" vs "the cat sat on a hat", n=3:
          shares 2 of 6 distinct trigrams -> 1/3
    """
    # TODO
    raise NotImplementedError


def dedupe_pairs(
    pairs: list[dict],
    *,
    threshold: float = 0.7,
    against: Iterable[str] = (),
    n: int = 3,
) -> list[dict]:
    """Greedy first-come dedup against a growing pool — Self-Instruct's
    core move, in miniature.

    A pair survives if its instruction is sufficiently *novel*: its
    maximum `ngram_overlap` against every instruction already in the
    pool stays below `threshold`. The pool starts as `against` (your
    hand-authored seeds — a synthetic pair that rephrases a seed adds
    nothing) and grows with each accepted pair, so the second copy of
    a repeated idea is measured against the first and rejected.

    Args:
        pairs: candidate pairs, in generation order. Order matters:
            first-come wins, later near-duplicates are dropped.
        threshold: overlap at or above this rejects. 0.7 is a loose
            gate (only near-rephrasings die); the notebook has you
            sweep it.
        against: instruction strings that pre-seed the pool.
        n: forwarded to `ngram_overlap`.

    Returns:
        The surviving pairs, in their original order.

    Recipe:

        1. pool = list(against)
           kept = []

        2. For each pair, in order:
               instruction = pair["user"]
               novel = all(
                   ngram_overlap(instruction, existing, n=n) < threshold
                   for existing in pool
               )
               if novel:
                   kept.append(pair)
                   pool.append(instruction)   # later candidates are
                                              # measured against it

        3. return kept

    The pool-append in step 2 is the whole design: dedup against a
    FIXED set would happily keep fifty copies of the same new idea,
    because none of them resemble a seed.
    """
    # TODO
    raise NotImplementedError
