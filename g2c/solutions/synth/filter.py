# Hand-maintained solutions mirror — canonical, lives in `main` (see AGENTS.md).
"""Solutions for g2c.synth.filter pedagogical scaffolds.

Patched onto the scaffold targets by g2c.solutions.apply().
"""
from __future__ import annotations

from collections.abc import Iterable


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
    words_a = a.casefold().split()
    words_b = b.casefold().split()
    if not words_a and not words_b:
        return 1.0
    if not words_a or not words_b:
        return 0.0

    def grams(words: list[str]) -> set[tuple[str, ...]]:
        if len(words) < n:
            return {tuple(words)}
        return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}

    set_a, set_b = grams(words_a), grams(words_b)
    return len(set_a & set_b) / len(set_a | set_b)


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

    (See the scaffold in g2c/synth/filter.py for the full argument
    documentation and recipe.)
    """
    pool = list(against)
    kept: list[dict] = []
    for pair in pairs:
        instruction = pair["user"]
        novel = all(
            ngram_overlap(instruction, existing, n=n) < threshold
            for existing in pool
        )
        if novel:
            kept.append(pair)
            pool.append(instruction)  # later candidates measured against it
    return kept
