"""Tests for Module 16B — synthetic instruction data (g2c/synth/).

Suggested order to implement & turn green:

1. `ngram_overlap` (g2c/synth/filter.py)
   -> test_overlap_identical_texts
   -> test_overlap_disjoint_texts
   -> test_overlap_known_value
   -> test_overlap_short_texts_and_edges
   -> test_overlap_is_symmetric_and_case_insensitive

2. `dedupe_pairs` (g2c/synth/filter.py)
   -> test_dedupe_drops_exact_and_near_duplicates
   -> test_dedupe_pool_grows_as_it_accepts
   -> test_dedupe_respects_the_seed_pool
   -> test_dedupe_threshold_semantics
   -> test_synthesize_dataset_funnel   (the full loop needs dedup)

`validate_pair`, the prompt builders, the parser, and the
`synthesize_dataset` orchestration are implemented boilerplate, so
those tests pass from the start.

The generation tests run against a `FakeTeacher` that misbehaves the
way real teachers do: it repeats itself, pads its lists with
commentary, and occasionally degenerates — the funnel has to catch
all of it without a real backend or a network.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from g2c.synth import (
    build_instruction_prompt,
    build_response_prompt,
    dedupe_pairs,
    ngram_overlap,
    parse_numbered_list,
    synthesize_dataset,
    validate_pair,
)

# ----------------------------------------------------------------------
# Boilerplate: validate_pair
# ----------------------------------------------------------------------


def _pair(user="What color is the sky?", assistant="Blue."):
    return {"user": user, "assistant": assistant}


def test_validate_pair_accepts_a_clean_pair():
    assert validate_pair(_pair()) == []


def test_validate_pair_rejects_shape_failures():
    assert validate_pair("not a dict") == ["not a dict"]
    assert validate_pair({"user": "hi"})  # missing assistant
    assert validate_pair(_pair(assistant="   "))  # blank
    assert validate_pair(_pair(user=""))


def test_validate_pair_rejects_runaway_length():
    assert validate_pair(_pair(assistant="x" * 500))
    assert validate_pair(_pair(assistant="x" * 500), max_chars=1000) == []


def test_validate_pair_rejects_echo_and_stutter():
    assert validate_pair(_pair(user="Say hi.", assistant="say hi."))
    assert validate_pair(_pair(assistant="it is the the the answer"))
    # Two in a row is emphasis, not degeneracy.
    assert validate_pair(_pair(assistant="it is very very blue")) == []


# ----------------------------------------------------------------------
# Boilerplate: prompts and parsing
# ----------------------------------------------------------------------


def test_instruction_prompt_shows_numbered_examples():
    prompt = build_instruction_prompt(["Alpha?", "Beta?"], count=5)
    assert "1. Alpha?" in prompt and "2. Beta?" in prompt
    assert "5 NEW instructions" in prompt


def test_response_prompt_embeds_the_instruction():
    assert "What is 2+2?" in build_response_prompt("What is 2+2?")


def test_parse_numbered_list_is_permissive():
    text = (
        "Sure! Here are some instructions:\n"
        "1. Name the largest planet.\n"
        "2)  Rewrite 'hello' in uppercase. \n"
        "some stray commentary\n"
        '3. "Count the vowels in banana."\n'
        "\n"
        "Hope these help!"
    )
    assert parse_numbered_list(text) == [
        "Name the largest planet.",
        "Rewrite 'hello' in uppercase.",
        "Count the vowels in banana.",
    ]
    assert parse_numbered_list("no list here") == []


# ----------------------------------------------------------------------
# Step 1: ngram_overlap
# ----------------------------------------------------------------------


def test_overlap_identical_texts():
    text = "name the largest planet in the solar system"
    assert ngram_overlap(text, text) == pytest.approx(1.0)
    assert ngram_overlap(text, text, n=2) == pytest.approx(1.0)


def test_overlap_disjoint_texts():
    assert ngram_overlap("alpha beta gamma delta", "one two three four") == 0.0


def test_overlap_known_value():
    a = "the cat sat on the mat"
    b = "the cat sat on a hat"
    # trigrams: 2 shared of 6 distinct -> 1/3
    assert ngram_overlap(a, b, n=3) == pytest.approx(1 / 3)


def test_overlap_short_texts_and_edges():
    # Shorter than n: the whole text is one gram.
    assert ngram_overlap("hello", "hello", n=3) == pytest.approx(1.0)
    assert ngram_overlap("hello", "goodbye", n=3) == 0.0
    assert ngram_overlap("", "", n=3) == pytest.approx(1.0)
    assert ngram_overlap("", "hello", n=3) == 0.0


def test_overlap_is_symmetric_and_case_insensitive():
    a = "Rewrite this sentence in the past tense"
    b = "rewrite THIS sentence in the future tense"
    assert ngram_overlap(a, b) == pytest.approx(ngram_overlap(b, a))
    assert ngram_overlap("Hello World", "hello world", n=2) == pytest.approx(1.0)


# ----------------------------------------------------------------------
# Step 2: dedupe_pairs
# ----------------------------------------------------------------------


def test_dedupe_drops_exact_and_near_duplicates():
    pairs = [
        _pair(user="Name the largest planet in the solar system."),
        _pair(user="Name the largest planet in the solar system."),  # exact
        _pair(user="Name the largest planet in our solar system."),  # near
        _pair(user="Translate the word 'cat' into French."),
    ]
    # The near-dup shares exactly 1/3 of its trigrams with the first
    # entry -- a 0.3 threshold catches it; the unrelated pair sails.
    kept = dedupe_pairs(pairs, threshold=0.3)
    assert [p["user"] for p in kept] == [
        "Name the largest planet in the solar system.",
        "Translate the word 'cat' into French.",
    ]


def test_dedupe_pool_grows_as_it_accepts():
    # Neither candidate resembles a seed — dedup against a FIXED pool
    # would keep both. The growing pool kills the second.
    pairs = [
        _pair(user="List three primary colors used in painting."),
        _pair(user="List three primary colors used in painting please."),
    ]
    kept = dedupe_pairs(pairs, threshold=0.5, against=["What is 2+2?"])
    assert len(kept) == 1


def test_dedupe_respects_the_seed_pool():
    seeds = ["Name the largest planet in the solar system."]
    pairs = [_pair(user="Name the largest planet in the solar system!")]
    assert dedupe_pairs(pairs, threshold=0.5, against=seeds) == []
    # The same pair sails through when the seeds don't cover it.
    assert len(dedupe_pairs(pairs, threshold=0.5, against=["Unrelated?"])) == 1


def test_dedupe_threshold_semantics():
    pairs = [
        _pair(user="the cat sat on the mat"),
        _pair(user="the cat sat on a hat"),  # overlap exactly 1/3
    ]
    assert len(dedupe_pairs(pairs, threshold=0.34)) == 2  # below -> novel
    assert len(dedupe_pairs(pairs, threshold=1 / 3)) == 1  # at -> duplicate
    assert dedupe_pairs([], threshold=0.5) == []


# ----------------------------------------------------------------------
# The full loop, against a misbehaving fake teacher
# ----------------------------------------------------------------------


@dataclass
class _Result:
    completion: str


class FakeTeacher:
    """Backend double that misbehaves like a real teacher model.

    Proposal calls cycle through canned batches containing repeats,
    commentary, and junk. Response calls answer cleanly except for
    marked instructions, which get degenerate answers the pair gate
    must catch.
    """

    BATCHES = [
        "Here you go!\n"
        "1. Name the largest planet in the solar system.\n"
        "2. Translate the word 'cat' into French.\n"
        "3. Name the largest planet in the solar system.\n"  # in-batch dup
        "4. STUTTER: repeat something\n",
        "1. Name the largest planet in our solar system.\n"  # near-dup of #1
        "2. Count the vowels in the word banana.\n"
        "3. \n"  # blank -> parser drops it
        "4. Rewrite the sentence 'I run fast' in the past tense.\n",
        "1. Give one use for a paperclip.\n"
        "2. What sound does a cow make?\n"
        "3. Spell the word 'rhythm' backwards.\n"
        "4. Give exactly one use for a paperclip.\n",  # near-dup of #1
    ]

    def __init__(self) -> None:
        self.proposal_calls = 0
        self.response_calls = 0

    def complete(self, prompt: str, *, max_new_tokens=128, temperature=1.0,
                 top_k=None, top_p=None) -> _Result:
        if "numbered list only" in prompt:
            batch = self.BATCHES[self.proposal_calls % len(self.BATCHES)]
            self.proposal_calls += 1
            return _Result(batch)
        self.response_calls += 1
        instruction = prompt.split("Instruction:")[1].split("Answer:")[0].strip()
        if instruction.startswith("STUTTER"):
            return _Result("yes yes yes yes yes")
        return _Result(f"A concise answer about: {instruction.rstrip('.?!').casefold()}.")


SEEDS = [
    _pair(user="What is the capital of France?", assistant="Paris."),
    _pair(user="What is 2+2?", assistant="4."),
]


def test_synthesize_dataset_funnel():
    teacher = FakeTeacher()
    pairs, funnel = synthesize_dataset(
        teacher, SEEDS, target=6, batch_count=4, threshold=0.5, seed=0
    )
    # Every accepted pair is clean and novel.
    assert len(pairs) == 6
    assert all(validate_pair(p) == [] for p in pairs)
    users = [p["user"] for p in pairs]
    assert len(set(users)) == len(users)
    # The teacher's repeats and degenerates were caught, and counted.
    assert funnel["accepted"] == 6
    assert funnel["duplicate"] >= 2  # in-batch dup + cross-batch near-dups
    assert funnel["bad_pair"] >= 1  # the stuttered answer
    assert funnel["proposed"] >= funnel["accepted"]
    assert teacher.response_calls >= funnel["accepted"]


def test_synthesize_dataset_stops_on_a_stuck_teacher():
    class StuckTeacher(FakeTeacher):
        BATCHES = ["1. Name the largest planet in the solar system.\n"]

    pairs, funnel = synthesize_dataset(
        StuckTeacher(), SEEDS, target=10, batch_count=4, threshold=0.5,
        max_rounds=5, seed=0,
    )
    # One novel idea, endlessly rephrased: the round budget ends the
    # loop instead of spinning forever.
    assert len(pairs) == 1
    assert funnel["rounds"] == 5
    assert funnel["duplicate"] >= 4
