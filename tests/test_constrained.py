"""Tests for constrained decoding (g2c/sampling/constrained.py, Module 18).

Suggested order to implement & turn green:

1. `allowed_token_mask` (g2c/sampling/constrained.py)
   -> test_mask_allows_exactly_the_grammatical_pieces
   -> test_mask_rejects_empty_pieces
   -> test_mask_is_all_false_on_a_complete_state
   -> test_mask_checks_multicharacter_pieces_as_units

2. `generate_json` (g2c/sampling/constrained.py)
   -> test_greedy_generation_is_forced_grammatical
   -> test_output_always_parses_or_is_a_valid_prefix
   -> test_generation_stops_when_the_root_object_closes
   -> test_warpers_compose_inside_the_allowed_set
   -> test_greedy_mode_still_respects_the_mask
   -> test_vocab_pieces_mismatch_raises

The `JsonPrefixAutomaton` is implemented boilerplate, so its tests
pass from the start and double as a readable spec of the grammar.

The generation tests use a rigged model whose logits always prefer
ungrammatical tokens — if the mask has a hole, the model drives
straight through it.
"""
from __future__ import annotations

import json

import pytest
import torch

from g2c.sampling import (
    JsonPrefixAutomaton,
    allowed_token_mask,
    generate_json,
    vocab_pieces,
)

# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

# Piece order IS the logit order (descending). The '"' piece outranks
# everything so greedy closes strings immediately; 'x' — the piece a
# free-running model would love — ranks dead last but is still the
# only *letter* available, so any hole in the mask surfaces as an 'x'
# where the grammar forbade one.
PIECES = ['"', "{", "}", ":", "a", " ", "1", "x"]
LOGITS = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]


class RiggedLM:
    """Returns the same fixed logits at every position, every step."""

    max_seq_len = 64
    device = torch.device("cpu")

    def __init__(self, logits: list[float] = LOGITS) -> None:
        self._logits = torch.tensor(logits, dtype=torch.float32)

    def __call__(self, ids: torch.Tensor) -> torch.Tensor:
        batch, seq = ids.shape
        return self._logits.view(1, 1, -1).expand(batch, seq, -1)


def continuation_text(full_ids: torch.Tensor, prompt_len: int) -> str:
    return "".join(PIECES[i] for i in full_ids[prompt_len:].tolist())


PROMPT = torch.tensor([0, 1, 2])  # ids are arbitrary; the model ignores them


class StubTokenizer:
    """Just enough tokenizer for vocab_pieces: decode of a one-id list."""

    def decode(self, ids: list[int]) -> str:
        return PIECES[ids[0]]


# ----------------------------------------------------------------------
# Boilerplate: the automaton is a readable spec of the grammar
# ----------------------------------------------------------------------


def test_automaton_accepts_valid_prefixes():
    auto = JsonPrefixAutomaton()
    for prefix in [
        "{",
        '{"',
        '{"name"',
        '{"name":',
        '{"name": "calc',
        '{"name": "calc", "arguments": {"expression": "2 + 2"',
        '{ "a" : 1 ',
        '{"n": -12.5',
        '{"a": {"b": {"c": "d"',
        '{"s": "with \\"escaped\\" quotes',
        " \n {",  # leading whitespace before the root (within the ration)
    ]:
        assert auto.advance(auto.initial(), prefix) is not None, prefix


def test_automaton_rejects_invalid_prefixes():
    auto = JsonPrefixAutomaton()
    for bad in [
        "x",  # root must be an object
        "[",  # arrays are outside the subset
        "{,",
        "{'a'",  # single quotes are not JSON
        '{"a" 1',  # missing colon
        '{"a": }',
        '{"a": .5',  # numbers need a leading digit
        '{"a": -',  # fine as prefix? no -- '-' alone IS a valid prefix
    ][:-1]:  # the last entry is documentation; see the prefix test below
        assert auto.advance(auto.initial(), bad) is None, bad
    # A bare minus IS a valid prefix (a digit can follow)...
    assert auto.advance(auto.initial(), '{"a": -') is not None
    # ...but it can't be closed without one.
    assert auto.advance(auto.initial(), '{"a": -}') is None


def test_automaton_enforces_json_number_rules():
    auto = JsonPrefixAutomaton()
    # Leading zeros are not JSON -- json.loads('{"a": 01}') raises, so
    # the automaton must reject it or the parse guarantee breaks.
    assert auto.advance(auto.initial(), '{"a": 01') is None
    assert auto.advance(auto.initial(), '{"a": 1.}') is None  # dot needs a digit
    assert auto.advance(auto.initial(), '{"a": 1.2.') is None
    for ok in ['{"a": 0}', '{"a": 0.5}', '{"a": -0.5}', '{"a": 10, "b": 2}']:
        state = auto.advance(auto.initial(), ok)
        assert state is not None and auto.is_complete(state), ok


def test_automaton_rations_whitespace():
    auto = JsonPrefixAutomaton()
    # Whitespace between tokens is legal JSON but RATIONED to 3 chars:
    # unbounded runs let a prose model hide in whitespace forever
    # instead of committing to a structural character.
    assert auto.advance(auto.initial(), "   {") is not None
    assert auto.advance(auto.initial(), "    {") is None
    assert auto.advance(auto.initial(), '{"a":\n\n\n\n1}') is None
    assert auto.advance(auto.initial(), '{"a":\n\n\n1}') is not None
    # Inside strings, spaces are content — no ration applies.
    state = auto.advance(auto.initial(), '{"a": "s p a     c e s"}')
    assert state is not None and auto.is_complete(state)


def test_automaton_rejects_raw_control_characters_in_strings():
    auto = JsonPrefixAutomaton()
    assert auto.advance(auto.initial(), '{"a": "line\none') is None
    assert auto.advance(auto.initial(), '{"a": "tab\tone') is None
    # The escaped forms are fine.
    assert auto.advance(auto.initial(), '{"a": "line\\none') is not None


def test_automaton_completion_is_terminal():
    auto = JsonPrefixAutomaton()
    state = auto.advance(auto.initial(), '{"name": "calc"}')
    assert state is not None and auto.is_complete(state)
    # Nothing at all is allowed after the root closes -- not even
    # whitespace. This is what makes constrained generation stop.
    for ch in [" ", "\n", "{", "}", "x"]:
        assert auto.advance(state, ch) is None


def test_automaton_complete_strings_always_parse():
    auto = JsonPrefixAutomaton()
    for text in [
        "{}",
        '{"name": "calc", "arguments": {"expression": "2 + 2"}}',
        '{ "a" : { "b" : -3.25 } }',
        '{"s": "esc \\\\ and \\" done"}',
    ]:
        state = auto.advance(auto.initial(), text)
        assert state is not None and auto.is_complete(state), text
        json.loads(text)  # the guarantee the grammar exists to provide


def test_vocab_pieces_decodes_each_id_once():
    assert vocab_pieces(StubTokenizer(), len(PIECES)) == PIECES


# ----------------------------------------------------------------------
# Step 1: allowed_token_mask
# ----------------------------------------------------------------------


def test_mask_allows_exactly_the_grammatical_pieces():
    auto = JsonPrefixAutomaton()
    mask = allowed_token_mask(auto, auto.initial(), PIECES)
    assert mask.dtype == torch.bool and mask.shape == (len(PIECES),)
    # At the very start only '{' and whitespace are grammatical.
    expected = {PIECES.index("{"), PIECES.index(" ")}
    assert set(torch.nonzero(mask).flatten().tolist()) == expected


def test_mask_rejects_empty_pieces():
    auto = JsonPrefixAutomaton()
    pieces = ["", "{", ""]
    mask = allowed_token_mask(auto, auto.initial(), pieces)
    # Empty pieces advance nothing -- allowing one is an infinite loop.
    assert mask.tolist() == [False, True, False]


def test_mask_is_all_false_on_a_complete_state():
    auto = JsonPrefixAutomaton()
    state = auto.advance(auto.initial(), "{}")
    assert auto.is_complete(state)
    mask = allowed_token_mask(auto, state, PIECES)
    assert not mask.any()


def test_mask_checks_multicharacter_pieces_as_units():
    auto = JsonPrefixAutomaton()
    state = auto.advance(auto.initial(), '{"name"')
    pieces = ['": "', ": ", '"', "}", "xy"]
    mask = allowed_token_mask(auto, state, pieces)
    # After a closed key: ': ' hops to the value slot in one piece;
    # '": "' would need a second quote; '}' needs a value first.
    assert mask.tolist() == [False, True, False, False, False]


# ----------------------------------------------------------------------
# Step 2: generate_json
# ----------------------------------------------------------------------


def test_greedy_generation_is_forced_grammatical():
    # The model's favorite token is '"' and its least favorite is 'x',
    # but at step 0 neither is grammatical -- the mask forces '{'.
    # Hand-tracing the greedy path through the grammar gives exactly:
    out = generate_json(
        RiggedLM(), PROMPT, PIECES, max_new_tokens=50, temperature=0.0
    )
    text = continuation_text(out, len(PROMPT))
    assert text == '{"":""}'
    json.loads(text)


def test_generation_stops_when_the_root_object_closes():
    out = generate_json(
        RiggedLM(), PROMPT, PIECES, max_new_tokens=50, temperature=0.0
    )
    # 7 tokens close the object; the remaining budget must NOT be spent.
    assert out.shape[0] == len(PROMPT) + 7
    assert torch.equal(out[: len(PROMPT)], PROMPT)


def test_output_always_parses_or_is_a_valid_prefix():
    auto = JsonPrefixAutomaton()
    for seed in range(20):
        out = generate_json(
            RiggedLM(),
            PROMPT,
            PIECES,
            max_new_tokens=40,
            temperature=1.5,  # hot -- the model WANTS to misbehave
            generator=torch.Generator().manual_seed(seed),
        )
        text = continuation_text(out, len(PROMPT))
        state = auto.advance(auto.initial(), text)
        assert state is not None, f"seed {seed} left the grammar: {text!r}"
        if auto.is_complete(state):
            json.loads(text)  # finished -> must parse, no exceptions


def test_warpers_compose_inside_the_allowed_set():
    # top_k=2 AFTER the mask keeps the 2 best *grammatical* tokens.
    # Applied before the mask, the 2 best tokens overall ('"', '{')
    # would both be ungrammatical at many states and softmax would NaN.
    for seed in range(10):
        out = generate_json(
            RiggedLM(),
            PROMPT,
            PIECES,
            max_new_tokens=40,
            temperature=1.0,
            top_k=2,
            generator=torch.Generator().manual_seed(seed),
        )
        text = continuation_text(out, len(PROMPT))
        auto = JsonPrefixAutomaton()
        assert auto.advance(auto.initial(), text) is not None


def test_greedy_mode_still_respects_the_mask():
    # Greedy bypasses the warpers -- but the mask is a constraint, not
    # a warper. The unconstrained greedy choice at step 0 would be '"'
    # (the highest logit); the constrained one must be '{'.
    out = generate_json(
        RiggedLM(), PROMPT, PIECES, max_new_tokens=1, temperature=0.0
    )
    assert PIECES[out[-1].item()] == "{"


def test_mask_cache_is_consulted_and_shared():
    # Seeding the cache with an all-False mask for the initial state
    # must halt generation before the first token -- proof the cache
    # is authoritative, not advisory.
    auto = JsonPrefixAutomaton()
    poisoned = {auto.initial(): torch.zeros(len(PIECES), dtype=torch.bool)}
    out = generate_json(
        RiggedLM(), PROMPT, PIECES,
        max_new_tokens=10, temperature=0.0, mask_cache=poisoned,
    )
    assert out.shape[0] == len(PROMPT)  # nothing generated

    # A shared cache across calls must not change outputs -- the mask
    # is a pure function of the automaton state -- and must fill up.
    cache: dict = {}
    first = generate_json(
        RiggedLM(), PROMPT, PIECES,
        max_new_tokens=50, temperature=0.0, mask_cache=cache,
    )
    populated = len(cache)
    second = generate_json(
        RiggedLM(), PROMPT, PIECES,
        max_new_tokens=50, temperature=0.0, mask_cache=cache,
    )
    assert torch.equal(first, second)
    assert populated > 0 and len(cache) == populated


def test_vocab_pieces_mismatch_raises():
    with pytest.raises(ValueError, match="pieces"):
        generate_json(
            RiggedLM(), PROMPT, PIECES[:-1], max_new_tokens=5, temperature=0.0
        )


def test_prompt_validation():
    with pytest.raises(ValueError):
        generate_json(
            RiggedLM(),
            torch.zeros(0, dtype=torch.long),
            PIECES,
            max_new_tokens=5,
        )
    with pytest.raises(ValueError):
        generate_json(
            RiggedLM(), PROMPT, PIECES, max_new_tokens=5, temperature=-1.0
        )


def test_budget_exhaustion_yields_a_valid_prefix():
    # Two tokens is not enough to close an object -- the result must be
    # truncated-but-grammatical, never malformed.
    out = generate_json(
        RiggedLM(), PROMPT, PIECES, max_new_tokens=2, temperature=0.0
    )
    text = continuation_text(out, len(PROMPT))
    auto = JsonPrefixAutomaton()
    state = auto.advance(auto.initial(), text)
    assert state is not None
    assert not auto.is_complete(state)
