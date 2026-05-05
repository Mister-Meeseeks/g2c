"""Tests for Module 04: BPE tokenizer.

Suggested order to implement & turn green:

  1. _get_pair_counts        → test_pair_counts_*
  2. _merge                  → test_merge_*
  3. train_step              → test_train_step_* and test_train_*
  4. encode                  → test_encode_*  (some need decode too for round-trip)
  5. decode                  → test_decode_*
  6. Round-trip + properties → test_round_trip_*

Construction, persistence, fast-path, and train-validation tests pass from
the start because that boilerplate is implemented. The BPE algorithm tests fail
with `NotImplementedError` until you fill in the TODOs.
"""
from __future__ import annotations

import pytest

from g2c.tokenizer import BPETokenizer

# ----------------------------------------------------------------------
# Construction (boilerplate — passes from the start)
# ----------------------------------------------------------------------

def test_initial_vocab_size():
    tok = BPETokenizer()
    assert len(tok.vocab) == 256


def test_initial_merges_empty():
    assert BPETokenizer().merges == {}


def test_initial_vocab_byte_identity():
    tok = BPETokenizer()
    for i in range(256):
        assert tok.vocab[i] == bytes([i])


# ----------------------------------------------------------------------
# Persistence (boilerplate — passes from the start)
# ----------------------------------------------------------------------

def test_to_dict_contains_ordered_merges():
    tok = BPETokenizer()
    tok.merges[(97, 98)] = 256
    tok.vocab[256] = b"ab"
    tok.merges[(256, 99)] = 257
    tok.vocab[257] = b"abc"

    payload = tok.to_dict()

    assert payload["format"] == "g2c.bpe"
    assert payload["version"] == 1
    assert payload["merges"] == [[97, 98, 256], [256, 99, 257]]


def test_save_load_round_trips_learned_merges(tmp_path):
    tok = BPETokenizer()
    tok.merges[(97, 98)] = 256
    tok.vocab[256] = b"ab"
    tok.merges[(256, 99)] = 257
    tok.vocab[257] = b"abc"

    path = tmp_path / "tokenizer.json"
    tok.save(path)
    loaded = BPETokenizer.load(path)

    assert loaded.merges == tok.merges
    assert loaded.vocab == tok.vocab


def test_from_dict_rejects_wrong_format():
    with pytest.raises(ValueError, match="BPE tokenizer"):
        BPETokenizer.from_dict({"format": "other", "version": 1, "merges": []})


# ----------------------------------------------------------------------
# Fast path (infrastructure — passes from the start)
# ----------------------------------------------------------------------

def test_encode_fast_no_merges_is_utf8_bytes():
    assert BPETokenizer().encode_fast("hi") == [104, 105]


def test_encode_fast_applies_learned_merges_in_priority_order():
    tok = BPETokenizer()
    tok.merges[(97, 98)] = 256
    tok.vocab[256] = b"ab"
    tok.merges[(256, 99)] = 257
    tok.vocab[257] = b"abc"

    assert tok.encode_fast("abcabc") == [257, 257]


def test_encode_fast_respects_non_overlapping_merges():
    tok = BPETokenizer()
    tok.merges[(97, 97)] = 256
    tok.vocab[256] = b"aa"

    assert tok.encode_fast("aaa") == [256, 97]


def test_train_fast_populates_course_merge_tables():
    text = "ababab abcabc " * 4
    tok = BPETokenizer()

    ids = tok.train_fast(text, vocab_size=270, show_progress=False)

    assert len(tok.vocab) > 256
    assert len(tok.vocab) == 256 + len(tok.merges)
    assert ids == tok.encode_fast(text)
    for (left_id, right_id), new_id in tok.merges.items():
        assert tok.vocab[new_id] == tok.vocab[left_id] + tok.vocab[right_id]


# ----------------------------------------------------------------------
# _get_pair_counts
# ----------------------------------------------------------------------

def test_pair_counts_empty():
    assert BPETokenizer._get_pair_counts([]) == {}


def test_pair_counts_single_id_no_pairs():
    assert BPETokenizer._get_pair_counts([42]) == {}


def test_pair_counts_two_ids_one_pair():
    assert BPETokenizer._get_pair_counts([1, 2]) == {(1, 2): 1}


def test_pair_counts_repeated_pair():
    assert BPETokenizer._get_pair_counts([1, 2, 1, 2]) == {(1, 2): 2, (2, 1): 1}


def test_pair_counts_distinct_pairs():
    counts = BPETokenizer._get_pair_counts([1, 2, 3, 4])
    assert counts == {(1, 2): 1, (2, 3): 1, (3, 4): 1}


# ----------------------------------------------------------------------
# _merge
# ----------------------------------------------------------------------

def test_merge_no_match_unchanged():
    assert BPETokenizer._merge([1, 2, 3], (4, 5), 99) == [1, 2, 3]


def test_merge_single_occurrence():
    assert BPETokenizer._merge([1, 2, 3], (1, 2), 99) == [99, 3]


def test_merge_multiple_non_overlapping():
    assert BPETokenizer._merge([1, 2, 3, 1, 2], (1, 2), 99) == [99, 3, 99]


def test_merge_at_end():
    assert BPETokenizer._merge([3, 1, 2], (1, 2), 99) == [3, 99]


def test_merge_overlapping_left_to_right():
    """Merging (1, 1) in [1, 1, 1] → [99, 1] — the middle `1` is consumed
    by the first match, not available for a second match."""
    assert BPETokenizer._merge([1, 1, 1], (1, 1), 99) == [99, 1]


def test_merge_chain_of_pairs():
    """Every adjacent pair in [1,2,1,2,1,2] is (1,2). All three should merge."""
    assert BPETokenizer._merge([1, 2, 1, 2, 1, 2], (1, 2), 99) == [99, 99, 99]


def test_merge_empty():
    assert BPETokenizer._merge([], (1, 2), 99) == []


# ----------------------------------------------------------------------
# train_step
# ----------------------------------------------------------------------

def test_train_step_no_pairs_returns_none():
    tok = BPETokenizer()
    assert tok.train_step([65], new_id=256) is None


def test_train_step_learns_most_frequent_pair():
    tok = BPETokenizer()
    result = tok.train_step([1, 2, 1, 2, 3], new_id=256)
    assert result == ([256, 256, 3], (1, 2), 2)
    assert tok.merges[(1, 2)] == 256


def test_train_step_updates_vocab_bytes():
    tok = BPETokenizer()
    result = tok.train_step([97, 98, 97, 98], new_id=256)
    assert result == ([256, 256], (97, 98), 2)
    assert tok.vocab[256] == b"ab"


# ----------------------------------------------------------------------
# train
# ----------------------------------------------------------------------

def test_train_grows_vocab_to_target():
    tok = BPETokenizer()
    tok.train("hello world " * 10, vocab_size=260)
    assert len(tok.vocab) == 260
    assert len(tok.merges) == 4


def test_train_no_merges_when_target_is_base_size():
    tok = BPETokenizer()
    tok.train("hello", vocab_size=256)
    assert len(tok.merges) == 0
    assert len(tok.vocab) == 256


def test_train_below_base_size_raises():
    tok = BPETokenizer()
    with pytest.raises(ValueError):
        tok.train("hello", vocab_size=200)


def test_train_assigns_sequential_ids():
    """Learned merge IDs should be 256, 257, 258, ... in order of learning."""
    tok = BPETokenizer()
    tok.train("the quick brown fox " * 5, vocab_size=261)
    learned_ids = sorted(tok.merges.values())
    assert learned_ids == [256, 257, 258, 259, 260]


def test_train_vocab_bytes_concatenation():
    """Each learned vocab entry's bytes should be the concatenation of its
    two parents' bytes."""
    tok = BPETokenizer()
    tok.train("ababab abcabc", vocab_size=260)
    for (a, b), new_id in tok.merges.items():
        assert tok.vocab[new_id] == tok.vocab[a] + tok.vocab[b]


def test_train_progress_callback_reports_final_state():
    tok = BPETokenizer()
    reports = []

    tok.train(
        "the the the " * 5,
        vocab_size=260,
        progress_callback=reports.append,
        progress_every=2,
    )

    assert reports
    final = reports[-1]
    assert final["done"] is True
    assert final["vocab_size"] == 260
    assert final["target_vocab_size"] == 260
    assert final["merges"] == 4
    assert final["target_merges"] == 4
    assert final["steps"] == 4
    assert final["tokens"] < len("the the the " * 5)
    assert isinstance(final["last_token_id"], int)
    assert isinstance(final["last_token_bytes"], bytes)
    assert isinstance(final["last_token_text"], str)
    assert isinstance(final["last_token_repr"], str)
    assert isinstance(final["elapsed_seconds"], float)


# ----------------------------------------------------------------------
# encode
# ----------------------------------------------------------------------

def test_encode_empty():
    assert BPETokenizer().encode("") == []


def test_encode_no_merges_is_utf8_bytes():
    """With no merges learned, encoding falls back to raw UTF-8 bytes."""
    tok = BPETokenizer()
    assert tok.encode("hi") == [104, 105]


def test_encode_uses_learned_merges():
    tok = BPETokenizer()
    tok.train("abcabcabc", vocab_size=260)
    encoded = tok.encode("abcabc")
    raw = list(b"abcabc")
    assert len(encoded) < len(raw)


# ----------------------------------------------------------------------
# decode
# ----------------------------------------------------------------------

def test_decode_empty():
    assert BPETokenizer().decode([]) == ""


def test_decode_single_byte():
    assert BPETokenizer().decode([65]) == "A"


def test_decode_no_merges_ascii():
    assert BPETokenizer().decode([104, 105]) == "hi"


# ----------------------------------------------------------------------
# Round-trip — the headline lossless property
# ----------------------------------------------------------------------

def test_round_trip_no_merges_ascii():
    tok = BPETokenizer()
    text = "Hello, World!"
    assert tok.decode(tok.encode(text)) == text


def test_round_trip_no_merges_unicode():
    """Round-trip works on multi-byte UTF-8 even with no merges."""
    tok = BPETokenizer()
    text = "héllo wörld 🌍"
    assert tok.decode(tok.encode(text)) == text


def test_round_trip_after_training():
    tok = BPETokenizer()
    tok.train("the quick brown fox jumps over the lazy dog " * 5, vocab_size=300)
    text = "the lazy fox jumps quick"
    assert tok.decode(tok.encode(text)) == text


def test_round_trip_held_out_text():
    """Trained on one corpus, must still round-trip held-out text losslessly
    (every byte is in the base vocab, so encoding can never fail)."""
    tok = BPETokenizer()
    tok.train("hello world " * 5, vocab_size=270)
    held_out = "completely different text 12345 !@#"
    assert tok.decode(tok.encode(held_out)) == held_out


def test_round_trip_unicode_after_training():
    tok = BPETokenizer()
    tok.train("the quick brown fox " * 5, vocab_size=270)
    text = "héllo wörld 🌍 — naïve"
    assert tok.decode(tok.encode(text)) == text


def test_compression_increases_with_vocab_size():
    """Bigger vocab → fewer tokens for the same text (more merges available)."""
    corpus = "the quick brown fox jumps over the lazy dog " * 20
    test_text = corpus[:200]

    small = BPETokenizer()
    small.train(corpus, vocab_size=270)

    big = BPETokenizer()
    big.train(corpus, vocab_size=350)

    assert len(big.encode(test_text)) <= len(small.encode(test_text))
