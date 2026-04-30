"""Byte-pair encoding (BPE) tokenizer — from scratch.

A `BPETokenizer` learns a subword vocabulary by iteratively merging the
most-frequent adjacent pair of token IDs into a new token ID, starting from
a base vocabulary of all 256 possible bytes. The result is a learned table
of merges that can be applied to any UTF-8 string to compress it into a
shorter sequence of integers — and reversed losslessly.

The algorithm is small but the implementation has a few subtleties (overlapping
pairs, greedy merge order at encode time, byte-level lossless round-trip). All
of those are the lesson, so they're scaffolded.

Boilerplate (constructor, base vocab) is implemented for you. Search for `# TODO`.
"""
from __future__ import annotations


class BPETokenizer:
    """A byte-pair encoding tokenizer.

    Attributes:
        merges: maps each learned pair `(id_a, id_b)` → the new ID assigned
                to that pair. Order of insertion encodes priority: lower IDs
                were learned earlier.
        vocab:  maps every ID (base bytes 0–255 plus learned merge IDs) to
                the raw byte sequence it represents. Used by `decode`.
    """

    merges: dict[tuple[int, int], int]
    vocab: dict[int, bytes]

    def __init__(self) -> None:
        # Base vocabulary: one ID for every possible byte value 0..255.
        # No merges learned yet — `train` will populate them.
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

    # ------------------------------------------------------------------
    # Algorithmic helpers — STUDENT IMPLEMENTS
    # ------------------------------------------------------------------

    @staticmethod
    def _get_pair_counts(ids: list[int]) -> dict[tuple[int, int], int]:
        """Return the frequency of each adjacent pair in `ids`.

        Example:
            _get_pair_counts([1, 2, 1, 2, 3])
              -> {(1, 2): 2, (2, 1): 1, (2, 3): 1}

        Hint: a single pass with a sliding window of size 2.
        """
        # TODO
        raise NotImplementedError

    @staticmethod
    def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        """Replace every non-overlapping occurrence of `pair` in `ids` with `new_id`.

        Example:
            _merge([1, 2, 3, 1, 2], (1, 2), 99)  ->  [99, 3, 99]

        IMPORTANT — overlap rule (left-to-right greedy):
            _merge([1, 1, 1], (1, 1), 99)  ->  [99, 1]    (NOT [99, 99]; the
            second `1` is consumed by the first match, leaving only one `1` left.)

        Hint: scan with an explicit index `i`, advancing by 2 on a match
        and by 1 otherwise.
        """
        # TODO
        raise NotImplementedError

    # ------------------------------------------------------------------
    # The main API — STUDENT IMPLEMENTS
    # ------------------------------------------------------------------

    def train(self, text: str, vocab_size: int) -> None:
        """Learn merges from `text` until the vocabulary reaches `vocab_size`.

        Algorithm:
          1. Encode `text` to UTF-8 bytes; treat as a list of int IDs (0–255).
          2. While the current vocab is smaller than `vocab_size`:
               a. Count adjacent pair frequencies in the current ID list.
               b. If no pairs remain (sequence too short), break.
               c. Pick the most-frequent pair (ties broken arbitrarily — by
                  insertion order in the dict, which is fine).
               d. Assign it a new ID (the next integer ≥ 256 not yet used).
               e. Apply the merge to the ID list (so subsequent counts are
                  computed on the post-merge sequence).
               f. Record the merge in `self.merges` and the corresponding
                  byte sequence in `self.vocab`.

        Args:
            text: training corpus.
            vocab_size: target vocabulary size. Must be ≥ 256 (the byte base);
                        exactly 256 means "no merges, just the base vocab."

        Raises:
            ValueError: if `vocab_size < 256`.
        """
        # TODO
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        """Encode `text` into a list of token IDs using the learned merges.

        Algorithm:
          1. Encode `text` to UTF-8 bytes → list of IDs (0–255).
          2. Repeatedly:
               a. Find all adjacent pairs in the current ID list.
               b. Among those that appear in `self.merges`, pick the one with
                  the LOWEST merge ID (i.e. the merge we learned earliest).
                  Earliest-learned merges have priority, just like in `train`.
               c. If no pair in the list is in `self.merges`, stop.
               d. Apply that merge to the ID list.
          3. Return the resulting list.

        Edge cases:
          - Empty `text` → empty list.
          - `text` containing characters never seen in training → still works,
            because every UTF-8 byte is in the base vocab.
        """
        # TODO
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        """Reverse of `encode`: reconstruct the original text from IDs.

        Algorithm:
          1. For each ID, look up its byte sequence in `self.vocab`.
          2. Concatenate them.
          3. Decode as UTF-8.

        Use `errors="replace"` on the UTF-8 decode to be safe against
        partial-byte ID sequences (would not arise from a correct round-trip,
        but is a robust default).

        Hint: a one-liner with `b"".join(...)` and `.decode("utf-8", errors="replace")`.
        """
        # TODO
        raise NotImplementedError
