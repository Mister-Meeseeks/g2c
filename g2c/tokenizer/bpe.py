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

from collections.abc import Callable
from os import PathLike
from time import perf_counter

ProgressCallback = Callable[[dict[str, object]], None]


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
    # Persistence — implemented boilerplate
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation of the learned tokenizer."""
        from .persistence import to_dict

        return to_dict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> BPETokenizer:
        """Construct a tokenizer from `to_dict()` output."""
        from .persistence import from_dict

        return from_dict(cls, payload)

    def save(self, path: str | PathLike[str]) -> None:
        """Save this tokenizer to a UTF-8 JSON file."""
        from .persistence import save

        save(self, path)

    @classmethod
    def load(cls, path: str | PathLike[str]) -> BPETokenizer:
        """Load a tokenizer saved by `save`."""
        from .persistence import load

        return load(cls, path)

    # ------------------------------------------------------------------
    # Fast path — implemented infrastructure
    # ------------------------------------------------------------------

    def encode_fast(self, text: str) -> list[int]:
        """Encode `text` with the Rust-backed BPE implementation."""
        from .fast import encode_fast

        return encode_fast(self, text)

    def train_fast(
        self,
        text: str,
        vocab_size: int,
        *,
        show_progress: bool = True,
        chunk_chars: int = 1_000_000,
    ) -> list[int]:
        """Train BPE with the Rust-backed implementation."""
        from .fast import train_fast

        return train_fast(
            self,
            text,
            vocab_size,
            show_progress=show_progress,
            chunk_chars=chunk_chars,
        )

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
        pair_counts = {}
        for i in range(len(ids) - 1):
            pair = (ids[i], ids[i + 1])
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
        return pair_counts

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
        left, right = pair
        merged: list[int] = []
        append = merged.append

        i = 0
        n = len(ids)
        while i < n:
            if i + 1 < n and ids[i] == left and ids[i + 1] == right:
                append(new_id)
                i += 2
            else:
                append(ids[i])
                i += 1

        return merged

    # ------------------------------------------------------------------
    # Main training step — STUDENT IMPLEMENTS
    # ------------------------------------------------------------------

    def train_step(
        self,
        ids: list[int],
        new_id: int,
    ) -> tuple[list[int], tuple[int, int], int] | None:
        """Learn one BPE merge from the current token ID sequence.

        Algorithm:
          1. Count adjacent pair frequencies in `ids`.
          2. If no pairs remain, return None.
          3. Pick the most-frequent pair (ties broken arbitrarily — by
             insertion order in the dict, which is fine).
          4. Apply the merge to the ID list with `_merge`.
          5. Record the merge in `self.merges`.
          6. Record the new token bytes in `self.vocab`.

        Args:
            ids: current token ID sequence.
            new_id: ID to assign to the selected pair.

        Returns:
            `(updated_ids, merged_pair, merge_count)`, or None if there are no
            adjacent pairs left to merge.
        """
        pair_counts = self._get_pair_counts(ids)
        if not pair_counts:
            return None

        best_pair = max(pair_counts, key=pair_counts.get)
        best_pair_count = pair_counts[best_pair]

        self.merges[best_pair] = new_id
        self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

        return self._merge(ids, best_pair, new_id), best_pair, best_pair_count

    # ------------------------------------------------------------------
    # The main training API — implemented scaffold
    # ------------------------------------------------------------------

    def train(
        self,
        text: str,
        vocab_size: int,
        *,
        progress_callback: ProgressCallback | None = None,
        progress_every: int = 100,
        on_progress: ProgressCallback | None = None,
    ) -> list[int]:
        """Learn merges from `text` until the vocabulary reaches `vocab_size`.

        The outer training loop is implemented scaffold. `train_step` performs
        the conceptual one-merge BPE operation.

        Args:
            text: training corpus.
            vocab_size: target vocabulary size. Must be ≥ 256 (the byte base);
                        exactly 256 means "no merges, just the base vocab."
            progress_callback: optional function called with a progress dict
                               every `progress_every` merge steps and at the
                               end of training.
            progress_every: callback cadence, in merge steps.
            on_progress: backward-compatible alias for `progress_callback`.

        Returns:
            The final token IDs for `text` after all learned merges have been
            applied.

        Raises:
            ValueError: if `vocab_size < 256`.
            ValueError: if `progress_every < 1`.
        """
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256")
        if progress_every < 1:
            raise ValueError("progress_every must be at least 1")
        if progress_callback is not None and on_progress is not None:
            raise ValueError("pass either progress_callback or on_progress, not both")

        callback = progress_callback if progress_callback is not None else on_progress
        ids = list(text.encode("utf-8"))
        target_vocab_size = vocab_size
        target_merges = max(0, target_vocab_size - 256)
        start = perf_counter()
        steps = 0
        last_pair: tuple[int, int] | None = None
        last_merge_count: int | None = None
        reported_done = False

        def report(done: bool) -> None:
            nonlocal reported_done
            if callback is None:
                return
            if done:
                reported_done = True
            last_token_id = len(self.vocab) - 1 if last_pair is not None else None
            last_token_bytes = (
                self.vocab[last_token_id]
                if last_token_id is not None and last_token_id in self.vocab
                else None
            )
            last_token_text = (
                last_token_bytes.decode("utf-8", errors="replace")
                if last_token_bytes is not None
                else None
            )
            callback(
                {
                    "vocab_size": len(self.vocab),
                    "target_vocab_size": target_vocab_size,
                    "merges": len(self.merges),
                    "target_merges": target_merges,
                    "steps": steps,
                    "tokens": len(ids),
                    "last_pair": last_pair,
                    "last_token_id": last_token_id,
                    "last_token_bytes": last_token_bytes,
                    "last_token_text": last_token_text,
                    "last_token_repr": repr(last_token_text)
                    if last_token_text is not None
                    else None,
                    "last_merge_count": last_merge_count,
                    "elapsed_seconds": perf_counter() - start,
                    "done": done,
                    # Legacy metric names used by earlier solutions notebooks.
                    "num_merges": len(self.merges),
                    "token_count": len(ids),
                    "last_pair_count": last_merge_count or 0,
                }
            )

        while len(self.vocab) < target_vocab_size:
            result = self.train_step(ids, len(self.vocab))
            if result is None:
                break

            ids, last_pair, last_merge_count = result
            steps += 1

            if steps % progress_every == 0 or len(self.vocab) >= target_vocab_size:
                report(done=len(self.vocab) >= target_vocab_size)

        if not reported_done:
            report(done=True)
        return ids

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
        ids = list(text.encode("utf-8"))

        while True:
            pairs = []
            for i in range(len(ids) - 1):
                pairs.append((ids[i], ids[i + 1]))
            merge_candidates = [pair for pair in pairs if pair in self.merges]

            if not merge_candidates:
                break

            best_pair = min(merge_candidates, key=lambda pair: self.merges[pair])
            ids = self._merge(ids, best_pair, self.merges[best_pair])

        return ids

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
        byte_text = b"".join(self.vocab[id] for id in ids)
        return byte_text.decode("utf-8", errors="replace")
