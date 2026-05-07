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
from typing import Self

ProgressCallback = Callable[[dict[str, object]], None]

"""Course-specific special tokens that later modules use for chat/document boundaries
"""
COURSE_SPECIAL_TOKENS: tuple[str, ...] = (
    "<|endoftext|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|tool_call|>",
    "<|tool_result|>",
    "<|end|>",
    "<|pad|>",
)


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

    def __init__(self, special_tokens: tuple[str, ...] | list[str] = ()) -> None:
        """Create a byte-level BPE tokenizer.

        Args:
            special_tokens: optional control-token strings that should encode as
                single atomic IDs. Later course modules use these for document
                boundaries, chat roles, and tool-call events. They are reserved
                immediately after the byte IDs, and BPE training does not learn
                merges that consume them.
        """
        self.special_tokens = tuple(special_tokens)
        self.special_to_id: dict[str, int] = {}
        self.id_to_special: dict[int, str] = {}
        self.special_token_ids: set[int] = set()
        self._special_tokens_by_length: tuple[str, ...] = ()
        self._validate_special_tokens()
        self._reset_to_base_vocab()

    @property
    def base_vocab_size(self) -> int:
        """Return byte vocabulary size plus reserved special-token count."""
        return 256 + len(self.special_tokens)

    @classmethod
    def with_course_special_tokens(cls) -> Self:
        """Return a tokenizer with the course chat/tool control tokens reserved."""
        return cls(special_tokens=COURSE_SPECIAL_TOKENS)

    def _validate_special_tokens(self) -> None:
        if len(set(self.special_tokens)) != len(self.special_tokens):
            raise ValueError("special tokens must be unique")
        for token in self.special_tokens:
            if not token:
                raise ValueError("special tokens must be non-empty")

    def _reset_to_base_vocab(self) -> None:
        """Reset learned merges while preserving byte IDs and special IDs."""
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.special_to_id = {}
        self.id_to_special = {}
        self.special_token_ids = set()
        for offset, token in enumerate(self.special_tokens):
            token_id = 256 + offset
            self.special_to_id[token] = token_id
            self.id_to_special[token_id] = token
            self.special_token_ids.add(token_id)
            self.vocab[token_id] = token.encode("utf-8")
        self._special_tokens_by_length = tuple(
            sorted(self.special_tokens, key=len, reverse=True)
        )

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
        chunk_chars: int = 8_192,
        encode_training_text: bool = True,
        progress_callback: ProgressCallback | None = None,
    ) -> list[int]:
        """Train BPE with the Rust-backed implementation."""
        from .fast import train_fast

        return train_fast(
            self,
            text,
            vocab_size,
            show_progress=show_progress,
            chunk_chars=chunk_chars,
            encode_training_text=encode_training_text,
            progress_callback=progress_callback,
        )

    # ------------------------------------------------------------------
    # Algorithmic helpers — STUDENT IMPLEMENTS
    # ------------------------------------------------------------------

    @staticmethod
    def _get_pair_counts(
        ids: list[int],
        *,
        barrier_ids: set[int] | None = None,
    ) -> dict[tuple[int, int], int]:
        """Return the frequency of each adjacent pair in `ids`.

        Example:
            _get_pair_counts([1, 2, 1, 2, 3])
              -> {(1, 2): 2, (2, 1): 1, (2, 3): 1}

        Optional `barrier_ids` are excluded from pair counting. This is how
        reserved special tokens remain atomic during training.

        Hint: a single pass with a sliding window of size 2.
        """
        pair_counts = {}
        for i in range(len(ids) - 1):
            pair = (ids[i], ids[i + 1])
            if barrier_ids and (pair[0] in barrier_ids or pair[1] in barrier_ids):
                continue
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
        pair_counts = self._get_pair_counts(ids, barrier_ids=self.special_token_ids)
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
            vocab_size: target vocabulary size. Must be at least
                        `self.base_vocab_size` (the byte base plus any reserved
                        special tokens). Exactly `self.base_vocab_size` means
                        "no merges, just the base vocab."
            progress_callback: optional function called with a progress dict
                               every `progress_every` merge steps and at the
                               end of training.
            progress_every: callback cadence, in merge steps.
            on_progress: backward-compatible alias for `progress_callback`.

        Returns:
            The final token IDs for `text` after all learned merges have been
            applied.

        Raises:
            ValueError: if `vocab_size` is smaller than the byte + special-token
                base vocabulary.
            ValueError: if `progress_every < 1`.
        """
        if vocab_size < self.base_vocab_size:
            raise ValueError(
                f"vocab_size must be at least {self.base_vocab_size} "
                "for this tokenizer's byte + special-token base vocabulary"
            )
        if progress_every < 1:
            raise ValueError("progress_every must be at least 1")
        if progress_callback is not None and on_progress is not None:
            raise ValueError("pass either progress_callback or on_progress, not both")

        callback = progress_callback if progress_callback is not None else on_progress
        ids = self._encode_initial_ids(text)
        target_vocab_size = vocab_size
        target_merges = max(0, target_vocab_size - self.base_vocab_size)
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
        ids = self._encode_initial_ids(text)

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

    # ------------------------------------------------------------------
    # Special-token segmentation — implemented boilerplate
    # ------------------------------------------------------------------

    def _special_aware_segments(self, text: str):
        """Yield text spans and special token IDs in left-to-right order."""
        if not self.special_tokens:
            if text:
                yield text
            return

        span_start = 0
        i = 0
        while i < len(text):
            matched = None
            for token in self._special_tokens_by_length:
                if text.startswith(token, i):
                    matched = token
                    break
            if matched is None:
                i += 1
                continue

            if span_start < i:
                yield text[span_start:i]
            yield self.special_to_id[matched]
            i += len(matched)
            span_start = i

        if span_start < len(text):
            yield text[span_start:]

    def _encode_initial_ids(self, text: str) -> list[int]:
        """Encode text to byte IDs while preserving atomic special tokens."""
        ids: list[int] = []
        for segment in self._special_aware_segments(text):
            if isinstance(segment, int):
                ids.append(segment)
            else:
                ids.extend(segment.encode("utf-8"))
        return ids
