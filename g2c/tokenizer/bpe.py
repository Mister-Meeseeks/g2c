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

import heapq
import json
from collections.abc import Callable
from pathlib import Path
from time import perf_counter

ProgressCallback = Callable[[dict[str, object]], None]
TOKENIZER_FORMAT = "g2c.bpe"
TOKENIZER_VERSION = 1


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
        return {
            "format": TOKENIZER_FORMAT,
            "version": TOKENIZER_VERSION,
            "merges": [
                [a, b, new_id]
                for (a, b), new_id in sorted(
                    self.merges.items(),
                    key=lambda item: item[1],
                )
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> BPETokenizer:
        """Construct a tokenizer from `to_dict()` output."""
        if payload.get("format") != TOKENIZER_FORMAT:
            raise ValueError("not a g2c BPE tokenizer payload")
        if payload.get("version") != TOKENIZER_VERSION:
            raise ValueError("unsupported BPE tokenizer version")

        tok = cls()
        merges = payload.get("merges")
        if not isinstance(merges, list):
            raise ValueError("tokenizer payload must contain a list of merges")

        for expected_id, merge in enumerate(merges, start=256):
            if (
                not isinstance(merge, list)
                or len(merge) != 3
                or not all(isinstance(part, int) for part in merge)
            ):
                raise ValueError("each tokenizer merge must be [a, b, new_id]")

            a, b, new_id = merge
            if new_id != expected_id:
                raise ValueError("tokenizer merge IDs must be sequential from 256")
            if a not in tok.vocab or b not in tok.vocab:
                raise ValueError("tokenizer merge references an unknown token ID")

            tok.merges[(a, b)] = new_id
            tok.vocab[new_id] = tok.vocab[a] + tok.vocab[b]

        return tok

    def save(self, path: str | Path) -> None:
        """Save this tokenizer to a UTF-8 JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        """Load a tokenizer saved by `save`."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("tokenizer file must contain a JSON object")
        return cls.from_dict(payload)

    # ------------------------------------------------------------------
    # Fast path — implemented infrastructure
    # ------------------------------------------------------------------

    def encode_fast(self, text: str) -> list[int]:
        """Encode `text` with a priority queue over merge ranks.

        This is the same BPE merge order as `encode`, but implemented as
        infrastructure for larger corpus/tokenizer artifact work. Students
        should implement `encode` first because it is easier to understand.
        """
        token_ids = list(text.encode("utf-8"))
        if len(token_ids) < 2 or not self.merges:
            return token_ids

        n = len(token_ids)
        prev = [-1] + list(range(n - 1))
        next_ = list(range(1, n)) + [-1]
        active = [True] * n
        heap: list[tuple[int, int, int]] = []

        def push_pair(left: int) -> None:
            right = next_[left]
            if right == -1:
                return
            new_id = self.merges.get((token_ids[left], token_ids[right]))
            if new_id is not None:
                heapq.heappush(heap, (new_id, left, right))

        for i in range(n - 1):
            push_pair(i)

        while heap:
            new_id, left, right = heapq.heappop(heap)
            if (
                not active[left]
                or not active[right]
                or next_[left] != right
                or self.merges.get((token_ids[left], token_ids[right])) != new_id
            ):
                continue

            token_ids[left] = new_id
            active[right] = False
            after = next_[right]
            next_[left] = after
            if after != -1:
                prev[after] = left

            before = prev[left]
            if before != -1:
                push_pair(before)
            push_pair(left)

        encoded: list[int] = []
        i = 0
        while i != -1:
            if active[i]:
                encoded.append(token_ids[i])
            i = next_[i]
        return encoded

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
        # TODO
        raise NotImplementedError

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
    ) -> None:
        """Learn merges from `text` until the vocabulary reaches `vocab_size`.

        The outer training loop is implemented scaffold. Students implement
        `train_step`, which performs the conceptual one-merge BPE operation.

        Args:
            text: training corpus.
            vocab_size: target vocabulary size. Must be ≥ 256 (the byte base);
                        exactly 256 means "no merges, just the base vocab."
            progress_callback: optional function called with a progress dict
                               every `progress_every` merge steps and at the
                               end of training. Notebook/script code can use
                               this to print progress without changing the BPE
                               algorithm students implement.
            progress_every: callback cadence, in merge steps.

        Raises:
            ValueError: if `vocab_size < 256`.
        """
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256")
        if progress_every < 1:
            raise ValueError("progress_every must be at least 1")

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
            if progress_callback is None:
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
            progress_callback(
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
