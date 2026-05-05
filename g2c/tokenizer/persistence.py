"""Persistence helpers for the Module 04 BPE tokenizer.

The tokenizer students build in `bpe.py` is just two tables: `merges` and
`vocab`. Saving and loading those tables is useful infrastructure for later
modules, but it is not the conceptual core of byte-pair encoding. Keeping the
serialization code here lets `bpe.py` stay focused on the algorithm students
write by hand.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .bpe import BPETokenizer


TOKENIZER_FORMAT = "g2c.bpe"
TOKENIZER_VERSION = 1


def to_dict(tokenizer: BPETokenizer) -> dict[str, object]:
    """Return a JSON-serializable representation of a learned tokenizer."""
    return {
        "format": TOKENIZER_FORMAT,
        "version": TOKENIZER_VERSION,
        "merges": [
            [a, b, new_id]
            for (a, b), new_id in sorted(
                tokenizer.merges.items(),
                key=lambda item: item[1],
            )
        ],
    }


def from_dict(
    tokenizer_cls: type[BPETokenizer],
    payload: dict[str, object],
) -> BPETokenizer:
    """Construct a tokenizer from `to_dict()` output."""
    if payload.get("format") != TOKENIZER_FORMAT:
        raise ValueError("not a g2c BPE tokenizer payload")
    if payload.get("version") != TOKENIZER_VERSION:
        raise ValueError("unsupported BPE tokenizer version")

    tok = tokenizer_cls()
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


def save(tokenizer: BPETokenizer, path: str | Path) -> None:
    """Save this tokenizer to a UTF-8 JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_dict(tokenizer), indent=2) + "\n", encoding="utf-8")


def load(tokenizer_cls: type[BPETokenizer], path: str | Path) -> BPETokenizer:
    """Load a tokenizer saved by `save`."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("tokenizer file must contain a JSON object")
    return from_dict(tokenizer_cls, payload)
