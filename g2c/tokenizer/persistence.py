"""Persistence helpers for the Module 04 BPE tokenizer.

The tokenizer students build in `bpe.py` is just two tables: `merges` and
`vocab`. Saving and loading those tables is useful infrastructure for later
modules, but it is not the conceptual core of byte-pair encoding. Keeping the
serialization code here lets `bpe.py` stay focused on the algorithm students
write by hand.
"""
from __future__ import annotations

import json
from array import array
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .bpe import BPETokenizer


TOKENIZER_FORMAT = "g2c.bpe"
TOKENIZER_VERSION = 1
TOKENIZER_FILENAME = "tokenizer.json"
TOKEN_IDS_FILENAME = "ids.uint32"
MANIFEST_FILENAME = "manifest.json"


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


def save_token_ids(ids: list[int], path: str | Path) -> None:
    """Save token IDs as a compact uint32 binary file."""
    values = _uint32_array(ids)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        values.tofile(f)


def load_token_ids(path: str | Path) -> list[int]:
    """Load token IDs saved by `save_token_ids`."""
    values = _uint32_array()
    path = Path(path)
    if path.stat().st_size % values.itemsize != 0:
        raise ValueError("token ID file size is not a multiple of uint32")
    with path.open("rb") as f:
        values.fromfile(f, path.stat().st_size // values.itemsize)
    return list(values)


def save_manifest(manifest: dict[str, object], path: str | Path) -> None:
    """Save an artifact manifest as pretty UTF-8 JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def load_manifest(path: str | Path) -> dict[str, object]:
    """Load an artifact manifest saved by `save_manifest`."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("artifact manifest must contain a JSON object")
    return payload


def save_artifact(
    tokenizer: BPETokenizer,
    ids: list[int],
    manifest: dict[str, object],
    artifact_dir: str | Path,
) -> None:
    """Save a reusable tokenizer artifact directory."""
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    save(tokenizer, artifact_dir / TOKENIZER_FILENAME)
    save_token_ids(ids, artifact_dir / TOKEN_IDS_FILENAME)
    save_manifest(manifest, artifact_dir / MANIFEST_FILENAME)


def load_artifact(
    tokenizer_cls: type[BPETokenizer],
    artifact_dir: str | Path,
) -> tuple[BPETokenizer, list[int], dict[str, object]]:
    """Load a reusable tokenizer artifact directory."""
    artifact_dir = Path(artifact_dir)
    tokenizer = load(tokenizer_cls, artifact_dir / TOKENIZER_FILENAME)
    ids = load_token_ids(artifact_dir / TOKEN_IDS_FILENAME)
    manifest = load_manifest(artifact_dir / MANIFEST_FILENAME)
    return tokenizer, ids, manifest


def _uint32_array(values=()) -> array:
    result = array("I", values)
    if result.itemsize != 4:
        raise RuntimeError("token ID persistence requires 32-bit unsigned ints")
    return result
