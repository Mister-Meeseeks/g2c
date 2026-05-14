"""Reusable tokenizer artifact helpers.

Module 04 notebooks use these helpers to train or load named tokenizer
artifacts. The notebook still owns the visual inspection, but the filesystem
layout, source-text lookup, manifest construction, and save/load plumbing live
here so later modules can reuse the same artifacts.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import torch

from g2c.tokenizer import COURSE_SPECIAL_TOKENS, BPETokenizer
from g2c.tokenizer.fast import MAX_FAST_TRAIN_CHUNK_CHARS
from g2c.tokenizer.persistence import (
    MANIFEST_FILENAME,
    TOKEN_IDS_FILENAME,
    TOKENIZER_FILENAME,
    load_artifact,
    load_manifest,
    load_token_ids,
    save_artifact,
)
from g2c.tokenizer.persistence import (
    load as load_tokenizer_file,
)

from .corpora import load_corpus_text
from .paths import find_repo_root, tokenizer_artifact_dir

ArtifactProgressCallback = Callable[[dict[str, object]], None]
ArtifactStatusCallback = Callable[[dict[str, object]], None]
DEFAULT_ENCODED_SAMPLE_CHARS = 1_000_000
DEFAULT_TOKENIZE_CHUNK_CHARS = 1_000_000
TextChunkFn = Callable[[str], Iterable[str]]


@dataclass(frozen=True)
class TokenizerArtifactConfig:
    """Configuration for a reusable tokenizer artifact."""

    name: str
    source: str
    vocab_size: int
    max_chars: int
    use_fast: bool = True
    chunk_chars: int = MAX_FAST_TRAIN_CHUNK_CHARS
    encoded_sample_chars: int = DEFAULT_ENCODED_SAMPLE_CHARS
    special_tokens: tuple[str, ...] = COURSE_SPECIAL_TOKENS
    notes: str = ""

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> TokenizerArtifactConfig:
        """Build a config from a notebook-style dictionary."""
        raw_special_tokens = values.get("special_tokens", COURSE_SPECIAL_TOKENS)
        if isinstance(raw_special_tokens, str):
            special_tokens = (raw_special_tokens,)
        else:
            special_tokens = tuple(str(token) for token in raw_special_tokens)
        return cls(
            name=str(values["name"]),
            source=str(values["source"]),
            vocab_size=int(values["vocab_size"]),
            max_chars=int(values["max_chars"]),
            use_fast=bool(values.get("use_fast", True)),
            chunk_chars=int(values.get("chunk_chars", MAX_FAST_TRAIN_CHUNK_CHARS)),
            encoded_sample_chars=int(
                values.get("encoded_sample_chars", DEFAULT_ENCODED_SAMPLE_CHARS)
            ),
            special_tokens=special_tokens,
            notes=str(values.get("notes", "")),
        )


@dataclass(frozen=True)
class TokenizerArtifact:
    """Loaded or newly trained tokenizer artifact plus optional source text.

    `ids` is an encoded sample for inspection, not necessarily the full
    training corpus. Large tokenized corpora are separate artifacts.
    """

    config: TokenizerArtifactConfig
    tokenizer: BPETokenizer
    ids: list[int]
    manifest: dict[str, object]
    artifact_dir: Path
    text: str | None


def tokenizer_artifact_exists(
    name: str,
    *,
    repo_root: str | Path | None = None,
) -> bool:
    """Return True when all files for a named tokenizer artifact are present."""
    artifact_dir = tokenizer_artifact_dir(name, repo_root)
    return all(
        (artifact_dir / filename).exists()
        for filename in (TOKENIZER_FILENAME, TOKEN_IDS_FILENAME, MANIFEST_FILENAME)
    )


def load_tokenizer_artifact(
    name: str,
    *,
    repo_root: str | Path | None = None,
    load_ids: bool = False,
    load_text: bool = False,
) -> TokenizerArtifact:
    """Load a tokenizer artifact without training it.

    Module 10 and later modules use this path when the durable artifact is the
    tokenizer table itself. Set `load_ids=True` only when you need the small
    encoded inspection sample saved by Module 04.
    """
    root = find_repo_root(repo_root)
    artifact_dir = tokenizer_artifact_dir(name, root)
    tokenizer_path = artifact_dir / TOKENIZER_FILENAME
    manifest_path = artifact_dir / MANIFEST_FILENAME
    ids_path = artifact_dir / TOKEN_IDS_FILENAME
    missing = [
        path.relative_to(root)
        for path in (tokenizer_path, manifest_path)
        if not path.exists()
    ]
    if load_ids and not ids_path.exists():
        missing.append(ids_path.relative_to(root))
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"missing tokenizer artifact {name!r}: {missing_text}")

    tokenizer = load_tokenizer_file(BPETokenizer, tokenizer_path)
    manifest = load_manifest(manifest_path)
    ids = load_token_ids(ids_path) if load_ids else []
    config = _config_from_manifest(name, manifest)
    text = None
    if load_text:
        text = load_tokenizer_source_text(config.source, config.max_chars, repo_root=root)
    return TokenizerArtifact(config, tokenizer, ids, manifest, artifact_dir, text)


def load_tokenizer_source_text(
    source: str,
    max_chars: int,
    *,
    repo_root: str | Path | None = None,
) -> str | None:
    """Load text for a named tokenizer source, capped at about `max_chars` bytes."""
    root = find_repo_root(repo_root)
    return load_corpus_text(source, byte_count=max_chars, split="train", repo_root=root)


def load_tinystories_text(
    *,
    train_max_chars: int,
    valid_max_chars: int,
    repo_root: str | Path | None = None,
    allow_sample: bool = True,
) -> tuple[str, str] | None:
    """Load TinyStories train/validation text from compressed shards or legacy files."""
    root = find_repo_root(repo_root)
    train_text = load_corpus_text(
        "tinystories",
        byte_count=train_max_chars,
        split="train",
        repo_root=root,
    )
    if train_text is not None and not allow_sample:
        tinystories_dir = root / "data" / "datasets" / "tinystories"
        has_full_train = (
            (tinystories_dir / "TinyStories-train-shards.json").exists()
            or (tinystories_dir / "TinyStories-train.txt").exists()
            or any(tinystories_dir.glob("TinyStories-train-[0-9][0-9][0-9][0-9].txt.gz"))
        )
        if not has_full_train:
            train_text = None
    valid_text = load_corpus_text(
        "tinystories",
        byte_count=valid_max_chars,
        split="val",
        repo_root=root,
    )
    if train_text is None or valid_text is None:
        return None
    return train_text, valid_text


def default_text_chunks(
    text: str,
    *,
    chunk_chars: int = DEFAULT_TOKENIZE_CHUNK_CHARS,
) -> Iterable[str]:
    """Yield fixed-size text chunks for bounded-memory tokenization."""
    if chunk_chars < 1:
        raise ValueError("chunk_chars must be at least 1")
    for start in range(0, len(text), chunk_chars):
        yield text[start : start + chunk_chars]


def encode_text_to_tensor(
    tokenizer: BPETokenizer,
    text: str,
    *,
    chunk_fn: TextChunkFn | None = None,
    dtype: torch.dtype = torch.long,
    progress_callback: ArtifactProgressCallback | None = None,
    vocab_size: int | None = None,
) -> torch.Tensor:
    """Encode text into one preallocated tensor using a two-pass chunk scan.

    This avoids materializing one giant Python `list[int]` before making the
    tensor. By default it chunks every 1M characters. A future corpus-aware
    caller can pass `chunk_fn` that yields story/document boundaries instead.

    `vocab_size` selects the tokenizer's effective vocab (passed through to
    ``BPETokenizer.encode_with_vocab_size``). ``None`` means the full trained
    vocab; ``0`` means byte + special tokens only; an integer ``N`` truncates
    learned merges to those with ID ``< N``.
    """
    chunk_fn = chunk_fn or default_text_chunks
    _emit(progress_callback, phase="encode_count_start", chars=len(text))
    total_tokens = 0
    chunks = 0
    chars_seen = 0
    for chunk in chunk_fn(text):
        chunk_ids = tokenizer.encode_with_vocab_size(chunk, vocab_size)
        chunks += 1
        chars_seen += len(chunk)
        total_tokens += len(chunk_ids)
        _emit(
            progress_callback,
            phase="encode_count_chunk",
            chunk_index=chunks,
            chars_seen=chars_seen,
            chars=len(text),
            tokens_seen=total_tokens,
        )
    _emit(
        progress_callback,
        phase="encode_count_done",
        chunks=chunks,
        chars=len(text),
        token_count=total_tokens,
    )

    result = torch.empty(total_tokens, dtype=dtype)
    offset = 0
    _emit(
        progress_callback,
        phase="encode_write_start",
        chunks=chunks,
        token_count=total_tokens,
    )
    for chunk_index, chunk in enumerate(chunk_fn(text), start=1):
        chunk_ids = tokenizer.encode_with_vocab_size(chunk, vocab_size)
        next_offset = offset + len(chunk_ids)
        result[offset:next_offset] = torch.as_tensor(chunk_ids, dtype=dtype)
        offset = next_offset
        _emit(
            progress_callback,
            phase="encode_write_chunk",
            chunk_index=chunk_index,
            chunks=chunks,
            tokens_written=offset,
            token_count=total_tokens,
        )
    if offset != total_tokens:
        raise ValueError("chunk_fn produced a different token count on the second pass")
    _emit(
        progress_callback,
        phase="encode_done",
        chunks=chunks,
        token_count=total_tokens,
    )
    return result


def load_required_tokenizer(
    tokenizer_name: str,
    *,
    repo_root: str | Path | None = None,
) -> BPETokenizer:
    """Load a required tokenizer artifact and return its tokenizer."""
    try:
        artifact = load_tokenizer_artifact(tokenizer_name, repo_root=repo_root)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Tokenizer artifact {tokenizer_name!r} is missing. "
            "Run Module 04's Mini Milestone tokenizer artifact cells first. "
            "For TinyStories, set RUN_TINYSTORIES_TOKENIZER = True in Module 04."
        ) from exc
    return artifact.tokenizer


def train_or_load_tokenizer_artifact(
    config: TokenizerArtifactConfig | Mapping[str, object],
    *,
    repo_root: str | Path | None = None,
    force: bool = False,
    progress_callback: ArtifactProgressCallback | None = None,
    status_callback: ArtifactStatusCallback | None = None,
) -> TokenizerArtifact | None:
    """Train or load one tokenizer artifact.

    Returns None when the configured source data is unavailable.
    """
    config = _normalize_config(config)
    root = find_repo_root(repo_root)
    artifact_dir = tokenizer_artifact_dir(config.name, root)

    if tokenizer_artifact_exists(config.name, repo_root=root) and not force:
        tokenizer, ids, manifest = load_artifact(BPETokenizer, artifact_dir)
        text = load_tokenizer_source_text(config.source, config.max_chars, repo_root=root)
        _emit(
            status_callback,
            phase="loaded",
            name=config.name,
            token_count=len(ids),
            artifact_dir=artifact_dir,
        )
        return TokenizerArtifact(config, tokenizer, ids, manifest, artifact_dir, text)

    text = load_tokenizer_source_text(config.source, config.max_chars, repo_root=root)
    if not text:
        _emit(status_callback, phase="missing_source", name=config.name, source=config.source)
        return None

    tokenizer = BPETokenizer(special_tokens=config.special_tokens)
    encoded_text = text[: min(len(text), config.encoded_sample_chars)]
    start = perf_counter()
    if config.vocab_size == tokenizer.base_vocab_size:
        ids = tokenizer.encode_base(encoded_text)
    elif config.use_fast:
        effective_chunk_chars = min(config.chunk_chars, MAX_FAST_TRAIN_CHUNK_CHARS)
        chunks = _num_text_chunks(text, effective_chunk_chars)
        _emit(
            status_callback,
            phase="fast_start",
            name=config.name,
            target_vocab_size=config.vocab_size,
            chars=len(text),
            chunk_chars=effective_chunk_chars,
            requested_chunk_chars=config.chunk_chars,
            chunks=chunks,
        )
        tokenizer.train_fast(
            text,
            vocab_size=config.vocab_size,
            show_progress=False,
            chunk_chars=effective_chunk_chars,
            encode_training_text=False,
            progress_callback=lambda info: _emit_fast_status(
                status_callback,
                name=config.name,
                info=info,
            ),
        )
        _emit(
            status_callback,
            phase="sample_encode_start",
            name=config.name,
            chars=len(encoded_text),
            training_chars=len(text),
        )
        ids = tokenizer.encode_fast(encoded_text)
        _emit(
            status_callback,
            phase="sample_encode_done",
            name=config.name,
            chars=len(encoded_text),
            token_count=len(ids),
        )
        _emit(
            status_callback,
            phase="fast_done",
            name=config.name,
            vocab_size=len(tokenizer.vocab),
            target_vocab_size=config.vocab_size,
            token_count=len(ids),
            chunks=chunks,
            elapsed_seconds=perf_counter() - start,
        )
    else:
        trained_ids = tokenizer.train(
            text,
            vocab_size=config.vocab_size,
            progress_callback=progress_callback,
            progress_every=_tokenizer_progress_every(config.vocab_size),
        )
        ids = (
            trained_ids
            if len(encoded_text) == len(text)
            else tokenizer.encode_fast(encoded_text)
        )

    manifest = _build_manifest(
        config,
        text,
        encoded_text,
        ids,
        tokenizer,
        perf_counter() - start,
    )
    save_artifact(tokenizer, ids, manifest, artifact_dir)
    _emit(
        status_callback,
        phase="saved",
        name=config.name,
        token_count=len(ids),
        artifact_dir=artifact_dir,
    )
    return TokenizerArtifact(config, tokenizer, ids, manifest, artifact_dir, text)


def _normalize_config(
    config: TokenizerArtifactConfig | Mapping[str, object],
) -> TokenizerArtifactConfig:
    if isinstance(config, TokenizerArtifactConfig):
        return config
    return TokenizerArtifactConfig.from_mapping(config)


def _config_from_manifest(name: str, manifest: Mapping[str, object]) -> TokenizerArtifactConfig:
    return TokenizerArtifactConfig(
        name=str(manifest.get("name", name)),
        source=str(manifest.get("source", "")),
        vocab_size=int(
            manifest.get(
                "requested_vocab_size",
                manifest.get("actual_vocab_size", 256),
            )
        ),
        max_chars=int(manifest.get("max_chars", manifest.get("actual_chars", 0))),
        use_fast=str(manifest.get("trainer", "")) == "rust-tokenizers",
        chunk_chars=int(manifest.get("chunk_chars") or MAX_FAST_TRAIN_CHUNK_CHARS),
        encoded_sample_chars=int(
            manifest.get("encoded_sample_chars") or DEFAULT_ENCODED_SAMPLE_CHARS
        ),
        special_tokens=tuple(
            str(token)
            for token in manifest.get("special_tokens", COURSE_SPECIAL_TOKENS)
        ),
        notes=str(manifest.get("notes", "")),
    )


def _build_manifest(
    config: TokenizerArtifactConfig,
    text: str,
    encoded_text: str,
    ids: list[int],
    tokenizer: BPETokenizer,
    elapsed_seconds: float,
) -> dict[str, object]:
    return {
        "name": config.name,
        "kind": "bpe-tokenizer-artifact",
        "created_at": datetime.now(UTC).isoformat(),
        "source": config.source,
        "requested_vocab_size": config.vocab_size,
        "actual_vocab_size": len(tokenizer.vocab),
        "special_tokens": list(config.special_tokens),
        "max_chars": config.max_chars,
        "actual_chars": len(text),
        "encoded_sample_chars": config.encoded_sample_chars,
        "encoded_chars": len(encoded_text),
        "encoded_full_training_text": len(encoded_text) == len(text),
        "token_count": len(ids),
        "chars_per_token": len(encoded_text) / max(1, len(ids)),
        "token_ids_file": TOKEN_IDS_FILENAME,
        "token_ids_dtype": "uint32",
        "token_ids_kind": "encoded_sample",
        "trainer": "rust-tokenizers" if config.use_fast else "python-scaffold",
        "chunk_chars": min(config.chunk_chars, MAX_FAST_TRAIN_CHUNK_CHARS)
        if config.use_fast
        else None,
        "requested_chunk_chars": config.chunk_chars if config.use_fast else None,
        "elapsed_seconds": elapsed_seconds,
        "notes": config.notes,
    }


def _tokenizer_progress_every(vocab_size: int, updates: int = 80) -> int:
    return max(1, (vocab_size - 256 - len(COURSE_SPECIAL_TOKENS)) // updates)


def _num_text_chunks(text: str, chunk_chars: int) -> int:
    return max(1, (len(text) + chunk_chars - 1) // chunk_chars)


def _emit(callback: ArtifactStatusCallback | None, **payload: object) -> None:
    if callback is not None:
        callback(payload)


def _emit_fast_status(
    callback: ArtifactStatusCallback | None,
    *,
    name: str,
    info: dict[str, object],
) -> None:
    # `fast_start` already reports the same coarse setup information with
    # artifact metadata attached. Keep the lower-level event available for
    # direct BPETokenizer callers, but do not force notebook status handlers to
    # understand both names.
    if info.get("phase") == "fast_chunks_start":
        return
    _emit(callback, name=name, **info)
