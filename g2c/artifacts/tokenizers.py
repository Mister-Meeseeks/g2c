"""Reusable tokenizer artifact helpers.

Module 04 notebooks use these helpers to train or load named tokenizer
artifacts. The notebook still owns the visual inspection, but the filesystem
layout, source-text lookup, manifest construction, and save/load plumbing live
here so later modules can reuse the same artifacts.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from g2c.tokenizer import BPETokenizer
from g2c.tokenizer.fast import MAX_FAST_TRAIN_CHUNK_CHARS
from g2c.tokenizer.persistence import (
    MANIFEST_FILENAME,
    TOKEN_IDS_FILENAME,
    TOKENIZER_FILENAME,
    load_artifact,
    save_artifact,
)

from .corpora import load_corpus_text
from .paths import find_repo_root, tokenizer_artifact_dir

ArtifactProgressCallback = Callable[[dict[str, object]], None]
ArtifactStatusCallback = Callable[[dict[str, object]], None]


@dataclass(frozen=True)
class TokenizerArtifactConfig:
    """Configuration for a reusable tokenizer artifact."""

    name: str
    source: str
    vocab_size: int
    max_chars: int
    use_fast: bool = True
    chunk_chars: int = MAX_FAST_TRAIN_CHUNK_CHARS
    notes: str = ""

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> TokenizerArtifactConfig:
        """Build a config from a notebook-style dictionary."""
        return cls(
            name=str(values["name"]),
            source=str(values["source"]),
            vocab_size=int(values["vocab_size"]),
            max_chars=int(values["max_chars"]),
            use_fast=bool(values.get("use_fast", True)),
            chunk_chars=int(values.get("chunk_chars", MAX_FAST_TRAIN_CHUNK_CHARS)),
            notes=str(values.get("notes", "")),
        )


@dataclass(frozen=True)
class TokenizerArtifact:
    """Loaded or newly trained tokenizer artifact plus optional source text."""

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
        tinystories_dir = root / "data" / "tinystories"
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

    tokenizer = BPETokenizer()
    start = perf_counter()
    if config.vocab_size == 256:
        ids = list(text.encode("utf-8"))
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
        ids = tokenizer.train_fast(
            text,
            vocab_size=config.vocab_size,
            show_progress=False,
            chunk_chars=effective_chunk_chars,
            progress_callback=lambda info: _emit_fast_status(
                status_callback,
                name=config.name,
                info=info,
            ),
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
        ids = tokenizer.train(
            text,
            vocab_size=config.vocab_size,
            progress_callback=progress_callback,
            progress_every=_tokenizer_progress_every(config.vocab_size),
        )

    manifest = _build_manifest(config, text, ids, tokenizer, perf_counter() - start)
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


def _build_manifest(
    config: TokenizerArtifactConfig,
    text: str,
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
        "max_chars": config.max_chars,
        "actual_chars": len(text),
        "token_count": len(ids),
        "chars_per_token": len(text) / max(1, len(ids)),
        "token_ids_file": TOKEN_IDS_FILENAME,
        "token_ids_dtype": "uint32",
        "trainer": "rust-tokenizers" if config.use_fast else "python-scaffold",
        "chunk_chars": min(config.chunk_chars, MAX_FAST_TRAIN_CHUNK_CHARS)
        if config.use_fast
        else None,
        "requested_chunk_chars": config.chunk_chars if config.use_fast else None,
        "elapsed_seconds": elapsed_seconds,
        "notes": config.notes,
    }


def _tokenizer_progress_every(vocab_size: int, updates: int = 80) -> int:
    return max(1, (vocab_size - 256) // updates)


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
