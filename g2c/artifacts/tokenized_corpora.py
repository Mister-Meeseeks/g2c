"""Disk-backed tokenized corpus artifacts for language-model training.

Raw corpora in ``data/`` stay as compressed text. This module builds one flat
token-ID artifact under ``artifacts/tokenized-corpora/``. Training notebooks can
then create cheap train/validation views over that file without re-tokenizing.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from codecs import getincrementaldecoder
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from g2c.tokenizer import BPETokenizer
from g2c.tokenizer.fast import FastBPEEncoder

from .corpora import iter_corpus_byte_chunks, resolve_corpus
from .models import atomic_json_save
from .paths import find_repo_root, tokenized_corpus_artifact_dir
from .tokenizers import load_required_tokenizer

TokenizedCorpusProgressCallback = Callable[[dict[str, object]], None]

_DTYPES: dict[str, np.dtype] = {
    "uint16": np.dtype("<u2"),
    "uint32": np.dtype("<u4"),
}

_WORKER_ENCODER: FastBPEEncoder | None = None


@dataclass
class TokenizedCorpus:
    """A view over a flat token-ID file with random-window sampling helpers."""

    name: str
    split: str
    path: Path
    dtype: str
    total_token_count: int
    start: int = 0
    end: int | None = None
    _array: np.memmap | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        end = self.total_token_count if self.end is None else self.end
        if not 0 <= self.start <= end <= self.total_token_count:
            raise ValueError("invalid tokenized corpus view bounds")
        self.end = end

    def __len__(self) -> int:
        assert self.end is not None
        return self.end - self.start

    @property
    def array(self) -> np.memmap:
        """Return a lazy memory map over the full token file."""
        if self._array is None:
            dtype = _numpy_dtype(self.dtype)
            self._array = np.memmap(
                self.path,
                dtype=dtype,
                mode="r",
                shape=(self.total_token_count,),
            )
        return self._array

    def view(self, split: str, start: int, end: int) -> TokenizedCorpus:
        """Return a cheap sub-view over this same token file."""
        return TokenizedCorpus(
            name=self.name,
            split=split,
            path=self.path,
            dtype=self.dtype,
            total_token_count=self.total_token_count,
            start=start,
            end=end,
        )

    def get_lm_batch(
        self,
        batch_size: int,
        context_length: int,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a language-model batch without loading the whole corpus."""
        n = len(self)
        if n <= context_length:
            raise ValueError(
                f"{self.name}/{self.split} has length {n}; need at least "
                f"context_length+1 = {context_length + 1}."
            )

        starts = torch.randint(
            0,
            n - context_length,
            (batch_size,),
            generator=generator,
        ).tolist()
        arr = self.array
        x = np.empty((batch_size, context_length), dtype=np.int64)
        y = np.empty((batch_size, context_length), dtype=np.int64)
        for row, relative_start in enumerate(starts):
            start = self.start + relative_start
            x[row, :] = arr[start : start + context_length]
            y[row, :] = arr[start + 1 : start + context_length + 1]
        return torch.from_numpy(x), torch.from_numpy(y)


@dataclass(frozen=True)
class TokenizedCorpusPair:
    """Train/validation views over one tokenized corpus artifact."""

    train: TokenizedCorpus
    val: TokenizedCorpus


@dataclass(frozen=True)
class TokenizedCorpusArtifact:
    """Loaded tokenized corpus artifact plus metadata."""

    name: str
    artifact_dir: Path
    tokens: TokenizedCorpus
    manifest: dict[str, Any]
    tokenizer: BPETokenizer | None = None

    def split(self, train_fraction: float = 0.9) -> TokenizedCorpusPair:
        """Return train/validation views over the same disk-backed token file."""
        if not 0.0 < train_fraction < 1.0:
            raise ValueError("train_fraction must be between 0 and 1")
        split_at = int(len(self.tokens) * train_fraction)
        if split_at == 0 or split_at == len(self.tokens):
            raise ValueError("split would produce an empty train or validation view")
        train = self.tokens.view("train", self.tokens.start, self.tokens.start + split_at)
        val = self.tokens.view("val", self.tokens.start + split_at, self.tokens.end)
        return TokenizedCorpusPair(train=train, val=val)


def tokenized_corpus_artifact_exists(
    name: str,
    *,
    repo_root: str | Path | None = None,
) -> bool:
    """Return True when a tokenized corpus artifact has manifest + token file."""
    artifact_dir = tokenized_corpus_artifact_dir(name, repo_root)
    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    filename = manifest.get("file")
    return isinstance(filename, str) and (artifact_dir / filename).exists()


def load_tokenized_corpus_artifact(
    name: str,
    *,
    repo_root: str | Path | None = None,
    load_tokenizer: bool = True,
) -> TokenizedCorpusArtifact:
    """Load a disk-backed tokenized corpus artifact."""
    root = find_repo_root(repo_root)
    artifact_dir = tokenized_corpus_artifact_dir(name, root)
    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing tokenized corpus manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    filename = manifest.get("file")
    dtype = manifest.get("dtype")
    token_count = manifest.get("token_count")
    if not isinstance(filename, str) or not isinstance(dtype, str):
        raise ValueError(f"tokenized corpus manifest is missing file/dtype: {manifest_path}")
    if not isinstance(token_count, int):
        raise ValueError(f"tokenized corpus manifest is missing token_count: {manifest_path}")

    path = artifact_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"missing tokenized corpus file: {path}")

    tokenizer = None
    if load_tokenizer:
        tokenizer_name = manifest.get("tokenizer_artifact")
        if isinstance(tokenizer_name, str) and tokenizer_name:
            tokenizer = load_required_tokenizer(tokenizer_name, repo_root=root)

    return TokenizedCorpusArtifact(
        name=name,
        artifact_dir=artifact_dir,
        tokens=TokenizedCorpus(
            name=name,
            split="all",
            path=path,
            dtype=dtype,
            total_token_count=token_count,
        ),
        manifest=manifest,
        tokenizer=tokenizer,
    )


def build_or_load_tokenized_corpus(
    name: str,
    *,
    corpus: str,
    tokenizer_name: str,
    byte_count: int | None,
    source_split: str = "train",
    vocab_size: int | None = None,
    repo_root: str | Path | None = None,
    force: bool = False,
    chunk_offset: int = 0,
    chunk_bytes: int = 1024 * 1024,
    workers: int = 1,
    progress_callback: TokenizedCorpusProgressCallback | None = None,
) -> TokenizedCorpusArtifact:
    """Build or load one reusable disk-backed tokenized corpus."""
    root = find_repo_root(repo_root)
    artifact_dir = tokenized_corpus_artifact_dir(name, root)
    if tokenized_corpus_artifact_exists(name, repo_root=root) and not force:
        _emit(progress_callback, phase="loaded", name=name, artifact_dir=artifact_dir)
        return load_tokenized_corpus_artifact(name, repo_root=root)

    tokenizer = load_required_tokenizer(tokenizer_name, repo_root=root)
    effective_vocab_size = tokenizer.effective_vocab_size(vocab_size)
    dtype = _dtype_for_vocab_size(effective_vocab_size)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    _emit(
        progress_callback,
        phase="build_start",
        name=name,
        corpus=corpus,
        source_split=source_split,
        tokenizer_artifact=tokenizer_name,
        vocab_size=effective_vocab_size,
        dtype=dtype,
        workers=workers,
    )

    token_count, chars_seen = _write_tokenized_stream(
        artifact_dir=artifact_dir,
        name=name,
        corpus=corpus,
        source_split=source_split,
        byte_count=byte_count,
        tokenizer=tokenizer,
        vocab_size=vocab_size,
        dtype=dtype,
        repo_root=root,
        chunk_offset=chunk_offset,
        chunk_bytes=chunk_bytes,
        workers=workers,
        progress_callback=progress_callback,
    )

    filename = f"tokens.{dtype}.bin"
    manifest: dict[str, Any] = {
        "name": name,
        "kind": "tokenized-corpus",
        "created_at": datetime.now(UTC).isoformat(),
        "corpus": corpus,
        "source_split": source_split,
        "byte_count": byte_count,
        "chars_seen": chars_seen,
        "tokenizer_artifact": tokenizer_name,
        "requested_vocab_size": vocab_size,
        "effective_vocab_size": effective_vocab_size,
        "dtype": dtype,
        "file": filename,
        "token_count": token_count,
        "tokenizer_digest": _tokenizer_digest(tokenizer),
        "chunk_bytes": chunk_bytes,
        "chunk_offset": chunk_offset,
        "workers": workers,
    }
    atomic_json_save(manifest, artifact_dir / "manifest.json")
    _emit(
        progress_callback,
        phase="build_done",
        name=name,
        token_count=token_count,
        chars_seen=chars_seen,
        artifact_dir=artifact_dir,
    )
    return load_tokenized_corpus_artifact(name, repo_root=root)


def _write_tokenized_stream(
    *,
    artifact_dir: Path,
    name: str,
    corpus: str,
    source_split: str,
    byte_count: int | None,
    tokenizer: BPETokenizer,
    vocab_size: int | None,
    dtype: str,
    repo_root: Path,
    chunk_offset: int,
    chunk_bytes: int,
    workers: int,
    progress_callback: TokenizedCorpusProgressCallback | None,
) -> tuple[int, int]:
    spec = resolve_corpus(corpus, split=source_split, repo_root=repo_root)
    if spec is None:
        raise FileNotFoundError(f"local corpus {corpus!r} has no {source_split!r} split")
    byte_target = byte_count if byte_count is not None else spec.uncompressed_bytes

    path = artifact_dir / f"tokens.{dtype}.bin"
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    start = time.perf_counter()
    _emit(
        progress_callback,
        phase="stream_start",
        name=name,
        corpus=corpus,
        source_split=source_split,
        byte_count=byte_count,
        byte_target=byte_target,
        bytes_seen=0,
        path=path,
    )

    chunks = _iter_text_chunks_with_byte_progress(
        corpus,
        byte_count,
        chunk_offset=chunk_offset,
        split=source_split,
        chunk_bytes=chunk_bytes,
        repo_root=repo_root,
        special_tokens=tokenizer.special_tokens,
    )
    token_count = 0
    chunk_count = 0
    chars_seen = 0
    bytes_seen = 0

    try:
        with tmp_path.open("wb") as f:
            if workers <= 1:
                encoder = FastBPEEncoder(tokenizer, vocab_size=vocab_size)
                for chunk_count, (text, bytes_seen) in enumerate(chunks, start=1):
                    ids = encoder.encode(text)
                    chars_seen += len(text)
                    token_count += _write_ids(f, ids, dtype=dtype)
                    _emit_stream_chunk(
                        progress_callback,
                        name=name,
                        corpus=corpus,
                        source_split=source_split,
                        chunk_count=chunk_count,
                        byte_target=byte_target,
                        bytes_seen=bytes_seen,
                        chars_seen=chars_seen,
                        token_count=token_count,
                        start=start,
                    )
            else:
                for chunk_count, chars, bytes_seen, written in _encode_parallel_in_order(
                    chunks,
                    tokenizer=tokenizer,
                    vocab_size=vocab_size,
                    workers=workers,
                    max_in_flight=max(2, workers * 2),
                ):
                    chars_seen += chars
                    token_count += _write_ids(f, written, dtype=dtype)
                    _emit_stream_chunk(
                        progress_callback,
                        name=name,
                        corpus=corpus,
                        source_split=source_split,
                        chunk_count=chunk_count,
                        byte_target=byte_target,
                        bytes_seen=bytes_seen,
                        chars_seen=chars_seen,
                        token_count=token_count,
                        start=start,
                    )
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    _emit(
        progress_callback,
        phase="stream_done",
        name=name,
        corpus=corpus,
        source_split=source_split,
        chunks=chunk_count,
        byte_target=byte_target,
        bytes_seen=bytes_seen,
        chars_seen=chars_seen,
        token_count=token_count,
        elapsed_seconds=time.perf_counter() - start,
    )
    if token_count == 0:
        raise ValueError("tokenized corpus produced zero tokens")
    return token_count, chars_seen


def _encode_parallel_in_order(
    chunks: Iterable[tuple[str, int]],
    *,
    tokenizer: BPETokenizer,
    vocab_size: int | None,
    workers: int,
    max_in_flight: int,
) -> Iterator[tuple[int, int, int, list[int]]]:
    futures: dict[Future[tuple[int, int, int, list[int]]], int] = {}
    pending: dict[int, tuple[int, int, list[int]]] = {}
    next_index = 1
    next_write = 1
    tokenizer_payload = tokenizer.to_dict()

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(tokenizer_payload, vocab_size),
    ) as executor:
        for text, bytes_seen in chunks:
            while len(futures) >= max_in_flight:
                _collect_completed(futures, pending)
                while next_write in pending:
                    chars, chunk_bytes_seen, ids = pending.pop(next_write)
                    yield next_write, chars, chunk_bytes_seen, ids
                    next_write += 1

            future = executor.submit(
                _encode_chunk_in_worker,
                next_index,
                text,
                bytes_seen,
            )
            futures[future] = next_index
            next_index += 1

        while futures:
            _collect_completed(futures, pending)
            while next_write in pending:
                chars, chunk_bytes_seen, ids = pending.pop(next_write)
                yield next_write, chars, chunk_bytes_seen, ids
                next_write += 1


def _iter_text_chunks_with_byte_progress(
    corpus: str,
    byte_count: int | None,
    *,
    chunk_offset: int,
    split: str,
    chunk_bytes: int,
    repo_root: Path,
    special_tokens: tuple[str, ...],
) -> Iterator[tuple[str, int]]:
    """Yield decoded text chunks and cumulative raw bytes consumed."""
    decoder = getincrementaldecoder("utf-8")(errors="replace")
    carry_len = max((len(token) for token in special_tokens), default=0)
    carry = ""
    bytes_seen = 0

    for byte_chunk in iter_corpus_byte_chunks(
        corpus,
        byte_count,
        chunk_offset=chunk_offset,
        split=split,
        chunk_bytes=chunk_bytes,
        repo_root=repo_root,
    ):
        bytes_seen += len(byte_chunk)
        decoded = decoder.decode(byte_chunk, final=False)
        if not decoded:
            continue
        text = carry + decoded
        if carry_len == 0:
            yield text, bytes_seen
            carry = ""
            continue
        if len(text) <= carry_len:
            carry = text
            continue
        emit_until = len(text) - carry_len
        yield text[:emit_until], bytes_seen
        carry = text[emit_until:]

    tail = decoder.decode(b"", final=True)
    if tail:
        text = carry + tail
        if carry_len == 0:
            yield text, bytes_seen
            carry = ""
        elif len(text) <= carry_len:
            carry = text
        else:
            emit_until = len(text) - carry_len
            yield text[:emit_until], bytes_seen
            carry = text[emit_until:]

    if carry:
        yield carry, bytes_seen


def _collect_completed(
    futures: dict[Future[tuple[int, int, int, list[int]]], int],
    pending: dict[int, tuple[int, int, list[int]]],
) -> None:
    done, _ = wait(futures, return_when=FIRST_COMPLETED)
    for future in done:
        futures.pop(future)
        index, chars, bytes_seen, ids = future.result()
        pending[index] = (chars, bytes_seen, ids)


def _init_worker(tokenizer_payload: dict[str, object], vocab_size: int | None) -> None:
    global _WORKER_ENCODER
    tokenizer = BPETokenizer.from_dict(tokenizer_payload)
    _WORKER_ENCODER = FastBPEEncoder(tokenizer, vocab_size=vocab_size)


def _encode_chunk_in_worker(
    index: int,
    text: str,
    bytes_seen: int,
) -> tuple[int, int, int, list[int]]:
    if _WORKER_ENCODER is None:
        raise RuntimeError("tokenized corpus worker encoder is not initialized")
    return index, len(text), bytes_seen, _WORKER_ENCODER.encode(text)


def _write_ids(f, ids: list[int], *, dtype: str) -> int:
    if not ids:
        return 0
    np_dtype = _numpy_dtype(dtype)
    max_value = int(np.iinfo(np_dtype).max)
    observed = max(ids)
    if observed > max_value:
        raise ValueError(f"token ID {observed} does not fit in {dtype}")
    np.asarray(ids, dtype=np_dtype).tofile(f)
    return len(ids)


def _dtype_for_vocab_size(vocab_size: int) -> str:
    if vocab_size <= 2**16:
        return "uint16"
    return "uint32"


def _tokenizer_digest(tokenizer: BPETokenizer) -> str:
    payload = json.dumps(tokenizer.to_dict(), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _numpy_dtype(dtype: str) -> np.dtype:
    try:
        return _DTYPES[dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported tokenized corpus dtype: {dtype}") from exc


def _emit_stream_chunk(
    callback: TokenizedCorpusProgressCallback | None,
    *,
    name: str,
    corpus: str,
    source_split: str,
    chunk_count: int,
    byte_target: int,
    bytes_seen: int,
    chars_seen: int,
    token_count: int,
    start: float,
) -> None:
    _emit(
        callback,
        phase="stream_chunk",
        name=name,
        corpus=corpus,
        source_split=source_split,
        chunks=chunk_count,
        byte_target=byte_target,
        bytes_seen=bytes_seen,
        chars_seen=chars_seen,
        token_count=token_count,
        elapsed_seconds=time.perf_counter() - start,
    )


def _emit(
    callback: TokenizedCorpusProgressCallback | None,
    **payload: object,
) -> None:
    if callback is not None:
        callback(payload)
