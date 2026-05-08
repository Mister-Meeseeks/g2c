"""Streaming loaders for reusable text corpora.

The public helpers return ordinary `str`/`bytes` objects because notebooks and
tokenizers usually want a concrete slice of text. Internally, they read only the
requested byte window from the underlying shards, starting from a shard offset
and wrapping when the request is larger than the local corpus.
"""
from __future__ import annotations

import gzip
import json
from codecs import getincrementaldecoder
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from .paths import find_repo_root


@dataclass(frozen=True)
class CorpusShard:
    """One physical shard in a local corpus."""

    path: Path
    uncompressed_bytes: int


@dataclass(frozen=True)
class CorpusSource:
    """A logical source within a corpus, such as TinyStories or Cosmopedia."""

    name: str
    shards: tuple[CorpusShard, ...]
    uncompressed_bytes: int


@dataclass(frozen=True)
class CorpusSpec:
    """Resolved local corpus metadata."""

    name: str
    sources: tuple[CorpusSource, ...]
    split: str
    root: Path

    @property
    def uncompressed_bytes(self) -> int:
        """Total available uncompressed bytes for this split."""
        return sum(source.uncompressed_bytes for source in self.sources)


def load_corpus_text(
    corpus: str,
    byte_count: int | None = None,
    *,
    chunk_offset: int = 0,
    split: str = "train",
    repo_root: str | Path | None = None,
) -> str | None:
    """Load a byte-bounded slice of a local corpus and decode it as UTF-8.

    `byte_count=None` loads the whole available split once. If `byte_count` is
    larger than the available split, shards wrap around and are oversampled.
    `chunk_offset` selects the starting shard modulo the shard count.
    """
    data = load_corpus_bytes(
        corpus,
        byte_count,
        chunk_offset=chunk_offset,
        split=split,
        repo_root=repo_root,
    )
    if data is None:
        return None
    return data.decode("utf-8", errors="replace")


def iter_corpus_text_chunks(
    corpus: str,
    byte_count: int | None = None,
    *,
    chunk_offset: int = 0,
    split: str = "train",
    chunk_bytes: int = 1024 * 1024,
    repo_root: str | Path | None = None,
) -> Iterator[str]:
    """Yield a byte-bounded corpus slice as decoded UTF-8 text chunks.

    This is the streaming companion to ``load_corpus_text``. It preserves the
    same corpus/source allocation rules, but avoids materializing the whole
    requested slice in memory. Chunk boundaries may fall in the middle of a
    multibyte UTF-8 sequence, so decoding uses an incremental decoder.
    """
    decoder = getincrementaldecoder("utf-8")(errors="replace")
    for chunk in iter_corpus_byte_chunks(
        corpus,
        byte_count,
        chunk_offset=chunk_offset,
        split=split,
        chunk_bytes=chunk_bytes,
        repo_root=repo_root,
    ):
        text = decoder.decode(chunk, final=False)
        if text:
            yield text
    tail = decoder.decode(b"", final=True)
    if tail:
        yield tail


def iter_corpus_byte_chunks(
    corpus: str,
    byte_count: int | None = None,
    *,
    chunk_offset: int = 0,
    split: str = "train",
    chunk_bytes: int = 1024 * 1024,
    repo_root: str | Path | None = None,
) -> Iterator[bytes]:
    """Yield a byte-bounded corpus slice without joining chunks in memory."""
    if byte_count is not None and byte_count < 0:
        raise ValueError("byte_count must be non-negative or None")
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")

    spec = resolve_corpus(corpus, split=split, repo_root=repo_root)
    if spec is None or byte_count == 0:
        return

    allocations = _source_byte_allocations(spec, byte_count)
    for source, source_bytes in allocations:
        if source_bytes <= 0:
            continue
        yield from _iter_source_byte_chunks(
            source,
            source_bytes,
            chunk_offset=chunk_offset,
            chunk_bytes=chunk_bytes,
        )


def load_corpus_bytes(
    corpus: str,
    byte_count: int | None = None,
    *,
    chunk_offset: int = 0,
    split: str = "train",
    repo_root: str | Path | None = None,
) -> bytes | None:
    """Load a byte-bounded slice of a local corpus.

    G2C corpus requests preserve source mix by allocating bytes to each
    sub-corpus in proportion to its local uncompressed bytes for the requested
    split. For example, if Cosmopedia is 20% of the local G2C train split, a
    250MB request includes about 50MB from Cosmopedia.
    """
    if byte_count is not None and byte_count < 0:
        raise ValueError("byte_count must be non-negative or None")

    spec = resolve_corpus(corpus, split=split, repo_root=repo_root)
    if spec is None:
        return None
    if byte_count == 0:
        return b""

    allocations = _source_byte_allocations(spec, byte_count)
    parts = [
        _read_source_bytes(source, source_bytes, chunk_offset=chunk_offset)
        for source, source_bytes in allocations
        if source_bytes > 0
    ]
    return b"".join(parts)


def resolve_corpus(
    corpus: str,
    *,
    split: str = "train",
    repo_root: str | Path | None = None,
) -> CorpusSpec | None:
    """Resolve a named corpus to local shard metadata.

    The logical `g2c` corpus prefers the full corpus when both full and small
    builds are present, and falls back to the small build otherwise. Explicit
    `g2c-corpus-full` and `g2c-corpus-small` requests resolve only that local
    corpus variant.
    """
    root = find_repo_root(repo_root)
    normalized_split = _normalize_split(split)

    if corpus == "g2c-corpus-full":
        return _resolve_g2c_dir(
            root / "data" / "g2c-corpus-v1",
            split=normalized_split,
            repo_root=root,
        )
    if corpus == "g2c-corpus-small":
        return _resolve_g2c_dir(
            root / "data" / "g2c-corpus-v1-small",
            split=normalized_split,
            repo_root=root,
        )

    normalized = _normalize_corpus_name(corpus)
    if normalized == "tinyshakespeare":
        return _resolve_single_file_corpus(
            name="tinyshakespeare",
            root=root,
            path=root / "data" / "tinyshakespeare.txt",
            split=normalized_split,
        )
    if normalized == "tinystories-100MB":
        return _resolve_tinystories(root, normalized_split, sample_only=True)
    if normalized == "tinystories":
        return _resolve_tinystories(root, normalized_split)
    if normalized == "g2c":
        return _resolve_g2c(root, normalized_split)
    raise ValueError(f"unknown corpus: {corpus}")


def _normalize_corpus_name(corpus: str) -> str:
    return corpus


def _normalize_split(split: str) -> str:
    if split == "valid":
        return "val"
    if split not in {"train", "val"}:
        raise ValueError("split must be 'train', 'val', or 'valid'")
    return split


def _resolve_single_file_corpus(
    *,
    name: str,
    root: Path,
    path: Path,
    split: str,
) -> CorpusSpec | None:
    if split != "train" or not path.exists():
        return None
    shard = CorpusShard(path=path, uncompressed_bytes=path.stat().st_size)
    source = CorpusSource(name=name, shards=(shard,), uncompressed_bytes=shard.uncompressed_bytes)
    return CorpusSpec(name=name, sources=(source,), split=split, root=root)


def _resolve_tinystories(
    root: Path,
    split: str,
    *,
    sample_only: bool = False,
) -> CorpusSpec | None:
    tinystories_dir = root / "data" / "tinystories"
    if split == "train" and sample_only:
        candidate_groups = (
            _tinystories_shards(tinystories_dir, "TinyStories-train-100MB"),
            _single_path_shards(tinystories_dir / "TinyStories-train-100MB.txt"),
        )
    elif split == "train":
        candidate_groups = (
            _tinystories_shards(tinystories_dir, "TinyStories-train"),
            _single_path_shards(tinystories_dir / "TinyStories-train.txt"),
            _tinystories_shards(tinystories_dir, "TinyStories-train-100MB"),
            _single_path_shards(tinystories_dir / "TinyStories-train-100MB.txt"),
        )
    elif sample_only:
        candidate_groups = ()
    else:
        candidate_groups = (
            _tinystories_shards(tinystories_dir, "TinyStories-valid"),
            _single_path_shards(tinystories_dir / "TinyStories-valid.txt"),
        )

    for shards in candidate_groups:
        if shards:
            source = CorpusSource(
                name="tinystories",
                shards=shards,
                uncompressed_bytes=sum(shard.uncompressed_bytes for shard in shards),
            )
            return CorpusSpec(name="tinystories", sources=(source,), split=split, root=root)
    return None


def _resolve_g2c(root: Path, split: str) -> CorpusSpec | None:
    for corpus_dir in (
        root / "data" / "g2c-corpus-v1",
        root / "data" / "g2c-corpus-v1-small",
    ):
        spec = _resolve_g2c_dir(corpus_dir, split=split, repo_root=root)
        if spec is not None:
            return spec
    return None


def _resolve_g2c_dir(corpus_dir: Path, *, split: str, repo_root: Path) -> CorpusSpec | None:
    manifest_path = corpus_dir / "manifest.json"
    if not manifest_path.exists():
        return None

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sources: list[CorpusSource] = []
    for source_row in manifest.get("sources", []):
        shards: list[CorpusShard] = []
        for shard_row in source_row.get("shards", []):
            if shard_row.get("split") != split:
                continue
            path = corpus_dir / shard_row["path"]
            if not path.exists():
                continue
            shards.append(
                CorpusShard(
                    path=path,
                    uncompressed_bytes=int(shard_row["uncompressed_bytes"]),
                )
            )
        if shards:
            sources.append(
                CorpusSource(
                    name=str(source_row["source"]),
                    shards=tuple(shards),
                    uncompressed_bytes=sum(shard.uncompressed_bytes for shard in shards),
                )
            )

    if not sources:
        return None
    return CorpusSpec(name="g2c", sources=tuple(sources), split=split, root=repo_root)


def _tinystories_shards(tinystories_dir: Path, prefix: str) -> tuple[CorpusShard, ...]:
    manifest_path = tinystories_dir / f"{prefix}-shards.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        shards = []
        for row in manifest.get("shards", []):
            path = tinystories_dir / row["path"]
            if path.exists():
                shards.append(
                    CorpusShard(
                        path=path,
                        uncompressed_bytes=int(row["uncompressed_bytes"]),
                    )
                )
        if shards:
            return tuple(shards)

    paths = sorted(tinystories_dir.glob(f"{prefix}-[0-9][0-9][0-9][0-9].txt.gz"))
    return tuple(
        CorpusShard(path=path, uncompressed_bytes=_uncompressed_size(path))
        for path in paths
    )


def _single_path_shards(path: Path) -> tuple[CorpusShard, ...]:
    if not path.exists():
        return ()
    return (CorpusShard(path=path, uncompressed_bytes=path.stat().st_size),)


def _uncompressed_size(path: Path) -> int:
    if path.suffix != ".gz":
        return path.stat().st_size
    total = 0
    with gzip.open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            total += len(chunk)
    return total


def _source_byte_allocations(
    spec: CorpusSpec,
    byte_count: int | None,
) -> list[tuple[CorpusSource, int]]:
    if byte_count is None:
        return [(source, source.uncompressed_bytes) for source in spec.sources]
    if len(spec.sources) <= 1:
        return [(source, byte_count) for source in spec.sources]

    total = spec.uncompressed_bytes
    if total <= 0:
        return []

    bases: list[int] = []
    remainders: list[tuple[int, int]] = []
    for index, source in enumerate(spec.sources):
        base, remainder = divmod(byte_count * source.uncompressed_bytes, total)
        bases.append(base)
        remainders.append((remainder, index))

    remaining = byte_count - sum(bases)
    for _, index in sorted(remainders, reverse=True)[:remaining]:
        bases[index] += 1

    return list(zip(spec.sources, bases, strict=True))


def _read_source_bytes(
    source: CorpusSource,
    byte_count: int,
    *,
    chunk_offset: int,
) -> bytes:
    if byte_count <= 0 or not source.shards:
        return b""

    result = bytearray()
    for chunk in _iter_source_byte_chunks(
        source,
        byte_count,
        chunk_offset=chunk_offset,
        chunk_bytes=1024 * 1024,
    ):
        result.extend(chunk)
    return bytes(result)


def _iter_source_byte_chunks(
    source: CorpusSource,
    byte_count: int,
    *,
    chunk_offset: int,
    chunk_bytes: int,
) -> Iterator[bytes]:
    if byte_count <= 0 or not source.shards:
        return

    produced = 0
    start = chunk_offset % len(source.shards)
    index = start
    empty_reads = 0
    while produced < byte_count and empty_reads < len(source.shards):
        shard = source.shards[index]
        needed = byte_count - produced
        wrote = False
        for chunk in _iter_shard_prefix_chunks(
            shard.path,
            needed,
            chunk_bytes=chunk_bytes,
        ):
            if not chunk:
                continue
            produced += len(chunk)
            wrote = True
            yield chunk
            if produced >= byte_count:
                break
        empty_reads = 0 if wrote else empty_reads + 1
        index = (index + 1) % len(source.shards)


def _read_shard_prefix(path: Path, byte_count: int) -> bytes:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        return f.read(byte_count)


def _iter_shard_prefix_chunks(
    path: Path,
    byte_count: int,
    *,
    chunk_bytes: int,
) -> Iterator[bytes]:
    opener = gzip.open if path.suffix == ".gz" else open
    remaining = byte_count
    with opener(path, "rb") as f:
        while remaining > 0:
            chunk = f.read(min(chunk_bytes, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk
