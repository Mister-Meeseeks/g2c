#!/usr/bin/env python3
"""Build raw G2C corpus shards from streamed upstream datasets.

The output is intentionally human-readable raw text, compressed into shards and
kept split by source. Later tokenizer/training scripts can sample from the
manifest without rebuilding the corpus.

Examples:

    python scripts/gen_corpus.py --preset small
    python scripts/gen_corpus.py --preset full
    python scripts/gen_corpus.py --fineweb-edu 5GB --cosmopedia 2.5GB \
        --tinystories 1GB --codesearchnet 1GB
    python scripts/gen_corpus.py --preset full --codesearchnet-js-ratio 0.2

The script streams Hugging Face datasets and stops once each source reaches its
requested uncompressed UTF-8 byte quota. It does not download full upstream
datasets.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import shutil
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.request import urlopen

END_OF_TEXT = "<|endoftext|>"

PRESETS: dict[str, dict[str, str]] = {
    "small": {
        "fineweb_edu": "512MB",
        "cosmopedia": "256MB",
        "tinystories": "128MB",
        "codesearchnet": "128MB",
    },
    "full": {
        "fineweb_edu": "5GB",
        "cosmopedia": "2.5GB",
        "tinystories": "1GB",
        "codesearchnet": "1GB",
    },
}


@dataclass(frozen=True)
class SourceSpec:
    name: str
    quota_bytes: int
    iterator_factory: Callable[[], Iterator[str]]


@dataclass
class ShardRecord:
    path: str
    split: str
    source: str
    documents: int
    uncompressed_bytes: int
    compressed_bytes: int
    sha256_uncompressed_text: str


@dataclass
class SourceResult:
    source: str
    requested_bytes: int
    train_requested_bytes: int
    val_requested_bytes: int
    train_actual_bytes: int
    val_actual_bytes: int
    documents: int
    exhausted: bool
    shards: list[ShardRecord] = field(default_factory=list)


class ShardWriter:
    """Write source/split text into size-bounded compressed shards."""

    def __init__(
        self,
        *,
        root: Path,
        source: str,
        split: str,
        shard_size_bytes: int,
        compression: str,
    ) -> None:
        self.root = root
        self.source = source
        self.split = split
        self.shard_size_bytes = shard_size_bytes
        self.compression = compression
        self.source_dir = root / "raw" / source
        self.source_dir.mkdir(parents=True, exist_ok=True)

        self.total_uncompressed_bytes = 0
        self.total_documents = 0
        self.records: list[ShardRecord] = []

        self._index = 0
        self._path: Path | None = None
        self._fh: Any | None = None
        self._sha = hashlib.sha256()
        self._shard_uncompressed_bytes = 0
        self._shard_documents = 0

    def write_document(self, text: str) -> int:
        payload = f"{text.rstrip()}\n{END_OF_TEXT}\n".encode()
        if (
            self._fh is not None
            and self._shard_uncompressed_bytes > 0
            and self._shard_uncompressed_bytes + len(payload) > self.shard_size_bytes
        ):
            self.close_shard()
        if self._fh is None:
            self.open_shard()

        assert self._fh is not None
        self._fh.write(payload)
        self._sha.update(payload)
        self._shard_uncompressed_bytes += len(payload)
        self._shard_documents += 1
        self.total_uncompressed_bytes += len(payload)
        self.total_documents += 1
        return len(payload)

    def open_shard(self) -> None:
        suffix = ".txt.gz" if self.compression == "gzip" else ".txt"
        name = f"{self.split}-{self._index:05d}{suffix}"
        self._path = self.source_dir / name
        if self.compression == "gzip":
            self._fh = gzip.open(self._path, "wb", compresslevel=6)
        elif self.compression == "none":
            self._fh = self._path.open("wb")
        else:
            raise ValueError(f"unsupported compression: {self.compression}")
        self._sha = hashlib.sha256()
        self._shard_uncompressed_bytes = 0
        self._shard_documents = 0

    def close_shard(self) -> None:
        if self._fh is None:
            return
        assert self._path is not None
        self._fh.close()
        compressed_bytes = self._path.stat().st_size
        self.records.append(
            ShardRecord(
                path=str(self._path.relative_to(self.root)),
                split=self.split,
                source=self.source,
                documents=self._shard_documents,
                uncompressed_bytes=self._shard_uncompressed_bytes,
                compressed_bytes=compressed_bytes,
                sha256_uncompressed_text=self._sha.hexdigest(),
            )
        )
        self._index += 1
        self._path = None
        self._fh = None
        self._shard_uncompressed_bytes = 0
        self._shard_documents = 0

    def close(self) -> None:
        self.close_shard()


def parse_size(value: str | int) -> int:
    """Parse human-readable byte sizes like `512MB`, `2.5GB`, or `1000000`."""

    if isinstance(value, int):
        if value < 0:
            raise ValueError("size must be non-negative")
        return value
    text = value.strip()
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*([KMGT]?I?B?|)", text, re.I)
    if not match:
        raise ValueError(f"invalid size: {value!r}")
    number = float(match.group(1))
    unit = match.group(2).upper()
    multipliers = {
        "": 1,
        "B": 1,
        "K": 1_000,
        "KB": 1_000,
        "M": 1_000_000,
        "MB": 1_000_000,
        "G": 1_000_000_000,
        "GB": 1_000_000_000,
        "T": 1_000_000_000_000,
        "TB": 1_000_000_000_000,
        "KIB": 1024,
        "MIB": 1024**2,
        "GIB": 1024**3,
        "TIB": 1024**4,
    }
    if unit not in multipliers:
        raise ValueError(f"invalid size unit in {value!r}")
    return int(number * multipliers[unit])


def human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(n)
    for unit in units:
        if abs(value) < 1000 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)}B"
            return f"{value:.2f}{unit}"
        value /= 1000
    raise AssertionError("unreachable")


def normalize_text(text: str) -> str:
    """Normalize one document without destroying its basic structure."""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\x00", "")
    text = text.replace(END_OF_TEXT, "")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def row_text(row: dict[str, Any], fields: tuple[str, ...] = ("text",)) -> str:
    for field_name in fields:
        value = row.get(field_name)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def format_codesearchnet(row: dict[str, Any], language: str) -> str:
    """Turn a CodeSearchNet row into markdown-ish text."""

    doc = row_text(
        row,
        (
            "func_documentation_string",
            "docstring",
            "documentation",
            "description",
        ),
    )
    code = row_text(
        row,
        (
            "whole_func_string",
            "func_code_string",
            "code",
            "text",
        ),
    )
    if not code:
        return ""

    repo = row.get("repo") or row.get("repository_name") or row.get("repo_name")
    path = row.get("path") or row.get("func_path_in_repository") or row.get("file_path")
    metadata_lines = [f"# Language: {language}"]
    if repo:
        metadata_lines.insert(0, f"# Repository: {repo}")
    if path:
        metadata_lines.append(f"# Path: {path}")

    fence_language = "javascript" if language == "javascript" else language
    parts = ["\n".join(metadata_lines)]
    if doc:
        parts.append(doc)
    parts.append(f"```{fence_language}\n{code.rstrip()}\n```")
    return "\n\n".join(parts)


def require_datasets() -> Callable[..., Any]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Building G2C Corpus requires the Hugging Face datasets package.\n"
            "Install dev dependencies with `uv pip install -e '.[dev]'`, or run "
            "`uv pip install datasets` inside the project venv."
        ) from exc
    return load_dataset


def iter_hf_dataset(
    *,
    dataset: str,
    config: str | None,
    split: str,
    seed: int,
    shuffle_buffer: int,
    extractor: Callable[[dict[str, Any]], str],
) -> Iterator[str]:
    load_dataset = require_datasets()
    kwargs: dict[str, Any] = {"split": split, "streaming": True}
    if config:
        kwargs["name"] = config
    stream = load_dataset(dataset, **kwargs)
    if shuffle_buffer > 0:
        stream = stream.shuffle(seed=seed, buffer_size=shuffle_buffer)
    for row in stream:
        if isinstance(row, dict):
            yield extractor(row)


def iter_tinystories_documents(path: str, url: str) -> Iterator[str]:
    """Yield TinyStories documents from a local file if present, otherwise URL."""
    marker = END_OF_TEXT.encode("utf-8")

    buffer = b""
    for stream_cm in iter_tinystories_streams(path, url):
        with stream_cm as stream:
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                buffer += chunk
                pieces = buffer.split(marker)
                for piece in pieces[:-1]:
                    text = piece.decode("utf-8", errors="replace")
                    if text.strip():
                        yield text
                buffer = pieces[-1]
    if buffer.strip():
        yield buffer.decode("utf-8", errors="replace")


def iter_tinystories_streams(path: str, url: str):
    source_path = Path(path)
    if source_path.exists():
        yield source_path.open("rb")
        return

    tinystories_dir = source_path.parent
    shard_paths = sorted(tinystories_dir.glob("TinyStories-train-[0-9][0-9][0-9][0-9].txt.gz"))
    if shard_paths:
        for shard_path in shard_paths:
            yield gzip.open(shard_path, "rb")
        return

    sample_shards = sorted(
        tinystories_dir.glob("TinyStories-train-100MB-[0-9][0-9][0-9][0-9].txt.gz")
    )
    if sample_shards:
        for shard_path in sample_shards:
            yield gzip.open(shard_path, "rb")
        return

    yield urlopen(url)


def collect_source(
    spec: SourceSpec,
    *,
    out_dir: Path,
    val_ratio: float,
    shard_size_bytes: int,
    compression: str,
    min_doc_chars: int,
    progress_every_bytes: int,
) -> SourceResult:
    val_quota = int(spec.quota_bytes * val_ratio)
    train_quota = spec.quota_bytes - val_quota
    train_writer = ShardWriter(
        root=out_dir,
        source=spec.name,
        split="train",
        shard_size_bytes=shard_size_bytes,
        compression=compression,
    )
    val_writer = ShardWriter(
        root=out_dir,
        source=spec.name,
        split="val",
        shard_size_bytes=shard_size_bytes,
        compression=compression,
    )

    exhausted = True
    next_progress = progress_every_bytes if progress_every_bytes > 0 else 0
    try:
        for raw in spec.iterator_factory():
            text = normalize_text(raw)
            if len(text) < min_doc_chars:
                continue

            if val_writer.total_uncompressed_bytes < val_quota:
                val_writer.write_document(text)
            elif train_writer.total_uncompressed_bytes < train_quota:
                train_writer.write_document(text)
            else:
                exhausted = False
                break

            total = train_writer.total_uncompressed_bytes + val_writer.total_uncompressed_bytes
            if next_progress and total >= next_progress:
                print(
                    f"{spec.name}: {human_bytes(total)}/{human_bytes(spec.quota_bytes)} "
                    f"from {train_writer.total_documents + val_writer.total_documents} docs",
                    flush=True,
                )
                while total >= next_progress:
                    next_progress += progress_every_bytes
    finally:
        train_writer.close()
        val_writer.close()

    shards = [*val_writer.records, *train_writer.records]
    return SourceResult(
        source=spec.name,
        requested_bytes=spec.quota_bytes,
        train_requested_bytes=train_quota,
        val_requested_bytes=val_quota,
        train_actual_bytes=train_writer.total_uncompressed_bytes,
        val_actual_bytes=val_writer.total_uncompressed_bytes,
        documents=train_writer.total_documents + val_writer.total_documents,
        exhausted=exhausted,
        shards=shards,
    )


def parse_quota(name: str, value: str | None, preset: str) -> int:
    if value is None:
        value = PRESETS[preset][name]
    return parse_size(value)


def build_specs(args: argparse.Namespace) -> list[SourceSpec]:
    specs: list[SourceSpec] = []
    seed = args.seed
    buffer = args.shuffle_buffer

    fineweb_quota = parse_quota("fineweb_edu", args.fineweb_edu, args.preset)
    if fineweb_quota > 0:
        specs.append(
            SourceSpec(
                name="fineweb-edu-dedup",
                quota_bytes=fineweb_quota,
                iterator_factory=lambda: iter_hf_dataset(
                    dataset=args.fineweb_dataset,
                    config=args.fineweb_config,
                    split=args.fineweb_split,
                    seed=seed,
                    shuffle_buffer=buffer,
                    extractor=lambda row: row_text(row, ("text",)),
                ),
            )
        )

    cosmopedia_quota = parse_quota("cosmopedia", args.cosmopedia, args.preset)
    if cosmopedia_quota > 0:
        specs.append(
            SourceSpec(
                name="cosmopedia-v2",
                quota_bytes=cosmopedia_quota,
                iterator_factory=lambda: iter_hf_dataset(
                    dataset=args.cosmopedia_dataset,
                    config=args.cosmopedia_config,
                    split=args.cosmopedia_split,
                    seed=seed + 1,
                    shuffle_buffer=buffer,
                    extractor=lambda row: row_text(row, ("text", "content")),
                ),
            )
        )

    tinystories_quota = parse_quota("tinystories", args.tinystories, args.preset)
    if tinystories_quota > 0:
        specs.append(
            SourceSpec(
                name="tinystories",
                quota_bytes=tinystories_quota,
                iterator_factory=lambda: iter_tinystories_documents(
                    args.tinystories_path,
                    args.tinystories_url,
                ),
            )
        )

    code_quota = parse_quota("codesearchnet", args.codesearchnet, args.preset)
    if code_quota > 0:
        js_quota = int(code_quota * args.codesearchnet_js_ratio)
        py_quota = code_quota - js_quota
        if py_quota > 0:
            specs.append(
                SourceSpec(
                    name="codesearchnet-python",
                    quota_bytes=py_quota,
                    iterator_factory=lambda: iter_hf_dataset(
                        dataset=args.codesearchnet_dataset,
                        config=args.codesearchnet_python_config,
                        split=args.codesearchnet_split,
                        seed=seed + 2,
                        shuffle_buffer=buffer,
                        extractor=lambda row: format_codesearchnet(row, "python"),
                    ),
                )
            )
        if js_quota > 0:
            specs.append(
                SourceSpec(
                    name="codesearchnet-javascript",
                    quota_bytes=js_quota,
                    iterator_factory=lambda: iter_hf_dataset(
                        dataset=args.codesearchnet_dataset,
                        config=args.codesearchnet_javascript_config,
                        split=args.codesearchnet_split,
                        seed=seed + 3,
                        shuffle_buffer=buffer,
                        extractor=lambda row: format_codesearchnet(row, "javascript"),
                    ),
                )
            )

    return specs


def write_manifest(
    *,
    out_dir: Path,
    args: argparse.Namespace,
    results: list[SourceResult],
) -> None:
    manifest = {
        "name": "g2c-corpus-v1",
        "kind": "raw-text-corpus",
        "created_at": datetime.now(UTC).isoformat(),
        "preset": args.preset,
        "end_of_text": END_OF_TEXT,
        "compression": args.compression,
        "val_ratio": args.val_ratio,
        "shard_size_bytes": args.shard_size_bytes,
        "min_doc_chars": args.min_doc_chars,
        "sources": [asdict(result) for result in results],
        "totals": {
            "requested_bytes": sum(result.requested_bytes for result in results),
            "actual_uncompressed_bytes": sum(
                result.train_actual_bytes + result.val_actual_bytes for result in results
            ),
            "actual_compressed_bytes": sum(
                shard.compressed_bytes for result in results for shard in result.shards
            ),
            "documents": sum(result.documents for result in results),
        },
        "upstream": {
            "fineweb_edu": {
                "dataset": args.fineweb_dataset,
                "config": args.fineweb_config,
                "split": args.fineweb_split,
            },
            "cosmopedia": {
                "dataset": args.cosmopedia_dataset,
                "config": args.cosmopedia_config,
                "split": args.cosmopedia_split,
            },
            "tinystories": {
                "local_path": args.tinystories_path,
                "url": args.tinystories_url,
            },
            "codesearchnet": {
                "dataset": args.codesearchnet_dataset,
                "python_config": args.codesearchnet_python_config,
                "javascript_config": args.codesearchnet_javascript_config,
                "split": args.codesearchnet_split,
                "javascript_ratio": args.codesearchnet_js_ratio,
            },
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def has_any_files(path: Path) -> bool:
    return any(child.is_file() for child in path.rglob("*"))


def prepare_output_dirs(out_dir: Path, *, force: bool) -> Path:
    """Return a clean staging directory for an atomic corpus build."""

    staging_dir = out_dir.with_name(f"{out_dir.name}.partial")

    if out_dir.exists():
        if force:
            shutil.rmtree(out_dir)
        elif has_any_files(out_dir):
            raise SystemExit(f"{out_dir} already exists and is not empty; pass --force to rebuild")
        else:
            shutil.rmtree(out_dir)

    if staging_dir.exists():
        print(f"Removing incomplete previous build at {staging_dir}")
        shutil.rmtree(staging_dir)

    staging_dir.mkdir(parents=True, exist_ok=True)
    return staging_dir


def dry_run(args: argparse.Namespace, specs: list[SourceSpec]) -> None:
    plan = {
        "out": str(args.out),
        "preset": args.preset,
        "compression": args.compression,
        "val_ratio": args.val_ratio,
        "shard_size": human_bytes(args.shard_size_bytes),
        "sources": [
            {
                "name": spec.name,
                "quota_bytes": spec.quota_bytes,
                "quota": human_bytes(spec.quota_bytes),
            }
            for spec in specs
        ],
        "total": human_bytes(sum(spec.quota_bytes for spec in specs)),
    }
    print(json.dumps(plan, indent=2))


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/g2c-corpus-v1"))
    parser.add_argument("--preset", choices=sorted(PRESETS), default="small")
    parser.add_argument(
        "--force",
        action="store_true",
        help="delete and rebuild --out if it exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the build plan without downloading",
    )

    parser.add_argument("--fineweb-edu", help="FineWeb-Edu-Dedup quota, e.g. 5GB or 0")
    parser.add_argument("--cosmopedia", help="Cosmopedia quota, e.g. 2.5GB or 0")
    parser.add_argument("--tinystories", help="TinyStories quota, e.g. 1GB or 0")
    parser.add_argument("--codesearchnet", help="CodeSearchNet quota, e.g. 1GB or 0")
    parser.add_argument(
        "--codesearchnet-js-ratio",
        type=float,
        default=0.0,
        help="fraction of the CodeSearchNet quota to spend on JavaScript; default 0",
    )

    parser.add_argument("--val-ratio", type=float, default=0.01)
    parser.add_argument("--shard-size", default="250MB")
    parser.add_argument("--compression", choices=("gzip", "none"), default="gzip")
    parser.add_argument("--min-doc-chars", type=int, default=80)
    parser.add_argument("--progress-every", default="50MB")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)

    parser.add_argument("--fineweb-dataset", default="HuggingFaceTB/smollm-corpus")
    parser.add_argument("--fineweb-config", default="fineweb-edu-dedup")
    parser.add_argument("--fineweb-split", default="train")
    parser.add_argument("--cosmopedia-dataset", default="HuggingFaceTB/smollm-corpus")
    parser.add_argument("--cosmopedia-config", default="cosmopedia-v2")
    parser.add_argument("--cosmopedia-split", default="train")
    parser.add_argument(
        "--tinystories-path",
        default="data/tinystories/TinyStories-train.txt",
        help=(
            "local TinyStories file to prefer before streaming from URL; "
            "if missing, compressed TinyStories-train-*.txt.gz shards are used"
        ),
    )
    parser.add_argument(
        "--tinystories-url",
        default=(
            "https://huggingface.co/datasets/roneneldan/TinyStories/"
            "resolve/main/TinyStories-train.txt"
        ),
    )
    parser.add_argument("--codesearchnet-dataset", default="claudios/code_search_net")
    parser.add_argument("--codesearchnet-python-config", default="python")
    parser.add_argument("--codesearchnet-javascript-config", default="javascript")
    parser.add_argument("--codesearchnet-split", default="train")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        args.shard_size_bytes = parse_size(args.shard_size)
        args.progress_every_bytes = parse_size(args.progress_every)
    except ValueError as exc:
        parser.error(str(exc))

    if not 0 <= args.codesearchnet_js_ratio <= 1:
        parser.error("--codesearchnet-js-ratio must be between 0 and 1")
    if not 0 <= args.val_ratio < 1:
        parser.error("--val-ratio must be >= 0 and < 1")

    specs = build_specs(args)
    if args.dry_run:
        dry_run(args, specs)
        return 0

    out_dir: Path = args.out
    staging_dir = prepare_output_dirs(out_dir, force=args.force)

    print(f"Writing G2C corpus to {out_dir}")
    print(f"Staging partial files in {staging_dir}")
    print(f"Total requested raw text: {human_bytes(sum(spec.quota_bytes for spec in specs))}")

    results: list[SourceResult] = []
    for spec in specs:
        print(f"\n==> {spec.name}: target {human_bytes(spec.quota_bytes)}", flush=True)
        result = collect_source(
            spec,
            out_dir=staging_dir,
            val_ratio=args.val_ratio,
            shard_size_bytes=args.shard_size_bytes,
            compression=args.compression,
            min_doc_chars=args.min_doc_chars,
            progress_every_bytes=args.progress_every_bytes,
        )
        results.append(result)
        actual = result.train_actual_bytes + result.val_actual_bytes
        status = "exhausted upstream before quota" if result.exhausted else "complete"
        print(f"{spec.name}: {human_bytes(actual)} from {result.documents} docs ({status})")

    write_manifest(out_dir=staging_dir, args=args, results=results)
    staging_dir.replace(out_dir)
    print(f"\nWrote manifest: {out_dir / 'manifest.json'}")
    compressed_bytes = sum(
        shard.compressed_bytes for result in results for shard in result.shards
    )
    print(f"Actual compressed size: {human_bytes(compressed_bytes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
