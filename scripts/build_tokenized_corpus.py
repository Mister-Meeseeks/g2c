#!/usr/bin/env python
"""Build a disk-backed tokenized corpus artifact from local raw text corpora.

Example:

    python scripts/build_tokenized_corpus.py \
      --name TinyLLM-g2c-30M-v8192 \
      --corpus g2c \
      --tokenizer G2CTokenizer \
      --vocab-size 8192 \
      --bytes 30M \
      --workers 4
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

from g2c.artifacts import (
    build_or_load_tokenized_corpus,
    find_repo_root,
    tokenizer_artifacts_root,
)


@dataclass(frozen=True)
class TokenizedCorpusJob:
    """One tokenized corpus artifact build request."""

    name: str
    corpus: str
    tokenizer: str
    vocab_size: int | None
    byte_count: int | None
    source_split: str = "train"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build uint16/uint32 token-ID files for LM pretraining. "
            "With no custom build args, builds standard corpora for every "
            "supported tokenizer artifact found under artifacts/tokenizers/."
        )
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Artifact name under artifacts/tokenized-corpora/",
    )
    parser.add_argument(
        "--corpus",
        default=None,
        help="Corpus name: tinystories or g2c",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Tokenizer artifact name from artifacts/tokenizers/",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="Effective vocab size. Omit for the tokenizer's full trained vocab.",
    )
    parser.add_argument(
        "--bytes",
        default=None,
        help="Corpus bytes to stream, e.g. 30M, 1G, or all.",
    )
    parser.add_argument(
        "--source-split",
        default="train",
        help="Raw corpus split to tokenize. Default: train.",
    )
    parser.add_argument(
        "--train-bytes",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--val-bytes",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--chunk-bytes",
        default="1M",
        help="Raw text chunk size handed to workers. Default: 1M.",
    )
    parser.add_argument(
        "--chunk-offset",
        type=int,
        default=0,
        help="Starting corpus shard offset, modulo shard count. Default: 0.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Parallel encoding worker processes. Default: all CPU cores.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print every N encoded chunks. Default: 10.",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild even if artifact exists.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root. Default: auto-detect from current directory.",
    )
    args = parser.parse_args()

    repo_root = find_repo_root(args.repo_root)
    progress = make_progress_printer(args.progress_every)

    if _has_custom_job_args(args):
        jobs = [_custom_job_from_args(parser, args)]
    else:
        jobs = default_jobs_for_available_tokenizers(repo_root)
        if not jobs:
            print(
                "No standard tokenizer artifacts found. Run Module 04 tokenizer "
                "artifact cells first, then rerun this script."
            )
            return 0

    for index, job in enumerate(jobs, start=1):
        print()
        print(f"==> [{index}/{len(jobs)}] {job.name}")
        try:
            artifact = build_or_load_tokenized_corpus(
                job.name,
                corpus=job.corpus,
                tokenizer_name=job.tokenizer,
                byte_count=job.byte_count,
                source_split=job.source_split,
                vocab_size=job.vocab_size,
                repo_root=repo_root,
                force=args.force,
                chunk_offset=args.chunk_offset,
                chunk_bytes=parse_size(args.chunk_bytes) or 1024 * 1024,
                workers=args.workers,
                progress_callback=progress,
            )
        except FileNotFoundError as exc:
            print(f"skipping {job.name}: {exc}")
            continue

        print()
        print(f"artifact: {artifact.artifact_dir.relative_to(repo_root)}")
        print(f"tokens:   {len(artifact.tokens):,}")
        print(f"dtype:        {artifact.manifest['dtype']}")
    return 0


def _has_custom_job_args(args: argparse.Namespace) -> bool:
    return any(
        value is not None
        for value in (
            args.name,
            args.corpus,
            args.tokenizer,
            args.bytes,
            args.train_bytes,
            args.val_bytes,
        )
    )


def _custom_job_from_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> TokenizedCorpusJob:
    missing = [
        flag
        for flag, value in (
            ("--name", args.name),
            ("--corpus", args.corpus),
            ("--tokenizer", args.tokenizer),
            ("--bytes", args.bytes or args.train_bytes),
        )
        if value is None
    ]
    if missing:
        parser.error(
            "custom builds must provide all of: "
            "--name, --corpus, --tokenizer, --bytes "
            f"(missing {', '.join(missing)})"
        )
    if args.val_bytes is not None:
        print("ignoring deprecated --val-bytes; split views are chosen at training time")
    return TokenizedCorpusJob(
        name=args.name,
        corpus=args.corpus,
        tokenizer=args.tokenizer,
        vocab_size=args.vocab_size,
        byte_count=parse_size(args.bytes or args.train_bytes),
        source_split=args.source_split,
    )


def default_jobs_for_available_tokenizers(
    repo_root: str | Path | None = None,
) -> list[TokenizedCorpusJob]:
    """Return standard tokenized corpus jobs for tokenizer artifacts on disk."""
    root = find_repo_root(repo_root)
    jobs: list[TokenizedCorpusJob] = []
    for tokenizer_name in available_tokenizer_names(root):
        job = standard_job_for_tokenizer(tokenizer_name, root)
        if job is not None:
            jobs.append(job)
    return jobs


def available_tokenizer_names(repo_root: str | Path | None = None) -> list[str]:
    """List tokenizer artifact names present under ``artifacts/tokenizers``."""
    root = tokenizer_artifacts_root(repo_root)
    if not root.exists():
        return []
    names = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        if (path / "tokenizer.json").exists() and (path / "manifest.json").exists():
            names.append(path.name)
    return names


def standard_job_for_tokenizer(
    tokenizer_name: str,
    repo_root: str | Path | None = None,
) -> TokenizedCorpusJob | None:
    """Map known course tokenizer artifacts to standard tokenized corpora."""
    root = tokenizer_artifacts_root(repo_root) / tokenizer_name
    manifest = _read_json(root / "manifest.json")
    source = manifest.get("source")

    if tokenizer_name == "StoryTokenizer" or source == "tinystories":
        return TokenizedCorpusJob(
            name="StoryLM-tinystories-500000000-v4096",
            corpus="tinystories",
            tokenizer=tokenizer_name,
            vocab_size=4096,
            byte_count=500_000_000,
        )
    if tokenizer_name == "G2CTokenizer" or source == "g2c":
        return TokenizedCorpusJob(
            name="TinyLLM-g2c-30000000-v8192",
            corpus="g2c",
            tokenizer=tokenizer_name,
            vocab_size=8192,
            byte_count=30_000_000,
        )
    if tokenizer_name == "ShakespeareTokenizer" or source == "tinyshakespeare":
        print(
            "skipping ShakespeareTokenizer: TinyShakespeare is small and has no "
            "separate validation split; Module 10 keeps it in memory."
        )
        return None
    print(f"skipping {tokenizer_name}: no standard tokenized corpus preset")
    return None


def _read_json(path: Path) -> dict[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def parse_size(value: str | None) -> int | None:
    if value is None:
        return None
    text = value.strip().lower().replace("_", "")
    if text in {"all", "none"}:
        return None
    suffixes = {
        "k": 1_000,
        "kb": 1_000,
        "m": 1_000_000,
        "mb": 1_000_000,
        "g": 1_000_000_000,
        "gb": 1_000_000_000,
    }
    for suffix, multiplier in sorted(suffixes.items(), key=lambda item: -len(item[0])):
        if text.endswith(suffix):
            return int(float(text[: -len(suffix)]) * multiplier)
    return int(text)


def make_progress_printer(progress_every: int):
    progress_every = max(1, progress_every)

    def progress(info: dict[str, object]) -> None:
        phase = info.get("phase")
        if phase == "loaded":
            print(f"loaded existing artifact: {info['artifact_dir']}", flush=True)
        elif phase == "build_start":
            print(
                "building "
                f"{info['name']} | corpus {info['corpus']} | "
                f"tokenizer {info['tokenizer_artifact']} | "
                f"vocab {info['vocab_size']:,} | dtype {info['dtype']} | "
                f"workers {info['workers']}",
                flush=True,
            )
        elif phase == "stream_start":
            byte_text = format_optional_count(info.get("byte_count"))
            source_split = info.get("source_split", "train")
            print(
                f"{source_split}: tokenizing {byte_text} bytes",
                flush=True,
            )
        elif phase == "stream_chunk":
            chunks = int(info["chunks"])
            if chunks % progress_every != 0:
                return
            source_split = info.get("source_split", "train")
            elapsed = max(float(info["elapsed_seconds"]), 1e-9)
            chars_seen = int(info["chars_seen"])
            token_count = int(info["token_count"])
            print(
                f"{source_split}: chunk {chunks:,} | "
                f"chars {chars_seen:,} | "
                f"tokens {token_count:,} | "
                f"{chars_seen / elapsed:,.0f} chars/s | "
                f"{token_count / elapsed:,.0f} tokens/s | "
                f"elapsed {elapsed:.1f}s",
                flush=True,
            )
        elif phase == "stream_done":
            source_split = info.get("source_split", "train")
            elapsed = max(float(info["elapsed_seconds"]), 1e-9)
            chars_seen = int(info["chars_seen"])
            token_count = int(info["token_count"])
            print(
                f"{source_split}: done | chunks {int(info['chunks']):,} | "
                f"chars {chars_seen:,} | tokens {token_count:,} | "
                f"{chars_seen / elapsed:,.0f} chars/s | "
                f"{token_count / elapsed:,.0f} tokens/s | "
                f"elapsed {elapsed:.1f}s",
                flush=True,
            )
        elif phase == "build_done":
            print(
                f"done | tokens {int(info['token_count']):,}",
                flush=True,
            )

    return progress


def format_optional_count(value: object) -> str:
    if value is None:
        return "all"
    return f"{int(value):,}"


if __name__ == "__main__":
    raise SystemExit(main())
