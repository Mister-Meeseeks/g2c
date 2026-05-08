#!/usr/bin/env python
"""Build a disk-backed tokenized corpus artifact from local raw text corpora.

Example:

    python scripts/build_tokenized_corpus.py \
      --name TinyLLM-g2c-full-v8192 \
      --corpus g2c \
      --tokenizer G2CTokenizer \
      --vocab-size 8192 \
      --bytes all \
      --workers 4
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
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
        help="Corpus bytes to stream, e.g. 30M, 1G, or all. Default: all.",
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
        help="Refresh progress every N encoded chunks. Default: 10.",
    )
    parser.add_argument(
        "--progress-style",
        choices=("line", "log"),
        default="line",
        help=(
            "Progress rendering style. 'line' updates one terminal line in place; "
            "'log' prints a new line for each update. Default: line."
        ),
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
    progress = make_progress_printer(args.progress_every, style=args.progress_style)

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
        except KeyboardInterrupt:
            _finish_progress_callback(progress)
            print(
                "interrupted; partial tokenized output was discarded. "
                "Re-run the script to load completed artifacts or rebuild this one.",
                file=sys.stderr,
                flush=True,
            )
            return 130

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
        )
        if value is None
    ]
    if missing:
        parser.error(
            "custom builds must provide all of: "
            "--name, --corpus, --tokenizer "
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
            name="StoryLM-tinystories-full-v4096",
            corpus="tinystories",
            tokenizer=tokenizer_name,
            vocab_size=4096,
            byte_count=None,
        )
    if tokenizer_name == "G2CTokenizer" or source == "g2c":
        return TokenizedCorpusJob(
            name="TinyLLM-g2c-full-v8192",
            corpus="g2c",
            tokenizer=tokenizer_name,
            vocab_size=8192,
            byte_count=None,
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


def make_progress_printer(
    progress_every: int,
    *,
    style: str = "line",
    terminal_width: int | None = None,
):
    progress_every = max(1, progress_every)
    if style not in {"line", "log"}:
        raise ValueError("progress style must be 'line' or 'log'")
    state = {
        "active": False,
        "last_len": 0,
        "style": style,
        "terminal_width": terminal_width,
    }

    def progress(info: dict[str, object]) -> None:
        phase = info.get("phase")
        if phase == "loaded":
            _finish_progress_line(state)
            print(f"loaded existing artifact: {info['artifact_dir']}", flush=True)
        elif phase == "build_start":
            _finish_progress_line(state)
            print(
                "building "
                f"{info['name']} | corpus {info['corpus']} | "
                f"tokenizer {info['tokenizer_artifact']} | "
                f"vocab {info['vocab_size']:,} | dtype {info['dtype']} | "
                f"workers {info['workers']}",
                flush=True,
            )
        elif phase == "stream_start":
            _render_progress_bar(info, state)
        elif phase == "stream_chunk":
            chunks = int(info["chunks"])
            if chunks % progress_every != 0:
                return
            _render_progress_bar(info, state)
        elif phase == "stream_done":
            _render_progress_bar(info, state, final=True)
            _finish_progress_line(state)
        elif phase == "build_done":
            _finish_progress_line(state)
            print(
                f"done | tokens {int(info['token_count']):,}",
                flush=True,
            )

    def finish() -> None:
        _finish_progress_line(state)

    setattr(progress, "finish", finish)
    return progress


def _finish_progress_callback(callback: object) -> None:
    finish = getattr(callback, "finish", None)
    if callable(finish):
        finish()


def _render_progress_bar(
    info: dict[str, object],
    state: dict[str, object],
    *,
    final: bool = False,
) -> None:
    name = str(info.get("name") or "")
    corpus = str(info.get("corpus") or "corpus")
    source_split = str(info.get("source_split") or "train")
    chunks = int(info.get("chunks") or 0)
    bytes_seen = int(info.get("bytes_seen") or 0)
    byte_target = int(info.get("byte_target") or 0)
    token_count = int(info.get("token_count") or 0)
    elapsed = max(float(info.get("elapsed_seconds") or 0.0), 1e-9)

    if byte_target > 0:
        fraction = min(1.0, bytes_seen / byte_target)
        percent = f"{fraction * 100:6.2f}%"
        byte_text = f"{_human_size(bytes_seen)}/{_human_size(byte_target)}"
    else:
        fraction = 1.0 if final else 0.0
        percent = "  n/a "
        byte_text = _human_size(bytes_seen)

    if bytes_seen and elapsed > 0:
        byte_rate = f"{_human_size(bytes_seen / elapsed)}/s"
    else:
        byte_rate = "--/s"
    token_rate = f"{token_count / elapsed:,.0f} tok/s" if token_count else "-- tok/s"
    terminal_width = _terminal_width(state)
    line = _progress_line(
        name=name,
        corpus=corpus,
        source_split=source_split,
        fraction=fraction,
        percent=percent,
        byte_text=byte_text,
        chunks=chunks,
        token_count=token_count,
        byte_rate=byte_rate,
        token_rate=token_rate,
        elapsed=elapsed,
        final=final,
        terminal_width=terminal_width,
    )

    if state.get("style") == "log":
        print(line, flush=True)
        state["active"] = False
        state["last_len"] = 0
        return

    # ``\r`` returns to column zero; the ANSI clear keeps the update tidy even
    # when the new line is shorter than the previous one. Padding is retained
    # as a fallback for terminals that do not honor the escape sequence.
    last_len = int(state.get("last_len") or 0)
    padding = " " * max(0, last_len - len(line))
    sys.stdout.write("\r\033[2K" + line + padding)
    sys.stdout.flush()
    state["active"] = True
    state["last_len"] = len(line)


def _progress_line(
    *,
    name: str,
    corpus: str,
    source_split: str,
    fraction: float,
    percent: str,
    byte_text: str,
    chunks: int,
    token_count: int,
    byte_rate: str,
    token_rate: str,
    elapsed: float,
    final: bool,
    terminal_width: int,
) -> str:
    max_width = max(30, terminal_width - 1)
    label = f"{name} | {corpus}:{source_split}"
    percent = percent.strip()

    candidates = []
    if max_width >= 120:
        candidates.append(
            _format_progress_line(
                label=_truncate_middle(label, 36),
                bar=_bar(fraction, width=28, final=final),
                percent=percent,
                byte_text=byte_text,
                parts=[
                    f"chunk {chunks:,}",
                    f"tokens {token_count:,}",
                    byte_rate,
                    token_rate,
                    f"{elapsed:.1f}s",
                ],
            )
        )
    if max_width >= 90:
        candidates.append(
            _format_progress_line(
                label=_truncate_middle(label, 28),
                bar=_bar(fraction, width=18, final=final),
                percent=percent,
                byte_text=byte_text,
                parts=[
                    f"chunk {chunks:,}",
                    f"tok {_compact_count(token_count)}",
                    f"{elapsed:.0f}s",
                ],
            )
        )

    compact_bar_width = 10 if max_width < 70 else 14
    fixed_text = (
        f" [{_bar(fraction, width=compact_bar_width, final=final)}] "
        f"{percent} {byte_text} tok {_compact_count(token_count)} {elapsed:.0f}s"
    )
    label_budget = max(8, max_width - len(fixed_text))
    candidates.append(
        f"{_truncate_middle(label, label_budget)}{fixed_text}"
    )

    for candidate in candidates:
        if len(candidate) <= max_width:
            return candidate
    return _truncate_middle(candidates[-1], max_width)


def _format_progress_line(
    *,
    label: str,
    bar: str,
    percent: str,
    byte_text: str,
    parts: list[str],
) -> str:
    suffix = " | ".join([byte_text, *parts])
    return f"{label} {bar} {percent} | {suffix}"


def _bar(fraction: float, *, width: int, final: bool) -> str:
    filled = width if final else round(width * fraction)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _compact_count(value: int) -> str:
    value_float = float(value)
    for suffix, divisor in (("B", 1_000_000_000), ("M", 1_000_000), ("K", 1_000)):
        if abs(value_float) >= divisor:
            compact = value_float / divisor
            if compact >= 100:
                return f"{compact:.0f}{suffix}"
            if compact >= 10:
                return f"{compact:.1f}{suffix}"
            return f"{compact:.2f}{suffix}"
    return str(value)


def _truncate_middle(text: str, width: int) -> str:
    if width <= 0:
        return ""
    if len(text) <= width:
        return text
    if width == 1:
        return "…"
    left = (width - 1) // 2
    right = width - 1 - left
    return text[:left] + "…" + text[-right:]


def _terminal_width(state: dict[str, object]) -> int:
    configured = state.get("terminal_width")
    if configured is not None:
        return max(30, int(configured))
    return max(30, shutil.get_terminal_size((100, 20)).columns)


def _finish_progress_line(state: dict[str, object]) -> None:
    if state.get("active"):
        sys.stdout.write("\n")
        sys.stdout.flush()
    state["active"] = False
    state["last_len"] = 0


def _human_size(value: float) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    size = float(value)
    for unit in units:
        if abs(size) < 1000.0 or unit == units[-1]:
            if unit == "B":
                return f"{size:.0f}{unit}"
            return f"{size:.2f}{unit}"
        size /= 1000.0


def format_optional_count(value: object) -> str:
    if value is None:
        return "all"
    return f"{int(value):,}"


if __name__ == "__main__":
    raise SystemExit(main())
