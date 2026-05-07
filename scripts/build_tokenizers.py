#!/usr/bin/env python
"""Build the course tokenizer artifacts non-interactively.

Called from setup.sh and datasets.sh after the source corpora are downloaded
so Module 10's pretraining can load each tokenizer artifact by name even when
a student skips Module 04's optional artifact-training cells.

Usage:
    scripts/build_tokenizers.py [name ...] [--force]

Names: ``shakespeare`` ``story`` ``g2c`` ``all``

Defaults to ``all`` when no names are passed. Idempotent: skips a tokenizer
when its artifact already exists, unless ``--force`` is given. Skips with a
warning (not an error) when the source corpus is missing — the script's job is
to make the artifact available *if possible*, not to police whether the data
was downloaded.

The configs here mirror the defaults in ``notebooks/clean/04-tokenizer.ipynb``.
The notebook is the canonical reference; if you want different tokenizer
parameters, change the notebook and rerun those cells (``force=True`` there
will overwrite whatever this script wrote).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from g2c.artifacts import (
    TokenizerArtifactConfig,
    find_repo_root,
    tokenizer_artifact_exists,
    train_or_load_tokenizer_artifact,
)


# ---------------------------------------------------------------------------
# Configs — keep in sync with notebooks/clean/04-tokenizer.ipynb
# ---------------------------------------------------------------------------

_CONFIGS: dict[str, TokenizerArtifactConfig] = {
    "shakespeare": TokenizerArtifactConfig(
        name="ShakespeareTokenizer",
        source="tinyshakespeare",
        vocab_size=4096,
        max_chars=1_000_000,
        use_fast=True,
        chunk_chars=8_192,
        notes="Built non-interactively by scripts/build_tokenizers.py",
    ),
    "story": TokenizerArtifactConfig(
        name="StoryTokenizer",
        source="tinystories",
        vocab_size=8192,
        max_chars=25_000_000,
        use_fast=True,
        chunk_chars=8_192,
        notes="Built non-interactively by scripts/build_tokenizers.py",
    ),
    "g2c": TokenizerArtifactConfig(
        name="G2CTokenizer",
        source="g2c",
        vocab_size=16384,
        max_chars=100_000_000,
        use_fast=True,
        chunk_chars=8_192,
        notes="Built non-interactively by scripts/build_tokenizers.py",
    ),
}

_BAR_WIDTH = 24
_LINE_WIDTH = 78


def _make_cli_progress():
    """Return a callback that renders progress to stdout for a single artifact."""
    start = time.perf_counter()
    state = {"line_in_progress": False}

    def _bar(current: int, total: int) -> str:
        if total <= 0:
            return "[" + "?" * _BAR_WIDTH + "]"
        filled = max(1, min(_BAR_WIDTH, round(_BAR_WIDTH * current / total)))
        return "[" + "#" * filled + "-" * (_BAR_WIDTH - filled) + "]"

    def _line(message: str) -> None:
        # Truncate or pad to a fixed width so subsequent \r overwrites cleanly
        # without wrapping on narrow terminals.
        if len(message) > _LINE_WIDTH:
            message = message[: _LINE_WIDTH - 1] + "…"
        sys.stdout.write("\r" + message.ljust(_LINE_WIDTH))
        sys.stdout.flush()
        state["line_in_progress"] = True

    def _end_line() -> None:
        if state["line_in_progress"]:
            sys.stdout.write("\n")
            sys.stdout.flush()
            state["line_in_progress"] = False

    def _milestone(message: str) -> None:
        _end_line()
        sys.stdout.write(f"  {message}\n")
        sys.stdout.flush()

    def update(info: dict) -> None:
        phase = info.get("phase")
        elapsed = time.perf_counter() - start
        if phase == "loaded":
            _milestone(f"loaded cached artifact ({info['token_count']:,} sample tokens)")
        elif phase == "missing_source":
            _milestone(f"source {info['source']!r} not available; skipping")
        elif phase == "fast_chunks_start":
            _line(f"chunks {info['chunks']:,} ({info['chars']:,} chars)")
        elif phase == "fast_chunk":
            cur = int(info["chunk_index"])
            tot = int(info["chunks"])
            _line(f"chunks {_bar(cur, tot)} {cur:,}/{tot:,} {elapsed:.1f}s")
        elif phase == "fast_native_train_start":
            _line(f"starting Rust trainer (target vocab {info['target_vocab_size']:,})")
        elif phase == "fast_native_progress":
            cur = int(info["current"])
            tot = max(1, int(info["total"]))
            stage = info.get("native_stage", "training")
            _line(f"{stage} {_bar(cur, tot)} {cur:,}/{tot:,} {elapsed:.1f}s")
        elif phase == "fast_chunks_done":
            _line("Rust BPE training complete; exporting...")
        elif phase == "fast_export_done":
            _line(f"exported {info['merge_count']:,} merges")
        elif phase == "fast_import_done":
            _line(
                f"imported vocab {info['vocab_size']:,}/{info['target_vocab_size']:,} "
                f"{elapsed:.1f}s"
            )
        elif phase == "sample_encode_start":
            _line(f"encoding inspection sample ({info['chars']:,} chars)")
        elif phase == "sample_encode_done":
            _milestone(
                f"sample encoded: {info['chars']:,} chars -> {info['token_count']:,} tokens"
            )
        elif phase == "fast_done":
            _milestone(
                f"trained vocab {info['vocab_size']:,}/{info['target_vocab_size']:,} "
                f"in {info['elapsed_seconds']:.1f}s"
            )
        elif phase == "saved":
            _milestone(f"saved -> {info['artifact_dir']}")
        elif phase is not None:
            _line(f"{phase} {elapsed:.1f}s")

    return update


def _build_one(key: str, repo_root: Path, *, force: bool) -> bool:
    """Build one tokenizer. Returns True on success, False on missing source."""
    config = _CONFIGS[key]
    print(
        f"\n=== {config.name} (source={config.source!r}, "
        f"vocab={config.vocab_size:,}, max_chars={config.max_chars:,}) ==="
    )
    if not force and tokenizer_artifact_exists(config.name, repo_root=repo_root):
        print(f"  artifact already exists; skipping (use --force to retrain)")
        return True

    progress = _make_cli_progress()
    artifact = train_or_load_tokenizer_artifact(
        config,
        repo_root=repo_root,
        force=force,
        progress_callback=progress,
        status_callback=progress,
    )
    return artifact is not None


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "names",
        nargs="*",
        default=[],
        help=f"tokenizers to build: {', '.join(_CONFIGS)}, or 'all'. Default: all.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain even when an artifact already exists.",
    )
    args = parser.parse_args(argv)

    valid = set(_CONFIGS) | {"all"}
    requested = args.names or ["all"]
    bad = [name for name in requested if name not in valid]
    if bad:
        parser.error(f"unknown tokenizer name(s): {bad}. Choose from: {sorted(valid)}")

    if "all" in requested:
        targets = list(_CONFIGS)
    else:
        # Preserve user-specified order, deduplicated.
        targets = []
        for name in requested:
            if name not in targets:
                targets.append(name)

    repo_root = find_repo_root()
    print(f"repo_root: {repo_root}")
    print(f"building tokenizers: {', '.join(targets)} (force={args.force})")

    skipped = []
    for key in targets:
        ok = _build_one(key, repo_root, force=args.force)
        if not ok:
            skipped.append(_CONFIGS[key].name)

    if skipped:
        print(
            f"\nSkipped (source corpus not present): {', '.join(skipped)}. "
            "Run the relevant ./datasets.sh target, then rerun this script."
        )
    else:
        print("\nAll requested tokenizers ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
