#!/usr/bin/env python3
"""Split a UTF-8 text file into gzip shards by uncompressed byte budget."""
from __future__ import annotations

import argparse
import gzip
import json
from datetime import UTC, datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--chunk-bytes", type=int, default=100_000_000)
    parser.add_argument("--manifest", type=Path)
    return parser.parse_args()


def shard_text_gzip(
    input_path: Path,
    *,
    output_dir: Path,
    prefix: str,
    chunk_bytes: int,
    manifest_path: Path | None = None,
) -> dict[str, object]:
    if chunk_bytes < 1:
        raise ValueError("chunk_bytes must be positive")
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    for existing in output_dir.glob(f"{prefix}-*.txt.gz"):
        existing.unlink()

    shards: list[dict[str, object]] = []
    total = 0
    writer = None
    shard_path: Path | None = None
    shard_bytes = 0
    shard_index = 0

    def open_next_shard():
        nonlocal writer, shard_path, shard_bytes, shard_index
        if writer is not None:
            writer.close()
            assert shard_path is not None
            shards[-1]["compressed_bytes"] = shard_path.stat().st_size

        shard_path = output_dir / f"{prefix}-{shard_index:04d}.txt.gz"
        writer = gzip.open(shard_path, "wb")
        shard_bytes = 0
        shards.append(
            {
                "path": shard_path.name,
                "uncompressed_bytes": 0,
                "compressed_bytes": 0,
            }
        )
        shard_index += 1

    with input_path.open("rb") as src:
        for line in src:
            if writer is None or (shard_bytes > 0 and shard_bytes + len(line) > chunk_bytes):
                open_next_shard()
            assert writer is not None
            writer.write(line)
            shard_bytes += len(line)
            total += len(line)
            shards[-1]["uncompressed_bytes"] = shard_bytes

    if writer is not None:
        writer.close()
        assert shard_path is not None
        shards[-1]["compressed_bytes"] = shard_path.stat().st_size

    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "prefix": prefix,
        "chunk_bytes": chunk_bytes,
        "total_uncompressed_bytes": total,
        "shards": shards,
    }
    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    args = parse_args()
    manifest = shard_text_gzip(
        args.input,
        output_dir=args.output_dir,
        prefix=args.prefix,
        chunk_bytes=args.chunk_bytes,
        manifest_path=args.manifest,
    )
    print(
        f"wrote {len(manifest['shards'])} shards "
        f"({manifest['total_uncompressed_bytes']} uncompressed bytes)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
