from __future__ import annotations

import gzip
import importlib.util
from pathlib import Path


def load_sharder():
    script = Path(__file__).resolve().parents[1] / "scripts" / "shard_text_gzip.py"
    spec = importlib.util.spec_from_file_location("shard_text_gzip", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shard_text_gzip_splits_by_uncompressed_budget(tmp_path):
    sharder = load_sharder()
    source = tmp_path / "source.txt"
    source.write_text("aaa\nbbb\nccc\n", encoding="utf-8")
    out_dir = tmp_path / "out"
    manifest_path = out_dir / "manifest.json"

    manifest = sharder.shard_text_gzip(
        source,
        output_dir=out_dir,
        prefix="TinyStories-train",
        chunk_bytes=8,
        manifest_path=manifest_path,
    )

    shards = sorted(out_dir.glob("TinyStories-train-*.txt.gz"))
    assert len(shards) == 2
    assert manifest_path.exists()
    assert manifest["total_uncompressed_bytes"] == len(b"aaa\nbbb\nccc\n")
    assert [row["uncompressed_bytes"] for row in manifest["shards"]] == [8, 4]

    restored = ""
    for shard in shards:
        with gzip.open(shard, "rt", encoding="utf-8") as f:
            restored += f.read()
    assert restored == "aaa\nbbb\nccc\n"
