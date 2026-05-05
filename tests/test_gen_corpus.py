"""Tests for the local G2C corpus builder helpers.

These do not stream upstream datasets. They only check the deterministic local
pieces: size parsing, text normalization, and compressed shard writing.
"""

from __future__ import annotations

import gzip
import importlib.util
import sys
from pathlib import Path


def load_gen_corpus_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "gen_corpus.py"
    spec = importlib.util.spec_from_file_location("gen_corpus", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_size_decimal_and_binary_units():
    gen_corpus = load_gen_corpus_module()

    assert gen_corpus.parse_size("512MB") == 512_000_000
    assert gen_corpus.parse_size("2.5GB") == 2_500_000_000
    assert gen_corpus.parse_size("1MiB") == 1024**2
    assert gen_corpus.parse_size("123") == 123


def test_normalize_text_removes_delimiter_and_preserves_basic_shape():
    gen_corpus = load_gen_corpus_module()

    text = "hello\r\nworld  \n\n\n\n<|endoftext|>\n"
    assert gen_corpus.normalize_text(text) == "hello\nworld"


def test_shard_writer_writes_gzip_documents(tmp_path):
    gen_corpus = load_gen_corpus_module()
    writer = gen_corpus.ShardWriter(
        root=tmp_path,
        source="unit-source",
        split="train",
        shard_size_bytes=10_000,
        compression="gzip",
    )

    writer.write_document("first document")
    writer.write_document("second document")
    writer.close()

    records = writer.records
    assert len(records) == 1
    assert records[0].documents == 2
    assert records[0].uncompressed_bytes > 0
    assert records[0].compressed_bytes > 0

    with gzip.open(tmp_path / records[0].path, "rt", encoding="utf-8") as fh:
        text = fh.read()

    assert "first document\n<|endoftext|>\n" in text
    assert "second document\n<|endoftext|>\n" in text


def test_prepare_output_dirs_allows_empty_directory_skeleton(tmp_path):
    gen_corpus = load_gen_corpus_module()
    out_dir = tmp_path / "g2c-corpus-v1"
    (out_dir / "raw" / "fineweb-edu-dedup").mkdir(parents=True)

    staging_dir = gen_corpus.prepare_output_dirs(out_dir, force=False)

    assert staging_dir == tmp_path / "g2c-corpus-v1.partial"
    assert staging_dir.exists()
    assert not out_dir.exists()
