from __future__ import annotations

import gzip
import json
from pathlib import Path

import torch

from g2c.artifacts import (
    TokenizerArtifactConfig,
    default_text_chunks,
    encode_text_to_tensor,
    find_repo_root,
    load_corpus_bytes,
    load_corpus_text,
    load_or_encode_tokenized_corpus,
    load_or_encode_tokenized_pair,
    load_required_tokenizer,
    load_tinystories_text,
    load_tokenizer_artifact,
    load_tokenizer_source_text,
    tokenizer_artifact_exists,
    train_or_load_tokenizer_artifact,
)
from g2c.tokenizer import BPETokenizer


def make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "g2c").mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname = 'tmp'\n", encoding="utf-8")
    return repo


def test_find_repo_root_walks_up_from_nested_path(tmp_path):
    repo = make_repo(tmp_path)
    nested = repo / "notebooks" / "solutions"
    nested.mkdir(parents=True)

    assert find_repo_root(nested) == repo


def test_tokenizer_config_accepts_mapping_defaults():
    config = TokenizerArtifactConfig.from_mapping(
        {
            "name": "Tiny",
            "source": "tinyshakespeare",
            "vocab_size": 256,
            "max_chars": 100,
        }
    )

    assert config.name == "Tiny"
    assert config.use_fast is True
    assert config.chunk_chars == 8_192
    assert config.encoded_sample_chars == 1_000_000


def test_load_tokenizer_source_text_reads_tinyshakespeare(tmp_path):
    repo = make_repo(tmp_path)
    data_path = repo / "data" / "tinyshakespeare.txt"
    data_path.parent.mkdir(parents=True)
    data_path.write_text("abcdef", encoding="utf-8")

    assert load_tokenizer_source_text("tinyshakespeare", 3, repo_root=repo) == "abc"


def test_default_text_chunks_uses_fixed_size_chunks():
    assert list(default_text_chunks("abcdef", chunk_chars=2)) == ["ab", "cd", "ef"]


def test_encode_text_to_tensor_uses_two_pass_chunking():
    tokenizer = BPETokenizer()
    calls = 0

    def chunk_fn(text: str):
        nonlocal calls
        calls += 1
        return (text[index : index + 2] for index in range(0, len(text), 2))

    events: list[dict[str, object]] = []
    ids = encode_text_to_tensor(
        tokenizer,
        "abcdef",
        chunk_fn=chunk_fn,
        progress_callback=events.append,
    )

    assert calls == 2
    assert torch.equal(ids, torch.tensor(list(b"abcdef"), dtype=torch.long))
    assert [event["phase"] for event in events] == [
        "encode_count_start",
        "encode_count_chunk",
        "encode_count_chunk",
        "encode_count_chunk",
        "encode_count_done",
        "encode_write_start",
        "encode_write_chunk",
        "encode_write_chunk",
        "encode_write_chunk",
        "encode_done",
    ]


def test_load_tinystories_text_reads_compressed_shards(tmp_path):
    repo = make_repo(tmp_path)
    data_dir = repo / "data" / "tinystories"
    data_dir.mkdir(parents=True)
    with gzip.open(data_dir / "TinyStories-train-0000.txt.gz", "wt", encoding="utf-8") as f:
        f.write("train shard one ")
    with gzip.open(data_dir / "TinyStories-train-0001.txt.gz", "wt", encoding="utf-8") as f:
        f.write("train shard two")
    with gzip.open(data_dir / "TinyStories-valid-0000.txt.gz", "wt", encoding="utf-8") as f:
        f.write("valid shard")

    assert load_tokenizer_source_text("tinystories", 20, repo_root=repo) == "train shard one trai"
    assert load_tinystories_text(
        train_max_chars=40,
        valid_max_chars=5,
        repo_root=repo,
    ) == ("train shard one train shard twotrain sha", "valid")


def test_load_corpus_bytes_wraps_and_uses_chunk_offset(tmp_path):
    repo = make_repo(tmp_path)
    data_dir = repo / "data" / "tinystories"
    data_dir.mkdir(parents=True)
    for index, text in enumerate(("aaa", "bbb", "ccc")):
        with gzip.open(
            data_dir / f"TinyStories-train-{index:04d}.txt.gz",
            "wt",
            encoding="utf-8",
        ) as f:
            f.write(text)
    (data_dir / "TinyStories-train-shards.json").write_text(
        json.dumps(
            {
                "shards": [
                    {
                        "path": f"TinyStories-train-{index:04d}.txt.gz",
                        "uncompressed_bytes": 3,
                    }
                    for index in range(3)
                ],
                "total_uncompressed_bytes": 9,
            }
        ),
        encoding="utf-8",
    )

    assert load_corpus_bytes(
        "tinystories",
        byte_count=11,
        chunk_offset=1,
        repo_root=repo,
    ) == b"bbbcccaaabb"


def test_load_g2c_corpus_preserves_source_weights_and_prefers_full(tmp_path):
    repo = make_repo(tmp_path)
    small_dir = repo / "data" / "g2c-corpus-v1-small"
    full_dir = repo / "data" / "g2c-corpus-v1"
    _write_g2c_manifested_shard(small_dir, "tinystories", "S" * 100)
    _write_g2c_manifested_shards(
        full_dir,
        {
            "fineweb-edu-dedup": "F" * 80,
            "cosmopedia-v2": "C" * 20,
        },
    )

    text = load_corpus_text("g2c", byte_count=50, repo_root=repo)

    assert text == ("F" * 40) + ("C" * 10)


def test_load_tokenizer_source_text_uses_logical_g2c_source(tmp_path):
    repo = make_repo(tmp_path)
    corpus_dir = repo / "data" / "g2c-corpus-v1-small"
    _write_g2c_manifested_shards(
        corpus_dir,
        {
            "fineweb-edu-dedup": "F" * 6,
            "cosmopedia-v2": "C" * 4,
        },
    )

    assert load_tokenizer_source_text("g2c", 5, repo_root=repo) == "FFFCC"
    assert load_tokenizer_source_text("g2c-corpus-small", 5, repo_root=repo) == "FFFCC"


def test_train_or_load_tokenizer_artifact_saves_and_reloads(tmp_path):
    repo = make_repo(tmp_path)
    data_path = repo / "data" / "tinyshakespeare.txt"
    data_path.parent.mkdir(parents=True)
    data_path.write_text("hello artifact", encoding="utf-8")
    config = TokenizerArtifactConfig(
        name="TestTokenizer",
        source="tinyshakespeare",
        vocab_size=256,
        max_chars=20,
    )

    artifact = train_or_load_tokenizer_artifact(config, repo_root=repo)

    assert artifact is not None
    assert artifact.manifest["name"] == "TestTokenizer"
    assert artifact.ids == list(b"hello artifacthello ")
    assert artifact.manifest["encoded_full_training_text"] is True
    assert artifact.manifest["token_ids_kind"] == "encoded_sample"
    assert tokenizer_artifact_exists("TestTokenizer", repo_root=repo)

    loaded = train_or_load_tokenizer_artifact(config, repo_root=repo)

    assert loaded is not None
    assert loaded.ids == artifact.ids
    assert loaded.manifest == artifact.manifest


def test_load_tokenizer_artifact_loads_without_source_text_or_ids(tmp_path):
    repo = make_repo(tmp_path)
    data_path = repo / "data" / "tinyshakespeare.txt"
    data_path.parent.mkdir(parents=True)
    data_path.write_text("hello artifact", encoding="utf-8")
    config = TokenizerArtifactConfig(
        name="LoadOnlyTokenizer",
        source="tinyshakespeare",
        vocab_size=256,
        max_chars=20,
    )
    saved = train_or_load_tokenizer_artifact(config, repo_root=repo)

    artifact = load_tokenizer_artifact("LoadOnlyTokenizer", repo_root=repo)

    assert saved is not None
    assert artifact.tokenizer.vocab == saved.tokenizer.vocab
    assert artifact.ids == []
    assert artifact.text is None
    assert artifact.manifest["name"] == "LoadOnlyTokenizer"


def test_load_tokenizer_artifact_can_load_sample_ids_and_text(tmp_path):
    repo = make_repo(tmp_path)
    data_path = repo / "data" / "tinyshakespeare.txt"
    data_path.parent.mkdir(parents=True)
    data_path.write_text("hello artifact", encoding="utf-8")
    config = TokenizerArtifactConfig(
        name="LoadFullTokenizer",
        source="tinyshakespeare",
        vocab_size=256,
        max_chars=20,
        encoded_sample_chars=5,
    )
    train_or_load_tokenizer_artifact(config, repo_root=repo)

    artifact = load_tokenizer_artifact(
        "LoadFullTokenizer",
        repo_root=repo,
        load_ids=True,
        load_text=True,
    )

    assert artifact.ids == list(b"hello")
    assert artifact.text == "hello artifacthello "
    assert artifact.config.source == "tinyshakespeare"
    assert artifact.config.encoded_sample_chars == 5


def test_load_required_tokenizer_returns_saved_tokenizer(tmp_path):
    repo = make_repo(tmp_path)
    data_path = repo / "data" / "tinyshakespeare.txt"
    data_path.parent.mkdir(parents=True)
    data_path.write_text("hello artifact", encoding="utf-8")
    config = TokenizerArtifactConfig(
        name="RequiredTokenizer",
        source="tinyshakespeare",
        vocab_size=256,
        max_chars=20,
    )
    train_or_load_tokenizer_artifact(config, repo_root=repo)

    tokenizer = load_required_tokenizer("RequiredTokenizer", repo_root=repo)

    assert tokenizer.encode_fast("abc") == list(b"abc")


def test_load_or_encode_tokenized_corpus_uses_artifact_and_cache(tmp_path):
    repo = make_repo(tmp_path)
    data_path = repo / "data" / "tinyshakespeare.txt"
    data_path.parent.mkdir(parents=True)
    data_path.write_text("hello artifact", encoding="utf-8")
    config = TokenizerArtifactConfig(
        name="CorpusTokenizer",
        source="tinyshakespeare",
        vocab_size=256,
        max_chars=20,
    )
    train_or_load_tokenizer_artifact(config, repo_root=repo)

    tokenizer, ids = load_or_encode_tokenized_corpus(
        "abcdef",
        "CorpusTokenizer",
        label="unit",
        repo_root=repo,
    )
    cached_tokenizer, cached_ids = load_or_encode_tokenized_corpus(
        "abcdef",
        "CorpusTokenizer",
        label="unit",
        repo_root=repo,
    )

    assert tokenizer.vocab == cached_tokenizer.vocab
    assert torch.equal(ids, torch.tensor(list(b"abcdef"), dtype=torch.long))
    assert torch.equal(cached_ids, ids)


def test_load_or_encode_tokenized_pair_uses_artifact_and_cache(tmp_path):
    repo = make_repo(tmp_path)
    data_path = repo / "data" / "tinyshakespeare.txt"
    data_path.parent.mkdir(parents=True)
    data_path.write_text("hello artifact", encoding="utf-8")
    config = TokenizerArtifactConfig(
        name="PairTokenizer",
        source="tinyshakespeare",
        vocab_size=256,
        max_chars=20,
    )
    train_or_load_tokenizer_artifact(config, repo_root=repo)

    tokenizer, train_ids, val_ids = load_or_encode_tokenized_pair(
        "abcd",
        "ef",
        "PairTokenizer",
        label="unit-pair",
        repo_root=repo,
    )
    cached_tokenizer, cached_train_ids, cached_val_ids = load_or_encode_tokenized_pair(
        "abcd",
        "ef",
        "PairTokenizer",
        label="unit-pair",
        repo_root=repo,
    )

    assert tokenizer.vocab == cached_tokenizer.vocab
    assert torch.equal(train_ids, torch.tensor(list(b"abcd"), dtype=torch.long))
    assert torch.equal(val_ids, torch.tensor(list(b"ef"), dtype=torch.long))
    assert torch.equal(cached_train_ids, train_ids)
    assert torch.equal(cached_val_ids, val_ids)


def test_train_or_load_tokenizer_artifact_saves_sample_ids(tmp_path):
    repo = make_repo(tmp_path)
    data_path = repo / "data" / "tinyshakespeare.txt"
    data_path.parent.mkdir(parents=True)
    data_path.write_text("abcdefghijklmnopqrstuvwxyz", encoding="utf-8")
    config = TokenizerArtifactConfig(
        name="SampleTokenizer",
        source="tinyshakespeare",
        vocab_size=256,
        max_chars=26,
        encoded_sample_chars=5,
    )

    artifact = train_or_load_tokenizer_artifact(config, repo_root=repo)

    assert artifact is not None
    assert artifact.ids == list(b"abcde")
    assert artifact.manifest["actual_chars"] == 26
    assert artifact.manifest["encoded_chars"] == 5
    assert artifact.manifest["encoded_sample_chars"] == 5
    assert artifact.manifest["encoded_full_training_text"] is False
    assert artifact.manifest["token_count"] == 5


def test_fast_tokenizer_artifact_skips_full_training_text_encode(tmp_path):
    repo = make_repo(tmp_path)
    text = "abab abcabc " * 20
    data_path = repo / "data" / "tinyshakespeare.txt"
    data_path.parent.mkdir(parents=True)
    data_path.write_text(text, encoding="utf-8")
    config = TokenizerArtifactConfig(
        name="FastSampleTokenizer",
        source="tinyshakespeare",
        vocab_size=260,
        max_chars=len(text),
        encoded_sample_chars=10,
        use_fast=True,
    )
    events: list[dict[str, object]] = []

    artifact = train_or_load_tokenizer_artifact(
        config,
        repo_root=repo,
        status_callback=events.append,
    )

    assert artifact is not None
    assert artifact.ids == artifact.tokenizer.encode_fast(text[:10])
    assert artifact.manifest["actual_chars"] == len(text)
    assert artifact.manifest["encoded_chars"] == 10
    phases = [event.get("phase") for event in events]
    assert "fast_encode_skipped" in phases
    assert "sample_encode_start" in phases
    assert "fast_encode_start" not in phases


def test_train_or_load_tokenizer_artifact_missing_source_returns_none(tmp_path):
    repo = make_repo(tmp_path)
    config = TokenizerArtifactConfig(
        name="Missing",
        source="tinyshakespeare",
        vocab_size=256,
        max_chars=20,
    )

    assert train_or_load_tokenizer_artifact(config, repo_root=repo) is None


def _write_g2c_manifested_shard(corpus_dir: Path, source: str, text: str) -> None:
    _write_g2c_manifested_shards(corpus_dir, {source: text})


def _write_g2c_manifested_shards(corpus_dir: Path, source_texts: dict[str, str]) -> None:
    sources = []
    for source, text in source_texts.items():
        shard_path = corpus_dir / "raw" / source / "train_0000.txt.gz"
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(shard_path, "wt", encoding="utf-8") as f:
            f.write(text)
        sources.append(
            {
                "source": source,
                "train_actual_bytes": len(text.encode("utf-8")),
                "val_actual_bytes": 0,
                "shards": [
                    {
                        "path": str(shard_path.relative_to(corpus_dir)),
                        "split": "train",
                        "source": source,
                        "uncompressed_bytes": len(text.encode("utf-8")),
                        "compressed_bytes": shard_path.stat().st_size,
                    }
                ],
            }
        )

    corpus_dir.mkdir(parents=True, exist_ok=True)
    (corpus_dir / "manifest.json").write_text(
        json.dumps({"sources": sources}),
        encoding="utf-8",
    )
