from __future__ import annotations

import gzip
import json
from pathlib import Path

from scripts.artifact_status import module_guidance, render_artifact_status, track_status


def make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "g2c").mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname = 'tmp'\n", encoding="utf-8")
    return repo


def test_artifact_status_reports_tiny_track_when_artifacts_exist(tmp_path):
    repo = make_repo(tmp_path)
    _write_tinystories_sample(repo)
    _write_tokenizer_artifact(repo, "StoryTokenizer", source="tinystories")
    _write_tokenized_corpus_artifact(repo, "StoryLM-tinystories-100MB-v4096")

    tracks = track_status(repo)
    report = render_artifact_status(repo, module="10")

    assert tracks[0].name == "Tiny"
    assert tracks[0].present is True
    assert tracks[1].name == "Standard"
    assert tracks[1].present is False
    assert "ok Tiny" in report
    assert "StoryLM-tinystories-100MB-v4096" in report
    assert "Module 10 readiness" in report
    assert "ok Tiny StoryLM tokenized corpus is ready." in report


def test_artifact_status_full_track_satisfies_smaller_tracks(tmp_path, monkeypatch):
    repo = make_repo(tmp_path)
    monkeypatch.setattr("scripts.artifact_status.GLOVE_MIN_BYTES", 1)
    _write_tinystories_full(repo)
    _write_g2c_corpus(repo, "data/datasets/g2c-corpus-v1")
    glove_path = repo / "data" / "embeddings" / "glove.6B.50d.txt"
    glove_path.parent.mkdir(parents=True, exist_ok=True)
    glove_path.write_text("x", encoding="utf-8")
    _write_tokenizer_artifact(repo, "StoryTokenizer", source="tinystories")
    _write_tokenizer_artifact(repo, "G2CTokenizer", source="g2c")
    _write_tokenized_corpus_artifact(repo, "StoryLM-tinystories-full-v4096")
    _write_tokenized_corpus_artifact(repo, "TinyLLM-g2c-full-v8192")

    tracks = track_status(repo)

    assert [track.name for track in tracks] == ["Tiny", "Standard", "Full"]
    assert all(track.present for track in tracks)


def test_artifact_status_module_13_recommends_baselm_without_model_artifact(tmp_path):
    repo = make_repo(tmp_path)

    guidance = module_guidance("13", repo)

    assert guidance == ["  next: run ./baselm.sh or save a Module 10 model artifact first."]


def test_artifact_status_module_13_accepts_baselm_artifact(tmp_path):
    repo = make_repo(tmp_path)
    _write_baselm_artifact(repo)

    guidance = module_guidance("13", repo)

    assert guidance == [
        "  ok BaseLM is available for the pretrained fallback path.",
        "  optional: save a Module 10 model artifact for comparison.",
    ]


def _write_tinystories_sample(repo: Path) -> None:
    data_dir = repo / "data" / "datasets" / "tinystories"
    data_dir.mkdir(parents=True)
    with gzip.open(
        data_dir / "TinyStories-train-100MB-0000.txt.gz",
        "wt",
        encoding="utf-8",
    ) as f:
        f.write("once upon a time")


def _write_tinystories_full(repo: Path) -> None:
    data_dir = repo / "data" / "datasets" / "tinystories"
    data_dir.mkdir(parents=True)
    (data_dir / "TinyStories-train.txt").write_text("once upon a time", encoding="utf-8")
    (data_dir / "TinyStories-valid.txt").write_text("the end", encoding="utf-8")


def _write_g2c_corpus(repo: Path, relative_dir: str) -> None:
    corpus_dir = repo / relative_dir
    corpus_dir.mkdir(parents=True)
    (corpus_dir / "train.txt").write_text("course text", encoding="utf-8")
    (corpus_dir / "manifest.json").write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "source": "course",
                        "shards": [
                            {
                                "split": "train",
                                "path": "train.txt",
                                "uncompressed_bytes": 11,
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def _write_tokenizer_artifact(repo: Path, name: str, *, source: str) -> None:
    artifact_dir = repo / "artifacts" / "tokenizers" / name
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (artifact_dir / "manifest.json").write_text(
        json.dumps({"name": name, "source": source}),
        encoding="utf-8",
    )
    (artifact_dir / "ids.uint32").write_bytes(b"")


def _write_tokenized_corpus_artifact(repo: Path, name: str) -> None:
    artifact_dir = repo / "data" / "cache" / name
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "tokens.uint16.bin").write_bytes(b"\x00\x00\x01\x00")
    (artifact_dir / "manifest.json").write_text(
        json.dumps({"name": name, "file": "tokens.uint16.bin"}),
        encoding="utf-8",
    )


def _write_baselm_artifact(repo: Path) -> None:
    artifact_dir = repo / "artifacts" / "models" / "BaseLM"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "config.json").write_text(
        json.dumps({"kind": "huggingface_causal_lm", "model_id": "org/test"}),
        encoding="utf-8",
    )
    (artifact_dir / "manifest.json").write_text(
        json.dumps({"kind": "huggingface_causal_lm", "name": "BaseLM"}),
        encoding="utf-8",
    )
