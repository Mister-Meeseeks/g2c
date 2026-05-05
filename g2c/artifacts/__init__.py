"""Reusable course artifact helpers.

Artifacts are generated outputs that later modules can reload without depending
on notebook state. Keep bulky path, detection, and load/save plumbing here so
notebooks can focus on the exercise workflow and visual inspection.
"""
from .corpora import (
    CorpusShard,
    CorpusSource,
    CorpusSpec,
    load_corpus_bytes,
    load_corpus_text,
    resolve_corpus,
)
from .paths import artifacts_root, find_repo_root, tokenizer_artifact_dir, tokenizer_artifacts_root
from .tokenizers import (
    ArtifactProgressCallback,
    ArtifactStatusCallback,
    TokenizerArtifact,
    TokenizerArtifactConfig,
    load_tinystories_text,
    load_tokenizer_source_text,
    tokenizer_artifact_exists,
    train_or_load_tokenizer_artifact,
)

__all__ = [
    "ArtifactProgressCallback",
    "ArtifactStatusCallback",
    "CorpusShard",
    "CorpusSource",
    "CorpusSpec",
    "TokenizerArtifact",
    "TokenizerArtifactConfig",
    "artifacts_root",
    "find_repo_root",
    "load_corpus_bytes",
    "load_corpus_text",
    "load_tinystories_text",
    "load_tokenizer_source_text",
    "resolve_corpus",
    "tokenizer_artifact_dir",
    "tokenizer_artifact_exists",
    "tokenizer_artifacts_root",
    "train_or_load_tokenizer_artifact",
]
