"""Filesystem path helpers for course artifacts."""
from __future__ import annotations

from pathlib import Path


def find_repo_root(start: str | Path | None = None) -> Path:
    """Find the repository root from a notebook, script, or package call site."""
    path = Path.cwd().resolve() if start is None else Path(start).resolve()
    if path.is_file():
        path = path.parent

    for candidate in (path, *path.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "g2c").exists():
            return candidate
    return path


def artifacts_root(repo_root: str | Path | None = None) -> Path:
    """Return the root directory for generated course artifacts."""
    return find_repo_root(repo_root) / "artifacts"


def tokenizer_artifacts_root(repo_root: str | Path | None = None) -> Path:
    """Return the directory containing tokenizer artifacts."""
    return artifacts_root(repo_root) / "tokenizers"


def tokenizer_artifact_dir(name: str, repo_root: str | Path | None = None) -> Path:
    """Return the directory for one named tokenizer artifact."""
    if not name:
        raise ValueError("tokenizer artifact name must be non-empty")
    return tokenizer_artifacts_root(repo_root) / name


def tokenized_corpora_root(repo_root: str | Path | None = None) -> Path:
    """Return the directory containing tokenized corpus cache directories.

    Tokenized corpora are reproducible caches keyed by tokenizer+source+vocab,
    not durable course artifacts, so they live under ``data/cache/`` rather
    than ``artifacts/``.
    """
    return find_repo_root(repo_root) / "data" / "cache"


def tokenized_corpus_artifact_dir(
    name: str,
    repo_root: str | Path | None = None,
) -> Path:
    """Return the directory for one named tokenized corpus cache."""
    if not name:
        raise ValueError("tokenized corpus artifact name must be non-empty")
    return tokenized_corpora_root(repo_root) / name
