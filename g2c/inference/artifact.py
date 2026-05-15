"""Backend helpers for saved course model artifacts.

Module 16's backend interface is broader than a raw ``TransformerLM``. By this
point the course can load self-trained models, SFT/DPO derivatives, and small
Hugging Face BaseLM artifacts through the common artifact layer. ``ArtifactBackend``
adapts any of those loaded artifacts to the same ``Backend.complete`` interface
used by RAG, tools, and agents.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from g2c.artifacts import (
    LoadedModelArtifact,
    baselm_artifact_exists,
    best_model_artifact,
    best_model_artifact_with_suffix,
    load_model_artifact_with_tokenizer,
    model_artifact_exists,
    resolve_artifact_name,
)

from .local import LocalTransformerBackend


class ArtifactBackend(LocalTransformerBackend):
    """A ``Backend`` over a saved ``LoadedModelArtifact``.

    This is the preferred in-process backend for notebooks after Module 16.
    It keeps the useful local behavior from ``LocalTransformerBackend`` but
    handles the course artifact details automatically:

    * uses ``encode_with_vocab_size`` when available, so large tokenizers can
      safely drive smaller model vocabularies;
    * picks a course end token as ``eos_id`` when the tokenizer exposes one;
    * carries artifact provenance in ``BackendInfo.extra``.
    """

    def __init__(
        self,
        artifact: LoadedModelArtifact,
        *,
        eos_id: int | None = None,
        name: str = "artifact",
        extra: dict[str, Any] | None = None,
    ) -> None:
        self._artifact = artifact
        tokenizer = _ArtifactTokenizer(artifact.tokenizer, artifact.model.vocab_size)
        if eos_id is None:
            eos_id = _default_eos_id(artifact.tokenizer)

        merged_extra = {
            "artifact_name": artifact.name,
            "canonical_name": artifact.canonical_name,
            "source": artifact.manifest.get("source"),
            "kind": artifact.manifest.get("kind", "course_transformer"),
        }
        if extra is not None:
            merged_extra.update(extra)

        if hasattr(artifact.model, "eval"):
            artifact.model.eval()

        super().__init__(
            artifact.model,
            tokenizer,
            model_id=artifact.name,
            eos_id=eos_id,
            name=name,
            extra=merged_extra,
        )

    @property
    def artifact(self) -> LoadedModelArtifact:
        """The loaded artifact backing this backend."""
        return self._artifact

    @property
    def raw_tokenizer(self):
        """The underlying artifact tokenizer, before vocab-size wrapping."""
        return self._artifact.tokenizer


def load_artifact_backend(
    artifact_name: str | None = None,
    *,
    repo_root: str | None = None,
    device: str | None = "auto",
    torch_dtype: str | None = None,
    prefer_suffixes: tuple[str, ...] = ("-DPO", "-SFT", ""),
    required: bool = True,
) -> ArtifactBackend | None:
    """Load a saved model artifact and wrap it as an ``ArtifactBackend``.

    If ``artifact_name`` is omitted, the strongest available post-training
    artifact is selected first (DPO, then SFT), then the strongest base Module
    10 artifact, then ``BaseLM`` if configured.
    """
    if artifact_name is None:
        artifact_name = _select_default_artifact_name(
            repo_root=repo_root,
            prefer_suffixes=prefer_suffixes,
        )
    if artifact_name is None:
        if not required:
            return None
        raise FileNotFoundError(
            "No model artifact found. Save a Module 10/13/14 artifact or run "
            "./baselm.sh for BaseLM."
        )

    loaded = load_model_artifact_with_tokenizer(
        artifact_name,
        repo_root=repo_root,
        device=device,
        torch_dtype=torch_dtype,
    )
    return ArtifactBackend(loaded)


def resolve_preferred_artifact_name(
    selection: str,
    *,
    repo_root: str | Path | None = None,
    suffixes: tuple[str, ...] = ("-DPO", "-SFT", ""),
) -> str:
    """Resolve an artifact selection to the best available concrete artifact.

    ``selection="course"`` means "the strongest course-trained artifact",
    excluding BaseLM, preferring DPO, then SFT, then the base model. A concrete
    base name such as ``"StoryLM-30M"`` resolves by trying
    ``StoryLM-30M-DPO``, then ``StoryLM-30M-SFT``, then ``StoryLM-30M``. If
    the caller passes an already-suffixed name such as ``"StoryLM-30M-SFT"``,
    that exact artifact is resolved directly.
    """
    selection = selection.strip()
    if not selection:
        raise ValueError("artifact selection must be non-empty")

    if selection == "course":
        for suffix in suffixes:
            if suffix:
                candidate = best_model_artifact_with_suffix(
                    suffix,
                    repo_root=repo_root,
                    extra_base_names=(),
                )
            else:
                candidate = best_model_artifact(repo_root=repo_root)
            if candidate is not None:
                return candidate.name
        raise FileNotFoundError(
            "No course-trained DPO, SFT, or base artifact found. Run Module 10 "
            "or select a concrete artifact name."
        )

    if any(selection.endswith(suffix) for suffix in suffixes if suffix):
        return resolve_artifact_name(selection, repo_root=repo_root)

    for suffix in suffixes:
        name = f"{selection}{suffix}"
        try:
            resolved = resolve_artifact_name(name, repo_root=repo_root)
        except FileNotFoundError:
            continue
        if model_artifact_exists(resolved, repo_root=repo_root):
            return resolved

    raise FileNotFoundError(
        f"No artifact found for {selection!r}. Tried DPO, SFT, and base variants."
    )


class _ArtifactTokenizer:
    """Tokenizer wrapper that respects a model's effective vocabulary size."""

    def __init__(self, tokenizer, vocab_size: int) -> None:
        self._tokenizer = tokenizer
        self._vocab_size = int(vocab_size)

    def encode(self, text: str) -> list[int]:
        if hasattr(self._tokenizer, "encode_with_vocab_size"):
            return self._tokenizer.encode_with_vocab_size(text, self._vocab_size)
        ids = self._tokenizer.encode(text)
        too_large = [token_id for token_id in ids if token_id >= self._vocab_size]
        if too_large:
            raise ValueError(
                "encoded prompt contains IDs outside the model vocab: "
                f"max={max(too_large)}, model.vocab_size={self._vocab_size}"
            )
        return ids

    def decode(self, ids: list[int]) -> str:
        return self._tokenizer.decode(ids)


def _default_eos_id(tokenizer) -> int | None:
    special_to_id = getattr(tokenizer, "special_to_id", {})
    for token in ("<|end|>", "<|endoftext|>"):
        token_id = special_to_id.get(token)
        if token_id is not None:
            return int(token_id)
    token_id = getattr(tokenizer, "eos_token_id", None)
    return int(token_id) if token_id is not None else None


def _select_default_artifact_name(
    *,
    repo_root: str | None,
    prefer_suffixes: tuple[str, ...],
) -> str | None:
    for suffix in prefer_suffixes:
        if suffix:
            candidate = best_model_artifact_with_suffix(suffix, repo_root=repo_root)
        else:
            candidate = best_model_artifact(repo_root=repo_root)
        if candidate is not None:
            return candidate.name
    if baselm_artifact_exists(repo_root=repo_root):
        return "BaseLM"
    return None
