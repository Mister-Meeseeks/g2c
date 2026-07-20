"""Notebook helpers for choosing saved model artifacts.

These helpers keep artifact-discovery plumbing out of the notebooks. The
selection rules are course workflow, not the concept under study.
"""

from __future__ import annotations

from pathlib import Path

from g2c.artifacts import (
    baselm_artifact_exists,
    best_model_artifact,
    best_model_artifact_with_suffix,
    model_artifact_exists,
    resolve_artifact_name,
    stage_root_name,
)
from g2c.inference import resolve_preferred_artifact_name


def select_base_artifact_name(selection: str, *, repo_root: str | Path | None = None) -> str:
    """Return the base artifact Module 13 should SFT.

    ``"auto"`` prefers the strongest base course artifact and falls back to
    BaseLM — the preference order the course design calls for, since the
    lesson is weight updates on your own model where that is possible.
    ``"BaseLM"`` selects the external pretrained BaseLM artifact. ``"course"``
    requires a course artifact and fails if there is none. Any other non-empty
    string is treated as an explicit base artifact name.
    """
    selection = selection.strip()
    if selection == "auto":
        candidate = best_model_artifact(repo_root=repo_root)
        if candidate is not None:
            return candidate.name
        if baselm_artifact_exists(repo_root=repo_root):
            return resolve_artifact_name("BaseLM", repo_root=repo_root)
        raise RuntimeError(
            "No course model artifact and no BaseLM. Run Module 10 to train "
            "one, or run ./baselm.sh to fetch BaseLM."
        )
    if selection == "BaseLM":
        if not baselm_artifact_exists(repo_root=repo_root):
            raise RuntimeError(
                "BaseLM is not configured. Run ./baselm.sh or set "
                "MODEL_SELECTION = 'course'."
            )
        return resolve_artifact_name("BaseLM", repo_root=repo_root)
    if selection == "course":
        candidate = best_model_artifact(repo_root=repo_root)
        if candidate is None:
            raise RuntimeError(
                "No course model artifact found. Run Module 10 or set "
                "MODEL_SELECTION = 'BaseLM'."
            )
        return candidate.name
    if not selection:
        raise ValueError("MODEL_SELECTION must be 'BaseLM', 'course', or an artifact name.")
    return selection


def select_sampling_artifact_name(
    selection: str | None,
    *,
    repo_root: str | Path | None = None,
) -> str | None:
    """Return the artifact Module 11 should sample from.

    ``None`` means "strongest base course artifact", preserving Module 11's
    focus on the models produced in Module 10. ``"BaseLM"`` is available as an
    explicit override for comparison.
    """
    if selection is None:
        candidate = best_model_artifact(repo_root=repo_root)
        return candidate.name if candidate is not None else None

    selection = selection.strip()
    if selection == "BaseLM":
        if not baselm_artifact_exists(repo_root=repo_root):
            raise RuntimeError("BaseLM is not configured. Run ./baselm.sh before selecting it.")
        return resolve_artifact_name("BaseLM", repo_root=repo_root)
    if not selection:
        raise ValueError("MODEL_ARTIFACT_NAME must be None, 'BaseLM', or an artifact name.")
    return selection


def select_sft_artifact_name(selection: str, *, repo_root: str | Path | None = None) -> str:
    """Return the SFT artifact Module 14 should use as its DPO starting point."""
    selection = selection.strip()
    if selection == "BaseLM":
        if not model_artifact_exists("BaseLM-SFT", repo_root=repo_root):
            raise RuntimeError(
                "BaseLM-SFT was not found. Run Module 13 with "
                "MODEL_SELECTION = 'BaseLM' first."
            )
        return "BaseLM-SFT"
    if selection == "course":
        candidate = best_model_artifact_with_suffix(
            "-SFT",
            repo_root=repo_root,
            extra_base_names=(),
        )
        if candidate is None:
            raise RuntimeError(
                "No course *-SFT artifact found. Run Module 13 on a course "
                "model or set MODEL_SELECTION = 'BaseLM'."
            )
        return candidate.name
    if not selection:
        raise ValueError("MODEL_SELECTION must be 'BaseLM', 'course', or an artifact name.")
    return selection if selection.endswith("-SFT") else f"{stage_root_name(selection)}-SFT"


def select_eval_artifact_name(selection: str, *, repo_root: str | Path | None = None) -> str:
    """Return the post-training artifact Module 15 should evaluate.

    Selection prefers DPO, then SFT, then base. This lets a student name a
    family/base artifact once and rerun evaluation as later stages appear.
    """
    selection = selection.strip()
    if not selection:
        raise ValueError("MODEL_SELECTION must be 'BaseLM', 'course', or an artifact name.")
    try:
        return resolve_preferred_artifact_name(selection, repo_root=repo_root)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "No matching evaluation artifact found. Run Module 13/14 or choose "
            "another MODEL_SELECTION."
        ) from exc


def select_inference_artifact_name(
    selection: str,
    *,
    repo_root: str | Path | None = None,
    load_artifact: bool = True,
) -> str | None:
    """Return the artifact backend Module 16 should compare against ProdLM."""
    if not load_artifact:
        return None
    selection = selection.strip()
    if not selection:
        raise ValueError("ARTIFACT_SELECTION must be 'BaseLM', 'course', or an artifact name.")
    try:
        return resolve_preferred_artifact_name(selection, repo_root=repo_root)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "No matching artifact found. Run ./baselm.sh, save a course model, "
            "or choose another ARTIFACT_SELECTION."
        ) from exc
