#!/usr/bin/env python
"""Report local course data/model artifacts and suggest the next setup step."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from g2c.artifacts import (
    DEFAULT_BASELM_NAME,
    available_model_artifacts,
    baselm_artifact_exists,
    best_model_artifact,
    best_model_artifact_with_suffix,
    find_repo_root,
    model_artifact_exists,
    resolve_corpus,
    tokenized_corpus_artifact_exists,
    tokenizer_artifact_exists,
)

GLOVE_MIN_BYTES = 171_350_079


@dataclass(frozen=True)
class StatusItem:
    """One thing the course can detect locally."""

    label: str
    present: bool
    detail: str = ""
    command: str = ""


@dataclass(frozen=True)
class TrackStatus:
    """One dataset/artifact preparation track."""

    name: str
    command: str
    present: bool
    detail: str


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--module",
        default=None,
        help="Optional module number/name for a focused readiness hint, e.g. 04, 10, 13.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root. Default: auto-detect from current directory.",
    )
    args = parser.parse_args(argv)

    repo_root = find_repo_root(args.repo_root)
    print(render_artifact_status(repo_root, module=args.module))
    return 0


def render_artifact_status(
    repo_root: str | Path | None = None,
    *,
    module: str | None = None,
) -> str:
    """Return a human-readable artifact status report."""
    root = find_repo_root(repo_root)
    datasets = dataset_status(root)
    tokenizers = tokenizer_status(root)
    tokenized = tokenized_corpus_status(root)
    tracks = track_status(root)
    models = available_model_artifacts(repo_root=root)
    best_model = best_model_artifact(repo_root=root)

    lines = [
        "Course artifact status",
        f"repo: {root}",
        "",
        "Dataset tracks",
    ]
    for track in tracks:
        marker = _marker(track.present)
        lines.append(f"  {marker} {track.name:<8} {track.command:<22} {track.detail}")

    lines.extend(["", "Datasets"])
    lines.extend(_render_items(datasets))
    lines.extend(["", "Tokenizers"])
    lines.extend(_render_items(tokenizers))
    lines.extend(["", "Tokenized corpora"])
    lines.extend(_render_items(tokenized))
    lines.extend(["", "Model artifacts"])
    has_baselm = baselm_artifact_exists(repo_root=root)
    if models:
        for artifact in models:
            lines.append(
                f"  ok {artifact.display_name:<18} {artifact.artifact_dir.relative_to(root)}"
            )
    elif not has_baselm:
        lines.append("  -- none yet")
    if best_model is not None:
        lines.append(f"  best: {best_model.display_name} ({best_model.name})")
    if has_baselm:
        baselm_dir = (
            DEFAULT_BASELM_NAME
            if model_artifact_exists(DEFAULT_BASELM_NAME, repo_root=root)
            else "BaseLM"
        )
        lines.append(f"  ok BaseLM             artifacts/models/{baselm_dir}")

    lines.extend(["", "Recommended commands"])
    lines.extend(_recommend_commands(datasets, tokenizers, tokenized))

    if module is not None:
        lines.extend(["", f"Module {module} readiness"])
        lines.extend(module_guidance(module, root))

    return "\n".join(lines)


def dataset_status(repo_root: str | Path | None = None) -> list[StatusItem]:
    """Return local dataset availability checks."""
    root = find_repo_root(repo_root)
    return [
        _corpus_item(
            "TinyShakespeare",
            "tinyshakespeare",
            root,
            command="./setup.sh",
        ),
        _corpus_item(
            "TinyStories 100MB",
            "tinystories-100MB",
            root,
            command="./datasets.sh --tiny",
        ),
        _tinystories_full_item(root),
        _corpus_item(
            "G2C Corpus small",
            "g2c-corpus-small",
            root,
            command="./datasets.sh --small",
        ),
        _corpus_item(
            "G2C Corpus full",
            "g2c-corpus-full",
            root,
            command="./datasets.sh",
        ),
        _glove_item(root),
    ]


def tokenizer_status(repo_root: str | Path | None = None) -> list[StatusItem]:
    """Return local tokenizer artifact availability checks."""
    root = find_repo_root(repo_root)
    return [
        _tokenizer_item("ShakespeareTokenizer", root, command="./setup.sh"),
        _tokenizer_item("StoryTokenizer", root, command="./datasets.sh --tiny"),
        _tokenizer_item("G2CTokenizer", root, command="./datasets.sh --small"),
    ]


def tokenized_corpus_status(repo_root: str | Path | None = None) -> list[StatusItem]:
    """Return disk-backed tokenized corpus artifact checks."""
    root = find_repo_root(repo_root)
    return [
        _tokenized_item(
            "StoryLM TinyStories 100MB",
            "StoryLM-tinystories-100MB-v4096",
            root,
            command="./datasets.sh --tiny",
        ),
        _tokenized_item(
            "StoryLM TinyStories full",
            "StoryLM-tinystories-full-v4096",
            root,
            command="./datasets.sh --small",
        ),
        _tokenized_item(
            "TinyLLM G2C small",
            "TinyLLM-g2c-small-v8192",
            root,
            command="./datasets.sh --small",
        ),
        _tokenized_item(
            "TinyLLM G2C full",
            "TinyLLM-g2c-full-v8192",
            root,
            command="./datasets.sh",
        ),
    ]


def track_status(repo_root: str | Path | None = None) -> list[TrackStatus]:
    """Return the three local dataset-preparation track statuses."""
    root = find_repo_root(repo_root)
    has_story_sample = (
        _has_corpus("tinystories-100MB", root)
        and tokenizer_artifact_exists("StoryTokenizer", repo_root=root)
        and tokenized_corpus_artifact_exists("StoryLM-tinystories-100MB-v4096", repo_root=root)
    )
    has_story_full = (
        _has_tinystories_full(root)
        and tokenizer_artifact_exists("StoryTokenizer", repo_root=root)
        and tokenized_corpus_artifact_exists("StoryLM-tinystories-full-v4096", repo_root=root)
    )
    has_g2c_small = (
        _has_tinystories_full(root)
        and _has_glove(root)
        and _has_corpus("g2c-corpus-small", root)
        and tokenizer_artifact_exists("StoryTokenizer", repo_root=root)
        and tokenizer_artifact_exists("G2CTokenizer", repo_root=root)
        and tokenized_corpus_artifact_exists("StoryLM-tinystories-full-v4096", repo_root=root)
        and tokenized_corpus_artifact_exists("TinyLLM-g2c-small-v8192", repo_root=root)
    )
    has_full = (
        _has_tinystories_full(root)
        and _has_glove(root)
        and _has_corpus("g2c-corpus-full", root)
        and tokenizer_artifact_exists("StoryTokenizer", repo_root=root)
        and tokenizer_artifact_exists("G2CTokenizer", repo_root=root)
        and tokenized_corpus_artifact_exists("StoryLM-tinystories-full-v4096", repo_root=root)
        and tokenized_corpus_artifact_exists("TinyLLM-g2c-full-v8192", repo_root=root)
    )
    has_tiny = has_story_sample or has_story_full
    has_small = has_g2c_small or has_full
    if has_story_sample:
        tiny_detail = "100MB TinyStories + StoryTokenizer + tokenized StoryLM artifact"
    elif has_story_full:
        tiny_detail = "covered by full TinyStories + StoryTokenizer + tokenized StoryLM artifact"
    else:
        tiny_detail = "100MB TinyStories + StoryTokenizer + tokenized StoryLM artifact"

    if has_g2c_small:
        standard_detail = "GloVe + full TinyStories + small G2C corpus + tokenized artifacts"
    elif has_full:
        standard_detail = "covered by full G2C corpus + tokenized artifacts"
    else:
        standard_detail = "GloVe + full TinyStories + small G2C corpus + tokenized artifacts"
    return [
        TrackStatus(
            "Tiny",
            "./datasets.sh --tiny",
            has_tiny,
            tiny_detail,
        ),
        TrackStatus(
            "Standard",
            "./datasets.sh --small",
            has_small,
            standard_detail,
        ),
        TrackStatus(
            "Full",
            "./datasets.sh",
            has_full,
            "GloVe + full TinyStories + full G2C corpus + tokenized artifacts",
        ),
    ]


def module_guidance(module: str, repo_root: str | Path | None = None) -> list[str]:
    """Return a short readiness hint for one module."""
    root = find_repo_root(repo_root)
    key = _normalize_module(module)
    lines: list[str] = []

    if key == "4":
        lines.append("  Module 04 works without large data for the conceptual exercises.")
        if tokenizer_artifact_exists("StoryTokenizer", repo_root=root):
            lines.append("  ok StoryTokenizer artifact is available for the mini milestone.")
        else:
            lines.append("  next: run ./datasets.sh --tiny for the light artifact path.")
        return lines

    if key == "5":
        if _has_glove(root):
            lines.append("  ok GloVe vectors are available for pretrained embedding exercises.")
        else:
            lines.append("  next: run ./datasets.sh glove for pretrained embedding exercises.")
        return lines

    if key == "10":
        if tokenized_corpus_artifact_exists("TinyLLM-g2c-full-v8192", repo_root=root):
            lines.append("  ok Full TinyLLM tokenized corpus is ready.")
        elif tokenized_corpus_artifact_exists("TinyLLM-g2c-small-v8192", repo_root=root):
            lines.append("  ok Standard TinyLLM tokenized corpus is ready.")
        elif tokenized_corpus_artifact_exists("StoryLM-tinystories-full-v4096", repo_root=root):
            lines.append("  ok full StoryLM tokenized corpus is ready.")
        elif tokenized_corpus_artifact_exists("StoryLM-tinystories-100MB-v4096", repo_root=root):
            lines.append("  ok Tiny StoryLM tokenized corpus is ready.")
        elif tokenizer_artifact_exists("ShakespeareTokenizer", repo_root=root):
            lines.append("  ok baseline TinyShakespeare path is ready.")
            lines.append("  optional next: run ./datasets.sh --tiny or ./datasets.sh --small.")
        else:
            lines.append("  next: run ./setup.sh before the baseline TinyShakespeare run.")
        return lines

    if key in {"11", "12"}:
        best = best_model_artifact(repo_root=root)
        if best is None:
            if baselm_artifact_exists(repo_root=root):
                lines.append("  ok BaseLM is available if you explicitly select it.")
                lines.append("  recommended: finish and save a Module 10 model artifact too.")
            else:
                lines.append(
                    "  next: finish and save at least one Module 10 model artifact."
                )
        else:
            lines.append(f"  ok strongest saved model: {best.display_name} ({best.name}).")
        return lines

    if key == "13":
        best = best_model_artifact(repo_root=root)
        if best is None:
            if baselm_artifact_exists(repo_root=root):
                lines.append("  ok BaseLM is available for the pretrained fallback path.")
                lines.append("  optional: save a Module 10 model artifact for comparison.")
            else:
                lines.append("  next: run ./baselm.sh or save a Module 10 model artifact first.")
        else:
            lines.append(f"  ok self-trained model available: {best.display_name} ({best.name}).")
            lines.append(
                "  fallback: run ./baselm.sh if this model is too weak."
            )
        return lines

    if key == "14":
        best_sft = best_model_artifact_with_suffix("-SFT", repo_root=root)
        if best_sft is not None:
            lines.append(f"  ok SFT artifact available: {best_sft.name}.")
        else:
            lines.append("  next: finish Module 13 and save an SFT artifact first.")
            lines.append("  expected artifact names look like TinyLLM-30M-SFT or BaseLM-SFT.")
        return lines

    if key == "15":
        best_dpo = best_model_artifact_with_suffix("-DPO", repo_root=root)
        best_sft = best_model_artifact_with_suffix("-SFT", repo_root=root)
        if best_dpo is not None:
            lines.append(f"  ok DPO artifact available: {best_dpo.name}.")
        elif best_sft is not None:
            lines.append(f"  ok SFT artifact available: {best_sft.name}.")
            lines.append("  recommended: finish Module 14 and save a DPO artifact too.")
        else:
            lines.append("  next: finish Module 13 or Module 14 and save a post-trained artifact.")
        return lines

    if key in {"16", "17", "18", "19", "20"}:
        lines.append(
            "  ProdLM is expected here: choose a local pretrained instruct model for your Mac."
        )
        lines.append("  TinyLLM/StoryLM can still be used as comparison backends when available.")
        return lines

    lines.append("  No special artifact requirements beyond ./setup.sh.")
    return lines


def _render_items(items: list[StatusItem]) -> list[str]:
    lines = []
    for item in items:
        detail = f" ({item.detail})" if item.detail else ""
        command = f" -> {item.command}" if item.command and not item.present else ""
        lines.append(f"  {_marker(item.present)} {item.label:<28}{detail}{command}")
    return lines


def _recommend_commands(
    datasets: list[StatusItem],
    tokenizers: list[StatusItem],
    tokenized: list[StatusItem],
) -> list[str]:
    missing_by_command: dict[str, list[str]] = {}
    all_items = [*datasets, *tokenizers, *tokenized]
    present_labels = {item.label for item in all_items if item.present}
    for item in all_items:
        if item.present or not item.command:
            continue
        if _covered_by_larger_artifact(item.label, present_labels):
            continue
        missing_by_command.setdefault(item.command, []).append(item.label)

    if not missing_by_command:
        return ["  ok local tracked datasets/artifacts are present."]

    priority = [
        "./setup.sh",
        "./datasets.sh --tiny",
        "./datasets.sh --small",
        "./datasets.sh",
        "./datasets.sh glove",
    ]
    commands = sorted(
        missing_by_command,
        key=lambda command: priority.index(command) if command in priority else len(priority),
    )
    return [
        f"  {command:<24} # prepares {', '.join(missing_by_command[command][:3])}"
        for command in commands
    ]


def _covered_by_larger_artifact(label: str, present_labels: set[str]) -> bool:
    if label in {"TinyStories 100MB", "StoryLM TinyStories 100MB"}:
        return "TinyStories full" in present_labels or "StoryLM TinyStories full" in present_labels
    if label in {"G2C Corpus small", "TinyLLM G2C small"}:
        return "G2C Corpus full" in present_labels or "TinyLLM G2C full" in present_labels
    return False


def _corpus_item(label: str, corpus: str, repo_root: Path, *, command: str) -> StatusItem:
    spec = resolve_corpus(corpus, repo_root=repo_root)
    if spec is None:
        return StatusItem(label, False, command=command)
    return StatusItem(label, True, _human_size(spec.uncompressed_bytes), command)


def _tinystories_full_item(repo_root: Path) -> StatusItem:
    if not _has_tinystories_full(repo_root):
        return StatusItem("TinyStories full", False, command="./datasets.sh --small")
    spec = resolve_corpus("tinystories", repo_root=repo_root)
    detail = _human_size(spec.uncompressed_bytes) if spec is not None else ""
    return StatusItem("TinyStories full", True, detail, "./datasets.sh --small")


def _glove_item(repo_root: Path) -> StatusItem:
    path = repo_root / "data" / "embeddings" / "glove.6B.50d.txt"
    if _has_glove(repo_root):
        return StatusItem(
            "GloVe 6B 50d",
            True,
            _human_size(path.stat().st_size),
            "./datasets.sh glove",
        )
    return StatusItem("GloVe 6B 50d", False, command="./datasets.sh glove")


def _tokenizer_item(name: str, repo_root: Path, *, command: str) -> StatusItem:
    present = tokenizer_artifact_exists(name, repo_root=repo_root)
    return StatusItem(name, present, command=command)


def _tokenized_item(label: str, name: str, repo_root: Path, *, command: str) -> StatusItem:
    present = tokenized_corpus_artifact_exists(name, repo_root=repo_root)
    return StatusItem(label, present, name, command)


def _has_corpus(corpus: str, repo_root: Path) -> bool:
    return resolve_corpus(corpus, repo_root=repo_root) is not None


def _has_tinystories_full(repo_root: Path) -> bool:
    spec = resolve_corpus("tinystories", repo_root=repo_root)
    if spec is None:
        return False
    return all("100MB" not in shard.path.name for source in spec.sources for shard in source.shards)


def _has_glove(repo_root: Path) -> bool:
    path = repo_root / "data" / "embeddings" / "glove.6B.50d.txt"
    return path.exists() and path.stat().st_size >= GLOVE_MIN_BYTES


def _marker(present: bool) -> str:
    return "ok" if present else "--"


def _normalize_module(module: str) -> str:
    text = module.strip().lower()
    if text.startswith("module"):
        text = text.removeprefix("module").strip(" -")
    if text.startswith("0") and text not in {"00"}:
        text = text.lstrip("0")
    return text


def _human_size(value: int) -> str:
    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size) < 1000.0 or unit == "TB":
            if unit == "B":
                return f"{size:.0f}{unit}"
            return f"{size:.2f}{unit}"
        size /= 1000.0


if __name__ == "__main__":
    raise SystemExit(main())
