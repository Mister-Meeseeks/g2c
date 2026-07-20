from __future__ import annotations

import pytest

from g2c.artifacts import model_artifact_dir, write_baselm_manifest
from g2c.notebook_extras.model_selection import select_base_artifact_name


def _write_course_artifact_marker(tmp_path, name: str) -> None:
    root = model_artifact_dir(name, tmp_path)
    root.mkdir(parents=True, exist_ok=True)
    (root / "model.pt").write_bytes(b"placeholder")
    (root / "config.json").write_text("{}", encoding="utf-8")
    (root / "manifest.json").write_text("{}", encoding="utf-8")


class TestNotebookModelSelection:
    def test_select_base_artifact_name_uses_baselm_when_configured(self, tmp_path) -> None:
        write_baselm_manifest(repo_root=tmp_path)

        assert select_base_artifact_name("BaseLM", repo_root=tmp_path) == "BaseLM-base"

    def test_select_base_artifact_name_errors_when_baselm_missing(self, tmp_path) -> None:
        with pytest.raises(RuntimeError, match="BaseLM is not configured"):
            select_base_artifact_name("BaseLM", repo_root=tmp_path)

    def test_select_base_name_course_uses_strongest_base_course_model(self, tmp_path) -> None:
        _write_course_artifact_marker(tmp_path, "StoryLM-5M-base")
        _write_course_artifact_marker(tmp_path, "TinyLLM-30M-base")

        assert select_base_artifact_name("course", repo_root=tmp_path) == "TinyLLM-30M-base"

    def test_select_base_artifact_name_explicit_name_passes_through(self, tmp_path) -> None:
        assert select_base_artifact_name("StoryLM-5M", repo_root=tmp_path) == "StoryLM-5M"

    def test_select_base_auto_prefers_course_artifact_over_baselm(self, tmp_path) -> None:
        """`auto` fine-tunes your own model when you have one — the lesson's point."""
        write_baselm_manifest(repo_root=tmp_path)
        _write_course_artifact_marker(tmp_path, "StoryLM-5M-base")

        assert select_base_artifact_name("auto", repo_root=tmp_path) == "StoryLM-5M-base"

    def test_select_base_auto_falls_back_to_baselm(self, tmp_path) -> None:
        """No Module 10 artifact yet: BaseLM stands in rather than hard-failing."""
        write_baselm_manifest(repo_root=tmp_path)

        assert select_base_artifact_name("auto", repo_root=tmp_path) == "BaseLM-base"

    def test_select_base_auto_errors_when_nothing_available(self, tmp_path) -> None:
        with pytest.raises(RuntimeError, match="No course model artifact and no BaseLM"):
            select_base_artifact_name("auto", repo_root=tmp_path)
