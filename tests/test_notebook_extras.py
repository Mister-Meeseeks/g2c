from __future__ import annotations

import pytest
import torch

from g2c.artifacts import checkpoint_backup_path, model_artifact_dir, write_baselm_manifest
from g2c.notebook_extras.model_selection import select_base_artifact_name
from g2c.notebook_extras.pretraining import milestone_checkpoints, train_with_progress
from g2c.transformer import TransformerLM

_TINY_MODEL_CONFIG = {
    "embedding_dim": 8,
    "num_layers": 1,
    "num_heads": 2,
    "max_seq_len": 16,
    "hidden_dim": 32,
}


def _tiny_run(checkpoint_path, *, max_steps: int, **milestone_kwargs):
    """Run a few real training steps against ``checkpoint_path``."""
    torch.manual_seed(0)
    model = TransformerLM(vocab_size=12, **_TINY_MODEL_CONFIG)
    ids = torch.randint(0, 12, (512,))
    return train_with_progress(
        "tiny",
        model,
        ids,
        ids,
        seed=0,
        checkpoint_path=checkpoint_path,
        checkpoint_every=2,
        checkpoint_extra={"model_config": _TINY_MODEL_CONFIG, "vocab_size": 12},
        batch_size=2,
        context_length=8,
        max_steps=max_steps,
        max_lr=1e-3,
        min_lr=1e-4,
        warmup_steps=1,
        weight_decay=0.0,
        grad_clip=1.0,
        eval_every=1000,
        eval_iters=1,
        log_every=1,
        device="cpu",
        optimizer="adamw",
        **milestone_kwargs,
    )


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


class TestMilestoneCheckpoints:
    def test_lists_milestones_in_step_order(self, tmp_path) -> None:
        checkpoint_path = tmp_path / "run.ckpt"
        for step in (5000, 200, 15000):
            (tmp_path / f"run.step{step}.ckpt").write_bytes(b"x")
        # Neither the rolling checkpoint nor its backup is a milestone.
        checkpoint_path.write_bytes(b"x")
        checkpoint_backup_path(checkpoint_path).write_bytes(b"x")

        found = milestone_checkpoints(checkpoint_path)

        assert [step for step, _ in found] == [200, 5000, 15000]

    def test_retains_only_the_newest_keep(self, tmp_path) -> None:
        checkpoint_path = tmp_path / "run.ckpt"
        # A near-zero interval fires a milestone on every log event.
        _tiny_run(
            checkpoint_path,
            max_steps=6,
            milestone_minutes=1e-9,
            milestone_keep=2,
        )

        found = milestone_checkpoints(checkpoint_path)

        assert len(found) == 2, f"expected 2 retained milestones, got {found}"
        assert [step for step, _ in found] == [5, 6]

    def test_disabled_by_none(self, tmp_path) -> None:
        checkpoint_path = tmp_path / "run.ckpt"
        _tiny_run(checkpoint_path, max_steps=4, milestone_minutes=None)

        assert milestone_checkpoints(checkpoint_path) == []


class TestResumeFallback:
    def test_resumes_from_backup_when_rolling_checkpoint_missing(
        self, tmp_path, capsys
    ) -> None:
        """The exact failure that silently discarded a 42k-step run.

        `atomic_torch_save` used to leave a window with no rolling checkpoint on
        disk, and the resume gate keyed on that file alone -- so a good backup
        sat unread while training restarted from random init.
        """
        checkpoint_path = tmp_path / "run.ckpt"
        _tiny_run(checkpoint_path, max_steps=4, milestone_minutes=None)
        assert checkpoint_backup_path(checkpoint_path).exists()
        checkpoint_path.unlink()
        capsys.readouterr()

        _tiny_run(checkpoint_path, max_steps=4, milestone_minutes=None)

        out = capsys.readouterr().out
        assert "resuming from step" in out
        assert "run.ckpt.bak" in out
        assert "starting fresh" not in out

    def test_resumes_from_newest_milestone_when_rolling_pair_gone(
        self, tmp_path, capsys
    ) -> None:
        checkpoint_path = tmp_path / "run.ckpt"
        _tiny_run(checkpoint_path, max_steps=6, milestone_minutes=1e-9, milestone_keep=2)
        checkpoint_path.unlink()
        checkpoint_backup_path(checkpoint_path).unlink(missing_ok=True)
        newest = milestone_checkpoints(checkpoint_path)[-1]
        capsys.readouterr()

        _tiny_run(checkpoint_path, max_steps=6, milestone_minutes=None)

        out = capsys.readouterr().out
        assert "resuming from step" in out
        assert newest[1].name in out

    def test_starts_fresh_only_when_nothing_survives(self, tmp_path, capsys) -> None:
        checkpoint_path = tmp_path / "run.ckpt"

        _tiny_run(checkpoint_path, max_steps=2, milestone_minutes=None)

        out = capsys.readouterr().out
        assert "starting fresh" in out
