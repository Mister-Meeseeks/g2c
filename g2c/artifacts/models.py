"""Save and load durable language-model artifacts.

Implements the artifact convention from
``docs/design/model-artifacts-and-tracks.md``::

    artifacts/models/<artifact-name>/
      model.pt        # vocab_size, model_config, params
      config.json     # model_config + training_config
      manifest.json   # provenance: source, tokenizer artifact, seed, git, timing
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from os import PathLike
from pathlib import Path
from typing import Any

import torch

from g2c.transformer import TransformerLM

from .paths import artifacts_root


def model_artifact_dir(name: str, repo_root: str | Path | None = None) -> Path:
    if not name:
        raise ValueError("model artifact name must be non-empty")
    return artifacts_root(repo_root) / "models" / name


def model_artifact_exists(name: str, repo_root: str | Path | None = None) -> bool:
    root = model_artifact_dir(name, repo_root)
    return (root / "model.pt").exists() and (root / "manifest.json").exists()


def _try_git_commit(repo_root: str | Path | None) -> str | None:
    cwd = Path(repo_root) if repo_root else None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def save_model_artifact(
    name: str,
    *,
    model: TransformerLM,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    tokenizer_artifact_name: str,
    source: str,
    history: dict[str, list] | None = None,
    seed: int | None = None,
    module: str = "module-10",
    notes: str = "",
    repo_root: str | Path | None = None,
) -> Path:
    """Save a TransformerLM as a durable course artifact.

    Writes ``model.pt``, ``config.json``, ``manifest.json`` under
    ``artifacts/models/<name>/``. The tokenizer is stored *by reference* in the
    manifest -- the artifact assumes ``artifacts/tokenizers/<tokenizer_artifact_name>/``
    exists and is the canonical source for the vocabulary.

    Args:
        name: artifact name (e.g. ``"ShakespeareLM-1M"``).
        model: trained TransformerLM to persist.
        model_config: kwargs used to construct the model. Excludes ``vocab_size``
            (saved separately on the state dict for fast lookup).
        training_config: trainer kwargs used for the run.
        tokenizer_artifact_name: name under ``artifacts/tokenizers/``.
        source: human-readable description of the training corpus slice.
        history: optional training history dict; final losses are summarized
            into the manifest if provided.
        seed: optional seed used for the run.
        module: which course module produced this artifact.
        notes: free-form text for intended use, caveats, etc.
        repo_root: override the detected repo root.

    Returns:
        The artifact directory path.
    """
    root = model_artifact_dir(name, repo_root)
    root.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "vocab_size": model.vocab_size,
            "model_config": dict(model_config),
            "params": [p.detach().cpu() for p in model.parameters()],
        },
        root / "model.pt",
    )

    config = {
        "model_config": dict(model_config),
        "training_config": dict(training_config),
    }
    (root / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    final_train = (history["train_loss"][-1] if history and history.get("train_loss") else None)
    final_val = (history["val_loss"][-1] if history and history.get("val_loss") else None)
    steps_completed = (history["step"][-1] + 1 if history and history.get("step") else 0)

    manifest = {
        "name": name,
        "module": module,
        "source": source,
        "tokenizer_artifact": tokenizer_artifact_name,
        "vocab_size": model.vocab_size,
        "model_config": dict(model_config),
        "training_config": dict(training_config),
        "seed": seed,
        "git_commit": _try_git_commit(repo_root),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "steps_completed": steps_completed,
        "final_train_loss": final_train,
        "final_val_loss": final_val,
        "notes": notes,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    return root


def load_model_artifact(
    name: str,
    *,
    repo_root: str | Path | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Reload a saved model artifact into a fresh ``TransformerLM``.

    Returns a dict with::

        {
            "model": TransformerLM (params loaded),
            "manifest": dict,
            "training_config": dict,
        }

    Tokenizer loading is not done here because tokenizers are independent
    artifacts; use ``load_tokenizer_artifact(manifest["tokenizer_artifact"])``.
    """
    root = model_artifact_dir(name, repo_root)
    if not (root / "model.pt").exists():
        raise FileNotFoundError(f"No model.pt at {root}")

    state = torch.load(root / "model.pt", map_location=map_location, weights_only=False)
    config = json.loads((root / "config.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())

    model = TransformerLM(vocab_size=state["vocab_size"], **state["model_config"])
    saved_params = state["params"]
    target_params = list(model.parameters())
    if len(saved_params) != len(target_params):
        raise ValueError(
            f"Artifact {name!r} has {len(saved_params)} params, "
            f"model has {len(target_params)}"
        )
    with torch.no_grad():
        for target, saved in zip(target_params, saved_params, strict=True):
            target.copy_(saved.to(device=target.device, dtype=target.dtype))

    return {
        "model": model,
        "manifest": manifest,
        "training_config": config["training_config"],
    }
