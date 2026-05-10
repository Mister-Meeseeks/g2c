"""ProdLM configuration helpers.

``ProdLM`` is the course name for the capable local inference model used in
Modules 17-20. The recommended implementation is an Ollama-served quantized
instruction model. We store only a lightweight manifest under
``artifacts/models/ProdLM``; Ollama owns the actual model weights.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from g2c.artifacts import model_artifact_dir
from g2c.artifacts.models import atomic_json_save

from .artifact import load_artifact_backend
from .backend import Backend
from .ollama import DEFAULT_OLLAMA_URL, OllamaBackend

PRODLM_NAME = "ProdLM"
PRODLM_KIND = "ollama_backend"
DEFAULT_PRODLM_MODEL_ID = "llama3.2:3b"


def write_prodlm_manifest(
    *,
    name: str = PRODLM_NAME,
    model_id: str = DEFAULT_PRODLM_MODEL_ID,
    base_url: str = DEFAULT_OLLAMA_URL,
    timeout: float = 120.0,
    repo_root: str | Path | None = None,
    notes: str = "",
) -> Path:
    """Write the lightweight configured-ProdLM manifest."""
    root = model_artifact_dir(name, repo_root)
    root.mkdir(parents=True, exist_ok=True)
    config = {
        "kind": PRODLM_KIND,
        "backend": "ollama",
        "model_id": model_id,
        "base_url": base_url,
        "timeout": float(timeout),
    }
    manifest = {
        "name": name,
        "display_name": name,
        "kind": PRODLM_KIND,
        "role": "ProdLM",
        "module": "module-16",
        "source": f"Ollama model {model_id} at {base_url}",
        "model_id": model_id,
        "backend": "ollama",
        "notes": notes,
    }
    atomic_json_save(config, root / "config.json")
    atomic_json_save(manifest, root / "manifest.json")
    return root


def prodlm_manifest_exists(
    *,
    name: str = PRODLM_NAME,
    repo_root: str | Path | None = None,
) -> bool:
    """Return whether a configured ProdLM manifest exists."""
    config_path = model_artifact_dir(name, repo_root) / "config.json"
    if not config_path.exists():
        return False
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return config.get("kind") == PRODLM_KIND


def load_prodlm_backend(
    *,
    name: str = PRODLM_NAME,
    repo_root: str | Path | None = None,
    model_id: str | None = None,
    base_url: str | None = None,
    timeout: float | None = None,
    required: bool = False,
) -> OllamaBackend:
    """Load the configured ProdLM backend, defaulting to Ollama."""
    config = _read_prodlm_config(name=name, repo_root=repo_root)
    if config is None:
        if required and model_id is None and base_url is None and timeout is None:
            raise FileNotFoundError(
                f"No ProdLM manifest found at {model_artifact_dir(name, repo_root)}. "
                "Run ./prodlm.sh to configure one."
            )
        config = {}

    return OllamaBackend(
        model_id=model_id or str(config.get("model_id", DEFAULT_PRODLM_MODEL_ID)),
        base_url=base_url or str(config.get("base_url", DEFAULT_OLLAMA_URL)),
        timeout=float(timeout if timeout is not None else config.get("timeout", 120.0)),
        name="prodlm",
        extra={"configured_name": name},
    )


def load_default_backend(
    kind: Literal["auto", "artifact", "prodlm"] = "auto",
    *,
    repo_root: str | Path | None = None,
    artifact_name: str | None = None,
    device: str | None = "auto",
    torch_dtype: str | None = None,
) -> Backend:
    """Load the backend most likely wanted by downstream notebooks.

    ``kind="auto"`` prefers a configured ProdLM because Modules 17-20 need a
    capable instruction model. If ProdLM has not been configured, it falls back
    to the strongest saved artifact.
    """
    if kind == "prodlm":
        return load_prodlm_backend(repo_root=repo_root)
    if kind == "artifact":
        backend = load_artifact_backend(
            artifact_name,
            repo_root=str(repo_root) if repo_root is not None else None,
            device=device,
            torch_dtype=torch_dtype,
            required=True,
        )
        assert backend is not None
        return backend
    if kind != "auto":
        raise ValueError(f"unknown backend kind: {kind!r}")
    if prodlm_manifest_exists(repo_root=repo_root):
        return load_prodlm_backend(repo_root=repo_root)
    backend = load_artifact_backend(
        artifact_name,
        repo_root=str(repo_root) if repo_root is not None else None,
        device=device,
        torch_dtype=torch_dtype,
        required=True,
    )
    assert backend is not None
    return backend


def _read_prodlm_config(
    *,
    name: str,
    repo_root: str | Path | None,
) -> dict | None:
    config_path = model_artifact_dir(name, repo_root) / "config.json"
    if not config_path.exists():
        return None
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("kind") != PRODLM_KIND:
        raise ValueError(f"{config_path} is not a ProdLM manifest")
    backend = config.get("backend")
    if backend != "ollama":
        raise ValueError(f"unsupported ProdLM backend: {backend!r}")
    return config
