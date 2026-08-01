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
import os
import pickle
import re
import subprocess
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from os import PathLike
from pathlib import Path
from typing import Any

import torch

from g2c.transformer import TransformerLM

from .paths import artifacts_root

CHECKPOINT_BACKUP_SUFFIX = ".bak"

_SIZE_TOKEN_RE = re.compile(r"^(\d+(?:\.\d+)?)([KMB])$")
BASE_STAGE = "base"
BASE_STAGE_SUFFIX = f"-{BASE_STAGE}"
_STAGE_NAMES = frozenset({BASE_STAGE, "SFT", "DPO"})
_SIZED_FAMILIES = frozenset({"ShakespeareLM", "StoryLM", "TinyLLM"})


@dataclass(frozen=True)
class ModelArtifactSpec:
    """One known reusable model tier for downstream notebooks."""

    canonical_name: str
    display_name: str
    rank: int
    aliases: tuple[str, ...] = ()

    @property
    def names(self) -> tuple[str, ...]:
        """Return base artifact names to search, new convention first."""
        return (self.base_artifact_name, self.canonical_name, *self.aliases)

    @property
    def stage_root_names(self) -> tuple[str, ...]:
        """Return names used as roots for SFT/DPO artifacts."""
        return (self.canonical_name, *self.aliases)

    @property
    def base_artifact_name(self) -> str:
        """Return the canonical on-disk name for the base-stage artifact."""
        return f"{self.canonical_name}{BASE_STAGE_SUFFIX}"


DEFAULT_MODEL_ARTIFACT_SPECS: tuple[ModelArtifactSpec, ...] = (
    ModelArtifactSpec(
        canonical_name="ShakespeareLM-1M",
        display_name="ShakespeareLM",
        rank=10,
        aliases=("ShakespeareLM",),
    ),
    ModelArtifactSpec(
        canonical_name="StoryLM-1M",
        display_name="StoryLM 1M",
        rank=15,
    ),
    ModelArtifactSpec(
        canonical_name="StoryLM-5M",
        display_name="StoryLM 5M",
        rank=20,
        aliases=("StoryLM-Small",),
    ),
    ModelArtifactSpec(
        canonical_name="StoryLM-10M",
        display_name="StoryLM 10M",
        rank=30,
    ),
    ModelArtifactSpec(
        canonical_name="StoryLM-30M",
        display_name="StoryLM 30M",
        rank=35,
        aliases=("StoryLM",),
    ),
    ModelArtifactSpec(
        canonical_name="TinyLLM-30M",
        display_name="TinyLLM 30M",
        rank=40,
        aliases=("TinyLLM",),
    ),
    ModelArtifactSpec(
        canonical_name="TinyLLM-100M",
        display_name="TinyLLM 100M",
        rank=50,
    ),
)


@dataclass(frozen=True)
class AvailableModelArtifact:
    """Resolved model artifact candidate on disk."""

    name: str
    canonical_name: str
    display_name: str
    rank: int
    artifact_dir: Path


@dataclass(frozen=True)
class LoadedModelArtifact:
    """Model artifact plus its tokenizer, ready for downstream notebooks."""

    name: str
    canonical_name: str
    display_name: str
    rank: int
    artifact_dir: Path
    model: Any
    tokenizer: Any
    tokenizer_artifact: Any
    manifest: dict[str, Any]
    training_config: dict[str, Any]


def model_artifact_dir(name: str, repo_root: str | Path | None = None) -> Path:
    if not name:
        raise ValueError("model artifact name must be non-empty")
    return artifacts_root(repo_root) / "models" / name


def model_artifact_exists(name: str, repo_root: str | Path | None = None) -> bool:
    root = model_artifact_dir(name, repo_root)
    if (
        (root / "model.pt").exists()
        and (root / "config.json").exists()
        and (root / "manifest.json").exists()
    ):
        return True
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return (
        manifest.get("kind") == "huggingface_causal_lm"
        and (root / "config.json").exists()
    )


def available_model_artifacts(
    *,
    repo_root: str | Path | None = None,
    specs: tuple[ModelArtifactSpec, ...] = DEFAULT_MODEL_ARTIFACT_SPECS,
) -> list[AvailableModelArtifact]:
    """Return known reusable model artifacts present on disk.

    The returned list is sorted from smallest/earliest course artifact to the
    strongest downstream candidate. Aliases let older notebooks keep working:
    for example ``StoryLM-Small`` resolves as the ``StoryLM-5M`` tier, and
    ``TinyLLM`` resolves as the ``TinyLLM-30M`` tier.
    """
    found: list[AvailableModelArtifact] = []
    for spec in specs:
        for name in spec.names:
            if model_artifact_exists(name, repo_root=repo_root):
                found.append(
                    AvailableModelArtifact(
                        name=name,
                        canonical_name=spec.canonical_name,
                        display_name=spec.display_name,
                        rank=spec.rank,
                        artifact_dir=model_artifact_dir(name, repo_root),
                    )
                )
                break
    return sorted(found, key=lambda artifact: artifact.rank)


def best_model_artifact(
    *,
    repo_root: str | Path | None = None,
    specs: tuple[ModelArtifactSpec, ...] = DEFAULT_MODEL_ARTIFACT_SPECS,
) -> AvailableModelArtifact | None:
    """Return the strongest known reusable model artifact, if any exist."""
    available = available_model_artifacts(repo_root=repo_root, specs=specs)
    return available[-1] if available else None


def available_model_artifacts_with_suffix(
    suffix: str,
    *,
    repo_root: str | Path | None = None,
    specs: tuple[ModelArtifactSpec, ...] = DEFAULT_MODEL_ARTIFACT_SPECS,
    extra_base_names: tuple[str, ...] = ("BaseLM",),
    extra_rank_start: int = 100,
) -> list[AvailableModelArtifact]:
    """Return known model artifacts with a post-training suffix.

    Modules 13-15 save derived artifacts by appending suffixes to the base
    artifact name, for example ``TinyLLM-30M-SFT`` or ``BaseLM-SFT``. This
    helper mirrors ``available_model_artifacts`` while keeping the base-model
    ranking, so downstream notebooks can load the strongest completed stage
    without hard-coding one track.
    """
    if not suffix:
        raise ValueError("suffix must be non-empty")

    found: list[AvailableModelArtifact] = []
    for spec in specs:
        for base_name in spec.stage_root_names:
            name = f"{base_name}{suffix}"
            if model_artifact_exists(name, repo_root=repo_root):
                found.append(
                    AvailableModelArtifact(
                        name=name,
                        canonical_name=f"{spec.canonical_name}{suffix}",
                        display_name=f"{spec.display_name}{suffix}",
                        rank=spec.rank,
                        artifact_dir=model_artifact_dir(name, repo_root),
                    )
                )
                break

    for offset, base_name in enumerate(extra_base_names):
        name = f"{base_name}{suffix}"
        if model_artifact_exists(name, repo_root=repo_root):
            found.append(
                AvailableModelArtifact(
                    name=name,
                    canonical_name=name,
                    display_name=name,
                    rank=extra_rank_start + offset,
                    artifact_dir=model_artifact_dir(name, repo_root),
                )
            )

    return sorted(found, key=lambda artifact: artifact.rank)


def best_model_artifact_with_suffix(
    suffix: str,
    *,
    repo_root: str | Path | None = None,
    specs: tuple[ModelArtifactSpec, ...] = DEFAULT_MODEL_ARTIFACT_SPECS,
    extra_base_names: tuple[str, ...] = ("BaseLM",),
    extra_rank_start: int = 100,
) -> AvailableModelArtifact | None:
    """Return the strongest known model artifact with ``suffix``."""
    available = available_model_artifacts_with_suffix(
        suffix,
        repo_root=repo_root,
        specs=specs,
        extra_base_names=extra_base_names,
        extra_rank_start=extra_rank_start,
    )
    return available[-1] if available else None


def artifact_spec_for_name(
    name: str,
    *,
    specs: tuple[ModelArtifactSpec, ...] = DEFAULT_MODEL_ARTIFACT_SPECS,
) -> ModelArtifactSpec | None:
    """Return the known artifact tier for ``name`` or one of its aliases."""
    for spec in specs:
        if name in spec.names:
            return spec
    return None


def parse_artifact_name(name: str) -> tuple[str, str | None, str | None]:
    """Split an artifact name into ``(family, size, stage)``.

    ``size`` is the size token (e.g. ``"5M"``, ``"30M"``) if present, else
    ``None``. ``stage`` is ``"base"``, ``"SFT"``, or ``"DPO"`` if present,
    else ``None``. Unrecognized components (legacy qualifiers like ``"Small"``)
    are ignored.
    """
    parts = name.split("-")
    family = parts[0]
    size: str | None = None
    stage: str | None = None
    for part in parts[1:]:
        if part in _STAGE_NAMES:
            stage = part
        elif _SIZE_TOKEN_RE.match(part):
            size = part
    return family, size, stage


def _size_token_to_int(token: str) -> int:
    match = _SIZE_TOKEN_RE.match(token)
    if not match:
        raise ValueError(f"not a size token: {token!r}")
    val, unit = float(match.group(1)), match.group(2)
    mult = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}[unit]
    return int(val * mult)


def is_base_stage_artifact_name(name: str) -> bool:
    """Return whether ``name`` is explicitly a base-stage artifact."""
    return name.endswith(BASE_STAGE_SUFFIX)


def stage_root_name(name: str) -> str:
    """Return the stage root used before ``-base``, ``-SFT``, or ``-DPO``."""
    if name.endswith(BASE_STAGE_SUFFIX):
        return name[: -len(BASE_STAGE_SUFFIX)]
    for stage in ("SFT", "DPO"):
        suffix = f"-{stage}"
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def canonical_base_artifact_name(
    name: str,
    *,
    specs: tuple[ModelArtifactSpec, ...] = DEFAULT_MODEL_ARTIFACT_SPECS,
) -> str:
    """Return the canonical save name for a base-stage artifact.

    SFT/DPO names and already-explicit ``*-base`` names are returned unchanged.
    Known course tiers are normalized to ``<tier>-base``. Unknown unsuffixed
    names also get ``-base`` so new base artifacts follow the same convention.
    """
    if not name:
        raise ValueError("artifact name must be non-empty")
    _, _, stage = parse_artifact_name(name)
    if stage is not None:
        return name
    spec = artifact_spec_for_name(name, specs=specs)
    if spec is not None:
        return spec.base_artifact_name
    return f"{name}{BASE_STAGE_SUFFIX}"


def resolve_artifact_name(
    name: str,
    *,
    repo_root: str | Path | None = None,
    specs: tuple[ModelArtifactSpec, ...] = DEFAULT_MODEL_ARTIFACT_SPECS,
) -> str:
    """Resolve ``name`` to a canonical on-disk artifact name.

    Resolution rules, applied in order:

    1. If ``name`` matches a known base spec or alias, prefer the explicit
       ``*-base`` artifact and fall back to the legacy unsuffixed artifact.
    2. If ``name`` exists on disk as-is, return it unchanged.
    3. If ``name`` is a family-level alias -- a sized family
       (``ShakespeareLM`` / ``StoryLM`` / ``TinyLLM``) with the size token
       omitted -- return the largest available ``<family>-<size>-base`` or
       legacy ``<family>-<size>`` artifact on disk at the same stage.

    Raises ``FileNotFoundError`` if no candidate exists.
    """
    if not name:
        raise ValueError("artifact name must be non-empty")

    spec = artifact_spec_for_name(name, specs=specs)
    if spec is not None:
        if model_artifact_exists(spec.base_artifact_name, repo_root=repo_root):
            return spec.base_artifact_name
        for legacy_name in (spec.canonical_name, *spec.aliases):
            if model_artifact_exists(legacy_name, repo_root=repo_root):
                return legacy_name

    if name == "BaseLM":
        if model_artifact_exists("BaseLM-base", repo_root=repo_root):
            return "BaseLM-base"

    if model_artifact_exists(name, repo_root=repo_root):
        return name

    if name == "BaseLM-base" and model_artifact_exists("BaseLM", repo_root=repo_root):
        return "BaseLM"

    family, size, stage = parse_artifact_name(name)
    if family in _SIZED_FAMILIES and size is None:
        models_dir = artifacts_root(repo_root) / "models"
        candidates: list[tuple[int, int, str]] = []
        if models_dir.exists():
            for entry in models_dir.iterdir():
                if not entry.is_dir():
                    continue
                ef, es, est = parse_artifact_name(entry.name)
                if ef != family or es is None:
                    continue
                if stage is None:
                    if est not in (BASE_STAGE, None):
                        continue
                    stage_priority = 1 if est == BASE_STAGE else 0
                elif est != stage:
                    continue
                else:
                    stage_priority = 0
                if model_artifact_exists(entry.name, repo_root=repo_root):
                    candidates.append((_size_token_to_int(es), stage_priority, entry.name))
        if candidates:
            candidates.sort(reverse=True)
            return candidates[0][2]

    raise FileNotFoundError(
        f"no model artifact resolves to {name!r}. Checked literal name, known "
        "aliases, and family-level alias scan."
    )


def load_model_artifact_with_tokenizer(
    name: str,
    *,
    repo_root: str | Path | None = None,
    device: str | torch.device | None = None,
    torch_dtype: str | torch.dtype | None = None,
    specs: tuple[ModelArtifactSpec, ...] = DEFAULT_MODEL_ARTIFACT_SPECS,
) -> LoadedModelArtifact:
    """Load one named model artifact plus its referenced tokenizer artifact.

    The ``name`` argument may be a canonical artifact name, a spec alias
    (e.g. ``"StoryLM-Small"``), or a family-level alias with the size token
    omitted (e.g. ``"StoryLM"``, ``"StoryLM-SFT"``); see
    :func:`resolve_artifact_name`.
    """
    from .baselm import huggingface_model_artifact_exists, load_huggingface_model_artifact

    try:
        name = resolve_artifact_name(name, repo_root=repo_root, specs=specs)
    except FileNotFoundError:
        # Fall through; the existence checks below will raise the original
        # not-found error from the underlying loader if the name truly is
        # missing. This preserves error messages for direct callers that
        # already handled non-existence.
        pass

    if huggingface_model_artifact_exists(name, repo_root=repo_root):
        return load_huggingface_model_artifact(
            name,
            repo_root=repo_root,
            device=device,
            torch_dtype=torch_dtype,
        )

    loaded = load_model_artifact(
        name,
        repo_root=repo_root,
        map_location="cpu",
    )
    model = loaded["model"]
    if device is not None:
        model.to(device)

    from .tokenizers import load_tokenizer_artifact

    tokenizer_name = loaded["manifest"].get("tokenizer_artifact")
    if not isinstance(tokenizer_name, str) or not tokenizer_name:
        raise ValueError(
            f"Model artifact {name!r} has no tokenizer_artifact in manifest"
        )
    tokenizer_artifact = load_tokenizer_artifact(tokenizer_name, repo_root=repo_root)

    spec = artifact_spec_for_name(name, specs=specs)
    canonical_name = spec.canonical_name if spec is not None else name
    display_name = spec.display_name if spec is not None else name
    rank = spec.rank if spec is not None else 0

    return LoadedModelArtifact(
        name=name,
        canonical_name=canonical_name,
        display_name=display_name,
        rank=rank,
        artifact_dir=model_artifact_dir(name, repo_root),
        model=model,
        tokenizer=tokenizer_artifact.tokenizer,
        tokenizer_artifact=tokenizer_artifact,
        manifest=loaded["manifest"],
        training_config=loaded["training_config"],
    )


def load_best_model_artifact(
    *,
    repo_root: str | Path | None = None,
    device: str | torch.device | None = None,
    specs: tuple[ModelArtifactSpec, ...] = DEFAULT_MODEL_ARTIFACT_SPECS,
    required: bool = True,
) -> LoadedModelArtifact | None:
    """Load the strongest available reusable model plus its tokenizer.

    Downstream notebooks use this to avoid hard-coding one Module 10 artifact.
    The search order is:

    ``ShakespeareLM-1M-base -> StoryLM-1M-base -> StoryLM-5M-base ->
    StoryLM-10M-base -> StoryLM-30M-base -> TinyLLM-30M-base ->
    TinyLLM-100M-base``.

    Current legacy names such as ``StoryLM-30M`` plus aliases such as
    ``StoryLM-Small``, ``StoryLM``, and ``TinyLLM`` are treated as fallbacks.
    """
    candidate = best_model_artifact(repo_root=repo_root, specs=specs)
    if candidate is None:
        if not required:
            return None
        expected = ", ".join(spec.base_artifact_name for spec in specs)
        raise FileNotFoundError(
            "No reusable model artifact found under artifacts/models/. "
            f"Expected one of: {expected}. Run Module 10 and save at least one model."
        )

    return load_model_artifact_with_tokenizer(
        candidate.name,
        repo_root=repo_root,
        device=device,
        specs=specs,
    )


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


def checkpoint_backup_path(path: str | PathLike[str]) -> Path:
    """Return the previous-good backup path for a rolling checkpoint."""
    checkpoint_path = Path(path)
    return checkpoint_path.with_name(checkpoint_path.name + CHECKPOINT_BACKUP_SUFFIX)


def _fsync_dir(directory: Path) -> None:
    """Force a directory entry change (rename, unlink) out to disk.

    Renaming is atomic with respect to ordering, but the rename itself lives
    in the directory's metadata and is not durable until the directory is
    synced. Best-effort: filesystems that reject `fsync` on a directory
    handle simply skip it.
    """
    try:
        dir_fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    except OSError:
        pass
    finally:
        os.close(dir_fd)


def atomic_torch_save(
    state: Any,
    path: str | PathLike[str],
    *,
    keep_backup: bool = True,
) -> Path:
    """Save a PyTorch payload without ever leaving `path` absent or partial.

    `torch.save` writes zip archives. If a notebook is interrupted while that
    archive is being written directly to the final path, later `torch.load`
    can fail with missing internal `data/N` records. This helper writes to a
    same-directory temporary file first, fsyncs it, and only then swaps it into
    place with a single atomic rename. The previous final file is kept as
    `<path>.bak`, so loaders have a previous-good checkpoint to fall back to.

    The backup is created by *hard-linking* the current final file rather than
    moving it. Moving would leave a window in which `path` does not exist at
    all -- and an interrupt landing in that window destroys the only complete
    checkpoint, since the temporary file has not been renamed into place yet.
    Adding a second name for the same inode closes that window: `path` is a
    loadable checkpoint at every instant. `os.replace` below repoints the
    directory entry rather than writing through the inode, so the linked
    backup is unaffected by the swap.
    """
    final_path = Path(path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_prefix = f".{final_path.name}.{os.getpid()}."
    tmp_path = final_path.with_name(f"{tmp_prefix}{uuid.uuid4().hex}.tmp")
    backup_path = checkpoint_backup_path(final_path)

    # Sweep temporaries this process orphaned in an earlier interrupted save.
    # Scoped to our own pid so a concurrent writer's tmp is never touched.
    for stale in final_path.parent.glob(f"{tmp_prefix}*.tmp"):
        stale.unlink(missing_ok=True)

    # Write the new payload and force it to disk. Only an *incomplete* write is
    # cleaned up here -- once `torch.save` returns, `tmp_path` holds a complete
    # checkpoint and is never deleted on the way out, even on KeyboardInterrupt.
    try:
        with open(tmp_path, "wb") as handle:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise

    if keep_backup and final_path.exists():
        backup_path.unlink(missing_ok=True)
        try:
            os.link(final_path, backup_path)
        except OSError:
            # Filesystem without hard-link support. Fall back to a move, which
            # reopens the absent-file window -- which is why both
            # `load_torch_checkpoint` and the resume gate consult the backup.
            os.replace(final_path, backup_path)

    os.replace(tmp_path, final_path)
    _fsync_dir(final_path.parent)
    return final_path


def load_torch_checkpoint(
    path: str | PathLike[str],
    *,
    map_location: str | torch.device = "cpu",
    fallback_to_backup: bool = True,
) -> Any:
    """Load a PyTorch checkpoint, optionally falling back to `<path>.bak`."""
    checkpoint_path = Path(path)
    try:
        return torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=False,
        )
    except (
        FileNotFoundError,
        RuntimeError,
        OSError,
        EOFError,
        ValueError,
        pickle.UnpicklingError,
    ):
        backup_path = checkpoint_backup_path(checkpoint_path)
        if not fallback_to_backup or not backup_path.exists():
            raise
        try:
            return torch.load(
                backup_path,
                map_location=map_location,
                weights_only=False,
            )
        except Exception as backup_exc:
            raise RuntimeError(
                f"Could not load checkpoint {checkpoint_path} or backup {backup_path}"
            ) from backup_exc


def atomic_json_save(payload: dict[str, Any], path: str | PathLike[str]) -> Path:
    """Atomically write a small JSON artifact file."""
    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = json_path.with_name(
        f".{json_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        tmp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp_path, json_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return json_path


def save_training_checkpoint(
    trainer,
    path: str | PathLike[str],
    *,
    history: dict[str, list] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Save a rolling Trainer checkpoint via the atomic artifact writer."""
    return atomic_torch_save(
        trainer.checkpoint_state(history=history, extra=extra),
        path,
        keep_backup=True,
    )


def load_training_checkpoint(
    path: str | PathLike[str],
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a rolling Trainer checkpoint, falling back to `<path>.bak`."""
    state = load_torch_checkpoint(path, map_location=map_location)
    if not isinstance(state, dict):
        raise ValueError("training checkpoint must contain a dict")
    return state


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
        name: artifact name (e.g. ``"ShakespeareLM-1M"``). Base-stage names
            are saved under the explicit ``*-base`` convention.
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
    name = canonical_base_artifact_name(name)
    root = model_artifact_dir(name, repo_root)
    root.mkdir(parents=True, exist_ok=True)

    atomic_torch_save(
        {
            "vocab_size": model.vocab_size,
            "model_config": dict(model_config),
            "params": [p.detach().cpu() for p in model.parameters()],
        },
        root / "model.pt",
        keep_backup=False,
    )

    config = {
        "model_config": dict(model_config),
        "training_config": dict(training_config),
    }
    atomic_json_save(config, root / "config.json")

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
        "created_at": datetime.now(UTC).isoformat(),
        "steps_completed": steps_completed,
        "final_train_loss": final_train,
        "final_val_loss": final_val,
        "notes": notes,
    }
    atomic_json_save(manifest, root / "manifest.json")

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

    state = load_torch_checkpoint(
        root / "model.pt",
        map_location=map_location,
        fallback_to_backup=False,
    )
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


def load_run_state(
    checkpoint_path: str | PathLike[str],
) -> tuple[TransformerLM, dict]:
    """Reconstruct ``(model, history)`` from a rolling training checkpoint.

    The checkpoint must include ``extra["model_config"]`` and
    ``extra["vocab_size"]``. If the main checkpoint file was interrupted during
    a previous direct write or otherwise corrupted, this loader falls back to
    ``<checkpoint_path>.bak`` when present.
    """
    path = Path(checkpoint_path)
    state = load_training_checkpoint(path)
    extra = state.get("extra") or {}
    model_config = extra.get("model_config")
    vocab_size = extra.get("vocab_size")
    if model_config is None or vocab_size is None:
        raise ValueError(
            f"Checkpoint at {path} is missing model_config / vocab_size in extra; "
            "training cell must pass these via checkpoint_extra."
        )

    model = TransformerLM(vocab_size=vocab_size, **model_config)
    saved = state["model_params"]
    target = list(model.parameters())
    if len(saved) != len(target):
        raise ValueError(
            f"Checkpoint has {len(saved)} params, model has {len(target)}"
        )
    with torch.no_grad():
        for p, s in zip(target, saved, strict=True):
            p.copy_(s.to(device=p.device, dtype=p.dtype))

    history = state.get("history")
    if history is None:
        raise ValueError(f"Checkpoint at {path} has no history")
    return model, history
