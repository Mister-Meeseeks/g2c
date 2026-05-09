"""BaseLM / Hugging Face model artifacts.

BaseLM is the small external pretrained base-model path for Modules 13-15.
Unlike the from-scratch ``TransformerLM`` artifacts, a Hugging Face artifact
keeps weights in their native ``save_pretrained`` format or in the local HF
cache. This module wraps that model so downstream notebooks can still call it
like a course language model: ``model(token_ids) -> (B, T, V)`` logits.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from g2c.nn import resolve_device
from g2c.tokenizer import COURSE_SPECIAL_TOKENS

from .models import LoadedModelArtifact, atomic_json_save, model_artifact_dir

DEFAULT_BASELM_NAME = "BaseLM"
DEFAULT_BASELM_MODEL_ID = "Qwen/Qwen3-0.6B-Base"
HUGGINGFACE_CAUSAL_LM_KIND = "huggingface_causal_lm"


@dataclass(frozen=True)
class HuggingFaceTokenizerAdapter:
    """Small adapter exposing the tokenizer API used by course notebooks."""

    inner: Any

    @property
    def special_to_id(self) -> dict[str, int]:
        """Return special-token string -> id for known special tokens."""
        values: dict[str, int] = {}
        for token in COURSE_SPECIAL_TOKENS:
            token_id = self.inner.convert_tokens_to_ids(token)
            if token_id is not None and token_id != self.inner.unk_token_id:
                values[token] = int(token_id)
        if self.inner.eos_token is not None and self.inner.eos_token_id is not None:
            values[self.inner.eos_token] = int(self.inner.eos_token_id)
        if self.inner.pad_token is not None and self.inner.pad_token_id is not None:
            values[self.inner.pad_token] = int(self.inner.pad_token_id)
        return values

    @property
    def eos_token_id(self) -> int | None:
        token_id = self.inner.eos_token_id
        return int(token_id) if token_id is not None else None

    @property
    def pad_token_id(self) -> int | None:
        token_id = self.inner.pad_token_id
        return int(token_id) if token_id is not None else None

    def encode(self, text: str) -> list[int]:
        """Encode text without adding model-family prompt/template tokens."""
        return [
            int(token_id)
            for token_id in self.inner.encode(text, add_special_tokens=False)
        ]

    def encode_with_vocab_size(self, text: str, vocab_size: int) -> list[int]:
        """Encode text and verify every ID fits the loaded model head."""
        ids = self.encode(text)
        too_large = [token_id for token_id in ids if token_id >= vocab_size]
        if too_large:
            raise ValueError(
                "prompt encoded to IDs outside the model vocabulary: "
                f"max={max(too_large)}, model.vocab_size={vocab_size}"
            )
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode IDs without dropping special tokens."""
        return self.inner.decode(
            [int(token_id) for token_id in ids],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    def save_pretrained(self, path: str | Path) -> None:
        self.inner.save_pretrained(path)


class HuggingFaceCausalLMAdapter(torch.nn.Module):
    """Wrap ``AutoModelForCausalLM`` behind the course model interface."""

    def __init__(self, inner: Any, *, max_seq_len: int | None = None) -> None:
        super().__init__()
        self.inner = inner
        config = inner.config
        self.vocab_size = int(getattr(config, "vocab_size", 0))
        if max_seq_len is None:
            max_seq_len = _infer_max_seq_len(config)
        self.max_seq_len = int(max_seq_len)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # The course sampling loop recomputes the cropped context each step and
        # does not pass `past_key_values` forward. Returning a KV cache would be
        # extra memory with no speed benefit here.
        return self.inner(input_ids=token_ids, use_cache=False).logits

    @property
    def device(self) -> torch.device:
        for parameter in self.inner.parameters():
            return parameter.device
        return torch.device("cpu")

    def to(self, device: str | torch.device | None = "auto") -> "HuggingFaceCausalLMAdapter":
        self.inner.to(resolve_device(device))
        return self

    def save_pretrained(self, path: str | Path) -> None:
        self.inner.save_pretrained(path)


def baselm_artifact_exists(
    name: str = DEFAULT_BASELM_NAME,
    *,
    repo_root: str | Path | None = None,
) -> bool:
    """Return whether ``name`` is an external Hugging Face artifact."""
    return huggingface_model_artifact_exists(name, repo_root=repo_root)


def huggingface_model_artifact_exists(
    name: str,
    *,
    repo_root: str | Path | None = None,
) -> bool:
    """Return whether ``artifacts/models/<name>`` is a HF causal-LM artifact."""
    root = model_artifact_dir(name, repo_root)
    manifest_path = root / "manifest.json"
    config_path = root / "config.json"
    if not manifest_path.exists() or not config_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return manifest.get("kind") == HUGGINGFACE_CAUSAL_LM_KIND


def write_baselm_manifest(
    *,
    name: str = DEFAULT_BASELM_NAME,
    model_id: str = DEFAULT_BASELM_MODEL_ID,
    repo_root: str | Path | None = None,
    revision: str | None = None,
    cache_dir: str | Path = "data/baselm/huggingface",
    torch_dtype: str = "float32",
    trust_remote_code: bool = False,
    tokenizer_path: str = "hf_tokenizer",
    notes: str = "",
) -> Path:
    """Create/update the lightweight BaseLM artifact metadata."""
    root = model_artifact_dir(name, repo_root)
    root.mkdir(parents=True, exist_ok=True)
    config = {
        "kind": HUGGINGFACE_CAUSAL_LM_KIND,
        "model_id": model_id,
        "revision": revision,
        "cache_dir": str(cache_dir),
        "torch_dtype": torch_dtype,
        "trust_remote_code": trust_remote_code,
        "tokenizer_path": tokenizer_path,
        "add_course_special_tokens": True,
        "special_tokens": list(COURSE_SPECIAL_TOKENS),
        "special_token_seed": 0,
    }
    atomic_json_save(config, root / "config.json")
    manifest = {
        "name": name,
        "display_name": name,
        "kind": HUGGINGFACE_CAUSAL_LM_KIND,
        "role": "BaseLM",
        "module": "external",
        "source": f"Hugging Face model {model_id}",
        "model_id": model_id,
        "revision": revision,
        "tokenizer_path": tokenizer_path,
        "tokenizer_artifact": None,
        "notes": notes,
    }
    atomic_json_save(manifest, root / "manifest.json")
    return root


def load_huggingface_model_artifact(
    name: str = DEFAULT_BASELM_NAME,
    *,
    repo_root: str | Path | None = None,
    device: str | torch.device | None = None,
    torch_dtype: str | torch.dtype | None = None,
) -> LoadedModelArtifact:
    """Load a HF causal-LM artifact and adapt it to the course interface."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "BaseLM loading requires the optional Hugging Face dependencies. "
            "Run `uv pip install --python .venv/bin/python -e '.[baselm]'` "
            "or rerun setup after installing the `baselm` extra."
        ) from exc

    root = model_artifact_dir(name, repo_root)
    if not huggingface_model_artifact_exists(name, repo_root=repo_root):
        raise FileNotFoundError(
            f"No Hugging Face model artifact named {name!r} at {root}. "
            "Run `./baselm.sh` to create BaseLM."
        )
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))

    model_source = _artifact_path_or_model_id(root, config, "model_path")
    tokenizer_source = _artifact_path_or_model_id(root, config, "tokenizer_path")
    if tokenizer_source is None:
        tokenizer_source = model_source
    if model_source is None:
        model_source = config["model_id"]

    cache_dir = _resolve_cache_dir(config.get("cache_dir"), repo_root)
    load_kwargs = {
        "revision": config.get("revision"),
        "cache_dir": str(cache_dir) if cache_dir is not None else None,
        "local_files_only": True,
        "trust_remote_code": bool(config.get("trust_remote_code", False)),
    }
    load_kwargs = {key: value for key, value in load_kwargs.items() if value is not None}
    hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **load_kwargs)
    if hf_tokenizer.pad_token is None and hf_tokenizer.eos_token is not None:
        hf_tokenizer.pad_token = hf_tokenizer.eos_token

    model_kwargs = dict(load_kwargs)
    dtype = _torch_dtype(
        torch_dtype if torch_dtype is not None else config.get("torch_dtype", "float32")
    )
    if dtype is not None:
        model_kwargs["dtype"] = dtype
    hf_model = AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs)
    if len(hf_tokenizer) > hf_model.get_input_embeddings().num_embeddings:
        seed = int(config.get("special_token_seed", 0))
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            hf_model.resize_token_embeddings(len(hf_tokenizer))
    wrapper = HuggingFaceCausalLMAdapter(
        hf_model,
        max_seq_len=_artifact_max_seq_len(config, hf_model.config),
    )
    if device is not None:
        wrapper.to(device)

    display_name = str(manifest.get("display_name") or name)
    rank = int(manifest.get("rank", 0))
    return LoadedModelArtifact(
        name=name,
        canonical_name=name,
        display_name=display_name,
        rank=rank,
        artifact_dir=root,
        model=wrapper,
        tokenizer=HuggingFaceTokenizerAdapter(hf_tokenizer),
        tokenizer_artifact=None,
        manifest=manifest,
        training_config={},
    )


def save_huggingface_model_artifact(
    name: str,
    *,
    model: Any,
    tokenizer: Any,
    base_artifact_name: str,
    source: str,
    training_config: dict[str, Any],
    history: dict[str, list] | None = None,
    repo_root: str | Path | None = None,
    module: str = "module-13",
    notes: str = "",
) -> Path:
    """Save a fine-tuned Hugging Face model in native format."""
    root = model_artifact_dir(name, repo_root)
    root.mkdir(parents=True, exist_ok=True)
    hf_model = getattr(model, "inner", model)
    hf_tokenizer = getattr(tokenizer, "inner", tokenizer)
    hf_model.save_pretrained(root / "hf_model")
    hf_tokenizer.save_pretrained(root / "hf_tokenizer")

    config = {
        "kind": HUGGINGFACE_CAUSAL_LM_KIND,
        "model_path": "hf_model",
        "tokenizer_path": "hf_tokenizer",
        "torch_dtype": "float32",
        "trust_remote_code": False,
        "base_artifact": base_artifact_name,
    }
    atomic_json_save(config, root / "config.json")
    final_train = (
        history["train_loss"][-1] if history and history.get("train_loss") else None
    )
    final_val = history["val_loss"][-1] if history and history.get("val_loss") else None
    steps_completed = history["step"][-1] + 1 if history and history.get("step") else 0
    manifest = {
        "name": name,
        "display_name": name,
        "kind": HUGGINGFACE_CAUSAL_LM_KIND,
        "role": "BaseLM-finetuned",
        "module": module,
        "source": source,
        "base_artifact": base_artifact_name,
        "tokenizer_artifact": None,
        "steps_completed": steps_completed,
        "final_train_loss": final_train,
        "final_val_loss": final_val,
        "notes": notes,
    }
    atomic_json_save(manifest, root / "manifest.json")
    return root


def _artifact_path_or_model_id(
    artifact_dir: Path,
    config: dict[str, Any],
    key: str,
) -> str | None:
    value = config.get(key)
    if not isinstance(value, str) or not value:
        return None
    path = artifact_dir / value
    return str(path) if path.exists() else value


def _resolve_cache_dir(value: Any, repo_root: str | Path | None) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    base = Path(repo_root) if repo_root is not None else Path.cwd()
    return base / path


def _torch_dtype(value: Any) -> torch.dtype | str | None:
    if value in (None, "none"):
        return None
    if isinstance(value, torch.dtype):
        return value
    if value == "auto":
        return "auto"
    if value in {"float32", "fp32"}:
        return torch.float32
    if value in {"float16", "fp16"}:
        return torch.float16
    if value in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise ValueError(f"unsupported torch_dtype for BaseLM: {value!r}")


def _infer_max_seq_len(config: Any) -> int:
    for attr in ("max_position_embeddings", "n_positions", "seq_length"):
        value = getattr(config, attr, None)
        if isinstance(value, int) and 0 < value < 1_000_000:
            return value
    return 2048


def _artifact_max_seq_len(config: dict[str, Any], hf_config: Any) -> int:
    value = config.get("max_seq_len")
    if isinstance(value, int) and value > 0:
        return value
    return _infer_max_seq_len(hf_config)
