#!/usr/bin/env python
"""Download/register the small pretrained BaseLM artifact.

BaseLM is the fallback base model for Modules 13-15 when a student's
from-scratch StoryLM/TinyLLM is too weak for post-training exercises.
The default model is intentionally small enough for local Mac experiments.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from g2c.artifacts import (
    DEFAULT_BASELM_MODEL_ID,
    DEFAULT_BASELM_NAME,
    find_repo_root,
    model_artifact_dir,
    write_baselm_manifest,
)
from g2c.tokenizer import COURSE_SPECIAL_TOKENS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-id",
        default=DEFAULT_BASELM_MODEL_ID,
        help=f"Hugging Face model id. Default: {DEFAULT_BASELM_MODEL_ID}",
    )
    parser.add_argument(
        "--name",
        default=DEFAULT_BASELM_NAME,
        help=f"Course artifact name. Default: {DEFAULT_BASELM_NAME}",
    )
    parser.add_argument("--revision", default=None, help="Optional HF revision.")
    parser.add_argument(
        "--cache-dir",
        default="data/cache/baselm/huggingface",
        help="HF cache directory, relative to repo root unless absolute.",
    )
    parser.add_argument(
        "--torch-dtype",
        default="float32",
        choices=("float32", "float16", "bfloat16", "auto"),
        help="dtype used when notebooks load the model. Default: float32.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Forward trust_remote_code=True to HF loaders.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Only write the artifact manifest/tokenizer from an existing HF cache.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root. Default: auto-detect.",
    )
    args = parser.parse_args(argv)

    repo_root = find_repo_root(args.repo_root)
    cache_dir = _resolve_cache_dir(args.cache_dir, repo_root)
    artifact_dir = model_artifact_dir(args.name, repo_root)
    tokenizer_dir = artifact_dir / "hf_tokenizer"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
        from transformers import AutoConfig, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "BaseLM setup requires optional dependencies. Run:\n"
            "    uv pip install --python .venv/bin/python -e '.[baselm]'\n"
            "or install `transformers` and `safetensors` in the active venv."
        ) from exc

    if not args.skip_download:
        print(f"Downloading/checking {args.model_id} in {cache_dir}")
        snapshot_download(
            repo_id=args.model_id,
            revision=args.revision,
            cache_dir=str(cache_dir),
        )
    else:
        print(f"Skipping download; using existing HF cache at {cache_dir}")

    print("Preparing tokenizer with course special tokens")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        revision=args.revision,
        cache_dir=str(cache_dir),
        local_files_only=True,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": list(COURSE_SPECIAL_TOKENS)})
    tokenizer.save_pretrained(tokenizer_dir)

    config = AutoConfig.from_pretrained(
        args.model_id,
        revision=args.revision,
        cache_dir=str(cache_dir),
        local_files_only=True,
        trust_remote_code=args.trust_remote_code,
    )
    artifact_path = write_baselm_manifest(
        name=args.name,
        model_id=args.model_id,
        repo_root=repo_root,
        revision=args.revision,
        cache_dir=args.cache_dir,
        torch_dtype=args.torch_dtype,
        trust_remote_code=args.trust_remote_code,
        tokenizer_path="hf_tokenizer",
        notes=(
            "Small external pretrained base model for Modules 13-15. "
            f"HF architecture: {getattr(config, 'model_type', 'unknown')}."
        ),
    )
    print(f"BaseLM artifact ready: {artifact_path.relative_to(repo_root)}")
    print(f"Tokenizer ready: {tokenizer_dir.relative_to(repo_root)}")
    return 0


def _resolve_cache_dir(value: str, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


if __name__ == "__main__":
    raise SystemExit(main())
