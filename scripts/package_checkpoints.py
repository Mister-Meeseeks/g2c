"""Package reference checkpoints for the GitHub release that checkpoints.sh fetches.

For each artifact this script:

1. Verifies the artifact directory is complete (model artifacts need
   ``model.pt`` + ``config.json`` + ``manifest.json``; tokenizer artifacts
   need ``tokenizer.json`` + ``manifest.json``).
2. Verifies the manifest carries the provenance the course promises students:
   ``git_commit``, ``seed``, and -- for models -- the golden
   ``final_train_loss`` / ``final_val_loss`` numbers.
3. Stages a copy with a ``distribution`` block injected into the manifest
   (release tag, asset name, packaging timestamp), so a downloaded artifact is
   distinguishable from a self-trained one. The local artifact is untouched.
4. Writes ``<name>.tar.gz`` into the output directory and prints sha256 sums.

With ``--update-script`` the matching ``*_SHA256=""`` lines in
``checkpoints.sh`` are rewritten in place, so the fetch script and the
packaged assets cannot drift apart.

Publish flow (maintainers):

    .venv/bin/python scripts/package_checkpoints.py --update-script
    gh release create checkpoints-v1 data/cache/checkpoint-dist/*.tar.gz \
        --title "Reference checkpoints v1" \
        --notes "Reference course checkpoints; fetched by ./checkpoints.sh"
    # re-publishing after a retrain: gh release upload checkpoints-v1 \
    #     data/cache/checkpoint-dist/*.tar.gz --clobber

Uses only the standard library, so it runs without the project venv.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "cache" / "checkpoint-dist"
FETCH_SCRIPT = REPO_ROOT / "checkpoints.sh"
DEFAULT_RELEASE_TAG = "checkpoints-v1"

# name -> (artifact kind, checkpoints.sh checksum variable)
DEFAULT_TARGETS: dict[str, tuple[str, str]] = {
    "StoryLM-1M-base": ("model", "STORYLM_1M_SHA256"),
    "StoryLM-5M-base": ("model", "STORYLM_5M_SHA256"),
    "StoryLM-30M-base": ("model", "STORYLM_30M_SHA256"),
    "TinyLLM-30M-base": ("model", "TINYLLM_30M_SHA256"),
    "StoryTokenizer": ("tokenizer", "STORYTOKENIZER_SHA256"),
    "G2CTokenizer": ("tokenizer", "G2CTOKENIZER_SHA256"),
}

REQUIRED_FILES = {
    "model": ("model.pt", "config.json", "manifest.json"),
    "tokenizer": ("tokenizer.json", "manifest.json"),
}

ARTIFACT_ROOTS = {
    "model": REPO_ROOT / "artifacts" / "models",
    "tokenizer": REPO_ROOT / "artifacts" / "tokenizers",
}

# Manifest fields a published reference run must carry. Golden losses are the
# course's calibration numbers; a manifest without them is a broken promise.
REQUIRED_PROVENANCE = {
    "model": ("git_commit", "seed", "final_train_loss", "final_val_loss"),
    "tokenizer": (),
}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_artifact(name: str, kind: str) -> Path:
    root = ARTIFACT_ROOTS[kind] / name
    if not root.is_dir():
        sys.exit(f"fail: no {kind} artifact at {root}")
    for filename in REQUIRED_FILES[kind]:
        if not (root / filename).exists():
            sys.exit(f"fail: {root} is missing {filename}")

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    missing = [
        field
        for field in REQUIRED_PROVENANCE[kind]
        if manifest.get(field) is None
    ]
    if missing:
        sys.exit(
            f"fail: {name} manifest is missing provenance fields: {missing}. "
            "Retrain/resave the artifact so the published manifest carries "
            "its golden numbers."
        )

    if kind == "model":
        _check_tokenizer_covers_model(name, manifest)
    return root


def _check_tokenizer_covers_model(name: str, manifest: dict) -> None:
    """Fail if the referenced tokenizer can't produce the model's vocab.

    A tokenizer trained *larger* than the model is fine by design: consumers
    encode via ``encode_with_vocab_size(text, model.vocab_size)``, which
    truncates BPE merges to the model's prefix vocab. The broken direction is
    a tokenizer smaller than the model — ids the model can emit that the
    tokenizer cannot decode.
    """
    tokenizer_name = manifest.get("tokenizer_artifact")
    model_vocab = manifest.get("vocab_size")
    if not tokenizer_name or not isinstance(model_vocab, int):
        sys.exit(f"fail: {name} manifest lacks tokenizer_artifact/vocab_size")

    tokenizer_manifest_path = (
        ARTIFACT_ROOTS["tokenizer"] / tokenizer_name / "manifest.json"
    )
    if not tokenizer_manifest_path.exists():
        sys.exit(
            f"fail: {name} references tokenizer {tokenizer_name!r} but "
            f"{tokenizer_manifest_path} does not exist; package the tokenizer "
            "alongside the model"
        )
    tokenizer_manifest = json.loads(
        tokenizer_manifest_path.read_text(encoding="utf-8")
    )
    tokenizer_vocab = tokenizer_manifest.get("actual_vocab_size")
    if isinstance(tokenizer_vocab, int) and tokenizer_vocab < model_vocab:
        sys.exit(
            f"fail: {name} has vocab_size={model_vocab} but its tokenizer "
            f"{tokenizer_name!r} only has actual_vocab_size={tokenizer_vocab}; "
            "this pair cannot decode the model's full output space"
        )


def package_artifact(
    name: str,
    kind: str,
    release_tag: str,
    out_dir: Path,
    packaged_at: str,
) -> Path:
    """Stage a copy with distribution provenance and tar it up."""
    source = validate_artifact(name, kind)
    asset_name = f"{name}.tar.gz"
    out_path = out_dir / asset_name

    with tempfile.TemporaryDirectory(prefix="g2c-checkpoint-") as tmp:
        staged = Path(tmp) / name
        shutil.copytree(source, staged)

        manifest_path = staged / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["distribution"] = {
            "release": release_tag,
            "asset": asset_name,
            "packaged_at": packaged_at,
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

        # Drop rolling-checkpoint backups; the release ships final weights only.
        for backup in staged.rglob("*.bak"):
            backup.unlink()

        with tarfile.open(out_path, "w:gz") as tar:
            tar.add(staged, arcname=name)

    return out_path


def update_fetch_script(checksums: dict[str, str]) -> None:
    text = FETCH_SCRIPT.read_text(encoding="utf-8")
    for var, sha in checksums.items():
        pattern = re.compile(rf'^{var}="[0-9a-f]*"$', re.MULTILINE)
        if not pattern.search(text):
            sys.exit(f"fail: {FETCH_SCRIPT.name} has no {var}= line to update")
        text = pattern.sub(f'{var}="{sha}"', text)
    FETCH_SCRIPT.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "names",
        nargs="*",
        default=list(DEFAULT_TARGETS),
        help="artifact names to package (default: the full ladder)",
    )
    parser.add_argument("--release-tag", default=DEFAULT_RELEASE_TAG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--update-script",
        action="store_true",
        help=f"rewrite the *_SHA256 lines in {FETCH_SCRIPT.name}",
    )
    args = parser.parse_args()

    unknown = [name for name in args.names if name not in DEFAULT_TARGETS]
    if unknown:
        sys.exit(f"fail: unknown artifact names {unknown}; known: {list(DEFAULT_TARGETS)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    packaged_at = datetime.now(UTC).isoformat()

    checksums: dict[str, str] = {}
    for name in args.names:
        kind, sha_var = DEFAULT_TARGETS[name]
        out_path = package_artifact(
            name, kind, args.release_tag, args.out_dir, packaged_at
        )
        sha = sha256_of(out_path)
        checksums[sha_var] = sha
        size_mb = out_path.stat().st_size / (1 << 20)
        print(f"  ok {out_path.name}  {size_mb:7.1f}MB  sha256={sha}")

    if args.update_script:
        update_fetch_script(checksums)
        print(f"  ok rewrote {len(checksums)} *_SHA256 lines in {FETCH_SCRIPT.name}")

    print(
        f"\nPublish with:\n  gh release create {args.release_tag} "
        f"{args.out_dir}/*.tar.gz --title 'Reference checkpoints' "
        "--notes 'Fetched by ./checkpoints.sh'"
    )


if __name__ == "__main__":
    main()
