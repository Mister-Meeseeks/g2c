#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="llama3.2:3b"
BASE_URL="http://localhost:11434"
TIMEOUT="120"
PULL=1

usage() {
  cat <<'EOF'
Usage: ./prodlm.sh [options]

Configure the course ProdLM backend for Modules 16-20.

Options:
  --model-id MODEL    Ollama model tag to use (default: llama3.2:3b)
  --base-url URL      Ollama server URL (default: http://localhost:11434)
  --timeout SECONDS   Backend timeout in seconds (default: 120)
  --no-pull           Do not run `ollama pull`; only write the manifest
  --pull              Run `ollama pull` before writing the manifest (default)
  -h, --help          Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-id)
      MODEL_ID="${2:?missing value for --model-id}"
      shift 2
      ;;
    --base-url)
      BASE_URL="${2:?missing value for --base-url}"
      shift 2
      ;;
    --timeout)
      TIMEOUT="${2:?missing value for --timeout}"
      shift 2
      ;;
    --no-pull)
      PULL=0
      shift
      ;;
    --pull)
      PULL=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$PULL" -eq 1 ]]; then
  if ! command -v ollama >/dev/null 2>&1; then
    echo "Ollama is not installed or not on PATH. Install it from https://ollama.com/download or rerun with --no-pull." >&2
    exit 1
  fi
  echo "==> Pulling ProdLM model with Ollama: ${MODEL_ID}"
  ollama pull "$MODEL_ID"
fi

PYTHON="${PYTHON:-.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python3"
fi

"$PYTHON" - "$MODEL_ID" "$BASE_URL" "$TIMEOUT" <<'PY'
from pathlib import Path
import sys

from g2c.inference import write_prodlm_manifest

model_id, base_url, timeout = sys.argv[1], sys.argv[2], float(sys.argv[3])
root = write_prodlm_manifest(
    model_id=model_id,
    base_url=base_url,
    timeout=timeout,
    repo_root=Path.cwd(),
    notes="Configured by prodlm.sh",
)
print(f" ok wrote {root / 'manifest.json'}")
print(f"ProdLM backend: ollama model {model_id} at {base_url}")
PY
