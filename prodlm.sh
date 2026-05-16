#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="llama3.2:3b"
EMBED_MODEL_ID="nomic-embed-text"
BASE_URL="http://localhost:11434"
TIMEOUT="120"
PULL=1

usage() {
  cat <<'EOF'
Usage: ./prodlm.sh [options]

Configure the course ProdLM backend for Modules 16-20.

Options:
  --model-id MODEL        Ollama generation model tag (default: llama3.2:3b)
  --embed-model-id MODEL  Ollama embedding model tag for RAG (default: nomic-embed-text)
  --base-url URL          Ollama server URL (default: http://localhost:11434)
  --timeout SECONDS       Backend timeout in seconds (default: 120)
  --no-pull               Do not run `ollama pull`; only write the manifest
  --pull                  Run `ollama pull` before writing the manifest (default)
  -h, --help              Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-id)
      MODEL_ID="${2:?missing value for --model-id}"
      shift 2
      ;;
    --embed-model-id)
      EMBED_MODEL_ID="${2:?missing value for --embed-model-id}"
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
  if ! ollama list >/dev/null 2>&1; then
    cat >&2 <<'EOF'
Ollama is installed, but the Ollama server is not reachable.

Start Ollama, then rerun this command:
  - macOS app: open Ollama from Applications
  - CLI:       ollama serve

Modules 16-20 need the Ollama server running before the notebook opens.
EOF
    exit 1
  fi
  echo "==> Pulling ProdLM model with Ollama: ${MODEL_ID}"
  ollama pull "$MODEL_ID"
  echo "==> Pulling RAG embedding model with Ollama: ${EMBED_MODEL_ID}"
  ollama pull "$EMBED_MODEL_ID"
fi

PYTHON="${PYTHON:-.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python3"
fi

"$PYTHON" - "$MODEL_ID" "$BASE_URL" "$TIMEOUT" "$EMBED_MODEL_ID" <<'PY'
from pathlib import Path
import sys

from g2c.inference import write_prodlm_manifest

model_id, base_url, timeout, embed_model_id = (
    sys.argv[1],
    sys.argv[2],
    float(sys.argv[3]),
    sys.argv[4],
)
root = write_prodlm_manifest(
    model_id=model_id,
    base_url=base_url,
    timeout=timeout,
    repo_root=Path.cwd(),
    notes="Configured by prodlm.sh",
)
print(f" ok wrote {root / 'manifest.json'}")
print(f"ProdLM backend: ollama model {model_id} at {base_url}")
print(f"RAG embedder: ollama model {embed_model_id} at {base_url}")
PY
