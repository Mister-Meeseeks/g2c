#!/usr/bin/env bash
# Bootstrap the From Gradients to ChatGPT course environment.
#
# Strategy:
#   - SYSTEM-LEVEL tools (python, uv, curl, ollama) are CHECKED, not installed.
#     Missing python/uv/curl print an install hint and exit. Missing ollama
#     only warns (it isn't needed until Module 16) — except with --full,
#     which runs ./prodlm.sh and therefore requires ollama up front.
#   - PROJECT-LEVEL tools (torch, pytest, ruff, jupyter, Hugging Face helpers,
#     and the g2c package itself) are installed into a project-local venv at
#     ./.venv via uv.
#
# Idempotent: safe to re-run.

set -euo pipefail

RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

info() { printf "${BLUE}==>${NC} %s\n" "$*"; }
ok()   { printf "${GREEN} ok${NC} %s\n" "$*"; }
warn() { printf "${YELLOW}warn${NC} %s\n" "$*"; }
fail() { printf "${RED}fail${NC} %s\n" "$*" >&2; exit 1; }

usage() {
    cat <<'EOF'
Usage:
    ./setup.sh [--full] [-h|--help]

Default:
    Bootstrap the venv, install Python dependencies, download TinyShakespeare,
    build the Shakespeare tokenizer artifact, and run the smoke test. Larger
    optional assets are left to be installed on demand.

--full:
    After the default setup, also run:
        ./datasets.sh    # GloVe + full TinyStories + full G2C corpus (Modules 05, 10)
        ./baselm.sh      # pretrained BaseLM artifact (Modules 13-16)
        ./prodlm.sh      # ProdLM Ollama runtime (Modules 16-20)

-h, --help:
    Show this help and exit.
EOF
}

MODE="default"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --full)
            MODE="full"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf "Unknown option: %s\n\n" "$1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

# ---- 1. Python 3.11+ ---------------------------------------------------------
info "Checking Python 3.11+"
if ! command -v python3 >/dev/null 2>&1; then
    fail "python3 not found.
       Install:  brew install python@3.11
       Or use pyenv:  https://github.com/pyenv/pyenv"
fi

PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')
if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then
    fail "Python 3.11+ required (found $PY_VERSION).
       Install:  brew install python@3.11"
fi
ok "Python $PY_VERSION"

# ---- 2. uv -------------------------------------------------------------------
info "Checking uv"
if ! command -v uv >/dev/null 2>&1; then
    fail "uv not found.
       Install:  brew install uv
       Or:       curl -LsSf https://astral.sh/uv/install.sh | sh"
fi
ok "uv $(uv --version | awk '{print $2}')"

# ---- 3. Ollama (optional until Module 16) ------------------------------------
info "Checking Ollama (optional; first needed in Module 16)"
if command -v ollama >/dev/null 2>&1; then
    ok "ollama $(ollama --version 2>/dev/null | head -n 1 | sed 's/^ollama version //')"
elif [[ "$MODE" == "full" ]]; then
    fail "ollama not found, and --full runs ./prodlm.sh, which requires it.
       Install:
         - Download the macOS app:
               https://ollama.com/download
         - OR:
               brew install ollama"
else
    warn "ollama not found. Modules 00-15 don't need it; ./prodlm.sh checks again
       before the ProdLM modules (16-20). Install when you get there:
         - Download the macOS app:
               https://ollama.com/download
         - OR:
               brew install ollama"
fi

# ---- 4. venv -----------------------------------------------------------------
PROJECT_ROOT=$(pwd -P)
if [[ ! -d .venv ]]; then
    info "Creating venv at .venv"
    uv venv --python 3.11 --prompt g2c .venv
    ok "venv created"
else
    ok "venv exists"
    REFRESH_VENV=0
    if [[ -f .venv/pyvenv.cfg ]] && ! grep -qx "prompt = g2c" .venv/pyvenv.cfg; then
        warn "venv prompt is stale; refreshing activation metadata"
        REFRESH_VENV=1
    fi
    if [[ -f .venv/bin/activate ]] && ! grep -q "VIRTUAL_ENV='$PROJECT_ROOT/.venv'" .venv/bin/activate; then
        warn "venv activation path is stale; refreshing activation metadata"
        REFRESH_VENV=1
    fi
    if [[ "$REFRESH_VENV" == "1" ]]; then
        uv venv --python 3.11 --prompt g2c --allow-existing .venv
        ok "venv metadata refreshed"
    fi
fi

REINSTALL_DEV=0
if [[ -d .venv/bin ]]; then
    for launcher in .venv/bin/*; do
        [[ -f "$launcher" ]] || continue
        IFS= read -r SHEBANG < "$launcher" || true
        case "$SHEBANG" in
            "#!"*".venv/bin/python"*)
                case "$SHEBANG" in
                    "#!$PROJECT_ROOT/.venv/bin/python"*) ;;
                    *)
                        warn "$(basename "$launcher") points outside this checkout; reinstalling dev deps"
                        REINSTALL_DEV=1
                        break
                        ;;
                esac
                ;;
        esac
    done
fi

# ---- 5. Project deps ---------------------------------------------------------
info "Installing g2c (editable) + dev/BaseLM deps"
if [[ "$REINSTALL_DEV" == "1" ]]; then
    uv pip install -e ".[dev,baselm]" --reinstall --quiet
else
    uv pip install -e ".[dev,baselm]" --quiet
fi
ok "deps installed"

# ---- 6. Small data assets ----------------------------------------------------
mkdir -p data

if ! command -v curl >/dev/null 2>&1; then
    fail "curl not found; needed to download TinyShakespeare"
fi

TINY_SHAKESPEARE_FILE="data/datasets/tinyshakespeare.txt"
TINY_SHAKESPEARE_URL="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/datasets/tinyshakespeare/input.txt"

info "Checking TinyShakespeare corpus"
if [[ -f "$TINY_SHAKESPEARE_FILE" ]]; then
    ok "$TINY_SHAKESPEARE_FILE exists"
else
    warn "$TINY_SHAKESPEARE_FILE not found; downloading TinyShakespeare"
    curl --fail --location --output "$TINY_SHAKESPEARE_FILE" "$TINY_SHAKESPEARE_URL"
    ok "$TINY_SHAKESPEARE_FILE ready"
fi

# Build the Shakespeare tokenizer artifact so Module 10's pretraining can load
# it by name even when a student skips Module 04's optional artifact cells.
# Idempotent: skips if the artifact already exists.
info "Building Shakespeare tokenizer artifact (~2 sec on first run)"
.venv/bin/python scripts/build_tokenizers.py shakespeare

# ---- 7. Smoke test -----------------------------------------------------------
info "Running smoke test"
.venv/bin/python scripts/smoke_test.py

# ---- 8. Completion sentinel --------------------------------------------------
# Module 00's environment-check exercise reads this file to confirm setup.sh
# ran end-to-end (smoke test passed, deps installed). Removed and rewritten on
# every successful run so it always reflects the latest setup.
date -u +"%Y-%m-%dT%H:%M:%SZ" > .venv/.g2c-setup-complete

if [[ "$MODE" == "full" ]]; then
    info "--full: running ./datasets.sh, ./baselm.sh, ./prodlm.sh"
    ./datasets.sh
    ./baselm.sh
    ./prodlm.sh
    ok "Full install complete."
fi

echo ""
ok "Setup complete."
echo ""
echo "Activate the venv with:"
echo "    source .venv/bin/activate"
echo ""
echo "Check clean notebooks without executing them with:"
echo "    python scripts/check_notebooks.py"
echo ""
echo "Download optional large datasets with:"
echo "    ./datasets.sh glove        # Module 05 pretrained vectors"
echo "    ./datasets.sh tinystories  # Module 10 scale-up corpus"
echo ""
echo "Download/register the optional BaseLM model with:"
echo "    ./baselm.sh                # Modules 13-16 pretrained base fallback"
echo ""
echo "Configure the ProdLM runtime with:"
echo "    ./prodlm.sh                # Modules 16-20 local production model"
echo ""
echo "Run the full suite with:"
echo "    python -m pytest    # many tests intentionally fail on the scaffold branch"
echo ""
echo "Run the smoke test again with:"
echo "    python scripts/smoke_test.py"
