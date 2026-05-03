#!/usr/bin/env bash
# Bootstrap the From Gradients to ChatGPT course environment.
#
# Strategy:
#   - SYSTEM-LEVEL tools (python, uv) are CHECKED, not installed. If missing,
#     the script prints how to install them and exits.
#   - PROJECT-LEVEL tools (torch, pytest, ruff, jupyter, the g2c package itself)
#     are installed into a project-local venv at ./.venv via uv.
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

# ---- 3. venv -----------------------------------------------------------------
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

# ---- 4. Project deps ---------------------------------------------------------
info "Installing g2c (editable) + dev deps"
if [[ "$REINSTALL_DEV" == "1" ]]; then
    uv pip install -e ".[dev]" --reinstall --quiet
else
    uv pip install -e ".[dev]" --quiet
fi
ok "deps installed"

# ---- 5. Optional data assets -------------------------------------------------
GLOVE_FILE="data/glove.6B.50d.txt"
GLOVE_ZIP="data/glove.6B.zip"
GLOVE_URL="https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip"

info "Checking GloVe vectors"
if [[ -f "$GLOVE_FILE" ]]; then
    ok "$GLOVE_FILE exists"
else
    mkdir -p data

    if ! command -v curl >/dev/null 2>&1; then
        fail "curl not found; needed to download $GLOVE_URL"
    fi
    if ! command -v unzip >/dev/null 2>&1; then
        fail "unzip not found; needed to extract $GLOVE_FILE"
    fi

    if [[ -f "$GLOVE_ZIP" ]] && unzip -t "$GLOVE_ZIP" glove.6B.50d.txt >/dev/null 2>&1; then
        ok "existing $GLOVE_ZIP contains glove.6B.50d.txt"
    else
        warn "$GLOVE_FILE not found; downloading GloVe 6B archive (~822MB)"
        if [[ -f "$GLOVE_ZIP" ]]; then
            info "Resuming existing download at $GLOVE_ZIP"
            if ! curl --fail --location --continue-at - --output "$GLOVE_ZIP" "$GLOVE_URL"; then
                warn "resume failed; restarting GloVe download"
                rm -f "$GLOVE_ZIP"
                curl --fail --location --output "$GLOVE_ZIP" "$GLOVE_URL"
            fi
        else
            curl --fail --location --output "$GLOVE_ZIP" "$GLOVE_URL"
        fi
    fi

    info "Extracting glove.6B.50d.txt"
    unzip -o "$GLOVE_ZIP" glove.6B.50d.txt -d data
    rm -f "$GLOVE_ZIP"

    if [[ ! -f "$GLOVE_FILE" ]]; then
        fail "expected $GLOVE_FILE after extraction, but it was not found"
    fi
    ok "$GLOVE_FILE ready"
fi

# ---- 6. Smoke test -----------------------------------------------------------
info "Running smoke test"
.venv/bin/python scripts/smoke_test.py

echo ""
ok "Setup complete."
echo ""
echo "Activate the venv with:"
echo "    source .venv/bin/activate"
echo ""
echo "Run tests with:"
echo "    python -m pytest"
echo ""
echo "Run the smoke test again with:"
echo "    python scripts/smoke_test.py"
