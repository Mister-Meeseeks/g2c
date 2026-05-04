#!/usr/bin/env bash
# Download optional large course datasets.
#
# Normal setup stays light and only prepares tinyshakespeare.txt. Run this
# script when a module asks for a larger local asset.

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
    ./datasets.sh [all|glove|tinystories]

Targets:
    glove        Download Stanford GloVe 6B and extract glove.6B.50d.txt.
    tinystories  Download TinyStories train/valid text files.
    all          Download both large datasets. This is the default.

Large files are written under data/, which is gitignored. Downloads are
resumed when possible and skipped when the final files already exist.
EOF
}

require_tool() {
    local tool="$1"
    if ! command -v "$tool" >/dev/null 2>&1; then
        fail "$tool not found; install it before downloading datasets"
    fi
}

download_resumable() {
    local url="$1"
    local output="$2"

    if [[ -f "$output" ]]; then
        info "Resuming existing download at $output"
        if curl --fail --location --continue-at - --output "$output" "$url"; then
            return
        fi
        warn "resume failed; restarting $output"
        rm -f "$output"
    fi

    curl --fail --location --output "$output" "$url"
}

download_glove() {
    local glove_file="data/glove.6B.50d.txt"
    local glove_zip="data/glove.6B.zip"
    local glove_url="https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip"

    info "Checking GloVe vectors"
    if [[ -f "$glove_file" ]]; then
        ok "$glove_file exists"
        return
    fi

    require_tool unzip

    if [[ -f "$glove_zip" ]] && unzip -t "$glove_zip" glove.6B.50d.txt >/dev/null 2>&1; then
        ok "existing $glove_zip contains glove.6B.50d.txt"
    else
        info "$glove_file not found; downloading GloVe 6B archive (~822MB)"
        download_resumable "$glove_url" "$glove_zip"
    fi

    info "Extracting glove.6B.50d.txt"
    unzip -o "$glove_zip" glove.6B.50d.txt -d data
    rm -f "$glove_zip"

    if [[ ! -f "$glove_file" ]]; then
        fail "expected $glove_file after extraction, but it was not found"
    fi
    ok "$glove_file ready"
}

download_tinystories() {
    local dir="data/tinystories"
    local train_file="$dir/TinyStories-train.txt"
    local valid_file="$dir/TinyStories-valid.txt"
    local train_url="https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt"
    local valid_url="https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt"

    info "Checking TinyStories corpus"
    mkdir -p "$dir"

    if [[ -f "$train_file" ]]; then
        ok "$train_file exists"
    else
        info "$train_file not found; downloading TinyStories train text (~1.9GB)"
        download_resumable "$train_url" "$train_file"
        ok "$train_file ready"
    fi

    if [[ -f "$valid_file" ]]; then
        ok "$valid_file exists"
    else
        info "$valid_file not found; downloading TinyStories validation text (~20MB)"
        download_resumable "$valid_url" "$valid_file"
        ok "$valid_file ready"
    fi
}

main() {
    local target="${1:-all}"
    if [[ "$target" == "-h" || "$target" == "--help" ]]; then
        usage
        exit 0
    fi
    if [[ $# -gt 1 ]]; then
        usage >&2
        exit 2
    fi

    require_tool curl
    mkdir -p data

    case "$target" in
        all)
            download_glove
            download_tinystories
            ;;
        glove)
            download_glove
            ;;
        tinystories)
            download_tinystories
            ;;
        *)
            usage >&2
            exit 2
            ;;
    esac

    echo ""
    ok "Dataset setup complete."
}

main "$@"
