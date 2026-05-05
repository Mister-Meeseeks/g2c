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
    ./datasets.sh [--small|--tiny|all|glove|tinystories|g2c-corpus-small|g2c-corpus-full]

Modes:
    ./datasets.sh        Download GloVe, full TinyStories, and full G2C Corpus v1.
    ./datasets.sh all    Same as no args.
    ./datasets.sh --small
                         Download GloVe, full TinyStories, and small G2C Corpus v1.
    ./datasets.sh --tiny Download only a 100MB TinyStories sample.

Individual targets:
    glove             Download Stanford GloVe 6B and extract glove.6B.50d.txt.
    tinystories       Download TinyStories train/valid text files.
    g2c-corpus-small  Build ~1GB raw G2C Corpus v1 shards.
    g2c-corpus-full   Build ~9.5GB raw G2C Corpus v1 shards.

G2C corpus targets accept extra flags passed to scripts/gen_corpus.py, e.g.:
    ./datasets.sh g2c-corpus-full --codesearchnet-js-ratio 0.2

Large files are written under data/, which is gitignored. Downloads are resumed
when possible and skipped when the final files already exist.
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

file_size() {
    wc -c < "$1" | tr -d ' '
}

file_at_least_bytes() {
    local path="$1"
    local min_bytes="$2"
    [[ -f "$path" ]] && [[ "$(file_size "$path")" -ge "$min_bytes" ]]
}

download_glove() {
    local glove_file="data/glove.6B.50d.txt"
    local glove_zip="data/glove.6B.zip"
    local glove_url="https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip"
    local glove_min_bytes=171350079

    require_tool curl
    info "Checking GloVe vectors"
    if file_at_least_bytes "$glove_file" "$glove_min_bytes"; then
        ok "$glove_file exists"
        return
    fi
    if [[ -f "$glove_file" ]]; then
        info "$glove_file is incomplete; rebuilding from archive"
        rm -f "$glove_file"
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

    if ! file_at_least_bytes "$glove_file" "$glove_min_bytes"; then
        fail "expected complete $glove_file after extraction, but it was not found"
    fi
    ok "$glove_file ready"
}

download_tinystories() {
    local dir="data/tinystories"
    local train_file="$dir/TinyStories-train.txt"
    local valid_file="$dir/TinyStories-valid.txt"
    local train_url="https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt"
    local valid_url="https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt"
    local train_min_bytes=1924281556
    local valid_min_bytes=19447282

    require_tool curl
    info "Checking TinyStories corpus"
    mkdir -p "$dir"

    if file_at_least_bytes "$train_file" "$train_min_bytes"; then
        ok "$train_file exists"
    else
        info "$train_file not found or incomplete; downloading TinyStories train text (~1.9GB)"
        download_resumable "$train_url" "$train_file"
        if ! file_at_least_bytes "$train_file" "$train_min_bytes"; then
            fail "$train_file is still incomplete after download"
        fi
        ok "$train_file ready"
    fi

    if file_at_least_bytes "$valid_file" "$valid_min_bytes"; then
        ok "$valid_file exists"
    else
        info "$valid_file not found or incomplete; downloading TinyStories validation text (~20MB)"
        download_resumable "$valid_url" "$valid_file"
        if ! file_at_least_bytes "$valid_file" "$valid_min_bytes"; then
            fail "$valid_file is still incomplete after download"
        fi
        ok "$valid_file ready"
    fi
}

download_tinystories_sample() {
    local dir="data/tinystories"
    local sample_file="$dir/TinyStories-train-100MB.txt"
    local train_url="https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt"
    local sample_bytes=100000000
    local actual_bytes

    require_tool curl
    info "Checking TinyStories 100MB sample"
    mkdir -p "$dir"

    if [[ -f "$sample_file" ]]; then
        actual_bytes="$(wc -c < "$sample_file" | tr -d ' ')"
        if [[ "$actual_bytes" -ge "$sample_bytes" ]]; then
            ok "$sample_file exists"
            return
        fi
        info "$sample_file is incomplete; restarting sample download"
        rm -f "$sample_file"
    fi

    info "$sample_file not found; downloading first 100MB of TinyStories train text"
    curl \
        --fail \
        --location \
        --range "0-$((sample_bytes - 1))" \
        --output "$sample_file" \
        "$train_url"

    actual_bytes="$(wc -c < "$sample_file" | tr -d ' ')"
    if [[ "$actual_bytes" -gt "$((sample_bytes + 1000000))" ]]; then
        fail "expected about 100MB, but $sample_file is $actual_bytes bytes"
    fi
    ok "$sample_file ready"
}

has_arg() {
    local needle="$1"
    shift
    local arg
    for arg in "$@"; do
        if [[ "$arg" == "$needle" ]]; then
            return 0
        fi
    done
    return 1
}

build_g2c_corpus() {
    local preset="$1"
    local out_dir="$2"
    shift
    shift
    local python_bin="python3"
    if [[ -x ".venv/bin/python" ]]; then
        python_bin=".venv/bin/python"
    fi

    info "Building G2C Corpus v1 ($preset preset)"
    if [[ -f "$out_dir/manifest.json" ]] && ! has_arg "--force" "$@" && ! has_arg "--dry-run" "$@"; then
        ok "$out_dir exists"
        return
    fi
    "$python_bin" scripts/gen_corpus.py --preset "$preset" --out "$out_dir" "$@"
}

download_all() {
    local corpus_preset="$1"
    local corpus_dir="$2"
    shift
    shift

    download_glove
    download_tinystories
    build_g2c_corpus "$corpus_preset" "$corpus_dir" "$@"
}

main() {
    local target="${1:-all}"
    if [[ "$target" == "-h" || "$target" == "--help" ]]; then
        usage
        exit 0
    fi
    shift || true

    mkdir -p data

    case "$target" in
        all)
            download_all full "data/g2c-corpus-v1" "$@"
            ;;
        --small)
            download_all small "data/g2c-corpus-v1-small" "$@"
            ;;
        --tiny)
            if [[ $# -gt 0 ]]; then
                usage >&2
                exit 2
            fi
            download_tinystories_sample
            ;;
        glove)
            if [[ $# -gt 0 ]]; then
                usage >&2
                exit 2
            fi
            download_glove
            ;;
        tinystories)
            if [[ $# -gt 0 ]]; then
                usage >&2
                exit 2
            fi
            download_tinystories
            ;;
        g2c-corpus-small)
            build_g2c_corpus small "data/g2c-corpus-v1-small" "$@"
            ;;
        g2c-corpus-full)
            build_g2c_corpus full "data/g2c-corpus-v1" "$@"
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
