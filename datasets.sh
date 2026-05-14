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
    ./datasets.sh --tiny Tiny track: download a 100MB TinyStories sample and
                         build matching StoryTokenizer/tokenized artifacts.
    ./datasets.sh --small
                         Standard track: download GloVe, full TinyStories, and
                         small G2C Corpus v1; build matching artifacts.
    ./datasets.sh        Full track: download GloVe, full TinyStories, and full
                         G2C Corpus v1; build matching artifacts.
    ./datasets.sh all    Same as no args.

Individual targets:
    glove             Download Stanford GloVe 6B and extract glove.6B.50d.txt.
    tinystories       Download TinyStories train/valid compressed shards.
    g2c-corpus-small  Build ~1GB raw G2C Corpus v1 shards.
    g2c-corpus-full   Build ~9.5GB raw G2C Corpus v1 shards.

G2C corpus targets accept extra flags passed to scripts/gen_corpus.py, e.g.:
    ./datasets.sh g2c-corpus-full --codesearchnet-js-ratio 0.2

Large files are written under data/, which is gitignored. Downloads are resumed
when possible and skipped when the final files already exist. TinyStories is
stored as gzip shards split by 100MB of uncompressed text per shard.

After each corpus is downloaded, scripts/build_tokenizers.py builds the
matching tokenizer artifact (StoryTokenizer for tinystories, G2CTokenizer for
the G2C corpus), then scripts/build_tokenized_corpus.py builds the matching
disk-backed tokenized corpus artifact. Both steps are idempotent and skipped
when complete. The Shakespeare tokenizer artifact is built by setup.sh.
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

python_bin() {
    if [[ -x ".venv/bin/python" ]]; then
        printf ".venv/bin/python"
    else
        printf "python3"
    fi
}

download_glove() {
    local glove_file="data/embeddings/glove.6B.50d.txt"
    local glove_zip="data/embeddings/glove.6B.zip"
    local glove_url="https://downloads.cs.stanford.edu/nlp/data/embeddings/glove.6B.zip"
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
    local dir="data/datasets/tinystories"
    local train_file="$dir/TinyStories-train.txt"
    local valid_file="$dir/TinyStories-valid.txt"
    local train_url="https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt"
    local valid_url="https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt"
    local train_min_bytes=1924281556
    local valid_min_bytes=19447282

    require_tool curl
    info "Checking TinyStories corpus"
    mkdir -p "$dir"

    download_tinystories_shards \
        "$train_url" \
        "$dir" \
        "TinyStories-train" \
        "$train_file" \
        "$train_min_bytes" \
        "TinyStories train text (~1.9GB)"

    download_tinystories_shards \
        "$valid_url" \
        "$dir" \
        "TinyStories-valid" \
        "$valid_file" \
        "$valid_min_bytes" \
        "TinyStories validation text (~20MB)"

    build_tokenizer_artifact "story"
    build_tokenized_corpus_artifact \
        "StoryLM-tinystories-full-v4096" \
        "tinystories" \
        "StoryTokenizer" \
        "4096"
}

download_tinystories_sample() {
    local dir="data/datasets/tinystories-100MB"
    local sample_file="$dir/TinyStories-train-100MB.txt"
    local train_url="https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt"
    local sample_bytes=100000000
    local actual_bytes
    local raw_tmp="$dir/.TinyStories-train-100MB.txt.download"

    require_tool curl
    info "Checking TinyStories 100MB sample"
    mkdir -p "$dir"

    if tinystories_shards_ready "$dir" "TinyStories-train-100MB" "$sample_bytes"; then
        ok "TinyStories 100MB sample shards exist"
    elif file_at_least_bytes "$sample_file" "$sample_bytes"; then
        info "$sample_file exists; converting to compressed shards"
        shard_tinystories_file "$sample_file" "$dir" "TinyStories-train-100MB"
        rm -f "$sample_file"
        ok "TinyStories 100MB sample shards ready"
    else
        if [[ -f "$sample_file" ]]; then
            info "$sample_file is incomplete; restarting sample download"
            rm -f "$sample_file"
        fi

        info "TinyStories 100MB sample not found; downloading first 100MB of train text"
        curl \
            --fail \
            --location \
            --range "0-$((sample_bytes - 1))" \
            --output "$raw_tmp" \
            "$train_url"

        actual_bytes="$(wc -c < "$raw_tmp" | tr -d ' ')"
        if [[ "$actual_bytes" -gt "$((sample_bytes + 1000000))" ]]; then
            fail "expected about 100MB, but sample download is $actual_bytes bytes"
        fi
        if [[ "$actual_bytes" -lt "$sample_bytes" ]]; then
            fail "TinyStories 100MB sample is incomplete after download"
        fi

        shard_tinystories_file "$raw_tmp" "$dir" "TinyStories-train-100MB"
        rm -f "$raw_tmp"
        ok "TinyStories 100MB sample shards ready"
    fi

    build_tokenizer_artifact "story"
    build_tokenized_corpus_artifact \
        "StoryLM-tinystories-100MB-v4096" \
        "tinystories-100MB" \
        "StoryTokenizer" \
        "4096"
}

tinystories_shards_ready() {
    local dir="$1"
    local prefix="$2"
    local min_bytes="$3"
    local manifest="$dir/$prefix-shards.json"

    [[ -f "$manifest" ]] || return 1
"$(python_bin)" - "$dir" "$manifest" "$min_bytes" <<'PY'
import json
import sys
from pathlib import Path

try:
    root = Path(sys.argv[1])
    manifest_path = Path(sys.argv[2])
    min_bytes = int(sys.argv[3])
    manifest = json.loads(manifest_path.read_text())
    shards = manifest.get("shards", [])
    if int(manifest.get("total_uncompressed_bytes", 0)) < min_bytes:
        raise SystemExit(1)
    if not shards:
        raise SystemExit(1)
    for shard in shards:
        if not (root / shard["path"]).exists():
            raise SystemExit(1)
except Exception:
    raise SystemExit(1)
PY
}

shard_tinystories_file() {
    local input="$1"
    local dir="$2"
    local prefix="$3"
    local chunk_bytes=100000000

    info "Compressing $prefix into 100MB uncompressed shards"
    "$(python_bin)" scripts/shard_text_gzip.py \
        "$input" \
        --output-dir "$dir" \
        --prefix "$prefix" \
        --chunk-bytes "$chunk_bytes" \
        --manifest "$dir/$prefix-shards.json"
}

download_tinystories_shards() {
    local url="$1"
    local dir="$2"
    local prefix="$3"
    local legacy_file="$4"
    local min_bytes="$5"
    local label="$6"
    local raw_tmp="$dir/.$prefix.txt.download"

    if tinystories_shards_ready "$dir" "$prefix" "$min_bytes"; then
        ok "$prefix compressed shards exist"
        return
    fi

    if file_at_least_bytes "$legacy_file" "$min_bytes"; then
        info "$legacy_file exists; converting to compressed shards"
        shard_tinystories_file "$legacy_file" "$dir" "$prefix"
        rm -f "$legacy_file"
    else
        if [[ -f "$legacy_file" ]]; then
            info "$legacy_file is incomplete; removing before redownload"
            rm -f "$legacy_file"
        fi
        info "$prefix shards not found; downloading $label"
        download_resumable "$url" "$raw_tmp"
        if ! file_at_least_bytes "$raw_tmp" "$min_bytes"; then
            fail "$prefix download is still incomplete"
        fi
        shard_tinystories_file "$raw_tmp" "$dir" "$prefix"
        rm -f "$raw_tmp"
    fi

    if ! tinystories_shards_ready "$dir" "$prefix" "$min_bytes"; then
        fail "$prefix compressed shards were not created correctly"
    fi
    ok "$prefix compressed shards ready"
}

build_tokenizer_artifact() {
    # Build one course tokenizer artifact non-interactively. Idempotent — skips
    # when the artifact already exists. Module 10 loads these by name.
    local name="$1"
    info "Building $name tokenizer artifact (skipped if it already exists)"
    "$(python_bin)" scripts/build_tokenizers.py "$name"
}

build_tokenized_corpus_artifact() {
    # Build one disk-backed token-ID corpus artifact. Idempotent — skips when
    # the artifact already exists. Later notebooks load these instead of
    # re-tokenizing large corpora in the kernel.
    local name="$1"
    local corpus="$2"
    local tokenizer="$3"
    local vocab_size="$4"

    info "Building $name tokenized corpus artifact (skipped if it already exists)"
    "$(python_bin)" scripts/build_tokenized_corpus.py \
        --name "$name" \
        --corpus "$corpus" \
        --tokenizer "$tokenizer" \
        --vocab-size "$vocab_size"
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
    local py
    py="$(python_bin)"

    info "Building G2C Corpus v1 ($preset preset)"
    if [[ -f "$out_dir/manifest.json" ]] && ! has_arg "--force" "$@" && ! has_arg "--dry-run" "$@"; then
        ok "$out_dir exists"
    else
        "$py" scripts/gen_corpus.py --preset "$preset" --out "$out_dir" "$@"
    fi

    if ! has_arg "--dry-run" "$@"; then
        build_tokenizer_artifact "g2c"
        build_tokenized_corpus_artifact \
            "TinyLLM-g2c-$preset-v8192" \
            "g2c-corpus-$preset" \
            "G2CTokenizer" \
            "8192"
    fi
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
            download_all full "data/datasets/g2c-corpus-v1" "$@"
            ;;
        --small)
            download_all small "data/datasets/g2c-corpus-v1-small" "$@"
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
            build_g2c_corpus small "data/datasets/g2c-corpus-v1-small" "$@"
            ;;
        g2c-corpus-full)
            build_g2c_corpus full "data/datasets/g2c-corpus-v1" "$@"
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
