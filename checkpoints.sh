#!/usr/bin/env bash
# Download the reference model checkpoints (the "checkpoint ladder").
#
# Module 10's scale-up runs cost hours-to-overnight. Training your own models
# is the point of Module 10 -- but Modules 12+ shouldn't be hostage to that
# wall-clock if your machine or schedule can't pay it. This script fetches the
# course's reference checkpoints: the same artifacts, trained with the
# reference implementations, with their training config and final losses
# recorded in each manifest.
#
# Existing artifacts are NEVER overwritten. If you trained your own
# StoryLM-30M-base, it stays yours.
#
# Assets are packaged by scripts/package_checkpoints.py and published as
# GitHub release assets. Checksums below are rewritten by that script's
# --update-script flag at packaging time.

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
    ./checkpoints.sh [--all|storylm-1m|storylm-5m|storylm-30m|tinyllm-30m]

Modes:
    ./checkpoints.sh        Fetch the StoryLM ladder (1M, 5M, 30M) plus the
                            StoryTokenizer artifact -- everything Module 12's
                            scaling comparison needs.
    ./checkpoints.sh --all  Also fetch TinyLLM-30M-base and the G2CTokenizer,
                            the strongest course-track model for Modules 13+.

Individual targets:
    storylm-1m    StoryLM-1M-base  (~5MB)
    storylm-5m    StoryLM-5M-base  (~22MB)
    storylm-30m   StoryLM-30M-base (~117MB)
    tinyllm-30m   TinyLLM-30M-base (~113MB)

Each target also fetches its tokenizer artifact if missing. Artifacts unpack
into artifacts/models/ and artifacts/tokenizers/. Existing artifacts are never
overwritten -- delete the artifact directory first if you really want the
reference copy instead of your own.

These are reference weights, not a substitute for the course: training your
own ladder is Module 10's deliverable. Fetch these when the overnight runs
aren't in reach, or when entering the course at Part II (see Module 12,
"Entering at Part II"). Each manifest records the reference run's training
config and final train/val losses, so you can calibrate your own runs against
the course's numbers.
EOF
}

CHECKPOINTS_RELEASE_TAG="checkpoints-v1"

STORYLM_1M_URL="https://github.com/Mister-Meeseeks/g2c/releases/download/checkpoints-v1/StoryLM-1M-base.tar.gz"
STORYLM_5M_URL="https://github.com/Mister-Meeseeks/g2c/releases/download/checkpoints-v1/StoryLM-5M-base.tar.gz"
STORYLM_30M_URL="https://github.com/Mister-Meeseeks/g2c/releases/download/checkpoints-v1/StoryLM-30M-base.tar.gz"
TINYLLM_30M_URL="https://github.com/Mister-Meeseeks/g2c/releases/download/checkpoints-v1/TinyLLM-30M-base.tar.gz"
STORYTOKENIZER_URL="https://github.com/Mister-Meeseeks/g2c/releases/download/checkpoints-v1/StoryTokenizer.tar.gz"
G2CTOKENIZER_URL="https://github.com/Mister-Meeseeks/g2c/releases/download/checkpoints-v1/G2CTokenizer.tar.gz"

# Rewritten by: scripts/package_checkpoints.py --update-script
STORYLM_1M_SHA256="9e25ec386ba0dde8112519fd0b8ea336065448d34324389c1f8f59298b26bc39"
STORYLM_5M_SHA256="798c37ecb7fb4713cf3984aaa0113fdf0b24021c738196905c3f7b6fd31448de"
STORYLM_30M_SHA256="4181ad2cf5acc222d68843009f3e64193748ad80c6476446af569ab4921d00cd"
TINYLLM_30M_SHA256="d2816f356371d7cc2a40b00497de010b7c7ec02269808b340fc694e09b24cd76"
STORYTOKENIZER_SHA256="99abd06b5c1766780b82d8908c177928b3c5bc90e457f4bb52b5890a05474c82"
G2CTOKENIZER_SHA256="ee268e24143c90e1e6c7681b9018875f2f0e5d06dc055fa026a000d8362d645d"

DOWNLOAD_DIR="data/cache/checkpoint-dist/downloads"

require_tool() {
    local tool="$1"
    if ! command -v "$tool" >/dev/null 2>&1; then
        fail "$tool not found; install it before downloading checkpoints"
    fi
}

sha256_of() {
    if command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    elif command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        fail "neither shasum nor sha256sum found; cannot verify downloads"
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

# Download one packaged artifact, verify its checksum, and unpack it under
# $dest_root. Refuses to touch an artifact directory that already exists.
fetch_artifact() {
    local name="$1"
    local url="$2"
    local expected_sha="$3"
    local dest_root="$4"

    local dest="$dest_root/$name"
    if [[ -e "$dest" ]]; then
        ok "$name already present -- keeping the existing artifact"
        return
    fi

    if [[ -z "$expected_sha" ]]; then
        fail "no checksum recorded for $name; the $CHECKPOINTS_RELEASE_TAG release has not been packaged yet (maintainers: run scripts/package_checkpoints.py --update-script)"
    fi

    require_tool curl
    require_tool tar
    mkdir -p "$DOWNLOAD_DIR" "$dest_root"

    local tarball="$DOWNLOAD_DIR/$name.tar.gz"
    info "Downloading $name from the $CHECKPOINTS_RELEASE_TAG release"
    download_resumable "$url" "$tarball"

    local actual_sha
    actual_sha="$(sha256_of "$tarball")"
    if [[ "$actual_sha" != "$expected_sha" ]]; then
        rm -f "$tarball"
        fail "checksum mismatch for $name (expected $expected_sha, got $actual_sha); the partial download was removed -- rerun to retry"
    fi

    tar -xzf "$tarball" -C "$dest_root"
    if [[ ! -f "$dest/manifest.json" ]]; then
        fail "expected $dest/manifest.json after unpacking; the archive layout is wrong"
    fi
    rm -f "$tarball"
    ok "$name ready at $dest"
}

fetch_model() {
    fetch_artifact "$1" "$2" "$3" "artifacts/models"
}

fetch_tokenizer() {
    fetch_artifact "$1" "$2" "$3" "artifacts/tokenizers"
}

fetch_storylm_1m() {
    fetch_tokenizer "StoryTokenizer" "$STORYTOKENIZER_URL" "$STORYTOKENIZER_SHA256"
    fetch_model "StoryLM-1M-base" "$STORYLM_1M_URL" "$STORYLM_1M_SHA256"
}

fetch_storylm_5m() {
    fetch_tokenizer "StoryTokenizer" "$STORYTOKENIZER_URL" "$STORYTOKENIZER_SHA256"
    fetch_model "StoryLM-5M-base" "$STORYLM_5M_URL" "$STORYLM_5M_SHA256"
}

fetch_storylm_30m() {
    fetch_tokenizer "StoryTokenizer" "$STORYTOKENIZER_URL" "$STORYTOKENIZER_SHA256"
    fetch_model "StoryLM-30M-base" "$STORYLM_30M_URL" "$STORYLM_30M_SHA256"
}

fetch_tinyllm_30m() {
    fetch_tokenizer "G2CTokenizer" "$G2CTOKENIZER_URL" "$G2CTOKENIZER_SHA256"
    fetch_model "TinyLLM-30M-base" "$TINYLLM_30M_URL" "$TINYLLM_30M_SHA256"
}

closing_note() {
    info "Reference checkpoints in place."
    info "Training your own ladder is still Module 10's deliverable -- these"
    info "exist so Module 12+ isn't hostage to an overnight run. Each manifest"
    info "records the reference run's final train/val losses for calibration."
}

main() {
    case "${1:-}" in
        -h|--help|help)
            usage
            ;;
        "")
            fetch_storylm_1m
            fetch_storylm_5m
            fetch_storylm_30m
            closing_note
            ;;
        --all)
            fetch_storylm_1m
            fetch_storylm_5m
            fetch_storylm_30m
            fetch_tinyllm_30m
            closing_note
            ;;
        storylm-1m)
            fetch_storylm_1m
            closing_note
            ;;
        storylm-5m)
            fetch_storylm_5m
            closing_note
            ;;
        storylm-30m)
            fetch_storylm_30m
            closing_note
            ;;
        tinyllm-30m)
            fetch_tinyllm_30m
            closing_note
            ;;
        *)
            usage
            fail "unrecognized target: $1"
            ;;
    esac
}

main "$@"
