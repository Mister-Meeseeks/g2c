#!/usr/bin/env bash
# notebook.sh — one-stop wrapper to set up the environment for a module
# notebook and launch it.
#
# Behavior:
#   1) Always run ./setup.sh (idempotent).
#   2) Run any additional setup scripts the module needs (datasets.sh,
#      baselm.sh, prodlm.sh).
#   3) Run the module's tests. Failures surface a warning but do not block
#      opening the notebook.
#   4) Hand off to scripts/open_notebook.py with the remaining args.

set -uo pipefail

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
Usage: ./notebook.sh <module> [open_notebook.py args]

One-stop wrapper for opening a module notebook:
  1) Always runs ./setup.sh.
  2) Runs the additional setup scripts the module needs (./datasets.sh,
     ./baselm.sh, ./prodlm.sh) based on the module id.
  3) Runs the module's tests. Failures surface a warning but do not block
     opening the notebook.
  4) Validates the working solutions notebook JSON + cell syntax. A malformed
     notebook aborts the launch with a restore command.
  5) Opens the working solutions copy via scripts/open_notebook.py, passing
     through any extra args (e.g. --fresh, --no-launch).

Module → extras mapping:
  04            ./datasets.sh --tiny
  05            ./datasets.sh glove
  10            ./datasets.sh
  13, 14, 15    ./baselm.sh
  16-20         ./prodlm.sh
  (others)      no extras

Examples:
  ./notebook.sh 01
  ./notebook.sh 03b --fresh
  ./notebook.sh 10 --no-launch
EOF
}

if [[ $# -eq 0 ]]; then
    usage >&2
    exit 1
fi
case "$1" in
    -h|--help)
        usage
        exit 0
        ;;
esac

MODULE="$1"
shift

# Normalize for case-statement matching: lowercase + zero-pad single-digit ids.
MODULE_NORM=$(printf "%s" "$MODULE" | tr 'A-Z' 'a-z')
if [[ "$MODULE_NORM" =~ ^[0-9]$ ]]; then
    MODULE_NORM="0$MODULE_NORM"
fi

EXTRAS=()
TEST_FILES=()
case "$MODULE_NORM" in
    00)  ;;
    01)  TEST_FILES=("tests/test_autodiff.py") ;;
    02)  TEST_FILES=("tests/test_tensors.py") ;;
    03)  TEST_FILES=("tests/test_nn.py") ;;
    03b) TEST_FILES=("tests/test_training.py") ;;
    04)  EXTRAS+=("./datasets.sh --tiny")
         TEST_FILES=("tests/test_tokenizer.py") ;;
    05)  EXTRAS+=("./datasets.sh glove")
         TEST_FILES=("tests/test_embeddings.py") ;;
    06)  TEST_FILES=("tests/test_lm.py") ;;
    07)  TEST_FILES=("tests/test_attention.py") ;;
    08)  TEST_FILES=("tests/test_multi_head_attention.py") ;;
    09)  TEST_FILES=("tests/test_transformer.py") ;;
    09b) TEST_FILES=("tests/test_pretraining_setup.py" "tests/test_pretraining.py") ;;
    10)  EXTRAS+=("./datasets.sh")
         TEST_FILES=("tests/test_training.py" "tests/test_pretraining_setup.py" "tests/test_pretraining.py") ;;
    11)  TEST_FILES=("tests/test_sampling.py") ;;
    13)  EXTRAS+=("./baselm.sh")
         TEST_FILES=("tests/test_sft.py") ;;
    14)  EXTRAS+=("./baselm.sh")
         TEST_FILES=("tests/test_dpo.py") ;;
    15)  EXTRAS+=("./baselm.sh")
         TEST_FILES=("tests/test_eval.py") ;;
    16)  EXTRAS+=("./prodlm.sh")
         TEST_FILES=("tests/test_inference.py") ;;
    17)  EXTRAS+=("./prodlm.sh")
         TEST_FILES=("tests/test_rag.py") ;;
    18)  EXTRAS+=("./prodlm.sh")
         TEST_FILES=("tests/test_tools.py") ;;
    19)  EXTRAS+=("./prodlm.sh")
         TEST_FILES=("tests/test_agent.py") ;;
    20)  EXTRAS+=("./prodlm.sh")
         TEST_FILES=("tests/test_assistant.py") ;;
    *)
        warn "Module '$MODULE' not recognized; running ./setup.sh only and skipping test verification."
        ;;
esac

printf "\n"
info "Setting up environment for notebook $MODULE"
printf "\n"

./setup.sh
SETUP_RC=$?
if [[ $SETUP_RC -ne 0 ]]; then
    fail "./setup.sh failed (exit $SETUP_RC). Aborting."
fi

if [[ ${#EXTRAS[@]} -gt 0 ]]; then
    for cmd in "${EXTRAS[@]}"; do
        printf "\n"
        info "Module $MODULE extra: $cmd"
        eval "$cmd"
        RC=$?
        if [[ $RC -ne 0 ]]; then
            fail "$cmd failed (exit $RC). Aborting."
        fi
    done
fi

printf "\n"
ok "Environment setup complete."
printf "\n"

TEST_FAILED=0
if [[ ${#TEST_FILES[@]} -gt 0 ]]; then
    info "Verifying module $MODULE tests"
    .venv/bin/python -m pytest "${TEST_FILES[@]}" -q
    TEST_FAILED=$?
fi

if [[ $TEST_FAILED -ne 0 ]]; then
    BAR=$(printf '#%.0s' {1..72})
    printf "\n"
    printf "${YELLOW}%s${NC}\n" "$BAR"
    printf "${YELLOW}# WARNING: module %s tests are NOT all passing.${NC}\n" "$MODULE"
    printf "${YELLOW}# The notebook will still open, but cells that depend on the${NC}\n"
    printf "${YELLOW}# module deliverable may fail until it is implemented.${NC}\n"
    printf "${YELLOW}%s${NC}\n" "$BAR"
    printf "\n"
fi

# Validate the working solutions copy if it already exists. A fresh copy is
# created by open_notebook.py on first open and inherits validation from the
# clean scaffold, so skip the check when the file is not on disk yet.
shopt -s nullglob
NB_FILES=(notebooks/solutions/${MODULE_NORM}-*.ipynb)
shopt -u nullglob
if [[ ${#NB_FILES[@]} -gt 0 ]]; then
    info "Validating notebook structure"
    if ! .venv/bin/python scripts/check_notebooks.py "${NB_FILES[@]}"; then
        BAR=$(printf '#%.0s' {1..72})
        printf "\n"
        printf "${RED}%s${NC}\n" "$BAR"
        printf "${RED}# Notebook is malformed. Restore with one of:${NC}\n"
        for nb in "${NB_FILES[@]}"; do
            printf "${RED}#   git checkout HEAD -- %s${NC}\n" "$nb"
        done
        printf "${RED}#   ./notebook.sh %s --fresh    # reset from notebooks/clean/${NC}\n" "$MODULE"
        printf "${RED}%s${NC}\n" "$BAR"
        printf "\n"
        exit 1
    fi
fi

info "Opening module $MODULE notebook"
exec .venv/bin/python scripts/open_notebook.py "$MODULE" "$@"
