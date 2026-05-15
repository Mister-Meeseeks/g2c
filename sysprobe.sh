#!/usr/bin/env bash
# Probe what your machine can handle: training sizes, inference throughput,
# and ProdLM model fit (pre-download). See scripts/sysprobe.py for details.

set -euo pipefail

if [[ -x ".venv/bin/python" ]]; then
    PYTHON=".venv/bin/python"
else
    PYTHON="python3"
fi

exec "$PYTHON" scripts/sysprobe.py "$@"
