#!/usr/bin/env bash
# Download/register the small pretrained BaseLM artifact for Modules 13-15.

set -euo pipefail

if [[ -x ".venv/bin/python" ]]; then
    PYTHON=".venv/bin/python"
else
    PYTHON="python3"
fi

"$PYTHON" scripts/setup_baselm.py "$@"
