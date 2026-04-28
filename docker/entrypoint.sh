#!/usr/bin/env bash
set -euo pipefail

# If the venv volume is empty (or the python binary is missing), run uv sync
# once. This pre-builds the three CUDA extensions; takes ~15 min on first
# container start, instant on subsequent runs.
if [[ ! -x "${UV_PROJECT_ENVIRONMENT:-/opt/venv}/bin/python" ]]; then
    echo "==> First run: building venv at ${UV_PROJECT_ENVIRONMENT:-/opt/venv}"
    echo "==> This compiles three CUDA extensions and may take ~15 minutes."
    uv sync
fi

exec "$@"
