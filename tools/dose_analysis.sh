#!/usr/bin/env bash
# Examples:
#   ./tools/dose_analysis.sh lateral --field 10
#   ./tools/dose_analysis.sh lateral --field 10 --combine-depths
#   ./tools/dose_analysis.sh pdd --field 10
# All arguments are forwarded unchanged; use --help to list commands.
# Set PYTHON_BIN to an interpreter path to override automatic selection:
#   PYTHON_BIN=/path/to/python ./tools/dose_analysis.sh --help
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
        PYTHON_BIN="$VIRTUAL_ENV/bin/python"
    elif [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
        PYTHON_BIN="$CONDA_PREFIX/bin/python"
    elif [[ -x "$SCRIPT_DIR/../.venv/bin/python" ]]; then
        PYTHON_BIN="$SCRIPT_DIR/../.venv/bin/python"
    else
        PYTHON_BIN=python3
    fi
fi

# Keep the caller's working directory so relative CLI paths retain their meaning.
exec "$PYTHON_BIN" "$SCRIPT_DIR/dose_analysis.py" "$@"
