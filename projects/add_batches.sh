#!/usr/bin/env bash
# Additional batches to create, NOT the final total. --count overrides this.
DEFAULT_ADDITIONAL_BATCHES=3
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec python3 "$SCRIPT_DIR/add_batches.py" --count "$DEFAULT_ADDITIONAL_BATCHES" "$@"
