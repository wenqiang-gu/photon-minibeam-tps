#!/usr/bin/env bash
# Editable defaults. CPU inference from TOPAS inputs takes priority.
DEFAULT_THROTTLE=10
DEFAULT_TIME="48:00:00"
DEFAULT_MEM=""
DEFAULT_PARTITION=""
DEFAULT_EXCLUDE=""
DEFAULT_JOB_NAME="topas_general"
DEFAULT_TOPAS_ENV="/data/maia/s245155/Applications/TOPAS/opentopas-env.sh"
DEFAULT_CPUS_PER_TASK=1

set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec python3 "$SCRIPT_DIR/submit_topas.py" \
  --throttle "$DEFAULT_THROTTLE" --time "$DEFAULT_TIME" \
  --mem "$DEFAULT_MEM" --partition "$DEFAULT_PARTITION" \
  --exclude "$DEFAULT_EXCLUDE" --job-name "$DEFAULT_JOB_NAME" \
  --topas-env "${TOPAS_ENV:-$DEFAULT_TOPAS_ENV}" \
  --default-cpus "$DEFAULT_CPUS_PER_TASK" "$@"
