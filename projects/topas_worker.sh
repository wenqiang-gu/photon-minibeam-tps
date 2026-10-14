#!/usr/bin/env bash
# Called by Slurm, not normally by users. No Python required on compute nodes.
set -eo pipefail
TASKS=$1
ENV_FILE=$2
[[ ${SLURM_ARRAY_TASK_ID:-} =~ ^[0-9]+$ ]] || { echo 'Missing/invalid SLURM_ARRAY_TASK_ID' >&2; exit 2; }
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASKS")
[[ -n $LINE ]] || { echo 'Array task is outside task manifest' >&2; exit 2; }
IFS=$'\t' read -r BUNDLE INPUT JOB <<< "$LINE"
printf 'Task %s: %s\nBundle: %s\nInput: %s\n' "$SLURM_ARRAY_TASK_ID" "$JOB" "$BUNDLE" "$INPUT"
# Publish status atomically in the setup, independent of submission/log location.
# A retry replaces old success before touching any simulation output.
[[ $JOB =~ ^batch_[0-9]+/[A-Za-z0-9_-]+$ ]] || { echo 'Invalid job identity' >&2; exit 2; }
STATUS="$BUNDLE/job_status/$JOB.status"
mkdir -p -- "$(dirname -- "$STATUS")"
write_status() {
    printf '%s\n' "$1" > "$STATUS.tmp.$$"
    mv -f -- "$STATUS.tmp.$$" "$STATUS"
}
write_status running
START=$SECONDS
finish() {
    status=$?
    trap - EXIT
    if [[ $status == 0 ]]; then write_status succeeded; else write_status failed; fi
    printf 'TOPAS worker finished: exit=%s elapsed=%ss\n' "$status" "$((SECONDS-START))"
    exit "$status"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
source "$ENV_FILE"
cd -- "$BUNDLE"
command -v topas >/dev/null || { echo 'topas not found after loading environment' >&2; exit 127; }
topas "$INPUT"
