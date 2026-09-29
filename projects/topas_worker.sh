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
START=$SECONDS
trap 'status=$?; printf "TOPAS worker finished: exit=%s elapsed=%ss\n" "$status" "$((SECONDS-START))"; exit "$status"' EXIT
source "$ENV_FILE"
cd -- "$BUNDLE"
command -v topas >/dev/null || { echo 'topas not found after loading environment' >&2; exit 127; }
topas "$INPUT"
