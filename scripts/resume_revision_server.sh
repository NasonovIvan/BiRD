#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "$0")/.." && pwd)
output_root=${1:-runs/revision_measured_20260909}
status_dir=$output_root/status
mkdir -p "$status_dir"
cd "$repo_dir"

on_error() {
  local code=$?
  printf '%s\n' "$(date -Is) resumed run failed (exit $code, line ${BASH_LINENO[0]})" \
    > "$status_dir/FAILED"
  exit "$code"
}
trap on_error ERR

printf '%s\n' "$(date -Is) resuming Gemma-2, Ministral, and Gemma-3" \
  > "$status_dir/supervisor.txt"
REVISION_MODELS="gemma2_9b ministral_3_8b gemma3_12b" \
  bash scripts/run_revision_server.sh full "$output_root"
printf '%s\n' "$(date -Is) full six-model run completed" \
  > "$status_dir/supervisor.txt"
touch "$status_dir/COMPLETE"
