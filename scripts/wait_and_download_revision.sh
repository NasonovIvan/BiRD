#!/usr/bin/env bash
set -uo pipefail

repo_dir=$(cd "$(dirname "$0")/.." && pwd)
remote=user@195.208.16.1
port=49139
key=/Users/ivnasonov/.ssh/id_ed25519
remote_repo=/home/user/BiRD
run_name=revision_measured_20260909
local_run=$repo_dir/runs/$run_name
poll_seconds=${REVISION_POLL_SECONDS:-300}
python_bin=${REVISION_LOCAL_PYTHON:-/opt/homebrew/bin/python3}
ssh_args=(-i "$key" -p "$port" -o BatchMode=yes -o ConnectTimeout=20)

timestamp() {
  date '+%Y-%m-%dT%H:%M:%S%z'
}

mkdir -p "$local_run/status"
log=$local_run/status/local_download.log
printf '%s\n' "$(timestamp) local completion watcher started" >> "$log"

sync_results() {
  rsync -a --partial \
    -e "ssh -i $key -p $port -o BatchMode=yes -o ConnectTimeout=20" \
    "$remote:$remote_repo/runs/$run_name/" "$local_run/"
}

sync_completed_model() {
  local model=$1
  local marker=$local_run/status/DOWNLOADED_$model
  [[ -f $marker ]] && return 0
  if ssh "${ssh_args[@]}" "$remote" \
      "grep -q 'completed full $model' $remote_repo/runs/$run_name/logs/full_$model.log 2>/dev/null"; then
    printf '%s\n' "$(timestamp) $model complete; downloading model results" >> "$log"
    mkdir -p "$local_run/$model" "$local_run/logs"
    rsync -a --partial \
      -e "ssh -i $key -p $port -o BatchMode=yes -o ConnectTimeout=20" \
      "$remote:$remote_repo/runs/$run_name/$model/" "$local_run/$model/" >> "$log" 2>&1
    rsync -a --partial \
      -e "ssh -i $key -p $port -o BatchMode=yes -o ConnectTimeout=20" \
      "$remote:$remote_repo/runs/$run_name/logs/full_$model.log" "$local_run/logs/" >> "$log" 2>&1
    if "$python_bin" "$repo_dir/experiments/audit_revision_run.py" \
        --run-dirs "$local_run/$model" --deep \
        --out "$local_run/${model}_local_audit.json" >> "$log" 2>&1; then
      touch "$marker"
      printf '%s\n' "$(timestamp) $model downloaded and verified" >> "$log"
    else
      printf '%s\n' "$(timestamp) $model local audit failed; will retry" >> "$log"
    fi
  fi
}

while true; do
  for model in llama32_3b qwen25_7b llama31_8b gemma2_9b ministral_3_8b gemma3_12b; do
    sync_completed_model "$model"
  done
  if ssh "${ssh_args[@]}" "$remote" \
      "test -f $remote_repo/runs/$run_name/status/COMPLETE"; then
    printf '%s\n' "$(timestamp) remote run complete; starting final download" >> "$log"
    if sync_results >> "$log" 2>&1; then
      model_dirs=()
      for model in llama32_3b qwen25_7b llama31_8b gemma2_9b ministral_3_8b gemma3_12b; do
        model_dirs+=("$local_run/$model")
      done
      if "$python_bin" "$repo_dir/experiments/audit_revision_run.py" \
          --run-dirs "${model_dirs[@]}" --deep \
          --out "$local_run/local_integrity_audit.json" >> "$log" 2>&1; then
        touch "$local_run/status/DOWNLOADED_AND_VERIFIED"
        printf '%s\n' "$(timestamp) download and local deep audit complete" >> "$log"
        # launchctl submit creates a keepalive job. Remain idle after the one
        # requested download so launchd cannot restart it and repeat rsync.
        while true; do sleep 86400; done
      fi
      printf '%s\n' "$(timestamp) local audit failed; retrying after $poll_seconds seconds" >> "$log"
    else
      printf '%s\n' "$(timestamp) rsync failed; retrying after $poll_seconds seconds" >> "$log"
    fi
  elif ssh "${ssh_args[@]}" "$remote" \
      "test -f $remote_repo/runs/$run_name/status/FAILED"; then
    printf '%s\n' "$(timestamp) remote supervisor reported failure; downloading diagnostics" >> "$log"
    sync_results >> "$log" 2>&1 || true
    touch "$local_run/status/REMOTE_FAILED"
    while true; do sleep 86400; done
  fi
  sleep "$poll_seconds"
done
