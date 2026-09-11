#!/usr/bin/env bash
set -euo pipefail

mode=${1:?usage: run_revision_server.sh smoke|pilot|full OUTPUT_ROOT}
output_root=${2:?usage: run_revision_server.sh smoke|pilot|full OUTPUT_ROOT}
repo_dir=$(cd "$(dirname "$0")/.." && pwd)
python_bin=${REVISION_PYTHON:-$repo_dir/.venv-revision/bin/python}

if [[ $mode != smoke && $mode != pilot && $mode != full ]]; then
  echo "mode must be smoke, pilot, or full" >&2
  exit 2
fi

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TOKENIZERS_PARALLELISM=false
mkdir -p "$output_root/logs"
cd "$repo_dir"

all_models=(llama32_3b qwen25_7b llama31_8b gemma2_9b ministral_3_8b gemma3_12b)
models=("${all_models[@]}")
if [[ -n ${REVISION_MODELS:-} ]]; then
  read -r -a models <<< "$REVISION_MODELS"
fi

batch_size() {
  case "$1" in
    llama32_3b) echo 48 ;;
    qwen25_7b|llama31_8b) echo 24 ;;
    gemma2_9b|ministral_3_8b) echo 16 ;;
    gemma3_12b) echo 10 ;;
  esac
}

model_path() {
  case "$1" in
    gemma2_9b) echo "${GEMMA2_9B_WEIGHTS:-}" ;;
    gemma3_12b) echo "${GEMMA3_12B_WEIGHTS:-}" ;;
    llama31_8b) echo "${LLAMA31_8B_WEIGHTS:-}" ;;
    llama32_3b) echo "${LLAMA32_3B_WEIGHTS:-}" ;;
    qwen25_7b) echo "${QWEN25_7B_WEIGHTS:-}" ;;
    ministral_3_8b) echo "${MINISTRAL_3_8B_WEIGHTS:-}" ;;
  esac
}

run_model() {
  local model=$1
  local out=$output_root/$model
  local weights
  weights=$(model_path "$model")
  local weight_args=()
  if [[ -n $weights ]]; then
    weight_args=(--model-name "$weights")
  fi

  local grid_args=(--coeffs 2 5 8 12 20 --max-multilayer-groups 1)
  if [[ $mode == smoke ]]; then
    grid_args=(--coeffs 5 --max-multilayer-groups 0)
  fi
  local common=(
    --model "$model"
    --out-dir "$out"
    --batch-size "$(batch_size "$model")"
    "${grid_args[@]}"
    --random-directions 1
    --selection-max-new-tokens 8
    --test-max-new-tokens 12
    --selection-checkpoint-every 20
    "${weight_args[@]}"
  )

  if [[ $mode == smoke ]]; then
    "$python_bin" experiments/run_revision_experiments.py \
      "${common[@]}" \
      --split-seeds 101 --counts 6 6 6 \
      --extraction-repeats 2 --selection-repeats 1 --test-repeats 1 \
      --sae-features 512 --sae-topk 16 --sae-selected 16 --sae-epochs 2
  elif [[ $mode == pilot ]]; then
    "$python_bin" experiments/run_revision_experiments.py \
      "${common[@]}" \
      --split-seeds 101 --counts 6 6 6 \
      --extraction-repeats 2 --selection-repeats 1 --test-repeats 1 \
      --sae-features 512 --sae-topk 16 --sae-selected 16 --sae-epochs 10
  else
    "$python_bin" experiments/run_revision_experiments.py \
      "${common[@]}" \
      --split-seeds 101 202 303 404 505 --counts 36 12 60 \
      --extraction-repeats 3 --selection-repeats 1 --test-repeats 3 \
      --sae-features 512 --sae-topk 16 --sae-selected 16 --sae-epochs 100
  fi

  "$python_bin" experiments/audit_revision_run.py --run-dirs "$out"

  if [[ $mode == full ]]; then
    "$python_bin" experiments/run_revision_validation.py \
      --run-dir "$out" "${weight_args[@]}" \
      --decision-max-new-tokens 12 --multiturn-max-new-tokens 8 \
      --capability-max-new-tokens 32 --explanation-max-new-tokens 96
    "$python_bin" experiments/audit_revision_run.py --run-dirs "$out"
  fi
}

if [[ $mode == pilot ]]; then
  # Exercise the smallest and largest architectures before committing to the
  # full six-model run.  Pilot outputs are never used as paper evidence.
  models=(llama32_3b gemma3_12b)
fi

for model in "${models[@]}"; do
  start=$(date +%s)
  echo "[$(date -Is)] starting $mode $model"
  run_model "$model" 2>&1 | tee "$output_root/logs/${mode}_${model}.log"
  end=$(date +%s)
  echo "[$(date -Is)] completed $mode $model in $((end-start)) s" | \
    tee -a "$output_root/logs/${mode}_${model}.log"
done

if [[ $mode == full ]]; then
  run_dirs=()
  # A resumed run may execute only a subset, but the final audit and analyses
  # must always cover the complete six-model suite.
  for model in "${all_models[@]}"; do run_dirs+=("$output_root/$model"); done
  "$python_bin" experiments/audit_revision_run.py \
    --run-dirs "${run_dirs[@]}" --deep --out "$output_root/integrity_audit.json"
  "$python_bin" experiments/analyze_revision.py \
    --run-dirs "${run_dirs[@]}" --out-dir "$output_root/analysis" --resamples 10000
  "$python_bin" experiments/analyze_revision_validation.py \
    --run-dirs "${run_dirs[@]}" --out "$output_root/validation_statistics.json" \
    --resamples 10000
fi
