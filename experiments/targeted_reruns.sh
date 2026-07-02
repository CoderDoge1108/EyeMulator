#!/usr/bin/env bash
# =============================================================================
# Targeted low-data reruns for selected (model, task) cells.
#
# The script waits for an available GPU, skips cells whose results already
# exist, and logs each job. Adjust the run_job lines at the bottom to target
# other cells or hyperparameters.
# =============================================================================
set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$ROOT"

TRAIN="${ROOT}/src/training/train_unified.py"
EVAL="${ROOT}/src/evaluation/eval_unified.py"
DATA="${DATA:-${ROOT}/corpus/data}"
WORK_BASE="${WORK_BASE:-${ROOT}/workspace/targeted_reruns}"
RESULT_BASE="${RESULT_BASE:-${ROOT}/results/targeted_reruns}"
LOG_DIR="${LOG_DIR:-${ROOT}/workspace/logs/targeted_reruns}"

mkdir -p "$WORK_BASE" "$RESULT_BASE" "$LOG_DIR"

if [ -z "${HF_TOKEN:-}" ]; then
  unset HF_TOKEN
fi
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

# GPUs to consider when searching for free memory.
GPUS="${GPUS:-0 1 2 3}"

timestamp() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

free_mem() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" | tr -d ' '
}

pick_gpu() {
  local min_free="$1" best_gpu="" best_free=0 gpu mem
  for gpu in $GPUS; do
    mem="$(free_mem "$gpu" || echo 0)"
    if [ "$mem" -ge "$min_free" ] && [ "$mem" -gt "$best_free" ]; then
      best_free="$mem"; best_gpu="$gpu"
    fi
  done
  [ -n "$best_gpu" ] && { echo "$best_gpu"; return 0; }
  return 1
}

wait_for_gpu() {
  local min_free="$1" gpu=""
  while true; do
    if gpu="$(pick_gpu "$min_free")"; then echo "$gpu"; return 0; fi
    echo "[$(timestamp)] Waiting for a GPU with >=${min_free} MiB free..."
    sleep 300
  done
}

run_job() {
  local tag="$1" model="$2" task="$3" max_samples="$4" epochs="$5" lr="$6"
  local lora_r="$7" lora_alpha="$8" batch="$9" grad="${10}" min_free="${11}"

  local work_root="${WORK_BASE}/${tag}"
  local result_root="${RESULT_BASE}/${tag}"
  local model_dir="${work_root}/${model}_eyemulator_${task}_seed42"
  local result_file="${result_root}/${model}_results/generated_results_eyemulator_${task}_seed42.json"
  local log_file="${LOG_DIR}/${tag}.log"
  local gpu

  if [ -f "$result_file" ]; then
    echo "[$(timestamp)] SKIP ${tag}: result exists" | tee -a "$log_file"; return 0
  fi

  gpu="$(wait_for_gpu "$min_free")"
  echo "[$(timestamp)] START ${tag} on GPU ${gpu} (model=${model} task=${task} epochs=${epochs} lr=${lr} r=${lora_r} samples=${max_samples})" | tee -a "$log_file"

  if [ ! -f "${model_dir}/adapter_config.json" ]; then
    CUDA_VISIBLE_DEVICES="$gpu" python "$TRAIN" \
      --model "$model" --task "$task" --method eyemulator --seed 42 \
      --data-folder "$DATA" --output-root "$work_root" \
      --epochs "$epochs" --batch-size "$batch" --grad-accum "$grad" --lr "$lr" \
      --max-length 1024 --lora-r "$lora_r" --lora-alpha "$lora_alpha" \
      --max-samples "$max_samples" --no-eval-during-training --device cuda:0 \
      2>&1 | tee -a "$log_file"
    train_status=${PIPESTATUS[0]}
    if [ "$train_status" -ne 0 ]; then
      echo "[$(timestamp)] FAIL train ${tag} exit=${train_status}" | tee -a "$log_file"; return "$train_status"
    fi
  else
    echo "[$(timestamp)] SKIP train ${tag}: adapter exists" | tee -a "$log_file"
  fi

  gpu="$(wait_for_gpu "$min_free")"
  CUDA_VISIBLE_DEVICES="$gpu" python "$EVAL" \
    --model "$model" --task "$task" --method eyemulator --seed 42 \
    --data-folder "$DATA" --model-root "$work_root" --output-root "$result_root" \
    --device cuda:0 --batch-size 1 --max-new-tokens 512 --use-test-set \
    2>&1 | tee -a "$log_file"
  eval_status=${PIPESTATUS[0]}
  if [ "$eval_status" -ne 0 ]; then
    echo "[$(timestamp)] FAIL eval ${tag} exit=${eval_status}" | tee -a "$log_file"; return "$eval_status"
  fi
  echo "[$(timestamp)] DONE ${tag}: ${result_file}" | tee -a "$log_file"
}

FAILED=0
# tag  model  task  max_samples  epochs  lr  lora_r  lora_alpha  batch  grad  min_free_MiB
run_job "low_starcoder_sum_r16_lr1e4_e5" "starcoder"  "summarization" 200 5 1e-4 16 32 2  8 12000 || FAILED=$((FAILED + 1))
run_job "low_starcoder_sum_r32_lr1e4_e5" "starcoder"  "summarization" 200 5 1e-4 32 64 2  8 12000 || FAILED=$((FAILED + 1))
run_job "low_deepseek_sum_r32_lr5e5_e5"  "deepseek"   "summarization" 200 5 5e-5 32 64 2  8 12000 || FAILED=$((FAILED + 1))
run_job "low_qwen_completion_r64_lr5e5"  "qwen35-2b"  "completion"    200 5 5e-5 64 128 1 16 16000 || FAILED=$((FAILED + 1))
run_job "low_smollm_completion_r64_lr5e5" "smollm3-3b" "completion"   200 5 5e-5 64 128 1 16 16000 || FAILED=$((FAILED + 1))

echo "[$(timestamp)] TARGETED RERUNS COMPLETE failed=${FAILED}"
exit "$FAILED"
