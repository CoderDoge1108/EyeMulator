#!/usr/bin/env bash
# =============================================================================
# EyeMulator - full experiment grid driver
#
# Drives the unified training + evaluation scripts across the
# (method x model x task) grid used for the main results (RQ1/RQ2/RQ4).
#
# Usage:
#   ./experiments/run_experiments.sh                                  # full grid
#   MODELS="llama" TASKS="completion" ./experiments/run_experiments.sh  # subset
#   METHODS="baseline eyemulator" ./experiments/run_experiments.sh
#
# Override any variable via the environment. Trained adapters land under
# workspace/, generated predictions under results/, both created on demand.
# =============================================================================
set -euo pipefail

# Resolve repo root (this script lives in <repo>/experiments/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$ROOT"

TRAIN="${ROOT}/src/training/train_unified.py"
EVAL="${ROOT}/src/evaluation/eval_unified.py"
DATA="${DATA:-${ROOT}/corpus/data}"
WORK="${WORK:-${ROOT}/workspace}"
RESULTS="${RESULTS:-${ROOT}/results}"
LOGS="${LOGS:-${ROOT}/workspace/logs}"

mkdir -p "$WORK" "$RESULTS" "$LOGS"

# Grid (override via env). Model keys must match train_unified.py --model choices.
MODELS="${MODELS:-llama deepseek starcoder qwen35-2b smollm3-3b}"
TASKS="${TASKS:-completion translation summarization}"
METHODS="${METHODS:-baseline eyemulator}"
SEED="${SEED:-42}"

# Training hyperparameters (defaults match the reported full-data runs).
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-1e-4}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
# Low-data setting: set MAX_SAMPLES to e.g. 200; empty means use all training data.
MAX_SAMPLES="${MAX_SAMPLES:-}"
DEVICE="${DEVICE:-cuda:0}"

# HuggingFace token: unset if empty to avoid an illegal empty Bearer header.
if [ -z "${HF_TOKEN:-}" ]; then
  unset HF_TOKEN
fi
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

max_samples_flag=()
if [ -n "$MAX_SAMPLES" ]; then
  max_samples_flag=(--max-samples "$MAX_SAMPLES")
fi

echo "=== EyeMulator experiment grid ==="
echo " models : $MODELS"
echo " tasks  : $TASKS"
echo " methods: $METHODS"
echo " seed   : $SEED   samples: ${MAX_SAMPLES:-all}"
echo "=================================="

for model in $MODELS; do
  for task in $TASKS; do
    for method in $METHODS; do
      tag="${model}_${method}_${task}_seed${SEED}"
      adapter_dir="${WORK}/${model}_${method}_${task}_seed${SEED}"
      result_file="${RESULTS}/${model}_results/generated_results_${method}_${task}_seed${SEED}.json"
      log="${LOGS}/${tag}.log"

      if [ -f "$result_file" ]; then
        echo "[SKIP] $tag (result exists)"
        continue
      fi

      echo "[TRAIN] $tag"
      python "$TRAIN" \
        --model "$model" \
        --task "$task" \
        --method "$method" \
        --seed "$SEED" \
        --data-folder "$DATA" \
        --output-root "$WORK" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --grad-accum "$GRAD_ACCUM" \
        --lr "$LR" \
        --max-length "$MAX_LENGTH" \
        --lora-r "$LORA_R" \
        --lora-alpha "$LORA_ALPHA" \
        --no-eval-during-training \
        --device "$DEVICE" \
        "${max_samples_flag[@]}" \
        2>&1 | tee "$log"

      echo "[EVAL] $tag"
      python "$EVAL" \
        --model "$model" \
        --task "$task" \
        --method "$method" \
        --seed "$SEED" \
        --data-folder "$DATA" \
        --model-root "$WORK" \
        --output-root "$RESULTS" \
        --device "$DEVICE" \
        --batch-size 1 \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --use-test-set \
        2>&1 | tee -a "$log"
    done
  done
done

echo "=== grid complete ==="
echo "predictions: $RESULTS"
echo "next: score with  python src/compute_metrics.py  (see README)"
