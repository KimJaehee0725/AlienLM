#!/bin/bash

set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)

DEFAULT_CACHE_ROOT=${HF_HOME:-"$REPO_DIR/.cache"}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-"$DEFAULT_CACHE_ROOT/hf_datasets"}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-"$DEFAULT_CACHE_ROOT/hf_models"}

MODEL_PATH="${MODEL_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/eval/runs/lm_eval}"
DEVICE="${DEVICE:-0,1}"
BATCH_SIZE="${BATCH_SIZE:-8}"
USE_CHAT_TEMPLATE="${USE_CHAT_TEMPLATE:-true}"
TOKENIZER_PATH="${TOKENIZER_PATH:-}"
PEFT_PATH="${PEFT_PATH:-}"
SKIP_TASKS="${SKIP_TASKS:-}"
SYSTEM_INSTRUCTION="${SYSTEM_INSTRUCTION:-}"

if [ -z "$MODEL_PATH" ]; then
  echo "Error: MODEL_PATH is required."
  echo "Example: MODEL_PATH=/path/to/model bash eval/scripts/run_lm_eval.sh"
  exit 1
fi

ARGS=(
  --model_path "$MODEL_PATH"
  --output_dir "$OUTPUT_DIR"
  --device "$DEVICE"
  --batch_size "$BATCH_SIZE"
)

if [ "$USE_CHAT_TEMPLATE" = "true" ]; then
  ARGS+=(--use_chat_template)
fi

if [ -n "$TOKENIZER_PATH" ]; then
  ARGS+=(--tokenizer_path "$TOKENIZER_PATH")
fi

if [ -n "$PEFT_PATH" ]; then
  ARGS+=(--peft_path "$PEFT_PATH")
fi

if [ -n "$SKIP_TASKS" ]; then
  ARGS+=(--skip_tasks "$SKIP_TASKS")
fi

if [ -n "$SYSTEM_INSTRUCTION" ]; then
  ARGS+=(--system_instruction "$SYSTEM_INSTRUCTION")
fi

bash "$SCRIPT_DIR/utils/evaluate_all.sh" "${ARGS[@]}"
