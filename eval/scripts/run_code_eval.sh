#!/bin/bash

set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)

DEFAULT_CACHE_ROOT=${HF_HOME:-"$REPO_DIR/.cache"}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-"$DEFAULT_CACHE_ROOT/hf_datasets"}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-"$DEFAULT_CACHE_ROOT/hf_models"}

MODEL_PATH="${MODEL_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/eval/runs/code_eval}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"

if [ -z "$MODEL_PATH" ]; then
  echo "Error: MODEL_PATH is required."
  echo "Example: MODEL_PATH=/path/to/model bash eval/scripts/run_code_eval.sh"
  exit 1
fi

bash "$SCRIPT_DIR/utils/evaluate_code_evalplus.sh" \
  --model_path "$MODEL_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --tensor_parallel_size "$TENSOR_PARALLEL_SIZE"
