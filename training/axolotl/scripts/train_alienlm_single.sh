#!/bin/bash

set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/../../.." && pwd)

DEFAULT_CACHE_ROOT=${HF_HOME:-"$REPO_DIR/.cache"}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-"$DEFAULT_CACHE_ROOT/hf_datasets"}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-"$DEFAULT_CACHE_ROOT/hf_models"}

export PYTHONPATH="$REPO_DIR/training/axolotl/tokenizers:$PYTHONPATH"

CONFIG="$REPO_DIR/training/axolotl/configs/alienlm/llama3-8b-instruct-alienlm-single.yaml"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3"}
export CUDA_VISIBLE_DEVICES

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "REPO_DIR=$REPO_DIR"
  echo "PYTHONPATH=$PYTHONPATH"
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "axolotl train $CONFIG"
  exit 0
fi

axolotl train "$CONFIG"
