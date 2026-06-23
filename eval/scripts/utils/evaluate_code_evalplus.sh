#!/bin/bash

# Run EvalPlus (MBPP/HumanEval) with vLLM backend.
run_evalplus() {
    local model_path=$1
    local dataset=$2
    local tp=${3:-1}
    local output_dir=$4
    local download_dir=$5

    echo "Running evaluation for model: $model_path on dataset: $dataset with tp=$tp"
    mkdir -p "$output_dir/$dataset"

    local args=(
        --model "$model_path"
        --dataset "$dataset"
        --backend vllm
        --greedy
        --tp "$tp"
        --samples "$output_dir/$dataset"
    )
    if [ -n "$download_dir" ]; then
        args+=(--download_dir "$download_dir")
    fi
    evalplus.evaluate "${args[@]}"
}

MODEL_PATH=""
OUTPUT_DIR=""
TENSOR_PARALLEL_SIZE=1

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model_path) MODEL_PATH="$2"; shift ;;
        --output_dir) OUTPUT_DIR="$2"; shift ;;
        --tensor_parallel_size) TENSOR_PARALLEL_SIZE="$2"; shift ;;
        *) shift ;;
    esac
    shift
done

if [ -z "$MODEL_PATH" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Error: --model_path and --output_dir are required."
    exit 1
fi

DEFAULT_CACHE_ROOT=${HF_HOME:-"$PWD/.cache"}
DOWNLOAD_DIR=${EVALPLUS_DOWNLOAD_DIR:-${TRANSFORMERS_CACHE:-"$DEFAULT_CACHE_ROOT/hf_models"}}

run_evalplus "$MODEL_PATH" "humaneval" "$TENSOR_PARALLEL_SIZE" "$OUTPUT_DIR" "$DOWNLOAD_DIR"
run_evalplus "$MODEL_PATH" "mbpp" "$TENSOR_PARALLEL_SIZE" "$OUTPUT_DIR" "$DOWNLOAD_DIR"
