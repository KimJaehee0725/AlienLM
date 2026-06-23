#!/bin/bash

# Evaluate a model on standard LM benchmarks.
# Benchmarks: MMLU, ARC-easy, ARC-challenge, HellaSwag, Winogrande, TruthfulQA, GSM8K.

set -e

MODEL_PATH=""
PEFT_PATH=""
DEVICE="0"
OUTPUT_DIR=""
BATCH_SIZE=8
USE_CHAT_TEMPLATE=false
SYSTEM_INSTRUCTION=""
RUN_CODE_EVAL=false
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.9

TASKS=("mmlu" "arc_easy" "arc_challenge" "hellaswag" "winogrande" "truthfulqa_mc1" "gsm8k_cot")
FEW_SHOTS=(5 25 25 10 5 0 5)

usage() {
    echo "Usage: $0 --model_path <path> --output_dir <path> [options]"
    echo
    echo "Options:"
    echo "  --model_path <path>          [required] Base model path"
    echo "  --output_dir <path>          [required] Output directory"
    echo "  --peft_path <path>           [optional] PEFT adapter path"
    echo "  --device <gpu_ids>           [optional] GPU ids (comma-separated). Default: '0'"
    echo "  --batch_size <num>           [optional] Batch size. Default: 8"
    echo "  --use_chat_template          [optional] Apply chat template + fewshot as multiturn"
    echo "  --system_instruction <str>   [optional] System instruction string (use \n for newlines)"
    echo "  --run_code_eval              [optional] Run code evaluation"
    echo "  --tensor_parallel_size <num> [optional] vLLM tensor parallel size. Default: 1"
    echo "  --gpu_memory_utilization <f> [optional] vLLM GPU memory utilization. Default: 0.9"
    exit 1
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model_path) MODEL_PATH="$2"; shift ;;
        --peft_path) PEFT_PATH="$2"; shift ;;
        --device) DEVICE="$2"; shift ;;
        --output_dir) OUTPUT_DIR="$2"; shift ;;
        --batch_size) BATCH_SIZE="$2"; shift ;;
        --use_chat_template) USE_CHAT_TEMPLATE=true ;;
        --system_instruction) SYSTEM_INSTRUCTION="$2"; shift ;;
        --run_code_eval) RUN_CODE_EVAL=true ;;
        --tensor_parallel_size) TENSOR_PARALLEL_SIZE="$2"; shift ;;
        --gpu_memory_utilization) GPU_MEMORY_UTILIZATION="$2"; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown parameter: $1"; usage ;;
    esac
    shift
done

if [ -z "$MODEL_PATH" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Error: --model_path and --output_dir are required."
    usage
fi

DEFAULT_CACHE_ROOT=${HF_HOME:-"$PWD/.cache"}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-"$DEFAULT_CACHE_ROOT/hf_datasets"}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-"$DEFAULT_CACHE_ROOT/hf_models"}
export CUDA_VISIBLE_DEVICES=$DEVICE

NUM_GPUS=$(echo $DEVICE | awk -F, '{print NF}')

MODEL_ARGS="pretrained=$MODEL_PATH,trust_remote_code=True,dtype=bfloat16"
if [ -n "$PEFT_PATH" ]; then
    MODEL_ARGS="$MODEL_ARGS,peft=$PEFT_PATH"
fi

for i in "${!TASKS[@]}"; do
    TASK_NAME=${TASKS[i]}
    NUM_FEWSHOT=${FEW_SHOTS[i]}

    LAUNCH_ARGS=(
        --main_process_port 29610
        --num_processes "$NUM_GPUS"
        -m lm_eval --model hf
        --model_args "$MODEL_ARGS"
        --tasks "$TASK_NAME"
        --num_fewshot "$NUM_FEWSHOT"
        --batch_size "$BATCH_SIZE"
        --output_path "$OUTPUT_DIR/$TASK_NAME/${NUM_FEWSHOT}-shot"
        --log_samples
    )

    if [ "$USE_CHAT_TEMPLATE" = true ] ; then
        LAUNCH_ARGS+=(--apply_chat_template --fewshot_as_multiturn)
    fi

    if [ -n "$SYSTEM_INSTRUCTION" ]; then
        PROCESSED_INSTRUCTION=$(printf '%b' "$SYSTEM_INSTRUCTION")
        LAUNCH_ARGS+=(--system_instruction "$PROCESSED_INSTRUCTION")
    fi

    echo "========================================================================="
    echo "Starting evaluation for task: $TASK_NAME ($NUM_FEWSHOT-shot)"
    echo "========================================================================="
    echo "Model: $MODEL_PATH"
    [ -n "$PEFT_PATH" ] && echo "PEFT adapter: $PEFT_PATH"
    echo "Device(s): $DEVICE ($NUM_GPUS GPUs)"
    echo "Batch size: $BATCH_SIZE"
    echo "Output path: $OUTPUT_DIR/$TASK_NAME/${NUM_FEWSHOT}-shot"
    echo "Using chat template: $USE_CHAT_TEMPLATE"
    [ -n "$SYSTEM_INSTRUCTION" ] && echo -e "System instruction:\n$SYSTEM_INSTRUCTION"
    echo "-------------------------------------------------------------------------"

    accelerate launch "${LAUNCH_ARGS[@]}"

    echo "-------------------------------------------------------------------------"
    echo "Finished evaluation for task: $TASK_NAME"
    echo "Results saved in $OUTPUT_DIR/$TASK_NAME/${NUM_FEWSHOT}-shot"
    echo "========================================================================="
    echo ""
done

if [ "$RUN_CODE_EVAL" = true ]; then
    CODE_EVAL_ARGS=(
        --model_path "$MODEL_PATH"
        --output_dir "$OUTPUT_DIR/code"
        --tensor_parallel_size "$TENSOR_PARALLEL_SIZE"
    )

    if [ -n "$PEFT_PATH" ]; then
        echo "Warning: PEFT_PATH is set, but evaluate_code_evalplus.sh may not use it directly."
        echo "Ensure MODEL_PATH points to a PEFT-merged model if PEFT is intended."
    fi

    echo "========================================================================="
    echo "Starting code evaluation"
    echo "========================================================================="
    echo "Model: $MODEL_PATH"
    [ -n "$PEFT_PATH" ] && echo "PEFT adapter: $PEFT_PATH"
    echo "Output path: $OUTPUT_DIR/code"
    echo "Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
    echo "-------------------------------------------------------------------------"

    bash "$(dirname "$0")/evaluate_code_evalplus.sh" "${CODE_EVAL_ARGS[@]}"

    echo "-------------------------------------------------------------------------"
    echo "Finished code evaluation"
    echo "Results saved in $OUTPUT_DIR/code"
    echo "========================================================================="
fi

