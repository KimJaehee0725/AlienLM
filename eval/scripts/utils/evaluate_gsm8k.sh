#!/bin/bash

set -e

MODEL_PATH=""
PEFT_PATH=""
DEVICE="0"
OUTPUT_DIR=""
FEW_SHOT=5
USE_CHAT_TEMPLATE=false
SYSTEM_INSTRUCTION=""

usage() {
    echo "Usage: $0 --model_path <path> --output_dir <path> [options]"
    echo
    echo "Options:"
    echo "  --model_path          [required] Base model path"
    echo "  --output_dir          [required] Output directory"
    echo "  --peft_path           [optional] PEFT adapter path"
    echo "  --device              [optional] GPU ids (comma-separated). Default: '0'"
    echo "  --few_shot            [optional] Few-shot examples. Default: 5"
    echo "  --use_chat_template   [optional] Apply chat template + fewshot as multiturn"
    echo "  --system_instruction  [optional] System instruction string"
    exit 1
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model_path) MODEL_PATH="$2"; shift ;;
        --peft_path) PEFT_PATH="$2"; shift ;;
        --device) DEVICE="$2"; shift ;;
        --output_dir) OUTPUT_DIR="$2"; shift ;;
        --few_shot) FEW_SHOT="$2"; shift ;;
        --use_chat_template) USE_CHAT_TEMPLATE=true ;;
        --system_instruction) SYSTEM_INSTRUCTION="$2"; shift ;;
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

if [ "$DEVICE" = "cpu" ]; then
    export CUDA_VISIBLE_DEVICES=""
    NUM_PROCESSES=1
    DEVICE_LABEL="cpu"
else
    export CUDA_VISIBLE_DEVICES=$DEVICE
    NUM_PROCESSES=$(echo $DEVICE | awk -F, '{print NF}')
    DEVICE_LABEL="$DEVICE ($NUM_PROCESSES GPUs)"
fi
TASK_NAMES=gsm8k_cot

MODEL_ARGS="pretrained=$MODEL_PATH,trust_remote_code=True,dtype=bfloat16"
if [ -n "$PEFT_PATH" ]; then
    MODEL_ARGS="$MODEL_ARGS,peft=$PEFT_PATH"
fi

CHAT_TEMPLATE_ARGS=""
if [ "$USE_CHAT_TEMPLATE" = true ] ; then
    CHAT_TEMPLATE_ARGS="--apply_chat_template --fewshot_as_multiturn"
fi

if [ -n "$SYSTEM_INSTRUCTION" ]; then
    SYSTEM_INSTRUCTION_ARGS="--system_instruction \"$SYSTEM_INSTRUCTION\""
fi

mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "Starting GSM8K-CoT evaluation"
echo "========================================"
echo "Model: $MODEL_PATH"
[ -n "$PEFT_PATH" ] && echo "PEFT adapter: $PEFT_PATH"
echo "Devices: $DEVICE_LABEL"
echo "Num processes: $NUM_PROCESSES"
echo "Few-shot: $FEW_SHOT"
echo "Output Dir: $OUTPUT_DIR"
echo "Use chat template: $USE_CHAT_TEMPLATE"
echo "System instruction: $SYSTEM_INSTRUCTION"
[ "$USE_CHAT_TEMPLATE" = true ] && echo "Chat template args: $CHAT_TEMPLATE_ARGS"
echo "Model args: $MODEL_ARGS"
echo "========================================"

accelerate launch --main_process_port 29610 --num_processes "$NUM_PROCESSES" -m lm_eval --model hf \
    --model_args "$MODEL_ARGS" \
    --tasks $TASK_NAMES \
    --batch_size 32 \
    --num_fewshot "$FEW_SHOT" \
    --output_path "$OUTPUT_DIR/$TASK_NAMES/${FEW_SHOT}-shot" \
    --log_samples \
    $CHAT_TEMPLATE_ARGS \
    $SYSTEM_INSTRUCTION_ARGS

echo "========================================"
echo "Evaluation finished."
echo "Results saved in $OUTPUT_DIR/$TASK_NAMES/${FEW_SHOT}-shot"
echo "========================================"
