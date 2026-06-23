#!/bin/bash

# Evaluate a model on standard LM benchmarks.
# Benchmarks: MMLU, ARC-easy, ARC-challenge, HellaSwag, Winogrande, TruthfulQA, GSM8K.

set -e

MODEL_PATH=""
TOKENIZER_PATH=""
PEFT_PATH=""
DEVICE="0"
OUTPUT_DIR=""
BATCH_SIZE=8
USE_CHAT_TEMPLATE=false
SYSTEM_INSTRUCTION=""
SKIP_TASKS=""

TASKS=("mmlu" "arc_easy" "arc_challenge" "hellaswag" "winogrande" "truthfulqa_mc1" "gsm8k_cot")
FEW_SHOTS=(5 25 25 10 5 0 5)

usage() {
    echo "Usage: $0 --model_path <path> --output_dir <path> [options]"
    echo
    echo "Options:"
    echo "  --model_path <path>         [required] Base model path"
    echo "  --tokenizer_path <path>     [optional] Tokenizer path (if different from model)"
    echo "  --output_dir <path>         [required] Output directory"
    echo "  --peft_path <path>          [optional] PEFT adapter path"
    echo "  --device <gpu_ids>          [optional] GPU ids (comma-separated). Default: '0'"
    echo "  --batch_size <num>          [optional] Batch size. Default: 8"
    echo "  --use_chat_template         [optional] Apply chat template + fewshot as multiturn"
    echo "  --system_instruction <str>  [optional] System instruction string (use \n for newlines)"
    echo "  --skip_tasks <tasks>         [optional] Comma-separated tasks to skip"
    exit 1
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model_path) MODEL_PATH="$2"; shift ;;
        --tokenizer_path) TOKENIZER_PATH="$2"; shift ;;
        --peft_path) PEFT_PATH="$2"; shift ;;
        --device) DEVICE="$2"; shift ;;
        --output_dir) OUTPUT_DIR="$2"; shift ;;
        --batch_size) BATCH_SIZE="$2"; shift ;;
        --use_chat_template) USE_CHAT_TEMPLATE=true ;;
        --system_instruction) SYSTEM_INSTRUCTION="$2"; shift ;;
        --skip_tasks) SKIP_TASKS="$2"; shift ;;
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
if [ -n "$TOKENIZER_PATH" ]; then
    MODEL_ARGS="$MODEL_ARGS,tokenizer=$TOKENIZER_PATH"
fi
if [ -n "$PEFT_PATH" ]; then
    MODEL_ARGS="$MODEL_ARGS,peft=$PEFT_PATH"
fi

for i in "${!TASKS[@]}"; do
    TASK_NAME=${TASKS[i]}
    NUM_FEWSHOT=${FEW_SHOTS[i]}

    if [ -n "$SKIP_TASKS" ]; then
        SKIP_ARRAY=($(echo "$SKIP_TASKS" | tr ',' ' '))
        SKIP_THIS=false
        for skip_task in "${SKIP_ARRAY[@]}"; do
            if [ "$TASK_NAME" == "$skip_task" ]; then
                SKIP_THIS=true
                break
            fi
        done
        if [ "$SKIP_THIS" == true ]; then
            echo "Skipping task: $TASK_NAME"
            continue
        fi
    fi

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
    [ -n "$TOKENIZER_PATH" ] && echo "Tokenizer: $TOKENIZER_PATH"
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

echo "All evaluations finished."
