#!/bin/bash

# Code generation evaluation using lm-evaluation-harness + vLLM.
# Benchmarks: MBPP, HumanEval-Instruct, HumanEval-64-Instruct.

set -e

MODEL_PATH=""
PEFT_PATH=""
OUTPUT_DIR=""
DEVICE="0"
BATCH_SIZE=16
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.9

TASKS=("mbpp" "humaneval_instruct" "humaneval_64_instruct")
FEW_SHOTS=(3 3 3)

usage() {
    echo "Usage: $0 --model_path <path> --output_dir <path> [options]"
    echo
    echo "Options:"
    echo "  --model_path <path>          [required] Model path (HF name or local path)."
    echo "  --output_dir <path>          [required] Output directory."
    echo "  --peft_path <path>           [optional] PEFT adapter path."
    echo "  --device <gpu_id>            [optional] GPU id. Default: '0'."
    echo "  --batch_size <num>           [optional] Batch size. Default: 16."
    echo "  --tensor_parallel_size <num> [optional] vLLM tensor parallel size. Default: 1."
    echo "  --gpu_memory_utilization <f> [optional] vLLM GPU memory utilization (0.0~1.0). Default: 0.9."
    echo "  -h|--help                    Show this message."
    exit 1
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model_path) MODEL_PATH="$2"; shift ;;
        --peft_path) PEFT_PATH="$2"; shift ;;
        --output_dir) OUTPUT_DIR="$2"; shift ;;
        --device) DEVICE="$2"; shift ;;
        --batch_size) BATCH_SIZE="$2"; shift ;;
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

export HF_ALLOW_CODE_EVAL="1"
if [ "$DEVICE" = "cpu" ]; then
    export CUDA_VISIBLE_DEVICES=""
    LM_EVAL_DEVICE="cpu"
else
    export CUDA_VISIBLE_DEVICES=$DEVICE
    LM_EVAL_DEVICE="cuda:0"
fi

MODEL_ARGS="pretrained=$MODEL_PATH,trust_remote_code=True,dtype=bfloat16,tensor_parallel_size=$TENSOR_PARALLEL_SIZE,gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"
if [ -n "$PEFT_PATH" ]; then
    MODEL_ARGS="$MODEL_ARGS,peft=$PEFT_PATH"
fi

for i in "${!TASKS[@]}"; do
    TASK_NAME=${TASKS[i]}
    NUM_FEWSHOT=${FEW_SHOTS[i]}

    SHOT_STRING="3-shot"
    OUTPUT_PATH="$OUTPUT_DIR/$TASK_NAME/${SHOT_STRING}-vllm"
    mkdir -p "$(dirname "$OUTPUT_PATH")"

    LAUNCH_ARGS=(
        --model vllm
        --model_args "$MODEL_ARGS"
        --tasks "$TASK_NAME"
        --device "$LM_EVAL_DEVICE"
        --batch_size "$BATCH_SIZE"
        --output_path "$OUTPUT_PATH"
        --log_samples
        --confirm_run_unsafe_code
        --apply_chat_template
        --fewshot_as_multiturn
    )

    if [ "$NUM_FEWSHOT" -ge 0 ]; then
        LAUNCH_ARGS+=(--num_fewshot "$NUM_FEWSHOT")
    fi

    SHOT_INFO=""
    if [ "$NUM_FEWSHOT" -ge 0 ]; then
        SHOT_INFO="${NUM_FEWSHOT}-shot"
    else
        SHOT_INFO="default (mbpp uses 3-shot)"
    fi

    echo "========================================================================="
    echo "Starting task: $TASK_NAME ($SHOT_INFO)"
    echo "========================================================================="
    echo "Model: $MODEL_PATH"
    [ -n "$PEFT_PATH" ] && echo "PEFT adapter: $PEFT_PATH"
    echo "Device: $DEVICE"
    echo "lm_eval device: $LM_EVAL_DEVICE"
    echo "Batch size: $BATCH_SIZE"
    echo "Output path: $OUTPUT_PATH"
    echo "-------------------------------------------------------------------------"

    lm_eval "${LAUNCH_ARGS[@]}"

    echo "-------------------------------------------------------------------------"
    echo "Finished task: $TASK_NAME"
    echo "Results saved to: $OUTPUT_PATH"
    echo "========================================================================="
    echo ""
done

echo "All code evaluations finished: $MODEL_PATH"
