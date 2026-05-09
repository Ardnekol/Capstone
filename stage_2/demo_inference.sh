#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PATH="$HOME/.conda/envs/Capstone/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1

echo "========================================================"
echo "Unified Florence-2 Inference Demo"
echo "========================================================"

SAMPLE_IMAGE=$(find ../datasets/detection/taco/TACO/data/batch_1 -name "*.jpg" -o -name "*.JPG" 2>/dev/null | head -1)
if [[ -z "$SAMPLE_IMAGE" ]]; then
    echo "ERROR: No TACO images found"
    exit 1
fi

echo "Sample image: $SAMPLE_IMAGE"
echo ""

MODEL_PATH="finetuned/florence2_unified_multitask_lora"
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    echo "Please run: bash train_unified.sh --skip-prep"
    exit 1
fi

echo "Model path: $MODEL_PATH"
echo ""

RUN_ID=$(date +%d_%m_%Y_%I_%M_%S_%p)
OUTPUT_DIR="./inference_outputs/$RUN_ID"
echo "Output folder: $OUTPUT_DIR"
echo ""

python3 inference_unified.py \
    --image "$SAMPLE_IMAGE" \
    --model-id "$MODEL_PATH" \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "✓ Demo complete! Check $OUTPUT_DIR/ for outputs"
