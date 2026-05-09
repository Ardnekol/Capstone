#!/bin/bash
# Step 3b: Fine-tune Florence-2 with LoRA on CCTV garbage dataset
# Run inside SLURM: srun -p cse-gpu-all --nodelist=dgx-a100-02 --pty bash

set -e

PYTHON="/u/student/2024/cs24mtech11024/.conda/envs/Capstone/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE2_DIR="/u/student/2024/cs24mtech11024/Capstone/stage_2"
FINETUNE_SCRIPT="$STAGE2_DIR/finetune_florence2_od_lora.py"

TRAIN_JSONL="$SCRIPT_DIR/data/finetune_data/cctv_od_train.jsonl"
VAL_JSONL="$SCRIPT_DIR/data/finetune_data/cctv_od_val.jsonl"
OUTPUT_DIR="$SCRIPT_DIR/finetuned/florence2_cctv_garbage_od_lora"

MODEL_ID="microsoft/Florence-2-large"
EPOCHS=10
BATCH=4
LR=1e-5

# Auto-select GPU with least memory used, or use CUDA_VISIBLE_DEVICES if already set
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "GPU memory usage:"
    nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader
    BEST_GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader \
               | awk -F',' '{gsub(/[^0-9]/, "", $2); print $2, $1}' \
               | sort -n | head -1 | awk '{print $2}')
    export CUDA_VISIBLE_DEVICES="$BEST_GPU"
    echo "Auto-selected GPU $BEST_GPU (least memory used)"
fi

echo "================================================"
echo "Florence-2 LoRA Fine-Tuning — CCTV Garbage OD"
echo "  Model   : $MODEL_ID"
echo "  Epochs  : $EPOCHS"
echo "  Batch   : $BATCH"
echo "  LR      : $LR"
echo "  GPU     : $CUDA_VISIBLE_DEVICES"
echo "  Output  : $OUTPUT_DIR"
echo "================================================"

# Ensure JSONL data is prepared
if [ ! -f "$TRAIN_JSONL" ] || [ ! -f "$VAL_JSONL" ]; then
    echo "Preparing Florence-2 JSONL data..."
    cd "$SCRIPT_DIR"
    $PYTHON prepare_cctv_florence2_od_jsonl.py --split all
fi

echo "Starting fine-tuning..."
$PYTHON "$FINETUNE_SCRIPT" \
    --train-jsonl                  "$TRAIN_JSONL" \
    --eval-jsonl                   "$VAL_JSONL" \
    --output-dir                   "$OUTPUT_DIR" \
    --model-id                     "$MODEL_ID" \
    --num-train-epochs             "$EPOCHS" \
    --per-device-train-batch-size  "$BATCH" \
    --learning-rate                "$LR" \
    --lora-r                       16 \
    --lora-alpha                   32 \
    --bf16

echo ""
echo "Fine-tuning done. Adapter saved to: $OUTPUT_DIR"
echo "Next: bash run_finetuned_eval.sh"
