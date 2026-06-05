#!/bin/bash
# Florence-2 LoRA fine-tuning v2 — CCTV garbage detection
# Improvements vs v1: hard negatives, color augmentation, partial DaViT unfreeze,
# expanded LoRA targets, longer cosine schedule, higher LR.
#
# Run inside SLURM:
#   srun -p cse-gpu-all --nodelist=dgx-a100-02 --pty bash
#   bash train_florence2_cctv_v2.sh

set -e

PYTHON="/u/student/2024/cs24mtech11024/.conda/envs/Capstone/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TRAIN_JSONL="$SCRIPT_DIR/data/finetune_data/cctv_od_train_v2.jsonl"
VAL_JSONL="$SCRIPT_DIR/data/finetune_data/cctv_od_val_v2.jsonl"
OUTPUT_DIR="$SCRIPT_DIR/finetuned/florence2_cctv_garbage_od_lora_v2"

MODEL_ID="microsoft/Florence-2-large"
EPOCHS=40
BATCH=2
GRAD_ACCUM=8        # effective batch = 16
LR=5e-5
LORA_R=32
LORA_ALPHA=64
UNFREEZE_BLOCKS=1

# GPU auto-select (least memory used)
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "GPU memory usage:"
    nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader
    BEST_GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader \
               | awk -F',' '{gsub(/[^0-9]/, "", $2); print $2, $1}' \
               | sort -n | head -1 | awk '{print $2}')
    export CUDA_VISIBLE_DEVICES="$BEST_GPU"
    echo "Auto-selected GPU $BEST_GPU"
fi

echo "================================================"
echo "Florence-2 LoRA v2 — CCTV Garbage OD"
echo "  Model            : $MODEL_ID"
echo "  Epochs           : $EPOCHS"
echo "  Batch / grad-acc : $BATCH / $GRAD_ACCUM  (eff $((BATCH*GRAD_ACCUM)))"
echo "  LR               : $LR  (cosine)"
echo "  LoRA r/alpha     : $LORA_R / $LORA_ALPHA"
echo "  Unfreeze blocks  : $UNFREEZE_BLOCKS (last DaViT)"
echo "  GPU              : $CUDA_VISIBLE_DEVICES"
echo "  Output           : $OUTPUT_DIR"
echo "================================================"

# Ensure v2 JSONL exists
if [ ! -f "$TRAIN_JSONL" ] || [ ! -f "$VAL_JSONL" ]; then
    echo "Preparing v2 JSONL (with hard negatives)..."
    cd "$SCRIPT_DIR"
    $PYTHON prepare_cctv_florence2_od_jsonl_v2.py --split all --neg-ratio 0.5
fi

echo "Starting v2 fine-tuning..."
$PYTHON "$SCRIPT_DIR/train_florence2_cctv_v2.py" \
    --train-jsonl              "$TRAIN_JSONL" \
    --eval-jsonl               "$VAL_JSONL" \
    --output-dir               "$OUTPUT_DIR" \
    --model-id                 "$MODEL_ID" \
    --num-train-epochs         "$EPOCHS" \
    --per-device-train-batch-size "$BATCH" \
    --gradient-accumulation-steps "$GRAD_ACCUM" \
    --learning-rate            "$LR" \
    --lora-r                   "$LORA_R" \
    --lora-alpha               "$LORA_ALPHA" \
    --unfreeze-vision-blocks   "$UNFREEZE_BLOCKS" \
    --bf16

echo ""
echo "Fine-tuning done. Adapter: $OUTPUT_DIR"
echo "Next: bash run_florence2_v2_eval.sh"
