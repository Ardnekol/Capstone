#!/bin/bash
# Evaluate Florence-2 v2 (zero-shot and fine-tuned) with size-conditional tile inference.
# Produces results in eval_results/florence2_{zeroshot|finetuned}_tiled_test_<ts>/

set -e

PYTHON="/u/student/2024/cs24mtech11024/.conda/envs/Capstone/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ADAPTER_DIR="$SCRIPT_DIR/finetuned/florence2_cctv_garbage_od_lora_v2"

MODEL_ID="microsoft/Florence-2-large"
TILE_MIN_SIDE=800
TILE_OVERLAP=0.2
NMS_IOU=0.5

# GPU auto-select
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    BEST_GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader \
               | awk -F',' '{gsub(/[^0-9]/, "", $2); print $2, $1}' \
               | sort -n | head -1 | awk '{print $2}')
    export CUDA_VISIBLE_DEVICES="$BEST_GPU"
    echo "Auto-selected GPU $BEST_GPU"
fi

echo "================================================"
echo "Florence-2 v2 tiled eval"
echo "  Model         : $MODEL_ID"
echo "  Adapter       : $ADAPTER_DIR"
echo "  Tile min side : $TILE_MIN_SIDE"
echo "  Tile overlap  : $TILE_OVERLAP"
echo "  NMS IoU       : $NMS_IOU"
echo "================================================"

cd "$SCRIPT_DIR"

# 1) Zero-shot tiled (baseline for tiling's effect alone)
echo ""
echo ">>> Florence-2 ZERO-SHOT (tiled)"
$PYTHON evaluate_florence2_cctv_tiled.py \
    --model-id      "$MODEL_ID" \
    --split         test \
    --tile-min-side "$TILE_MIN_SIDE" \
    --tile-overlap  "$TILE_OVERLAP" \
    --nms-iou       "$NMS_IOU"

# 2) Fine-tuned tiled (only if v2 adapter exists)
if [ -d "$ADAPTER_DIR" ]; then
    echo ""
    echo ">>> Florence-2 FINE-TUNED v2 (tiled)"
    $PYTHON evaluate_florence2_cctv_tiled.py \
        --model-id      "$MODEL_ID" \
        --adapter       "$ADAPTER_DIR" \
        --split         test \
        --tile-min-side "$TILE_MIN_SIDE" \
        --tile-overlap  "$TILE_OVERLAP" \
        --nms-iou       "$NMS_IOU"
else
    echo ""
    echo "WARNING: $ADAPTER_DIR not found — skipping fine-tuned eval."
    echo "Run train_florence2_cctv_v2.sh first."
fi

echo ""
echo "Done. Results in: $SCRIPT_DIR/eval_results/"
