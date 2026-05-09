#!/bin/bash
#
# Run remaining detection experiments one at a time
# 
# YOLOv8 is already complete! Run remaining models separately.
#
# Usage:
#   bash run_remaining.sh fasterrcnn
#   bash run_remaining.sh retinanet
#   bash run_remaining.sh all
#

cd "$(dirname "$0")"

MODEL=${1:-"all"}
EPOCHS=50
BATCH_RCNN=2  # Reduced batch size to avoid OOM
DEVICE="cuda:0"

echo "============================================"
echo "Running: $MODEL"
echo "============================================"

if [ "$MODEL" == "fasterrcnn" ] || [ "$MODEL" == "all" ]; then
    echo ""
    echo "🏋️ Training Faster R-CNN (batch=$BATCH_RCNN)..."
    CUDA_VISIBLE_DEVICES=0 python train_fasterrcnn.py \
        --epochs $EPOCHS \
        --batch $BATCH_RCNN \
        --device $DEVICE
fi

if [ "$MODEL" == "retinanet" ] || [ "$MODEL" == "all" ]; then
    echo ""
    echo "🏋️ Training RetinaNet (batch=$BATCH_RCNN)..."
    CUDA_VISIBLE_DEVICES=0 python train_retinanet.py \
        --epochs $EPOCHS \
        --batch $BATCH_RCNN \
        --device $DEVICE
fi

if [ "$MODEL" == "compare" ] || [ "$MODEL" == "all" ]; then
    echo ""
    echo "📊 Generating comparison report..."
    python evaluate.py --compare
fi

echo ""
echo "✅ Done!"
