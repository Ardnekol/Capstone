#!/bin/bash
#
# Detection Experiments Quick Start
# ==================================
#
# This script runs the complete detection experiment pipeline.
# 
# Usage:
#   bash quick_start.sh           # Full run (50 epochs)
#   bash quick_start.sh --quick   # Quick test (5 epochs)
#

set -e  # Exit on error

cd "$(dirname "$0")"

echo "========================================"
echo "Object Detection Experiment Pipeline"
echo "========================================"
echo ""

# Check for quick mode
if [ "$1" == "--quick" ]; then
    echo "Running in QUICK TEST mode (5 epochs)"
    EPOCHS=5
    BATCH_YOLO=8
    BATCH_RCNN=2
else
    echo "Running FULL experiments (50 epochs)"
    EPOCHS=50
    BATCH_YOLO=16
    BATCH_RCNN=4
fi

echo ""
echo "Step 1: Installing requirements..."
pip install -r requirements.txt

echo ""
echo "Step 2: Converting annotations to YOLO format..."
python convert_annotations.py --all

echo ""
echo "Step 3: Training YOLOv8..."
python train_yolov8.py --epochs $EPOCHS --batch $BATCH_YOLO

echo ""
echo "Step 4: Training Faster R-CNN..."
python train_fasterrcnn.py --epochs $EPOCHS --batch $BATCH_RCNN

echo ""
echo "Step 5: Training RetinaNet..."
python train_retinanet.py --epochs $EPOCHS --batch $BATCH_RCNN

echo ""
echo "Step 6: Evaluating Grounding-DINO..."
python eval_grounding_dino.py

echo ""
echo "Step 7: Evaluating Florence-2..."
python eval_florence2.py

echo ""
echo "Step 8: Generating comparison report..."
python evaluate.py --compare

echo ""
echo "========================================"
echo "COMPLETE!"
echo "========================================"
echo "Results saved to: results/COMPARISON_REPORT.md"
echo ""
