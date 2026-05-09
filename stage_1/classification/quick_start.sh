#!/bin/bash
#
# Classification Experiments Quick Start
# =======================================
#
# This script runs the complete classification experiment pipeline.
#
# Usage:
#   bash quick_start.sh           # Full run (50 epochs)
#   bash quick_start.sh --quick   # Quick test (5 epochs)
#

set -e  # Exit on error

cd "$(dirname "$0")"

echo "========================================"
echo "Waste Classification Experiment Pipeline"
echo "========================================"
echo ""

# Check for quick mode
if [ "$1" == "--quick" ]; then
    echo "Running in QUICK TEST mode (5 epochs)"
    EPOCHS=5
    BATCH_SIZE=16
else
    echo "Running FULL experiments (50 epochs)"
    EPOCHS=50
    BATCH_SIZE=32
fi

echo ""
echo "Step 1: Installing requirements..."
pip install -r requirements.txt

echo ""
echo "Step 2: Preprocessing datasets..."
bash scripts/preprocess.sh

echo ""
echo "Step 3: Training ResNet-50..."
python scripts/train_resnet50.py --epochs $EPOCHS --batch $BATCH_SIZE

echo ""
echo "Step 4: Training EfficientNet-B0..."
python scripts/train_efficientnetb0.py --epochs $EPOCHS --batch $BATCH_SIZE

echo ""
echo "Step 5: Evaluating CLIP zero-shot..."
python scripts/eval_clip.py

echo ""
echo "Step 6: Generating comparison report..."
python scripts/evaluate.py

echo ""
echo "🎉 All experiments complete!"
echo "📁 Results: results/"
echo "📊 Report: results/COMPARISON_REPORT.md"