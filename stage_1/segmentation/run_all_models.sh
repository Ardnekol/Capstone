#!/bin/bash

# Run All Segmentation Models Sequentially
# Trains all models with 50 epochs each, then generates final report

set -e

cd /u/student/2024/cs24mtech11024/Capstone/stage_1/segmentation

echo "🚀 Starting Complete Segmentation Pipeline"
echo "=========================================="

# Run U-Net
echo ""
echo "1️⃣ Training U-Net (50 epochs)..."
python scripts/train_unet.py --epochs 50

# Run DeepLabV3+
echo ""
echo "2️⃣ Training DeepLabV3+ (50 epochs)..."
python scripts/train_deeplabv3plus.py --epochs 50

# Run Mask R-CNN
echo ""
echo "3️⃣ Training Mask R-CNN (50 epochs)..."
python scripts/train_maskrcnn.py --epochs 50

# Run SAM Evaluation
echo ""
echo "4️⃣ Evaluating SAM (zero-shot)..."
python scripts/eval_sam.py

# Generate Final Report
echo ""
echo "📊 Generating Comprehensive Report..."
python scripts/evaluate_segmentation.py

echo ""
echo "🎉 All Done! Results Summary:"
echo "📁 Results: Capstone/stage_1/segmentation/results/"
echo "📄 Report: SEGMENTATION_REPORT.md"
echo "📊 Plot: segmentation_iou_comparison.png"