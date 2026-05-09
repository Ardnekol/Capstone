#!/bin/bash

# Segmentation Model Runner
# Usage: bash run_one_model.sh <model_name> [epochs] [gpu]
# Models: unet, deeplabv3plus, maskrcnn, sam
# Default epochs: 50, Default GPU: 7

set -e

MODEL_NAME=$1
EPOCHS=${2:-50}  # Default to 50 epochs if not specified
GPU=${3:-7}      # Default to GPU 7 if not specified

if [ -z "$MODEL_NAME" ]; then
    echo "❌ Error: Please specify a model name"
    echo "Usage: bash run_one_model.sh <model_name> [epochs] [gpu]"
    echo "Available models: unet, deeplabv3plus, maskrcnn, sam"
    echo "Default epochs: 50, Default GPU: 7"
    exit 1
fi

cd /u/student/2024/cs24mtech11024/Capstone/stage_1/segmentation

echo "🚀 Running segmentation model: $MODEL_NAME (${EPOCHS} epochs) on GPU $GPU"
echo "Working directory: $(pwd)"

case $MODEL_NAME in
    "unet")
        echo "🔄 Training U-Net model (${EPOCHS} epochs) on GPU ${GPU}..."
        CUDA_VISIBLE_DEVICES=$GPU python scripts/train_unet.py --epochs $EPOCHS --batch 8
        ;;
    "deeplabv3plus")
        echo "🔄 Training DeepLabV3+ model (${EPOCHS} epochs) on GPU ${GPU}..."
        CUDA_VISIBLE_DEVICES=$GPU python scripts/train_deeplabv3plus.py --epochs $EPOCHS --batch 8
        ;;
    "maskrcnn")
        echo "🔄 Training Mask R-CNN model (${EPOCHS} epochs) on GPU ${GPU}..."
        CUDA_VISIBLE_DEVICES=$GPU python scripts/train_maskrcnn.py --epochs $EPOCHS --batch 2
        ;;
    "sam")
        echo "🔄 Evaluating SAM (zero-shot) on GPU ${GPU}..."
        CUDA_VISIBLE_DEVICES=$GPU python scripts/eval_sam.py
        ;;
    *)
        echo "❌ Error: Unknown model '$MODEL_NAME'"
        echo "Available models: unet, deeplabv3plus, maskrcnn, sam"
        exit 1
        ;;
esac

echo "✅ Model $MODEL_NAME completed successfully!"

# Generate report if all models are done
if [ -f "results/unet_results.json" ] && [ -f "results/deeplabv3plus_results.json" ] && [ -f "results/maskrcnn_results.json" ] && [ -f "results/sam_results.json" ]; then
    echo "🎉 All models completed! Generating final report..."
    python scripts/evaluate_segmentation.py
    echo "📄 Report generated: results/SEGMENTATION_REPORT.md"
else
    echo "📊 Waiting for remaining models to complete before generating report"
fi