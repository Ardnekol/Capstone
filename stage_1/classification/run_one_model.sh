#!/bin/bash
#
# ============================================================================
# RUN ONE CLASSIFICATION MODEL AT A TIME
# ============================================================================
#
# This script runs classification models one by one to avoid GPU memory issues.
#
# Usage on A100:
#   1. srun -p cse-gpu-all --nodelist=dgx-a100-02 --pty bash
#   2. tmux new -s classification_train
#   3. cd ~/Capstone/stage_1/classification
#   4. bash run_one_model.sh resnet50       # Run ResNet-50 only
#   5. bash run_one_model.sh efficientnet   # Run EfficientNet-B0 only
#   6. bash run_one_model.sh clip           # Run CLIP evaluation only
#   7. bash run_one_model.sh report         # Generate final report
#
# ============================================================================

set -e

cd "$(dirname "$0")"

MODEL=${1:-"help"}
EPOCHS=50
BATCH_SIZE=32
GPU_ID=${2:-6}  # Default to GPU 6 (usually free)
DEVICE="cuda:0"

echo "Using GPU $GPU_ID"

echo ""
echo "============================================"
echo "🗂️ Classification Experiment: $MODEL"
echo "============================================"
echo ""

case $MODEL in
    resnet50)
        echo "🏋️ Training ResNet-50 (epochs=$EPOCHS, batch=$BATCH_SIZE)"
        echo ""
        pip install -r ../requirements.txt -q
        CUDA_VISIBLE_DEVICES=$GPU_ID python scripts/train_resnet50.py \
            --epochs $EPOCHS \
            --batch $BATCH_SIZE
        echo ""
        echo "✅ ResNet-50 training complete!"
        echo "📁 Results: results/resnet50_results.json"
        ;;

    efficientnet)
        echo "🏋️ Training EfficientNet-B0 (epochs=$EPOCHS, batch=$BATCH_SIZE)"
        echo ""
        pip install -r ../requirements.txt -q
        CUDA_VISIBLE_DEVICES=$GPU_ID python scripts/train_efficientnetb0.py \
            --epochs $EPOCHS \
            --batch $BATCH_SIZE
        echo ""
        echo "✅ EfficientNet-B0 training complete!"
        echo "📁 Results: results/efficientnetb0_results.json"
        ;;

    vit_base)
        echo "🏋️ Training ViT-Base (epochs=$EPOCHS, batch=$BATCH_SIZE)"
        echo ""
        pip install -r ../requirements.txt -q
        CUDA_VISIBLE_DEVICES=$GPU_ID python scripts/train_vit_base.py \
            --epochs $EPOCHS \
            --batch $BATCH_SIZE
        echo ""
        echo "✅ ViT-Base training complete!"
        echo "📁 Results: results/vit_base_results.json"
        ;;

    clip)
        echo "🔍 Evaluating CLIP (ViT-B/16) zero-shot"
        echo ""
        pip install -r ../requirements.txt -q
        CUDA_VISIBLE_DEVICES=$GPU_ID python scripts/eval_clip.py
        echo ""
        echo "✅ CLIP evaluation complete!"
        echo "📁 Results: results/clip_results.json"
        ;;

    report)
        echo "📊 Generating comparison report"
        echo ""
        python scripts/evaluate.py
        echo ""
        echo "✅ Report generation complete!"
        echo "📁 Report: results/COMPARISON_REPORT.md"
        echo "📊 Plot: results/accuracy_comparison.png"
        ;;

    help|*)
        echo "Usage: $0 {resnet50|efficientnet|vit_base|clip|report}"
        echo ""
        echo "Models:"
        echo "  resnet50      - Train ResNet-50 on TrashNet"
        echo "  efficientnet  - Train EfficientNet-B0 on TrashNet"
        echo "  vit_base      - Train ViT-Base on TrashNet"
        echo "  clip          - Evaluate CLIP zero-shot on both datasets"
        echo "  report        - Generate comparison report and plots"
        echo ""
        echo "Examples:"
        echo "  $0 resnet50       # Train ResNet-50"
        echo "  $0 efficientnet   # Train EfficientNet-B0"
        echo "  $0 vit_base       # Train ViT-Base"
        echo "  $0 clip           # Evaluate CLIP"
        echo "  $0 report         # Generate report"
        ;;
esac