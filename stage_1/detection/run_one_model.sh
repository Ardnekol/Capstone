#!/bin/bash
#
# ============================================================================
# RUN ONE MODEL AT A TIME
# ============================================================================
#
# This script runs detection models one by one to avoid GPU memory issues.
#
# Usage on A100:
#   1. srun -p cse-gpu-all --nodelist=dgx-a100-02 --pty bash
#   2. tmux new -s detection_train
#   3. cd ~/Capstone/stage_1/detection
#   4. bash run_one_model.sh yolov8       # Run YOLOv8 only
#   5. bash run_one_model.sh fasterrcnn   # Run Faster R-CNN only
#   6. bash run_one_model.sh retinanet    # Run RetinaNet only
#   7. bash run_one_model.sh report       # Generate final report
#
# ============================================================================

set -e

cd "$(dirname "$0")"

MODEL=${1:-"help"}
EPOCHS=50
GPU_ID=${2:-6}  # Default to GPU 6 (usually free)
DEVICE="cuda:0"

echo "Using GPU $GPU_ID"

echo ""
echo "============================================"
echo "🔬 Detection Experiment: $MODEL"
echo "============================================"
echo ""

case $MODEL in
    yolov8)
        echo "🏋️ Training YOLOv8-M (epochs=$EPOCHS, batch=16)"
        echo ""
        pip install ultralytics -q
        CUDA_VISIBLE_DEVICES=$GPU_ID python train_yolov8.py \
            --epochs $EPOCHS \
            --batch 16 \
            --device $DEVICE
        echo ""
        echo "✅ YOLOv8 training complete!"
        echo "📁 Results: runs/detect/"
        ;;
        
    fasterrcnn)
        echo "🏋️ Training Faster R-CNN (epochs=$EPOCHS, batch=2)"
        echo ""
        CUDA_VISIBLE_DEVICES=$GPU_ID python train_fasterrcnn.py \
            --epochs $EPOCHS \
            --batch 2 \
            --device $DEVICE
        echo ""
        echo "✅ Faster R-CNN training complete!"
        echo "📁 Results: runs/fasterrcnn/"
        ;;
        
    retinanet)
        echo "🏋️ Training RetinaNet (epochs=$EPOCHS, batch=2)"
        echo ""
        CUDA_VISIBLE_DEVICES=$GPU_ID python train_retinanet.py \
            --epochs $EPOCHS \
            --batch 2 \
            --device $DEVICE
        echo ""
        echo "✅ RetinaNet training complete!"
        echo "📁 Results: runs/retinanet/"
        ;;
    
    grounding-dino)
        echo "🔬 Evaluating Grounding-DINO (Zero-Shot Foundation Model)"
        echo ""
        CUDA_VISIBLE_DEVICES=$GPU_ID python eval_grounding_dino.py \
            --device $DEVICE
        echo ""
        echo "✅ Grounding-DINO evaluation complete!"
        echo "📁 Results: results/"
        ;;
    
    owlvit)
        echo "🔬 Evaluating OWL-ViT (Zero-Shot Foundation Model)"
        echo ""
        CUDA_VISIBLE_DEVICES=$GPU_ID python eval_owlvit.py
        echo ""
        echo "✅ OWL-ViT evaluation complete!"
        echo "📁 Results: results/"
        ;;
        
    report)
        echo "📊 Generating comparison report..."
        echo ""
        python evaluate.py --compare
        echo ""
        echo "✅ Report generated!"
        echo "📁 Results: results/"
        ;;
        
    all)
        echo "Running all models sequentially..."
        echo ""
        echo "=== TASK-SPECIFIC MODELS ==="
        bash $0 yolov8 $GPU_ID
        bash $0 fasterrcnn $GPU_ID
        bash $0 retinanet $GPU_ID
        echo ""
        echo "=== FOUNDATION MODELS ==="
        bash $0 grounding-dino $GPU_ID
        bash $0 florence2 $GPU_ID
        echo ""
        bash $0 report
        ;;
        
    *)
        echo "Usage: bash run_one_model.sh <model> [gpu_id]"
        echo ""
        echo "TASK-SPECIFIC MODELS (require training):"
        echo "  yolov8         - Train YOLOv8-M (recommended first)"
        echo "  fasterrcnn     - Train Faster R-CNN ResNet50-FPN"
        echo "  retinanet      - Train RetinaNet ResNet50-FPN"
        echo ""
    echo "FOUNDATION MODELS (zero-shot, no training):"
    echo "  grounding-dino - Evaluate Grounding-DINO"
    echo "  owlvit         - Evaluate OWL-ViT"
        echo ""
        echo "OTHER:"
        echo "  report         - Generate comparison report"
        echo "  all            - Run all models sequentially"
        echo ""
        echo "Example:"
        echo "  bash run_one_model.sh yolov8 6    # Use GPU 6"
        echo "  bash run_one_model.sh all 6       # Run all on GPU 6"
        echo ""
        ;;
esac
