#!/bin/bash
#
# ============================================================================
# COMPLETE DETECTION EXPERIMENT PIPELINE
# ============================================================================
# 
# This script runs ALL detection experiments:
# - Task-Specific: YOLOv8, Faster R-CNN, RetinaNet
# - Foundation Models: Grounding-DINO, Florence-2
#
# Usage:
#   On A100 GPU:
#   1. srun -p cse-gpu-all --nodelist=dgx-a100-02 --pty bash
#   2. tmux new -s detection_train
#   3. cd ~/Capstone/stage_1/detection
#   4. bash run_detection_full.sh
#
# To detach: Ctrl+b, then d
# To reattach: tmux attach -t detection_train
#
# ============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
EPOCHS=50
BATCH_YOLO=16
BATCH_RCNN=4
DEVICE="cuda:0"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Timestamp
START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
LOG_FILE="experiment_log_$(date '+%Y%m%d_%H%M%S').txt"

# Function to log and print
log() {
    echo -e "$1" | tee -a "$LOG_FILE"
}

# ============================================================================
# HEADER
# ============================================================================
clear
log "============================================================================"
log "${BLUE}🔬 WASTE DETECTION: TASK-SPECIFIC vs FOUNDATION MODELS${NC}"
log "============================================================================"
log "Start Time: $START_TIME"
log "Working Dir: $SCRIPT_DIR"
log "GPU Device: $DEVICE"
log "Epochs: $EPOCHS"
log "Log File: $LOG_FILE"
log "============================================================================"
log ""

# Check GPU
log "${YELLOW}🔍 Checking GPU...${NC}"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv | tee -a "$LOG_FILE"
log ""

# ============================================================================
# STEP 0: Install Requirements
# ============================================================================
log "============================================================================"
log "${BLUE}📦 STEP 0: Installing Requirements${NC}"
log "============================================================================"

pip install ultralytics transformers torch torchvision tqdm pandas Pillow PyYAML 2>&1 | tee -a "$LOG_FILE"

log "${GREEN}✅ Requirements installed${NC}"
log ""

# ============================================================================
# STEP 1: Prepare Data (Convert Annotations)
# ============================================================================
log "============================================================================"
log "${BLUE}📁 STEP 1: Preparing Data - Converting Annotations to YOLO Format${NC}"
log "============================================================================"

python convert_annotations.py --all 2>&1 | tee -a "$LOG_FILE"

log "${GREEN}✅ Data preparation complete${NC}"
log ""

# ============================================================================
# STEP 2: Train YOLOv8
# ============================================================================
log "============================================================================"
log "${BLUE}🏋️ STEP 2: Training YOLOv8-M${NC}"
log "============================================================================"
log "Epochs: $EPOCHS, Batch: $BATCH_YOLO"
log ""

CUDA_VISIBLE_DEVICES=0 python train_yolov8.py \
    --epochs $EPOCHS \
    --batch $BATCH_YOLO \
    --device $DEVICE \
    2>&1 | tee -a "$LOG_FILE"

log "${GREEN}✅ YOLOv8 training complete${NC}"
log ""

# ============================================================================
# STEP 3: Train Faster R-CNN
# ============================================================================
log "============================================================================"
log "${BLUE}🏋️ STEP 3: Training Faster R-CNN ResNet50-FPN${NC}"
log "============================================================================"
log "Epochs: $EPOCHS, Batch: $BATCH_RCNN"
log ""

CUDA_VISIBLE_DEVICES=0 python train_fasterrcnn.py \
    --epochs $EPOCHS \
    --batch $BATCH_RCNN \
    --device $DEVICE \
    2>&1 | tee -a "$LOG_FILE"

log "${GREEN}✅ Faster R-CNN training complete${NC}"
log ""

# ============================================================================
# STEP 4: Train RetinaNet
# ============================================================================
log "============================================================================"
log "${BLUE}🏋️ STEP 4: Training RetinaNet ResNet50-FPN${NC}"
log "============================================================================"
log "Epochs: $EPOCHS, Batch: $BATCH_RCNN"
log ""

CUDA_VISIBLE_DEVICES=0 python train_retinanet.py \
    --epochs $EPOCHS \
    --batch $BATCH_RCNN \
    --device $DEVICE \
    2>&1 | tee -a "$LOG_FILE"

log "${GREEN}✅ RetinaNet training complete${NC}"
log ""

# ============================================================================
# STEP 5: Evaluate Grounding-DINO (Zero-Shot)
# ============================================================================
log "============================================================================"
log "${BLUE}🔍 STEP 5: Evaluating Grounding-DINO (Zero-Shot)${NC}"
log "============================================================================"

CUDA_VISIBLE_DEVICES=0 python eval_grounding_dino.py \
    --device $DEVICE \
    --dataset all \
    2>&1 | tee -a "$LOG_FILE"

log "${GREEN}✅ Grounding-DINO evaluation complete${NC}"
log ""

# ============================================================================
# STEP 6: Evaluate Florence-2 (Zero-Shot)
# ============================================================================
log "============================================================================"
log "${BLUE}🔍 STEP 6: Evaluating Florence-2 (Zero-Shot)${NC}"
log "============================================================================"

CUDA_VISIBLE_DEVICES=0 python eval_florence2.py \
    --device $DEVICE \
    --dataset all \
    2>&1 | tee -a "$LOG_FILE"

log "${GREEN}✅ Florence-2 evaluation complete${NC}"
log ""

# ============================================================================
# STEP 7: Generate Comparison Report
# ============================================================================
log "============================================================================"
log "${BLUE}📊 STEP 7: Generating Comparison Report${NC}"
log "============================================================================"

python evaluate.py --compare 2>&1 | tee -a "$LOG_FILE"

log "${GREEN}✅ Comparison report generated${NC}"
log ""

# ============================================================================
# FINAL SUMMARY
# ============================================================================
END_TIME=$(date '+%Y-%m-%d %H:%M:%S')

log "============================================================================"
log "${GREEN}🎉 ALL EXPERIMENTS COMPLETED!${NC}"
log "============================================================================"
log ""
log "Start Time: $START_TIME"
log "End Time:   $END_TIME"
log ""
log "Results saved to:"
log "  📄 results/COMPARISON_REPORT.md"
log "  📄 results/model_comparison.csv"
log "  📄 results/all_results.json"
log ""
log "Training outputs:"
log "  📁 runs/detect/      (YOLOv8)"
log "  📁 runs/fasterrcnn/  (Faster R-CNN)"
log "  📁 runs/retinanet/   (RetinaNet)"
log "  📁 runs/grounding_dino/ (G-DINO)"
log "  📁 runs/florence2/   (Florence-2)"
log ""
log "Log file: $LOG_FILE"
log "============================================================================"

# Display results if available
if [ -f "results/model_comparison.csv" ]; then
    log ""
    log "${BLUE}📊 RESULTS SUMMARY:${NC}"
    log ""
    cat results/model_comparison.csv | column -t -s',' | tee -a "$LOG_FILE"
fi

log ""
log "${GREEN}Done! Check results/COMPARISON_REPORT.md for detailed analysis.${NC}"
