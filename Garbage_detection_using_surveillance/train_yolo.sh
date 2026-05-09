#!/bin/bash
# Step 3a: Train YOLOv8m on CCTV garbage dataset (improved config)
# Run inside SLURM: srun -p cse-gpu-all --nodelist=dgx-a100-02 --pty bash
# Then: tmux attach -t CCTV && bash train_yolo.sh

set -e

PYTHON="/u/student/2024/cs24mtech11024/.conda/envs/Capstone/bin/python"
PIP="/u/student/2024/cs24mtech11024/.conda/envs/Capstone/bin/pip"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_SIZE="m"
EPOCHS=150
IMGSZ=1024
BATCH=4
PATIENCE=40

# Auto-select GPU with least memory used
echo "GPU memory usage:"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader
DEVICE=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader \
         | awk -F',' '{gsub(/[^0-9]/, "", $2); print $2, $1}' \
         | sort -n | head -1 | awk '{print $2}')
echo "Auto-selected GPU: $DEVICE (least memory used)"

echo "================================================"
echo "YOLOv8m CCTV Garbage Detection Training"
echo "  Model  : yolov8${MODEL_SIZE}"
echo "  Epochs : $EPOCHS"
echo "  ImgSz  : $IMGSZ"
echo "  Batch  : $BATCH"
echo "  Device : cuda:$DEVICE"
echo "================================================"

$PIP install -q ultralytics "numpy<2" opencv-python-headless

cd "$SCRIPT_DIR"

# Ensure YOLO dataset exists
if [ ! -d "$SCRIPT_DIR/yolo_data/images/train" ]; then
    echo "YOLO dataset not found — converting COCO → YOLO..."
    bash "$SCRIPT_DIR/convert_to_yolo.sh"
fi

$PYTHON train_yolo_cctv.py \
    --model  "$MODEL_SIZE" \
    --epochs "$EPOCHS" \
    --imgsz  "$IMGSZ" \
    --batch  "$BATCH" \
    --device "$DEVICE" \
    --patience "$PATIENCE"

echo ""
echo "Training done. Best model: runs/detect/cctv_garbage_yolo_v2/weights/best.pt"
echo "Next: bash run_yolo_eval.sh"
