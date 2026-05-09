#!/bin/bash
# Step 4c: Evaluate trained YOLO model on the test set

set -e

PYTHON="/u/student/2024/cs24mtech11024/.conda/envs/Capstone/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${1:-$SCRIPT_DIR/runs/detect/cctv_garbage_yolo_v42/weights/best.pt}"
cd "$SCRIPT_DIR"

if [ ! -f "$MODEL" ]; then
    echo "Model not found at: $MODEL"
    echo "Run train_yolo.sh first, or pass the model path as argument:"
    echo "  bash run_yolo_eval.sh /path/to/best.pt"
    exit 1
fi

echo "YOLO Evaluation..."
echo "  Model: $MODEL"
$PYTHON evaluate_yolo_cctv.py \
    --model "$MODEL" \
    --split test \
    --conf  0.15

echo "Done. Results in: eval_results/yolo/"
