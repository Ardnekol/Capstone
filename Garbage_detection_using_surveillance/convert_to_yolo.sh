#!/bin/bash
# Step 0: Convert COCO annotations to YOLO format
# Must run split_cctv_coco.py first.

set -e

PYTHON="/u/student/2024/cs24mtech11024/.conda/envs/Capstone/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR"

echo "Step 1: Splitting COCO dataset..."
$PYTHON split_cctv_coco.py

echo "Step 2: Converting to YOLO format..."
$PYTHON coco_to_yolo.py

echo "Done. YOLO dataset at: $SCRIPT_DIR/yolo_data/"
