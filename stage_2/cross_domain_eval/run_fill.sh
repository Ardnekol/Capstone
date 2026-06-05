#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Fill the missing specialist cells in the master grid:
#   - Faster R-CNN  → ZeroWaste-f detection, WaRP-D detection
#   - U-Net + Mask R-CNN → ZeroWaste-f segmentation
#
# Run on the GPU node (inside tmux), from this directory:
#     cd ~/Capstone/stage_2/cross_domain_eval && bash run_fill.sh
#
# Knobs:  MAX=30 bash run_fill.sh   (quick sanity check on 30 images)
#         PYTHON=... DEVICE=cuda:0  (override interpreter / device)
# ─────────────────────────────────────────────────────────────────────────────
set -u
cd "$(dirname "$0")"

PYTHON="${PYTHON:-/u/student/2024/cs24mtech11024/.conda/envs/Capstone/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
MAX="${MAX:-0}"   # 0 = all images

# Shared node: pick the GPU with the most free memory unless caller pinned one.
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  BEST=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
         | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')
  export CUDA_VISIBLE_DEVICES="${BEST:-0}"
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="../eval_results/logs_fill_${STAMP}"
mkdir -p "$LOGDIR"
echo "=================================================================="
echo " Filling specialist cells   ($STAMP)"
echo " python=$PYTHON  GPU=$CUDA_VISIBLE_DEVICES  max_images=$MAX"
echo " logs → $LOGDIR"
echo "=================================================================="

run_step () {
  local name="$1"; shift
  echo ""
  echo ">>> [$(date +%H:%M:%S)] START $name"
  ( "$@" ) 2>&1 | tee "$LOGDIR/${name}.log"
  echo ">>> [$(date +%H:%M:%S)] DONE  $name (exit ${PIPESTATUS[0]})"
}

# 1) Faster R-CNN — ZeroWaste-f detection
run_step "fill_fasterrcnn_zerowaste" \
  $PYTHON eval_detection.py --dataset zerowaste \
    --skip yolov8,grounding_dino,florence2_ft \
    --output-dir ../eval_results/detection_zerowaste_fill \
    --max-images "$MAX" --device "$DEVICE"

# 2) Faster R-CNN — WaRP-D detection
run_step "fill_fasterrcnn_warpd" \
  $PYTHON eval_detection.py --dataset warpd \
    --skip yolov8,grounding_dino,florence2_ft \
    --output-dir ../eval_results/detection_warpd_fill \
    --max-images "$MAX" --device "$DEVICE"

# 3) U-Net + Mask R-CNN — ZeroWaste-f segmentation
run_step "fill_unet_maskrcnn_zerowaste" \
  $PYTHON eval_segmentation.py \
    --skip deeplabv3plus,sam,florence2_ft \
    --output-dir ../eval_results/segmentation_zerowaste_fill \
    --max-images "$MAX" --device "$DEVICE"

echo ""
echo "=================================================================="
echo " FILL COMPLETE — summaries:"
echo "=================================================================="
for f in ../eval_results/detection_zerowaste_fill/*_summary.md \
         ../eval_results/detection_warpd_fill/*_summary.md \
         ../eval_results/segmentation_zerowaste_fill/*_summary.md; do
  [ -f "$f" ] && { echo ""; echo "### $f"; cat "$f"; }
done
echo ""
echo "Logs in: $LOGDIR"
