#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Fill the missing specialist cells, using TWO GPUs in parallel.
#   GPU A  → both Faster R-CNN detection runs (ZeroWaste-f, WaRP-D)
#   GPU B  → U-Net + Mask R-CNN segmentation (ZeroWaste-f)
#
# Run on the GPU node (inside tmux), from this directory:
#     cd ~/Capstone/stage_2/cross_domain_eval && bash run_fill_2gpu.sh
#
# Pick which physical GPUs to use (default 5 and 6):
#     GPUA=5 GPUB=6 bash run_fill_2gpu.sh
# Quick check:  MAX=30 bash run_fill_2gpu.sh
# ─────────────────────────────────────────────────────────────────────────────
set -u
cd "$(dirname "$0")"

PYTHON="${PYTHON:-/u/student/2024/cs24mtech11024/.conda/envs/Capstone/bin/python}"
GPUA="${GPUA:-5}"
GPUB="${GPUB:-6}"
MAX="${MAX:-0}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="../eval_results/logs_fill2_${STAMP}"
mkdir -p "$LOGDIR"

echo "=================================================================="
echo " Parallel fill   ($STAMP)   GPU A=$GPUA (detection)  GPU B=$GPUB (segmentation)"
echo " python=$PYTHON  max_images=$MAX   logs → $LOGDIR"
echo "=================================================================="

# ── Group A: detection (Faster R-CNN), both datasets, on GPU A ────────────────
(
  export CUDA_VISIBLE_DEVICES="$GPUA"
  echo "[A/GPU$GPUA] $(date +%H:%M:%S) Faster R-CNN ZeroWaste-f"
  $PYTHON eval_detection.py --dataset zerowaste \
    --skip yolov8,grounding_dino,florence2_ft \
    --output-dir ../eval_results/detection_zerowaste_fill \
    --max-images "$MAX" --device cuda:0
  echo "[A/GPU$GPUA] $(date +%H:%M:%S) Faster R-CNN WaRP-D"
  $PYTHON eval_detection.py --dataset warpd \
    --skip yolov8,grounding_dino,florence2_ft \
    --output-dir ../eval_results/detection_warpd_fill \
    --max-images "$MAX" --device cuda:0
  echo "[A/GPU$GPUA] $(date +%H:%M:%S) DONE detection"
) > "$LOGDIR/groupA_detection.log" 2>&1 &
PIDA=$!

# ── Group B: segmentation (U-Net + Mask R-CNN), on GPU B ──────────────────────
(
  export CUDA_VISIBLE_DEVICES="$GPUB"
  echo "[B/GPU$GPUB] $(date +%H:%M:%S) U-Net + Mask R-CNN ZeroWaste-f"
  $PYTHON eval_segmentation.py \
    --skip deeplabv3plus,sam,florence2_ft \
    --output-dir ../eval_results/segmentation_zerowaste_fill \
    --max-images "$MAX" --device cuda:0
  echo "[B/GPU$GPUB] $(date +%H:%M:%S) DONE segmentation"
) > "$LOGDIR/groupB_segmentation.log" 2>&1 &
PIDB=$!

echo "Launched: detection PID=$PIDA (GPU $GPUA), segmentation PID=$PIDB (GPU $GPUB)"
echo "Tailing both logs (Ctrl-C just stops the tail, jobs keep running)..."
tail -f "$LOGDIR/groupA_detection.log" "$LOGDIR/groupB_segmentation.log" &
TAILPID=$!

wait $PIDA; echo ">>> detection group finished (exit $?)"
wait $PIDB; echo ">>> segmentation group finished (exit $?)"
kill $TAILPID 2>/dev/null

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
