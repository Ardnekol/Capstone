#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Florence multi-instance-cascade segmentation on BOTH cross-domain seg sets.
# Only re-runs Florence (skips the specialists/SAM already on record), so the
# cascade is compared fairly against the existing Mask R-CNN / DeepLab numbers.
#
#     cd ~/Capstone/stage_2/cross_domain_eval && bash run_seg_cascade.sh
#
# Knobs: MAX=20 (sanity check), PYTHON=..., DEVICE=cuda:0
# ─────────────────────────────────────────────────────────────────────────────
set -u
cd "$(dirname "$0")"

PYTHON="${PYTHON:-/u/student/2024/cs24mtech11024/.conda/envs/Capstone/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
MAX="${MAX:-0}"

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  BEST=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
         | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')
  export CUDA_VISIBLE_DEVICES="${BEST:-0}"
fi
echo "GPU=$CUDA_VISIBLE_DEVICES  python=$PYTHON  max=$MAX"

echo "===== ZeroWaste-f  (Florence cascade) ====="
$PYTHON eval_segmentation.py --dataset zerowaste --seg-method cascade \
  --skip deeplabv3plus,unet,maskrcnn,sam \
  --output-dir ../eval_results/segmentation_zerowaste_cascade \
  --max-images "$MAX" --device "$DEVICE"

echo "===== DWSD  (Florence cascade) ====="
$PYTHON eval_segmentation.py --dataset dwsd --seg-method cascade \
  --skip deeplabv3plus,unet,maskrcnn,sam \
  --output-dir ../eval_results/segmentation_dwsd_cascade \
  --max-images "$MAX" --device "$DEVICE"

echo ""
echo "===== CASCADE SUMMARIES ====="
for f in ../eval_results/segmentation_zerowaste_cascade/*_summary.md \
         ../eval_results/segmentation_dwsd_cascade/*_summary.md; do
  [ -f "$f" ] && { echo ""; echo "### $f"; cat "$f"; }
done
echo ""
echo "Compare against current (referring) Florence: ZeroWaste 0.136, DWSD 0.221"
echo "Specialist to beat on ZeroWaste: Mask R-CNN 0.169"
