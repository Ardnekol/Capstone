#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Matched-protocol recompute of the two carried-over baselines (closes #2):
#   - RealWaste classification (3-regime)
#   - Trash-ICRA19 detection   (3-regime, region-proposal head for Florence)
#
# Run in tmux on the GPU node:
#   cd ~/Capstone/stage_2/cross_domain_eval && bash run_recompute.sh
#
# Knobs:
#   MAX=300 bash run_recompute.sh        # quick preview (subsample)
#   CUDA_VISIBLE_DEVICES=6 bash ...       # pin a specific (idle) GPU
# ─────────────────────────────────────────────────────────────────────────────
set -u
cd "$(dirname "$0")"
CPY="${PYTHON:-/u/student/2024/cs24mtech11024/.conda/envs/Capstone/bin/python}"
DEV="${DEVICE:-cuda:0}"
MAX="${MAX:-0}"          # 0 = all images
LORA="../finetuned/florence2_unified_multitask_lora"

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  BEST=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
         | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')
  export CUDA_VISIBLE_DEVICES="${BEST:-0}"
fi
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="../eval_results/logs_recompute_${STAMP}"; mkdir -p "$LOG"
echo "=================================================================="
echo " Matched recompute ($STAMP)  GPU=$CUDA_VISIBLE_DEVICES  max=$MAX  logs=$LOG"
echo "=================================================================="

echo ""; echo ">>> [$(date +%H:%M:%S)] RealWaste classification (matched 3-regime)"
$CPY eval_realwaste_classification.py \
  --lora "$LORA" --max-per-class "$MAX" \
  --output-dir ../eval_results/realwaste_matched --device "$DEV" \
  2>&1 | tee "$LOG/realwaste.log"

echo ""; echo ">>> [$(date +%H:%M:%S)] Trash-ICRA19 detection (matched 3-regime, region-proposal)"
$CPY eval_detection.py --dataset icra19 --det-method region_proposal \
  --lora "$LORA" --max-images "$MAX" \
  --output-dir ../eval_results/detection_icra19_matched --device "$DEV" \
  2>&1 | tee "$LOG/icra19.log"

echo ""
echo "=================================================================="
echo " RECOMPUTE COMPLETE — summaries:"
echo "=================================================================="
for f in ../eval_results/realwaste_matched/*_summary.md \
         ../eval_results/detection_icra19_matched/*_summary.md; do
  [ -f "$f" ] && { echo ""; echo "### $f"; cat "$f"; }
done
echo ""
echo "Carried-over numbers to compare against (currently in the paper):"
echo "  RealWaste cls: ViT 39.98 / ResNet 17.85 / EffNet 32.89 / CLIP 42.68 / Florence 56.68"
echo "  ICRA19  det F1: YOLO 0.139 / FasterRCNN 0.137 / G-DINO 0.372 / Florence 0.505"
echo "Logs: $LOG"
