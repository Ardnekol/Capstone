#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Overnight experiment batch for the DSAA submission. EVAL-ONLY (no training),
# so it is fast and reliable, and every step writes its own output — partial
# completion still gives usable results.
#
# Produces:
#   1. Deployment-cost table (unified vs specialist stack)            -> #4
#   2. Single-task vs unified ablation (cls/det/seg-only LoRAs)       -> #3, #5
#   3. Detection head comparison (<OD> vs region-proposal)           -> #5c
#   4. DWSD segmentation recomputed in one matched pass              -> #2 (partial)
#
# Run in tmux on the GPU node:
#   cd ~/Capstone/stage_2/cross_domain_eval && bash run_overnight.sh
# ─────────────────────────────────────────────────────────────────────────────
set -u
cd "$(dirname "$0")"
CPY="${PYTHON:-/u/student/2024/cs24mtech11024/.conda/envs/Capstone/bin/python}"
DEV="${DEVICE:-cuda:0}"
MAX="${MAX:-0}"   # 0 = all images; set MAX=50 for a quick smoke test

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  BEST=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
         | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')
  export CUDA_VISIBLE_DEVICES="${BEST:-0}"
fi
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="../eval_results/logs_overnight_${STAMP}"; mkdir -p "$LOG"
UNI="../finetuned/florence2_unified_multitask_lora"
CLS="../finetuned/florence2_cls_only_lora"
DET="../finetuned/florence2_det_only_lora"
SEG="../finetuned/florence2_seg_only_lora"

echo "=================================================================="
echo " Overnight batch ($STAMP)  GPU=$CUDA_VISIBLE_DEVICES  max=$MAX  logs=$LOG"
echo "=================================================================="
step(){ echo ""; echo ">>> [$(date +%H:%M:%S)] $1"; }

# 1) Deployment metrics (fast) ────────────────────────────────────────────────
step "Deployment metrics"
$CPY measure_deployment.py --device "$DEV" --n 20 \
  --output-dir ../eval_results/deployment 2>&1 | tee "$LOG/1_deployment.log"

# 2) Single-task vs unified ablation ──────────────────────────────────────────
step "Ablation: classification (cls-only LoRA)"
$CPY eval_warpc_classification.py --lora "$CLS" --skip vit_base,resnet50,efficientnetb0,clip \
  --output-dir ../eval_results/ablation_cls_only --max-per-class "$MAX" --device "$DEV" \
  2>&1 | tee "$LOG/2a_ablation_cls.log"

step "Ablation: detection (det-only LoRA) — ZeroWaste-f + WaRP-D"
for ds in zerowaste warpd; do
  $CPY eval_detection.py --dataset $ds --det-method region_proposal --lora "$DET" \
    --skip yolov8,faster_rcnn,grounding_dino \
    --output-dir ../eval_results/ablation_det_only_$ds --max-images "$MAX" --device "$DEV" \
    2>&1 | tee "$LOG/2b_ablation_det_$ds.log"
done

step "Ablation: segmentation (seg-only LoRA) — ZeroWaste-f + DWSD"
for ds in zerowaste dwsd; do
  $CPY eval_segmentation.py --dataset $ds --seg-method cascade --lora "$SEG" \
    --skip deeplabv3plus,unet,maskrcnn,sam \
    --output-dir ../eval_results/ablation_seg_only_$ds --max-images "$MAX" --device "$DEV" \
    2>&1 | tee "$LOG/2c_ablation_seg_$ds.log"
done

# 3) Detection head comparison (unified model) ────────────────────────────────
step "Detection head comparison (<OD> vs region_proposal vs dense_region_caption)"
for m in od region_proposal dense_region_caption; do
  $CPY eval_detection.py --dataset zerowaste --det-method $m --lora "$UNI" \
    --skip yolov8,faster_rcnn,grounding_dino \
    --output-dir ../eval_results/head_$m --max-images "$MAX" --device "$DEV" \
    2>&1 | tee "$LOG/3_head_$m.log"
done

# 4) DWSD segmentation recomputed in one matched pass (all regimes) ────────────
step "DWSD segmentation — full matched 3-regime recompute"
$CPY eval_segmentation.py --dataset dwsd --seg-method cascade --lora "$UNI" \
  --output-dir ../eval_results/segmentation_dwsd_matched --max-images "$MAX" --device "$DEV" \
  2>&1 | tee "$LOG/4_dwsd_matched.log"

echo ""
echo "=================================================================="
echo " OVERNIGHT BATCH COMPLETE — key summaries:"
echo "=================================================================="
for f in ../eval_results/deployment/deployment.md \
         ../eval_results/ablation_cls_only/*_summary.md \
         ../eval_results/ablation_det_only_*/*_summary.md \
         ../eval_results/ablation_seg_only_*/*_summary.md \
         ../eval_results/head_*/*_summary.md \
         ../eval_results/segmentation_dwsd_matched/*_summary.md; do
  [ -f "$f" ] && { echo ""; echo "### $f"; cat "$f"; }
done
echo ""
echo "Reference (unified, for the ablation comparison):"
echo "  cls WaRP-C 60.35% | det ZeroWaste 0.272 / WaRP-D 0.281 | seg ZeroWaste 0.160 / DWSD 0.180"
echo "Logs: $LOG"
