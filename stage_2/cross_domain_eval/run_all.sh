#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Overnight 3-regime cross-domain evaluation runner (run inside tmux on the GPU).
#
#   1. ssh / srun onto the GPU node:
#        srun -p cse-gpu-all --nodelist=dgx-a100-02 --pty bash
#   2. tmux new -s eval
#   3. cd .../Capstone/stage_2/cross_domain_eval && bash run_all.sh
#   4. detach with Ctrl-b d ; reattach in the morning with: tmux attach -t eval
#
# Override the interpreter if your working env is a different python:
#        PYTHON=python3.11 bash run_all.sh
# Skip regimes whose deps are missing, e.g. no CLIP:  SKIP_CLS=clip bash run_all.sh
# Quick smoke test on a few images:  MAX=20 bash run_all.sh
# ─────────────────────────────────────────────────────────────────────────────
set -u
cd "$(dirname "$0")"

# Default to the Capstone conda env (complete stack: Florence, YOLO, CLIP,
# DeepLab, SAM). Override with PYTHON=... if needed.
PYTHON="${PYTHON:-/u/student/2024/cs24mtech11024/.conda/envs/Capstone/bin/python}"

# This is a SHARED node — pick the GPU with the most free memory (unless the
# caller already pinned one). After this, scripts use cuda:0 = that physical GPU.
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  BEST=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
         | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')
  export CUDA_VISIBLE_DEVICES="${BEST:-0}"
fi
echo "Using physical GPU(s): CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
DEVICE="${DEVICE:-cuda:0}"
MAX="${MAX:-0}"                 # 0 = all images
LORA="${LORA:-../finetuned/florence2_unified_multitask_lora}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="../eval_results/logs_${STAMP}"
mkdir -p "$LOGDIR"

echo "=================================================================="
echo " Cross-domain 3-regime evaluation   ($STAMP)"
echo " python=$PYTHON  device=$DEVICE  max_images=$MAX"
echo " logs → $LOGDIR"
echo "=================================================================="

echo ">>> Environment check"
$PYTHON check_env.py 2>&1 | tee "$LOGDIR/00_env.log"

run_step () {
  local name="$1"; shift
  echo ""
  echo ">>> [$(date +%H:%M:%S)] START $name"
  # Do not abort the whole night if one step fails.
  ( "$@" ) 2>&1 | tee "$LOGDIR/${name}.log"
  echo ">>> [$(date +%H:%M:%S)] DONE  $name (exit ${PIPESTATUS[0]})"
}

# 1) Classification — WaRP-C (specialists + CLIP + Florence-FT)
run_step "01_classification_warpc" \
  $PYTHON eval_warpc_classification.py \
    --lora "$LORA" --max-per-class "$MAX" --device "$DEVICE" \
    --skip "${SKIP_CLS:-}"

# 2) Detection — ZeroWaste-f (YOLOv8 + G-DINO + Florence-FT)
run_step "02_detection_zerowaste" \
  $PYTHON eval_detection.py --dataset zerowaste \
    --lora "$LORA" --max-images "$MAX" --device "$DEVICE" \
    --skip "${SKIP_DET:-}"

# 3) Detection — WaRP-D
run_step "03_detection_warpd" \
  $PYTHON eval_detection.py --dataset warpd \
    --lora "$LORA" --max-images "$MAX" --device "$DEVICE" \
    --skip "${SKIP_DET:-}"

# 4) Segmentation — ZeroWaste-f (DeepLabV3+ + SAM + Florence-FT)
run_step "04_segmentation_zerowaste" \
  $PYTHON eval_segmentation.py \
    --lora "$LORA" --max-images "$MAX" --device "$DEVICE" \
    --skip "${SKIP_SEG:-}"

echo ""
echo "=================================================================="
echo " ALL STEPS COMPLETE — collected summaries:"
echo "=================================================================="
for f in ../eval_results/*/*_summary.md; do
  [ -f "$f" ] && { echo ""; echo "### $f"; cat "$f"; }
done
echo ""
echo "Per-step logs in: $LOGDIR"
