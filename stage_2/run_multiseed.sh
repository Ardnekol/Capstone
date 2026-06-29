#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Multi-seed results for the unified Florence-2 + LoRA model.
#
# Re-trains the UNIFIED model with several random seeds (paper v1 config:
# epochs=3, lora_r=16, alpha=32, lr=1e-4, batch=1, grad_accum=8, balance=none)
# and re-evaluates ONLY the Florence rows on the 7 cross-domain benchmarks.
# Baselines (ViT/YOLO/DeepLab/...) are fixed across seeds, so they are skipped.
#
# Run on the GPU node, in tmux:
#   srun -p cse-gpu-all --nodelist=dgx-a100-02 --pty bash
#   cd ~/Capstone/stage_2 && tmux new -s seeds
#   bash run_multiseed.sh 2>&1 | tee eval_results/multiseed_$(date +%Y%m%d).log
#
# Knobs:
#   SEEDS="1 2 3"   seeds to run (default). Each is one full train + 7 evals.
#   MAX=300         subsample for a quick smoke test (0 = all images = paper).
#   SKIP_TRAIN=1    only re-evaluate existing seed checkpoints.
#
# Estimated cost: ~3-5 GPU-h train + ~1-2 GPU-h eval PER seed on one A100.
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PATH="$HOME/.conda/envs/Capstone/bin:$PATH"
CPY="$HOME/.conda/envs/Capstone/bin/python"

SEEDS="${SEEDS:-1 2 3}"
MAX="${MAX:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

# Pin the idlest GPU on the shared node unless the caller already chose one.
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  BEST=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
         | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')
  export CUDA_VISIBLE_DEVICES="${BEST:-0}"
fi
echo "Using GPU ${CUDA_VISIBLE_DEVICES} | seeds=[${SEEDS}] | max=${MAX}"

MODEL_ID="microsoft/Florence-2-large-ft"
SKIP_CLS="vit_base,resnet50,efficientnetb0,clip"
SKIP_DET="yolov8,faster_rcnn,grounding_dino"
SKIP_SEG="deeplabv3plus,unet,maskrcnn,sam"

for SEED in $SEEDS; do
  echo ""
  echo "=================================================================="
  echo " SEED ${SEED}  [$(date +%H:%M:%S)]"
  echo "=================================================================="
  OUT="finetuned/florence2_unified_multitask_lora_seed${SEED}"
  RES="eval_results/seed${SEED}"
  mkdir -p "$RES"

  # ── Train (paper v1 config) ───────────────────────────────────────────────
  if [ "$SKIP_TRAIN" != "1" ] && [ ! -f "$OUT/adapter_model.safetensors" ] && [ ! -f "$OUT/adapter_model.bin" ]; then
    echo ">>> [seed ${SEED}] training unified model -> $OUT"
    $CPY finetune_florence2_od_lora.py \
        --model-id "$MODEL_ID" \
        --train-jsonl finetune_data/unified_multitask_train.jsonl \
        --eval-jsonl  finetune_data/unified_multitask_val.jsonl \
        --output-dir  "$OUT" \
        --num-train-epochs 3 \
        --per-device-train-batch-size 1 \
        --gradient-accumulation-steps 8 \
        --learning-rate 1e-4 \
        --lora-r 16 --lora-alpha 32 \
        --seed "$SEED" \
        --device cuda:0
  else
    echo ">>> [seed ${SEED}] reusing existing checkpoint at $OUT"
  fi

  # ── Evaluate ONLY Florence on the 7 cross-domain benchmarks ───────────────
  pushd cross_domain_eval >/dev/null
  L="../$OUT"

  echo ">>> [seed ${SEED}] classification: RealWaste, WaRP-C"
  $CPY eval_realwaste_classification.py --lora "$L" --skip "$SKIP_CLS" \
       --max-per-class "$MAX" --output-dir "../$RES/realwaste" --device cuda:0
  $CPY eval_warpc_classification.py     --lora "$L" --skip "$SKIP_CLS" \
       --max-per-class "$MAX" --output-dir "../$RES/warpc"     --device cuda:0

  echo ">>> [seed ${SEED}] detection: ICRA19, ZeroWaste-f, WaRP-D (region-proposal head)"
  for D in icra19 zerowaste warpd; do
    $CPY eval_detection.py --dataset "$D" --det-method region_proposal \
         --lora "$L" --skip "$SKIP_DET" \
         --max-images "$MAX" --output-dir "../$RES/det_${D}" --device cuda:0
  done

  echo ">>> [seed ${SEED}] segmentation: DWSD, ZeroWaste-f (cascade head)"
  for S in dwsd zerowaste; do
    $CPY eval_segmentation.py --dataset "$S" --seg-method cascade \
         --lora "$L" --skip "$SKIP_SEG" \
         --max-images "$MAX" --output-dir "../$RES/seg_${S}" --device cuda:0
  done
  popd >/dev/null

  echo ">>> [seed ${SEED}] done. Summaries under $RES/"
done

echo ""
echo "=================================================================="
echo " ALL SEEDS DONE. Aggregate with:"
echo "   $CPY aggregate_seeds.py --seeds ${SEEDS// /,}"
echo "=================================================================="
