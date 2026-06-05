#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Stage-2 Florence-2 UNIFIED retrain v2 — legitimate cross-domain improvement.
#
# Levers (all on the ORIGINAL training data; test domains stay untouched):
#   • epochs        3  → 10   (model was under-trained)
#   • LoRA rank     16 → 64   (alpha 32 → 128) — capacity for the seg/polygon task
#   • task balance  none → oversample (det/cls repeated up to seg's count)
#   • bf16 for memory/speed on the shared A100
#
# v1 is PRESERVED: v2 writes to finetuned/florence2_unified_multitask_lora_v2.
#
# Run on the GPU node, in tmux:
#   cd ~/Capstone/stage_2 && bash train_unified_v2.sh
#
# Override: EPOCHS=8 LORA_R=64 bash train_unified_v2.sh
# ──────────────────────────────────────────────────────────────────────────────
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PATH="$HOME/.conda/envs/Capstone/bin:$PATH"
CPY="$HOME/.conda/envs/Capstone/bin/python"

# Pick the GPU with the most FREE memory (shared node).
BEST_GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
           | sort -t',' -k2 -nr | head -1 | cut -d',' -f1 | tr -d ' ')
export CUDA_VISIBLE_DEVICES="${BEST_GPU}"
echo "Using GPU ${BEST_GPU} (most free memory)"

EPOCHS="${EPOCHS:-10}"
LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
LR="${LR:-1e-4}"
OUT="finetuned/florence2_unified_multitask_lora_v2"
TRAIN_JSONL="finetune_data/unified_multitask_train_balanced.jsonl"
VAL_JSONL="finetune_data/unified_multitask_val_balanced.jsonl"

echo "============================================================"
echo " v2 retrain: epochs=$EPOCHS  lora_r=$LORA_R alpha=$LORA_ALPHA  balance=oversample bf16"
echo " output → $OUT   (v1 untouched)"
echo "============================================================"

# ── Step 1: build the task-balanced unified JSONL (oversample) ──
echo "──── building balanced JSONL (oversample) ────"
$CPY prepare_unified_multitask_jsonl.py \
    --balance oversample \
    --out-train "$TRAIN_JSONL" \
    --out-val   "$VAL_JSONL"

# ── Step 2: fine-tune v2 ──
echo "──── training v2 ────"
$CPY finetune_florence2_od_lora.py \
    --model-id microsoft/Florence-2-large-ft \
    --train-jsonl "$TRAIN_JSONL" \
    --eval-jsonl  "$VAL_JSONL" \
    --output-dir  "$OUT" \
    --num-train-epochs "$EPOCHS" \
    --per-device-train-batch-size 1 \
    --gradient-accumulation-steps 8 \
    --learning-rate "$LR" \
    --lora-r "$LORA_R" \
    --lora-alpha "$LORA_ALPHA" \
    --bf16 \
    --device cuda:0

echo ""
echo "============================================================"
echo " v2 training complete → $OUT"
echo " Next: evaluate v2 vs v1 (test domains untouched), e.g.:"
echo "   cd cross_domain_eval"
echo "   python eval_segmentation.py --dataset zerowaste --seg-method cascade \\"
echo "     --skip deeplabv3plus,unet,maskrcnn,sam \\"
echo "     --lora ../$OUT --output-dir ../eval_results/seg_zerowaste_v2"
echo "============================================================"
