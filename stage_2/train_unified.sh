#!/bin/bash
# ──────────────────────────────────────────────────────────────
# Unified Florence-2 Multi-Task Training Pipeline
#
# Usage:
#   bash train_unified.sh                    # Full pipeline
#   bash train_unified.sh --skip-prep        # Skip data prep, go to training
#   bash train_unified.sh --eval-only        # Only run evaluation
#   bash train_unified.sh --quick            # Quick test with 50 images
# ──────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Activate environment ──
# Directly prepend Capstone conda env to PATH so torch is found
export PATH="$HOME/.conda/envs/Capstone/bin:$PATH"
export CONDA_DEFAULT_ENV="Capstone"
# Restrict to a single GPU to avoid DataParallel issues with Florence-2 + LoRA
# Auto-select the GPU with lowest memory usage
BEST_GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t',' -k2 -n | head -1 | cut -d',' -f1 | tr -d ' ')
export CUDA_VISIBLE_DEVICES="${BEST_GPU}"
export CUDA_LAUNCH_BLOCKING=1
echo "Auto-selected GPU ${BEST_GPU} (least memory usage)"
python3 -c "import torch; print('torch:', torch.__version__)" || { echo "ERROR: torch not found."; exit 1; }

# ── Defaults ──
MODEL_ID="microsoft/Florence-2-large-ft"
EPOCHS=3
LR="1e-4"
BATCH_SIZE=1
GRAD_ACCUM=8
LORA_R=16
LORA_ALPHA=32
BALANCE="none"
MAX_IMAGES=0  # 0 = all
DEVICE="cuda:0"

# ── Parse flags ──
SKIP_PREP=false
EVAL_ONLY=false
QUICK=false

for arg in "$@"; do
    case $arg in
        --skip-prep) SKIP_PREP=true ;;
        --eval-only) EVAL_ONLY=true ;;
        --quick)     QUICK=true; MAX_IMAGES=50 ;;
        --balance=*) BALANCE="${arg#*=}" ;;
        --epochs=*)  EPOCHS="${arg#*=}" ;;
        --lr=*)      LR="${arg#*=}" ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

echo "============================================================"
echo "Unified Florence-2 Multi-Task Training Pipeline"
echo "============================================================"
echo "Model:    $MODEL_ID"
echo "Epochs:   $EPOCHS"
echo "LR:       $LR"
echo "Balance:  $BALANCE"
echo "Device:   $DEVICE"
echo "Quick:    $QUICK (max_images=$MAX_IMAGES)"
echo "============================================================"

# ── Step 1: Prepare Segmentation JSONL ──
if [[ "$EVAL_ONLY" == false && "$SKIP_PREP" == false ]]; then
    echo ""
    echo "──── Step 1: Prepare Segmentation JSONL ────"
    python prepare_taco_florence2_seg_jsonl.py
    echo ""

    echo "──── Step 2: Combine into Unified Multi-Task JSONL ────"
    python prepare_unified_multitask_jsonl.py --balance "$BALANCE"
    echo ""
fi

# ── Step 3: Fine-tune Unified LoRA ──
if [[ "$EVAL_ONLY" == false ]]; then
    echo "──── Step 3: Fine-tune Unified Florence-2 LoRA ────"
    python finetune_florence2_od_lora.py \
        --model-id "$MODEL_ID" \
        --train-jsonl finetune_data/unified_multitask_train.jsonl \
        --eval-jsonl  finetune_data/unified_multitask_val.jsonl \
        --output-dir  finetuned/florence2_unified_multitask_lora \
        --num-train-epochs "$EPOCHS" \
        --per-device-train-batch-size "$BATCH_SIZE" \
        --gradient-accumulation-steps "$GRAD_ACCUM" \
        --learning-rate "$LR" \
        --lora-r "$LORA_R" \
        --lora-alpha "$LORA_ALPHA" \
        --device "$DEVICE"
    echo ""
fi

# ── Step 4: Evaluate Unified Model ──
echo "──── Step 4: Evaluate Unified Model ────"

EVAL_MODEL="finetuned/florence2_unified_multitask_lora"
if [[ ! -d "$EVAL_MODEL" ]]; then
    echo "Fine-tuned model not found at $EVAL_MODEL, using base: $MODEL_ID"
    EVAL_MODEL="$MODEL_ID"
fi

python evaluate_unified_model.py \
    --model-id "$EVAL_MODEL" \
    --device "$DEVICE" \
    --max-images "$MAX_IMAGES"

echo ""
echo "============================================================"
echo "Pipeline complete!"
echo "============================================================"
