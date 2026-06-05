#!/bin/bash
# ──────────────────────────────────────────────────────────────
# E1 — Single-Task Ablation Training
#
# Trains THREE LoRAs separately (one per task) using identical hyperparams
# to the unified run (train_unified.sh), against the SAME base model
# (microsoft/Florence-2-large-ft). Outputs:
#
#   finetuned/florence2_cls_only_lora/   (classification on TrashNet)
#   finetuned/florence2_det_only_lora/   (detection on TACO OD)
#   finetuned/florence2_seg_only_lora/   (segmentation on TACO seg)
#
# These three + the existing florence2_unified_multitask_lora form the
# 4-way comparison for the workshop paper's multi-task ablation.
#
# Usage:
#   bash train_single_task_ablation.sh                  # train all 3
#   bash train_single_task_ablation.sh cls              # train only cls
#   bash train_single_task_ablation.sh det seg          # train det and seg
#   bash train_single_task_ablation.sh --skip-prep all  # skip val-split prep
# ──────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Activate environment (matches train_unified.sh) ──
export PATH="$HOME/.conda/envs/Capstone/bin:$PATH"
export CONDA_DEFAULT_ENV="Capstone"

# Auto-select least-loaded GPU
BEST_GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t',' -k2 -n | head -1 | cut -d',' -f1 | tr -d ' ')
export CUDA_VISIBLE_DEVICES="${BEST_GPU}"
export CUDA_LAUNCH_BLOCKING=1
echo "Auto-selected GPU ${BEST_GPU} (least memory usage)"

python3 -c "import torch; print('torch:', torch.__version__)" || { echo "ERROR: torch not found."; exit 1; }

# ── Defaults (MUST match train_unified.sh for fair comparison) ──
MODEL_ID="microsoft/Florence-2-large-ft"
EPOCHS=3
LR="1e-4"
BATCH_SIZE=1
GRAD_ACCUM=8
LORA_R=16
LORA_ALPHA=32
DEVICE="cuda:0"

# ── Parse args ──
SKIP_PREP=false
declare -a TASKS=()
for arg in "$@"; do
    case $arg in
        --skip-prep) SKIP_PREP=true ;;
        cls|det|seg|all) TASKS+=("$arg") ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

# Default: train all three
if [[ ${#TASKS[@]} -eq 0 ]] || [[ " ${TASKS[*]} " =~ " all " ]]; then
    TASKS=(cls det seg)
fi

echo "============================================================"
echo "E1 — Single-Task Ablation Training"
echo "============================================================"
echo "Base model:   $MODEL_ID"
echo "Tasks:        ${TASKS[*]}"
echo "Epochs:       $EPOCHS"
echo "LR:           $LR"
echo "Batch/Accum:  $BATCH_SIZE / $GRAD_ACCUM"
echo "LoRA:         r=$LORA_R, alpha=$LORA_ALPHA"
echo "Device:       $DEVICE (GPU $BEST_GPU)"
echo "============================================================"

# ── Step 0: Ensure val splits exist for det and seg ──
if [[ "$SKIP_PREP" == false ]]; then
    echo ""
    echo "──── Step 0: Prepare 90/10 val splits ────"
    python3 prepare_single_task_splits.py
fi

run_one_task() {
    local task=$1
    local train_jsonl=$2
    local val_jsonl=$3
    local out_dir=$4

    echo ""
    echo "──── Training $task-only LoRA ────"
    echo "  train: $train_jsonl"
    echo "  val:   $val_jsonl"
    echo "  out:   $out_dir"

    python3 finetune_florence2_od_lora.py \
        --model-id "$MODEL_ID" \
        --train-jsonl "$train_jsonl" \
        --eval-jsonl  "$val_jsonl" \
        --output-dir  "$out_dir" \
        --num-train-epochs "$EPOCHS" \
        --per-device-train-batch-size "$BATCH_SIZE" \
        --gradient-accumulation-steps "$GRAD_ACCUM" \
        --learning-rate "$LR" \
        --lora-r "$LORA_R" \
        --lora-alpha "$LORA_ALPHA" \
        --device "$DEVICE"
}

# ── Step 1: Classification-only ──
if [[ " ${TASKS[*]} " =~ " cls " ]]; then
    run_one_task "cls" \
        "finetune_data/trashnet_caption_train.jsonl" \
        "finetune_data/trashnet_caption_val.jsonl" \
        "finetuned/florence2_cls_only_lora"
fi

# ── Step 2: Detection-only ──
if [[ " ${TASKS[*]} " =~ " det " ]]; then
    run_one_task "det" \
        "finetune_data/taco_od_train_split.jsonl" \
        "finetune_data/taco_od_val_split.jsonl" \
        "finetuned/florence2_det_only_lora"
fi

# ── Step 3: Segmentation-only ──
if [[ " ${TASKS[*]} " =~ " seg " ]]; then
    run_one_task "seg" \
        "finetune_data/taco_seg_train_split.jsonl" \
        "finetune_data/taco_seg_val_split.jsonl" \
        "finetuned/florence2_seg_only_lora"
fi

echo ""
echo "============================================================"
echo "E1 training complete!"
echo "Next: bash eval_single_task_ablation.sh"
echo "============================================================"
