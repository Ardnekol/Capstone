#!/bin/bash
# ──────────────────────────────────────────────────────────────
# E1 — Single-Task Ablation Evaluation
#
# Evaluates each LoRA (cls-only, det-only, seg-only, unified) on ALL
# five benchmarks, so the ablation table has the same denominator across
# every row. Each run writes to its own eval_results subdirectory; the
# table is built afterwards by make_ablation_table.py.
#
# Usage:
#   bash eval_single_task_ablation.sh                # eval all 4 LoRAs
#   bash eval_single_task_ablation.sh cls            # only cls-only
#   bash eval_single_task_ablation.sh --max-images=200
# ──────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PATH="$HOME/.conda/envs/Capstone/bin:$PATH"
export CONDA_DEFAULT_ENV="Capstone"

BEST_GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t',' -k2 -n | head -1 | cut -d',' -f1 | tr -d ' ')
export CUDA_VISIBLE_DEVICES="${BEST_GPU}"
echo "Auto-selected GPU ${BEST_GPU}"

DEVICE="cuda:0"
MAX_IMAGES=200  # per benchmark; keeps full ablation under ~6 hr; set 0 for full

declare -a MODELS_TO_EVAL=()
for arg in "$@"; do
    case $arg in
        --max-images=*) MAX_IMAGES="${arg#*=}" ;;
        cls)     MODELS_TO_EVAL+=("florence2_cls_only_lora") ;;
        det)     MODELS_TO_EVAL+=("florence2_det_only_lora") ;;
        seg)     MODELS_TO_EVAL+=("florence2_seg_only_lora") ;;
        unified) MODELS_TO_EVAL+=("florence2_unified_multitask_lora") ;;
        all)     MODELS_TO_EVAL+=("florence2_cls_only_lora" "florence2_det_only_lora" "florence2_unified_multitask_lora") ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

# Default model set excludes seg-only (mode-collapsed during training).
if [[ ${#MODELS_TO_EVAL[@]} -eq 0 ]]; then
    MODELS_TO_EVAL=(florence2_cls_only_lora florence2_det_only_lora florence2_unified_multitask_lora)
fi

echo "============================================================"
echo "E1 — Single-Task Ablation Evaluation"
echo "============================================================"
echo "Models:      ${MODELS_TO_EVAL[*]}"
echo "Device:      $DEVICE (GPU $BEST_GPU)"
echo "Max images:  $MAX_IMAGES per benchmark (0 = full)"
echo "============================================================"

mkdir -p eval_results/e1_ablation

for model_dir in "${MODELS_TO_EVAL[@]}"; do
    full_path="finetuned/$model_dir"
    if [[ ! -d "$full_path" ]]; then
        echo "[skip] Adapter not found: $full_path"
        continue
    fi

    out_dir="eval_results/e1_ablation/${model_dir}"
    mkdir -p "$(dirname "$out_dir")"
    echo ""
    echo "──── Evaluating $model_dir ────"
    echo "  adapter: $full_path"
    echo "  output:  $out_dir/"

    python3 evaluate_unified_model.py \
        --model-id "$full_path" \
        --device "$DEVICE" \
        --max-images "$MAX_IMAGES" \
        --output-dir "$out_dir" \
        2>&1 | tee "${out_dir}_console.log" || true
done

echo ""
echo "============================================================"
echo "E1 evaluation complete!"
echo "Next: python3 make_ablation_table.py"
echo "============================================================"
