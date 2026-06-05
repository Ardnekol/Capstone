#!/bin/bash
# ──────────────────────────────────────────────────────────────
# E1 — Multi-Task Ablation: full pipeline
#
# Trains 3 single-task LoRAs (cls/det/seg) and evaluates each on all 5
# benchmarks, then builds the comparison table. Designed to be launched
# in tmux on the deploy GPU — total wall time ~15-18 hours.
#
# Usage:
#   bash run_e1_ablation.sh                         # full pipeline
#   bash run_e1_ablation.sh --skip-train            # eval-only (use existing LoRAs)
#   bash run_e1_ablation.sh --skip-train --skip-eval  # just rebuild the table
#   bash run_e1_ablation.sh --max-images=400        # control eval cost
# ──────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SKIP_TRAIN=false
SKIP_EVAL=false
MAX_IMAGES=200

for arg in "$@"; do
    case $arg in
        --skip-train)    SKIP_TRAIN=true ;;
        --skip-eval)     SKIP_EVAL=true ;;
        --max-images=*)  MAX_IMAGES="${arg#*=}" ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

echo "============================================================"
echo "E1 — Multi-Task Ablation: full pipeline"
echo "============================================================"
echo "Skip train:  $SKIP_TRAIN"
echo "Skip eval:   $SKIP_EVAL"
echo "Max images:  $MAX_IMAGES"
echo "Started at:  $(date)"
echo "============================================================"

if [[ "$SKIP_TRAIN" == false ]]; then
    bash train_single_task_ablation.sh all
fi

if [[ "$SKIP_EVAL" == false ]]; then
    bash eval_single_task_ablation.sh all --max-images="$MAX_IMAGES"
fi

echo ""
echo "──── Building ablation comparison table ────"
python3 make_ablation_table.py

echo ""
echo "============================================================"
echo "E1 ablation done at $(date)"
echo "Comparison table:    eval_results/e1_ablation/comparison.md"
echo "Per-model results:   eval_results/e1_ablation/<lora_name>/results.json"
echo "============================================================"
