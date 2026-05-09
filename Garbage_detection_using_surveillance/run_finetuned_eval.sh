#!/bin/bash
# Step 4b: Evaluate Florence-2 FINE-TUNED on the test set

set -e

PYTHON="/u/student/2024/cs24mtech11024/.conda/envs/Capstone/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ADAPTER="${1:-$SCRIPT_DIR/finetuned/florence2_cctv_garbage_od_lora}"
export PYTHONPATH="/u/student/2024/cs24mtech11024/Capstone/stage_2:$PYTHONPATH"
cd "$SCRIPT_DIR"

if [ ! -d "$ADAPTER" ]; then
    echo "Adapter not found at: $ADAPTER"
    echo "Run train_florence2_cctv.sh first, or pass the adapter path as argument:"
    echo "  bash run_finetuned_eval.sh /path/to/adapter"
    exit 1
fi

echo "Florence-2 Fine-Tuned Evaluation..."
echo "  Adapter: $ADAPTER"
$PYTHON evaluate_florence2_cctv.py \
    --model-id "microsoft/Florence-2-large" \
    --adapter  "$ADAPTER" \
    --split test \
    --visualize

echo "Done. Results in: eval_results/florence2_finetuned_test_*/"
