#!/bin/bash
# Step 4a: Evaluate Florence-2 ZERO-SHOT on the test set

set -e

PYTHON="/u/student/2024/cs24mtech11024/.conda/envs/Capstone/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="/u/student/2024/cs24mtech11024/Capstone/stage_2:$PYTHONPATH"
cd "$SCRIPT_DIR"

echo "Florence-2 Zero-Shot Evaluation..."
$PYTHON evaluate_florence2_cctv.py \
    --model-id "microsoft/Florence-2-large" \
    --split test \
    --visualize

echo "Done. Results in: eval_results/florence2_zeroshot_test_*/"
