#!/bin/bash
# Run Florence-2 evaluation on benchmark datasets and generate baseline comparison matrix.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PATH="$HOME/.conda/envs/Capstone/bin:$PATH"
export CONDA_DEFAULT_ENV="Capstone"
export CUDA_LAUNCH_BLOCKING=1

MAX_IMAGES=100
RUN_ID=$(date +%d_%m_%Y_%I_%M_%S_%p)
MODE="finetuned"
MODEL_ID="finetuned/florence2_unified_multitask_lora"
MODEL_LABEL="Florence-2 Fine-Tuned"

for arg in "$@"; do
    case $arg in
        --zero-shot)
            MODE="zeroshot"
            MODEL_ID="microsoft/Florence-2-large-ft"
            MODEL_LABEL="Florence-2 Zero-Shot"
            ;;
        --finetuned)
            MODE="finetuned"
            MODEL_ID="finetuned/florence2_unified_multitask_lora"
            MODEL_LABEL="Florence-2 Fine-Tuned"
            ;;
        --max-images=*)
            MAX_IMAGES="${arg#*=}"
            ;;
        --model-id=*)
            MODEL_ID="${arg#*=}"
            ;;
        *)
            echo "Unknown arg: $arg"
            echo "Usage: bash run_unified_100.sh [--zero-shot|--finetuned] [--max-images=N] [--model-id=...]"
            exit 1
            ;;
    esac
done

if [[ "$MODE" == "zeroshot" ]]; then
    EVAL_DIR="./eval_results/zeroshot_${RUN_ID}_${MAX_IMAGES}"
else
    EVAL_DIR="./eval_results/unified_${RUN_ID}_${MAX_IMAGES}"
fi

echo "============================================================"
echo "Florence-2 Benchmark Evaluation"
echo "============================================================"
echo "Mode:        $MODE"
echo "Run ID:      $RUN_ID"
echo "Max images:  $MAX_IMAGES per dataset"
echo "Model:       $MODEL_ID"
echo "Label:       $MODEL_LABEL"
echo "Eval output: $EVAL_DIR"
echo "============================================================"

if [[ "$MODE" == "finetuned" && ! -d "$MODEL_ID" ]]; then
    echo "ERROR: Model not found at $MODEL_ID"
    echo "Please fine-tune the model first or use --zero-shot"
    exit 1
fi

python3 evaluate_unified_model.py \
    --model-id "$MODEL_ID" \
    --max-images "$MAX_IMAGES" \
    --output-dir "$EVAL_DIR"

python3 generate_comparison_matrix.py \
    --eval-results "$EVAL_DIR/results.json" \
    --output-dir "$EVAL_DIR" \
    --model-label "$MODEL_LABEL"

echo ""
echo "============================================================"
echo "Complete"
echo "============================================================"
echo "Results:   $EVAL_DIR/results.json"
echo "Matrix:    $EVAL_DIR/comparison_matrix.md"
echo "JSON:      $EVAL_DIR/comparison_matrix.json"