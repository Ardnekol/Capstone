#!/bin/bash
# Step 5: Generate three-way comparison report
# Run this AFTER all three evaluations are done.

set -e

PYTHON="/u/student/2024/cs24mtech11024/.conda/envs/Capstone/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Generating comparison report..."
$PYTHON generate_comparison.py

echo "Report saved to: reports/"
