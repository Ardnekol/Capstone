#!/bin/bash
#
# Preprocess Classification Datasets
# ==================================
#
# This script runs preprocessing for both TrashNet and RealWaste datasets.
#

set -e  # Exit on error

cd "$(dirname "$0")"

echo "========================================"
echo "Preprocessing Classification Datasets"
echo "========================================"
echo ""

echo "Step 1: Preprocessing TrashNet..."
python preprocess_trashnet.py

echo ""
echo "Step 2: Preprocessing RealWaste..."
python preprocess_realwaste.py

echo ""
echo "All preprocessing complete!"