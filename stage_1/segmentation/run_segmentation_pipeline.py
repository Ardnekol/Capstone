#!/usr/bin/env python3
"""
Segmentation Pipeline Runner

Orchestrates the complete segmentation pipeline:
1. Preprocess datasets
2. Train models (U-Net, DeepLabV3+, Mask R-CNN)
3. Evaluate SAM (zero-shot)
4. Generate comprehensive report

Usage:
    python run_segmentation_pipeline.py
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

def check_requirements():
    """Check if required packages are installed."""
    print("🔍 Checking requirements...")

    try:
        import torch
        import torchvision
        import segmentation_models_pytorch as smp
        import pycocotools
        import matplotlib
        import seaborn
        print("✅ All required packages are available")
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

def preprocess_data():
    """Run data preprocessing."""
    cmd = "cd /u/student/2024/cs24mtech11024/Capstone/stage_1/segmentation && python scripts/preprocess_segmentation.py"
    return run_command(cmd, "Preprocessing TACO and BePLi datasets")

def train_unet():
    """Train U-Net model."""
    cmd = "cd /u/student/2024/cs24mtech11024/Capstone/stage_1/segmentation && python scripts/train_unet.py"
    return run_command(cmd, "Training U-Net model")

def train_deeplabv3plus():
    """Train DeepLabV3+ model."""
    cmd = "cd /u/student/2024/cs24mtech11024/Capstone/stage_1/segmentation && python scripts/train_deeplabv3plus.py"
    return run_command(cmd, "Training DeepLabV3+ model")

def train_maskrcnn():
    """Train Mask R-CNN model."""
    cmd = "cd /u/student/2024/cs24mtech11024/Capstone/stage_1/segmentation && python scripts/train_maskrcnn.py"
    return run_command(cmd, "Training Mask R-CNN model")

def evaluate_sam():
    """Evaluate SAM model (zero-shot)."""
    cmd = "cd /u/student/2024/cs24mtech11024/Capstone/stage_1/segmentation && python scripts/eval_sam.py"
    return run_command(cmd, "Evaluating SAM (zero-shot)")

def generate_report():
    """Generate final report."""
    cmd = "cd /u/student/2024/cs24mtech11024/Capstone/stage_1/segmentation && python scripts/evaluate_segmentation.py"
    return run_command(cmd, "Generating comprehensive report")

def main():
    print("🚀 Starting Segmentation Pipeline")
    print("=" * 60)

    # Check requirements
    if not check_requirements():
        sys.exit(1)

    # Create results directory
    os.makedirs('results', exist_ok=True)

    # Pipeline steps
    steps = [
        ("Data Preprocessing", preprocess_data),
        ("U-Net Training", train_unet),
        ("DeepLabV3+ Training", train_deeplabv3plus),
        ("Mask R-CNN Training", train_maskrcnn),
        ("SAM Evaluation", evaluate_sam),
        ("Report Generation", generate_report)
    ]

    results = {}
    start_time = time.time()

    for step_name, step_func in steps:
        step_start = time.time()
        success = step_func()
        step_time = time.time() - step_start

        results[step_name] = {
            'success': success,
            'time': step_time
        }

        if not success:
            print(f"\n❌ Pipeline failed at: {step_name}")
            break

    # Summary
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print("📊 PIPELINE SUMMARY")
    print(f"{'='*60}")

    successful_steps = sum(1 for r in results.values() if r['success'])
    total_steps = len(steps)

    print(f"Completed: {successful_steps}/{total_steps} steps")
    print(".2f")

    for step_name, result in results.items():
        status = "✅" if result['success'] else "❌"
        print(".2f")

    if successful_steps == total_steps:
        print("\n🎉 All steps completed successfully!")
        print("📁 Results available in: Capstone/stage_1/segmentation/results/")
        print("📄 Report: SEGMENTATION_REPORT.md")
        print("📊 Plot: segmentation_iou_comparison.png")
    else:
        print(f"\n⚠️  Pipeline completed {successful_steps}/{total_steps} steps")
        print("Check the output above for error details")

if __name__ == '__main__':
    main()