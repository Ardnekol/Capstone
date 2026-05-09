#!/usr/bin/env python3
"""
Run All Detection Experiments

Master script to run the complete detection experiment pipeline.

Usage:
    python run_experiments.py --prepare      # Convert annotations only
    python run_experiments.py --train        # Train all task-specific models
    python run_experiments.py --eval         # Evaluate foundation models
    python run_experiments.py --all          # Run everything
    python run_experiments.py --quick        # Quick test run (fewer epochs)
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

# ============================================================================
# Configuration
# ============================================================================

EXPERIMENT_CONFIG = {
    # Training settings
    'epochs': 50,
    'batch_yolo': 16,
    'batch_rcnn': 4,  # Lower for memory
    
    # Quick test settings
    'quick_epochs': 5,
    'quick_batch': 8,
    
    # Device
    'device': 'cuda:0',
}


# ============================================================================
# Experiment Steps
# ============================================================================

def run_command(cmd: str, description: str):
    """Run a shell command and handle errors"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"❌ Error: {description} failed")
        return False
    
    print(f"✅ {description} completed")
    return True


def prepare_data(detection_dir: Path):
    """Step 1: Convert annotations to YOLO format"""
    
    print("\n" + "="*60)
    print("📁 STEP 1: Preparing Data")
    print("="*60)
    
    cmd = f"cd {detection_dir} && python convert_annotations.py --all"
    return run_command(cmd, "Converting annotations to YOLO format")


def train_yolov8(detection_dir: Path, epochs: int, batch: int, device: str):
    """Train YOLOv8"""
    
    cmd = f"cd {detection_dir} && python train_yolov8.py --epochs {epochs} --batch {batch} --device {device}"
    return run_command(cmd, "Training YOLOv8")


def train_fasterrcnn(detection_dir: Path, epochs: int, batch: int, device: str):
    """Train Faster R-CNN"""
    
    cmd = f"cd {detection_dir} && python train_fasterrcnn.py --epochs {epochs} --batch {batch} --device {device}"
    return run_command(cmd, "Training Faster R-CNN")


def train_retinanet(detection_dir: Path, epochs: int, batch: int, device: str):
    """Train RetinaNet"""
    
    cmd = f"cd {detection_dir} && python train_retinanet.py --epochs {epochs} --batch {batch} --device {device}"
    return run_command(cmd, "Training RetinaNet")


def eval_grounding_dino(detection_dir: Path, device: str):
    """Evaluate Grounding-DINO"""
    
    cmd = f"cd {detection_dir} && python eval_grounding_dino.py --device {device}"
    return run_command(cmd, "Evaluating Grounding-DINO (Zero-Shot)")


def eval_florence2(detection_dir: Path, device: str):
    """Evaluate Florence-2"""
    
    cmd = f"cd {detection_dir} && python eval_florence2.py --device {device}"
    return run_command(cmd, "Evaluating Florence-2 (Zero-Shot)")


def generate_report(detection_dir: Path):
    """Generate comparison report"""
    
    cmd = f"cd {detection_dir} && python evaluate.py --compare"
    return run_command(cmd, "Generating Comparison Report")


# ============================================================================
# Main Pipeline
# ============================================================================

def run_all_experiments(detection_dir: Path, config: dict):
    """Run the complete experiment pipeline"""
    
    print("\n" + "="*80)
    print("🔬 WASTE DETECTION EXPERIMENT PIPELINE")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Working Directory: {detection_dir}")
    print(f"Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    results = {}
    
    # Step 1: Prepare data
    results['prepare'] = prepare_data(detection_dir)
    if not results['prepare']:
        print("❌ Data preparation failed. Exiting.")
        return results
    
    # Step 2: Train task-specific models
    print("\n" + "="*60)
    print("🏋️ STEP 2: Training Task-Specific Models")
    print("="*60)
    
    results['yolov8'] = train_yolov8(
        detection_dir, 
        config['epochs'], 
        config['batch_yolo'],
        config['device']
    )
    
    results['fasterrcnn'] = train_fasterrcnn(
        detection_dir,
        config['epochs'],
        config['batch_rcnn'],
        config['device']
    )
    
    results['retinanet'] = train_retinanet(
        detection_dir,
        config['epochs'],
        config['batch_rcnn'],
        config['device']
    )
    
    # Step 3: Evaluate foundation models
    print("\n" + "="*60)
    print("🔍 STEP 3: Evaluating Foundation Models")
    print("="*60)
    
    results['grounding_dino'] = eval_grounding_dino(detection_dir, config['device'])
    results['florence2'] = eval_florence2(detection_dir, config['device'])
    
    # Step 4: Generate comparison report
    print("\n" + "="*60)
    print("📊 STEP 4: Generating Comparison Report")
    print("="*60)
    
    results['report'] = generate_report(detection_dir)
    
    # Summary
    print("\n" + "="*80)
    print("📋 EXPERIMENT SUMMARY")
    print("="*80)
    
    for step, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {step}")
    
    success_count = sum(results.values())
    total_count = len(results)
    
    print(f"\nCompleted: {success_count}/{total_count} steps")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if success_count == total_count:
        print("\n🎉 All experiments completed successfully!")
        print(f"📄 Check the report at: {detection_dir}/results/COMPARISON_REPORT.md")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run detection experiments")
    
    parser.add_argument('--prepare', action='store_true', help='Prepare data only')
    parser.add_argument('--train', action='store_true', help='Train task-specific models')
    parser.add_argument('--eval', action='store_true', help='Evaluate foundation models')
    parser.add_argument('--compare', action='store_true', help='Generate comparison report')
    parser.add_argument('--all', action='store_true', help='Run complete pipeline')
    parser.add_argument('--quick', action='store_true', help='Quick test with fewer epochs')
    
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda:0')
    
    args = parser.parse_args()
    
    detection_dir = Path(__file__).parent
    
    # Build config
    config = EXPERIMENT_CONFIG.copy()
    
    if args.quick:
        config['epochs'] = config['quick_epochs']
        config['batch_yolo'] = config['quick_batch']
        config['batch_rcnn'] = 2
    
    if args.epochs:
        config['epochs'] = args.epochs
    
    config['device'] = args.device
    
    # Run selected steps
    if args.all:
        run_all_experiments(detection_dir, config)
    else:
        if args.prepare:
            prepare_data(detection_dir)
        
        if args.train:
            train_yolov8(detection_dir, config['epochs'], config['batch_yolo'], config['device'])
            train_fasterrcnn(detection_dir, config['epochs'], config['batch_rcnn'], config['device'])
            train_retinanet(detection_dir, config['epochs'], config['batch_rcnn'], config['device'])
        
        if args.eval:
            eval_grounding_dino(detection_dir, config['device'])
            eval_florence2(detection_dir, config['device'])
        
        if args.compare:
            generate_report(detection_dir)
        
        if not any([args.prepare, args.train, args.eval, args.compare]):
            print("Usage:")
            print("  python run_experiments.py --all      # Run complete pipeline")
            print("  python run_experiments.py --prepare  # Prepare data only")
            print("  python run_experiments.py --train    # Train models")
            print("  python run_experiments.py --eval     # Evaluate foundation models")
            print("  python run_experiments.py --compare  # Generate report")
            print("  python run_experiments.py --quick    # Quick test run")


if __name__ == "__main__":
    main()
