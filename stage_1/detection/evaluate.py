#!/usr/bin/env python3
"""
Unified Evaluation Script for Object Detection

Computes and compares metrics across all models (task-specific and foundation).

Usage:
    python evaluate.py --all
    python evaluate.py --model yolov8 --weights runs/detect/best.pt
    python evaluate.py --compare
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

import torch
import numpy as np
from PIL import Image

# ============================================================================
# Configuration
# ============================================================================

METRICS = ['mAP50', 'mAP50-95', 'precision', 'recall']

MODELS = {
    'task_specific': ['yolov8', 'fasterrcnn', 'retinanet'],
    'foundation': ['grounding_dino', 'florence2'],
}

DATASETS = ['taco', 'icra19']


# ============================================================================
# Results Collection
# ============================================================================

def find_latest_results(runs_dir: str, model_name: str) -> Optional[Path]:
    """Find the latest results file for a model"""
    
    runs_path = Path(runs_dir)
    
    if not runs_path.exists():
        return None
    
    # Look for results files
    patterns = [
        f"**/cross_domain_results.json",
        f"**/*results*.json",
    ]
    
    results_files = []
    for pattern in patterns:
        results_files.extend(runs_path.glob(pattern))
    
    if not results_files:
        return None
    
    # Return most recent
    results_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return results_files[0]


def load_all_results(detection_dir: Path) -> Dict:
    """Load results from all model runs"""
    
    all_results = {}
    
    # Task-specific models
    for model in MODELS['task_specific']:
        runs_dir = detection_dir / 'runs' / model.replace('_', '')
        
        if runs_dir.exists():
            # Find subdirectories with results
            for run_dir in runs_dir.iterdir():
                if run_dir.is_dir():
                    results_file = run_dir / 'cross_domain_results.json'
                    if results_file.exists():
                        with open(results_file) as f:
                            all_results[model] = json.load(f)
                        break
    
    # Foundation models
    for model in MODELS['foundation']:
        runs_dir = detection_dir / 'runs' / model
        results_file = find_latest_results(str(runs_dir), model)
        
        if results_file:
            with open(results_file) as f:
                all_results[model] = json.load(f)
    
    return all_results


# ============================================================================
# Comparison Tables
# ============================================================================

def create_comparison_table(results: Dict) -> pd.DataFrame:
    """Create a comparison table of all models"""
    
    rows = []
    
    for model, data in results.items():
        row = {'Model': model, 'Type': 'Task-Specific' if model in MODELS['task_specific'] else 'Foundation'}
        
        # In-domain metrics
        if 'taco_val' in data:
            in_domain = data['taco_val']
        elif 'in_domain' in data:
            in_domain = data['in_domain']
        elif 'datasets' in data and 'taco' in data['datasets']:
            in_domain = data['datasets']['taco']
        else:
            in_domain = {}
        
        row['In-Domain mAP50'] = in_domain.get('mAP50', in_domain.get('mAP', 'N/A'))
        row['In-Domain mAP50-95'] = in_domain.get('mAP50-95', 'N/A')
        
        # Cross-domain metrics
        if 'icra19_test' in data:
            cross_domain = data['icra19_test']
        elif 'cross_domain' in data and isinstance(data['cross_domain'], dict):
            cross_domain = data['cross_domain']
        elif 'datasets' in data and 'icra19' in data['datasets']:
            cross_domain = data['datasets']['icra19']
        else:
            cross_domain = {}
        
        row['Cross-Domain mAP50'] = cross_domain.get('mAP50', cross_domain.get('mAP', 'N/A'))
        row['Cross-Domain mAP50-95'] = cross_domain.get('mAP50-95', 'N/A')
        
        # Generalization gap
        if isinstance(row['In-Domain mAP50'], (int, float)) and isinstance(row['Cross-Domain mAP50'], (int, float)):
            row['Gap'] = row['In-Domain mAP50'] - row['Cross-Domain mAP50']
            row['Retention %'] = (row['Cross-Domain mAP50'] / row['In-Domain mAP50'] * 100) if row['In-Domain mAP50'] > 0 else 0
        else:
            row['Gap'] = 'N/A'
            row['Retention %'] = 'N/A'
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


def print_comparison_table(df: pd.DataFrame):
    """Pretty print the comparison table"""
    
    print("\n" + "="*100)
    print("📊 Model Comparison: Task-Specific vs Foundation Models")
    print("="*100)
    
    # Format numeric columns
    for col in df.columns:
        if col not in ['Model', 'Type'] and col != 'Retention %':
            df[col] = df[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
        elif col == 'Retention %':
            df[col] = df[col].apply(lambda x: f"{x:.1f}%" if isinstance(x, (int, float)) else x)
    
    print(df.to_string(index=False))
    print("="*100)


def save_comparison_report(results: Dict, output_dir: Path):
    """Generate and save a comprehensive comparison report"""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create comparison table
    df = create_comparison_table(results)
    
    # Save as CSV
    csv_path = output_dir / 'model_comparison.csv'
    df.to_csv(csv_path, index=False)
    
    # Save as JSON
    json_path = output_dir / 'all_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate markdown report
    report = generate_markdown_report(results, df)
    md_path = output_dir / 'COMPARISON_REPORT.md'
    with open(md_path, 'w') as f:
        f.write(report)
    
    print(f"\n✅ Reports saved to:")
    print(f"   CSV: {csv_path}")
    print(f"   JSON: {json_path}")
    print(f"   Markdown: {md_path}")


def generate_markdown_report(results: Dict, df: pd.DataFrame) -> str:
    """Generate a comprehensive markdown report"""
    
    report = f"""# Object Detection: Model Comparison Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report compares **task-specific models** (trained on TACO dataset) with 
**foundation models** (zero-shot) for waste detection across different domains.

### Key Findings

| Metric | Task-Specific | Foundation |
|--------|--------------|------------|
| Avg In-Domain mAP@0.5 | - | - |
| Avg Cross-Domain mAP@0.5 | - | - |
| Avg Retention | - | - |

## Detailed Results

### Model Comparison Table

{df.to_markdown(index=False)}

## Analysis

### In-Domain Performance (TACO Validation)

Models evaluated on the same distribution as training data (or similar for foundation models).

### Cross-Domain Performance (Trash-ICRA19 Test)

Models evaluated on a completely different dataset to measure generalization.

### Generalization Gap

The difference between in-domain and cross-domain performance indicates how well
models generalize to unseen domains.

## Methodology

### Datasets
- **Training**: TACO (1,500 images, unified to 2 classes)
- **In-Domain Test**: TACO validation split
- **Cross-Domain Test**: Trash-ICRA19 test set (underwater trash)

### Models Evaluated

#### Task-Specific (Trained on TACO)
1. **YOLOv8-M**: Single-stage detector, fast inference
2. **Faster R-CNN ResNet50-FPN**: Two-stage detector, strong baseline
3. **RetinaNet ResNet50-FPN**: Single-stage with focal loss

#### Foundation Models (Zero-Shot)
1. **Grounding-DINO**: Open-vocabulary detector with text prompts
2. **Florence-2**: Microsoft's vision foundation model

### Metrics
- **mAP@0.5**: Mean Average Precision at IoU threshold 0.5
- **mAP@0.5:0.95**: Mean AP across IoU thresholds 0.5 to 0.95
- **Retention %**: Cross-domain performance / In-domain performance × 100

## Conclusions

[To be filled after running experiments]

---

*This report is part of the Waste Management Foundation Model vs Task-Specific Model comparison study.*
"""
    
    return report


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Unified evaluation and comparison")
    
    parser.add_argument('--compare', action='store_true', 
                        help='Compare all available results')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for reports')
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    output_dir = script_dir / args.output_dir
    
    if args.compare:
        print("\n🔄 Loading results from all model runs...")
        
        results = load_all_results(script_dir)
        
        if not results:
            print("❌ No results found. Run training/evaluation scripts first.")
            print("   Example:")
            print("   python convert_annotations.py --all")
            print("   python train_yolov8.py --epochs 50")
            sys.exit(1)
        
        print(f"Found results for {len(results)} models: {list(results.keys())}")
        
        df = create_comparison_table(results)
        print_comparison_table(df)
        
        save_comparison_report(results, output_dir)
    else:
        print("Usage: python evaluate.py --compare")


if __name__ == "__main__":
    main()
