#!/usr/bin/env python3
"""
YOLOv8 Training Script for Object Detection

Train YOLOv8 on TACO dataset and evaluate on both TACO (in-domain) 
and Trash-ICRA19 (cross-domain).

Usage:
    python train_yolov8.py --epochs 50 --batch 16
    python train_yolov8.py --resume runs/detect/train/weights/last.pt
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import json

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CONFIG = {
    # Model
    'model': 'yolov8m.pt',  # YOLOv8 Medium
    'model_name': 'yolov8m',
    
    # Training
    'epochs': 50,
    'batch': 16,
    'imgsz': 640,
    'patience': 20,  # Early stopping
    
    # Optimizer
    'optimizer': 'AdamW',
    'lr0': 0.001,
    'lrf': 0.01,  # Final learning rate (lr0 * lrf)
    'momentum': 0.937,
    'weight_decay': 0.0005,
    
    # Augmentation
    'augment': True,
    'mosaic': 1.0,
    'mixup': 0.1,
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    'degrees': 0.0,
    'translate': 0.1,
    'scale': 0.5,
    'fliplr': 0.5,
    
    # Device
    'device': 0,  # GPU 0, or 'cpu'
    
    # Misc
    'workers': 8,
    'seed': 42,
    'verbose': True,
}


# ============================================================================
# Training Functions
# ============================================================================

def train_yolov8(config: dict, data_yaml: str, output_dir: str):
    """Train YOLOv8 model"""
    
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ ultralytics not installed. Install with:")
        print("   pip install ultralytics")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("🚀 Training YOLOv8")
    print("="*60)
    print(f"Model: {config['model']}")
    print(f"Data: {data_yaml}")
    print(f"Epochs: {config['epochs']}")
    print(f"Batch size: {config['batch']}")
    print(f"Image size: {config['imgsz']}")
    
    # Load model
    model = YOLO(config['model'])
    
    # Train
    results = model.train(
        data=data_yaml,
        epochs=config['epochs'],
        batch=config['batch'],
        imgsz=config['imgsz'],
        patience=config['patience'],
        optimizer=config['optimizer'],
        lr0=config['lr0'],
        lrf=config['lrf'],
        momentum=config['momentum'],
        weight_decay=config['weight_decay'],
        augment=config['augment'],
        mosaic=config['mosaic'],
        mixup=config['mixup'],
        hsv_h=config['hsv_h'],
        hsv_s=config['hsv_s'],
        hsv_v=config['hsv_v'],
        degrees=config['degrees'],
        translate=config['translate'],
        scale=config['scale'],
        fliplr=config['fliplr'],
        device=config['device'],
        workers=config['workers'],
        seed=config['seed'],
        verbose=config['verbose'],
        project=output_dir,
        name=f"{config['model_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        exist_ok=True,
    )
    
    print("\n✅ Training complete!")
    print(f"Best model: {results.save_dir}/weights/best.pt")
    
    return model, results


def evaluate_yolov8(model_path: str, data_yaml: str, split: str = 'val', 
                    save_dir: str = None, conf: float = 0.25):
    """Evaluate YOLOv8 model on a dataset"""
    
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ ultralytics not installed")
        sys.exit(1)
    
    print(f"\n📊 Evaluating on {split} split...")
    
    model = YOLO(model_path)
    
    results = model.val(
        data=data_yaml,
        split=split,
        conf=conf,
        iou=0.5,
        save_json=True,
        project=save_dir,
        name=f"eval_{split}",
        exist_ok=True,
    )
    
    # Extract metrics
    metrics = {
        'mAP50': float(results.box.map50),
        'mAP50-95': float(results.box.map),
        'precision': float(results.box.mp),
        'recall': float(results.box.mr),
    }
    
    print(f"  mAP@0.5: {metrics['mAP50']:.4f}")
    print(f"  mAP@0.5:0.95: {metrics['mAP50-95']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    
    return metrics


def cross_domain_evaluation(model_path: str, 
                            taco_yaml: str, 
                            icra19_yaml: str,
                            output_dir: str):
    """Evaluate model on both in-domain (TACO) and cross-domain (ICRA19)"""
    
    print("\n" + "="*60)
    print("🔄 Cross-Domain Evaluation")
    print("="*60)
    
    results = {}
    
    # In-domain: TACO val
    print("\n📍 In-Domain: TACO validation set")
    results['taco_val'] = evaluate_yolov8(
        model_path, taco_yaml, split='val', 
        save_dir=output_dir
    )
    
    # Cross-domain: ICRA19 test
    print("\n📍 Cross-Domain: Trash-ICRA19 test set")
    results['icra19_test'] = evaluate_yolov8(
        model_path, icra19_yaml, split='test',
        save_dir=output_dir
    )
    
    # Calculate generalization gap
    gap = results['taco_val']['mAP50'] - results['icra19_test']['mAP50']
    relative_gap = (gap / results['taco_val']['mAP50']) * 100 if results['taco_val']['mAP50'] > 0 else 0
    
    print("\n" + "="*60)
    print("📊 Generalization Analysis")
    print("="*60)
    print(f"In-domain (TACO) mAP@0.5:    {results['taco_val']['mAP50']:.4f}")
    print(f"Cross-domain (ICRA19) mAP@0.5: {results['icra19_test']['mAP50']:.4f}")
    print(f"Absolute Drop:               {gap:.4f}")
    print(f"Relative Drop:               {relative_gap:.1f}%")
    
    results['generalization'] = {
        'absolute_drop': gap,
        'relative_drop_percent': relative_gap,
        'retention_ratio': results['icra19_test']['mAP50'] / results['taco_val']['mAP50'] if results['taco_val']['mAP50'] > 0 else 0
    }
    
    # Save results
    results_path = Path(output_dir) / "cross_domain_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {results_path}")
    
    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 for object detection")
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--model', type=str, default='yolov8m.pt', help='Model variant')
    
    # Paths
    parser.add_argument('--data', type=str, default=None, help='Path to data.yaml')
    parser.add_argument('--output-dir', type=str, default='runs/detect', help='Output directory')
    
    # Modes
    parser.add_argument('--train', action='store_true', help='Train model')
    parser.add_argument('--eval', action='store_true', help='Evaluate model')
    parser.add_argument('--cross-eval', action='store_true', help='Cross-domain evaluation')
    parser.add_argument('--weights', type=str, default=None, help='Model weights for evaluation')
    parser.add_argument('--resume', type=str, default=None, help='Resume training from checkpoint')
    
    # Device
    parser.add_argument('--device', type=str, default='0', help='Device (0, 1, cpu)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Setup paths
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    taco_yaml = data_dir / "taco_yolo" / "data.yaml"
    icra19_yaml = data_dir / "icra19_yolo" / "data.yaml"
    
    # Update config
    config = DEFAULT_CONFIG.copy()
    config['epochs'] = args.epochs
    config['batch'] = args.batch
    config['imgsz'] = args.imgsz
    config['model'] = args.model
    config['device'] = args.device
    config['seed'] = args.seed
    
    # Default: train mode
    if not args.eval and not args.cross_eval:
        args.train = True
    
    if args.train:
        # Check data exists
        if args.data:
            data_yaml = args.data
        elif taco_yaml.exists():
            data_yaml = str(taco_yaml)
        else:
            print(f"❌ Data not found at {taco_yaml}")
            print("   Run: python convert_annotations.py --all")
            sys.exit(1)
        
        model, results = train_yolov8(config, data_yaml, args.output_dir)
        
        # Auto cross-domain evaluation after training
        best_weights = Path(results.save_dir) / "weights" / "best.pt"
        if icra19_yaml.exists():
            cross_domain_evaluation(
                str(best_weights),
                str(taco_yaml),
                str(icra19_yaml),
                str(results.save_dir)
            )
    
    elif args.eval or args.cross_eval:
        if not args.weights:
            print("❌ Please provide --weights path")
            sys.exit(1)
        
        if args.cross_eval:
            cross_domain_evaluation(
                args.weights,
                str(taco_yaml),
                str(icra19_yaml),
                args.output_dir
            )
        else:
            data_yaml = args.data or str(taco_yaml)
            evaluate_yolov8(args.weights, data_yaml, save_dir=args.output_dir)


if __name__ == "__main__":
    main()
