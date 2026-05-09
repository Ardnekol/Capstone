#!/usr/bin/env python3
"""
Grounding-DINO Evaluation Script for Object Detection

Zero-shot evaluation of Grounding-DINO on TACO and Trash-ICRA19 datasets.

Grounding-DINO is an open-set object detection model that can detect 
objects based on text prompts without task-specific training.

Usage:
    python eval_grounding_dino.py --prompt "trash . garbage . litter . waste"
    python eval_grounding_dino.py --dataset taco
    python eval_grounding_dino.py --dataset icra19
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

import torch
import numpy as np
from PIL import Image

# ============================================================================
# Configuration
# ============================================================================

# Text prompts for detection
# Grounding-DINO uses text descriptions to detect objects
PROMPT_TEMPLATES = {
    'simple': "trash",
    'detailed': "trash . garbage . litter . waste . rubbish",
    'categorical': "plastic bottle . paper . cardboard . metal can . food waste . glass . textile . organic waste",
    'unified': "trash . bio . organic waste",
}

# Default model
DEFAULT_MODEL = "IDEA-Research/grounding-dino-tiny"
MODELS = {
    'tiny': "IDEA-Research/grounding-dino-tiny",
    'base': "IDEA-Research/grounding-dino-base",
}


# ============================================================================
# Dataset Loading
# ============================================================================

class DetectionDataset:
    """Simple dataset for evaluation"""
    
    def __init__(self, images_dir: str, labels_dir: str = None):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir) if labels_dir else None
        
        # Get image files
        self.image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            self.image_files.extend(self.images_dir.glob(ext))
        self.image_files = sorted(self.image_files)
        
        print(f"Found {len(self.image_files)} images")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        
        # Load ground truth if available
        gt_boxes = []
        gt_labels = []
        
        if self.labels_dir:
            label_path = self.labels_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                w, h = image.size
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center = float(parts[1]) * w
                            y_center = float(parts[2]) * h
                            width = float(parts[3]) * w
                            height = float(parts[4]) * h
                            
                            x1 = x_center - width/2
                            y1 = y_center - height/2
                            x2 = x_center + width/2
                            y2 = y_center + height/2
                            
                            gt_boxes.append([x1, y1, x2, y2])
                            gt_labels.append(class_id)
        
        return image, {
            'image_path': str(img_path),
            'boxes': np.array(gt_boxes),
            'labels': np.array(gt_labels),
        }


# ============================================================================
# Grounding-DINO Model
# ============================================================================

def load_grounding_dino(model_name: str = DEFAULT_MODEL, device: str = 'cuda'):
    """Load Grounding-DINO model from HuggingFace"""
    
    try:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    except ImportError:
        print("❌ transformers not installed. Install with:")
        print("   pip install transformers")
        sys.exit(1)
    
    print(f"Loading Grounding-DINO: {model_name}")
    
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    
    print("✅ Model loaded successfully")
    
    return model, processor


@torch.no_grad()
def detect_objects(model, processor, image: Image.Image, text_prompt: str,
                   box_threshold: float = 0.25, text_threshold: float = 0.25,
                   device: str = 'cuda'):
    """Run detection on a single image"""
    
    inputs = processor(images=image, text=text_prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    outputs = model(**inputs)
    
    # Post-process (API uses 'threshold' not 'box_threshold')
    target_sizes = torch.tensor([image.size[::-1]]).to(device)
    results = processor.post_process_grounded_object_detection(
        outputs,
        input_ids=inputs['input_ids'],
        target_sizes=target_sizes,
        threshold=box_threshold,
        text_threshold=text_threshold,
    )[0]
    
    return {
        'boxes': results['boxes'].cpu().numpy(),
        'scores': results['scores'].cpu().numpy(),
        'labels': results['labels'],  # text labels
    }


# ============================================================================
# Evaluation
# ============================================================================

def compute_iou(box1, box2):
    """Compute IoU between two boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0


def compute_map(predictions: List[Dict], targets: List[Dict],
                iou_threshold: float = 0.5) -> Dict:
    """Compute mAP at given IoU threshold"""
    
    all_tp = []
    all_fp = []
    all_scores = []
    n_gt = 0
    
    for pred, target in zip(predictions, targets):
        pred_boxes = pred['boxes']
        pred_scores = pred['scores']
        gt_boxes = target['boxes']
        
        n_gt += len(gt_boxes)
        matched = set()
        
        order = np.argsort(-pred_scores)
        
        for i in order:
            all_scores.append(pred_scores[i])
            
            best_iou = 0
            best_gt = -1
            
            for j, gt_box in enumerate(gt_boxes):
                if j in matched:
                    continue
                iou = compute_iou(pred_boxes[i], gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt = j
            
            if best_iou >= iou_threshold and best_gt not in matched:
                all_tp.append(1)
                all_fp.append(0)
                matched.add(best_gt)
            else:
                all_tp.append(0)
                all_fp.append(1)
    
    if n_gt == 0 or len(all_tp) == 0:
        return {'mAP': 0, 'precision': 0, 'recall': 0}
    
    # Sort by scores
    order = np.argsort(-np.array(all_scores))
    tp = np.array(all_tp)[order]
    fp = np.array(all_fp)[order]
    
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall = tp_cumsum / n_gt
    
    # 11-point interpolation AP
    ap = 0
    for r in np.arange(0, 1.1, 0.1):
        prec_at_recall = precision[recall >= r]
        ap += np.max(prec_at_recall) / 11 if len(prec_at_recall) > 0 else 0
    
    return {
        'mAP': ap,
        'precision': float(np.sum(tp)) / (np.sum(tp) + np.sum(fp)) if (np.sum(tp) + np.sum(fp)) > 0 else 0,
        'recall': float(np.sum(tp)) / n_gt if n_gt > 0 else 0,
        'total_predictions': len(all_tp),
        'total_gt': n_gt,
    }


def evaluate_dataset(model, processor, dataset: DetectionDataset,
                     text_prompt: str, box_threshold: float = 0.25,
                     device: str = 'cuda'):
    """Evaluate on entire dataset"""
    
    predictions = []
    targets = []
    
    for i in tqdm(range(len(dataset)), desc="Evaluating"):
        image, target = dataset[i]
        
        # Run detection
        result = detect_objects(
            model, processor, image, text_prompt,
            box_threshold=box_threshold,
            device=device
        )
        
        predictions.append(result)
        targets.append(target)
    
    # Compute metrics at multiple IoU thresholds
    map50 = compute_map(predictions, targets, iou_threshold=0.5)
    
    # Compute mAP@0.5:0.95
    aps = []
    for iou in np.arange(0.5, 1.0, 0.05):
        result = compute_map(predictions, targets, iou_threshold=iou)
        aps.append(result['mAP'])
    
    metrics = {
        'mAP50': map50['mAP'],
        'mAP50-95': np.mean(aps),
        'precision': map50['precision'],
        'recall': map50['recall'],
        'total_predictions': map50['total_predictions'],
        'total_gt': map50['total_gt'],
    }
    
    return metrics, predictions


def run_evaluation(config: dict):
    """Main evaluation function"""
    
    print("\n" + "="*60)
    print("🔍 Grounding-DINO Zero-Shot Evaluation")
    print("="*60)
    
    device = config['device'] if torch.cuda.is_available() else 'cpu'
    
    # Load model
    model, processor = load_grounding_dino(config['model'], device)
    
    # Prepare results
    results = {
        'model': config['model'],
        'prompt': config['prompt'],
        'box_threshold': config['box_threshold'],
        'datasets': {},
    }
    
    # Evaluate on TACO
    if config.get('taco_images'):
        print(f"\n📍 Evaluating on TACO")
        print(f"   Prompt: '{config['prompt']}'")
        
        taco_dataset = DetectionDataset(
            config['taco_images'],
            config['taco_labels']
        )
        
        metrics, _ = evaluate_dataset(
            model, processor, taco_dataset,
            config['prompt'], config['box_threshold'],
            device
        )
        
        results['datasets']['taco'] = metrics
        print(f"   mAP@0.5: {metrics['mAP50']:.4f}")
        print(f"   mAP@0.5:0.95: {metrics['mAP50-95']:.4f}")
    
    # Evaluate on ICRA19
    if config.get('icra19_images'):
        print(f"\n📍 Evaluating on Trash-ICRA19")
        print(f"   Prompt: '{config['prompt']}'")
        
        icra19_dataset = DetectionDataset(
            config['icra19_images'],
            config['icra19_labels']
        )
        
        metrics, _ = evaluate_dataset(
            model, processor, icra19_dataset,
            config['prompt'], config['box_threshold'],
            device
        )
        
        results['datasets']['icra19'] = metrics
        print(f"   mAP@0.5: {metrics['mAP50']:.4f}")
        print(f"   mAP@0.5:0.95: {metrics['mAP50-95']:.4f}")
    
    # Cross-domain analysis
    if 'taco' in results['datasets'] and 'icra19' in results['datasets']:
        taco_map = results['datasets']['taco']['mAP50']
        icra19_map = results['datasets']['icra19']['mAP50']
        
        print("\n" + "="*60)
        print("📊 Cross-Domain Analysis (Zero-Shot)")
        print("="*60)
        print(f"TACO mAP@0.5:    {taco_map:.4f}")
        print(f"ICRA19 mAP@0.5:  {icra19_map:.4f}")
        print(f"Difference:      {taco_map - icra19_map:.4f}")
        
        results['cross_domain'] = {
            'taco_map50': taco_map,
            'icra19_map50': icra19_map,
            'difference': taco_map - icra19_map,
        }
    
    # Save results
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = output_dir / f"grounding_dino_results_{timestamp}.json"
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {results_path}")
    
    return results


# ============================================================================
# Prompt Experiments
# ============================================================================

def run_prompt_ablation(config: dict):
    """Run experiments with different prompts"""
    
    print("\n" + "="*60)
    print("🧪 Prompt Ablation Study")
    print("="*60)
    
    device = config['device'] if torch.cuda.is_available() else 'cpu'
    model, processor = load_grounding_dino(config['model'], device)
    
    all_results = {}
    
    for prompt_name, prompt in PROMPT_TEMPLATES.items():
        print(f"\n📝 Testing prompt: '{prompt_name}'")
        print(f"   Prompt text: '{prompt}'")
        
        config['prompt'] = prompt
        
        prompt_results = {}
        
        if config.get('taco_images'):
            taco_dataset = DetectionDataset(
                config['taco_images'],
                config['taco_labels']
            )
            metrics, _ = evaluate_dataset(
                model, processor, taco_dataset,
                prompt, config['box_threshold'], device
            )
            prompt_results['taco'] = metrics
            print(f"   TACO mAP@0.5: {metrics['mAP50']:.4f}")
        
        if config.get('icra19_images'):
            icra19_dataset = DetectionDataset(
                config['icra19_images'],
                config['icra19_labels']
            )
            metrics, _ = evaluate_dataset(
                model, processor, icra19_dataset,
                prompt, config['box_threshold'], device
            )
            prompt_results['icra19'] = metrics
            print(f"   ICRA19 mAP@0.5: {metrics['mAP50']:.4f}")
        
        all_results[prompt_name] = prompt_results
    
    # Save results
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = output_dir / "prompt_ablation_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ Ablation results saved to: {results_path}")
    
    return all_results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate Grounding-DINO")
    
    parser.add_argument('--model', type=str, default='tiny',
                        choices=['tiny', 'base'],
                        help='Model variant')
    parser.add_argument('--prompt', type=str, 
                        default="trash . garbage . litter . waste",
                        help='Text prompt for detection')
    parser.add_argument('--box-threshold', type=float, default=0.25,
                        help='Box confidence threshold')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--output-dir', type=str, default='runs/grounding_dino')
    
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['taco', 'icra19', 'all'])
    parser.add_argument('--ablation', action='store_true',
                        help='Run prompt ablation study')
    
    args = parser.parse_args()
    
    # Setup paths
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    
    config = {
        'model': MODELS[args.model],
        'prompt': args.prompt,
        'box_threshold': args.box_threshold,
        'device': args.device,
        'output_dir': args.output_dir,
    }
    
    # Dataset paths
    if args.dataset in ['taco', 'all']:
        config['taco_images'] = str(data_dir / 'taco_yolo' / 'val' / 'images')
        config['taco_labels'] = str(data_dir / 'taco_yolo' / 'val' / 'labels')
    
    if args.dataset in ['icra19', 'all']:
        config['icra19_images'] = str(data_dir / 'icra19_yolo' / 'test' / 'images')
        config['icra19_labels'] = str(data_dir / 'icra19_yolo' / 'test' / 'labels')
    
    if args.ablation:
        run_prompt_ablation(config)
    else:
        run_evaluation(config)


if __name__ == "__main__":
    main()
