#!/usr/bin/env python3
"""
Florence-2 Evaluation Script for Object Detection

Zero-shot evaluation of Florence-2 on TACO and Trash-ICRA19 datasets.

Florence-2 is a foundation vision model from Microsoft that can perform 
multiple vision tasks including object detection based on text prompts.

Usage:
    python eval_florence2.py --task object_detection
    python eval_florence2.py --task open_vocabulary --prompt "trash, garbage"
    python eval_florence2.py --dataset taco
    python eval_florence2.py --dataset icra19
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

# Florence-2 Models
MODELS = {
    'base': "microsoft/Florence-2-base",
    'large': "microsoft/Florence-2-large",
    'base-ft': "microsoft/Florence-2-base-ft",  # Fine-tuned
    'large-ft': "microsoft/Florence-2-large-ft",  # Fine-tuned
}

# Task prompts for Florence-2
TASK_PROMPTS = {
    'object_detection': "<OD>",  # General object detection
    'open_vocabulary': "<OPEN_VOCABULARY_DETECTION>",  # With text prompt
    'dense_region_caption': "<DENSE_REGION_CAPTION>",  # Detailed captions
    'region_proposal': "<REGION_PROPOSAL>",  # Just proposals
}

# Text prompts for open vocabulary detection
TEXT_PROMPTS = [
    "trash",
    "garbage", 
    "litter",
    "waste",
    "rubbish",
    "plastic bottle",
    "can",
    "paper",
    "cardboard",
]


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
        
        # Load ground truth
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
# Florence-2 Model
# ============================================================================

def load_florence2(model_name: str = "microsoft/Florence-2-base", device: str = 'cpu'):
    """Load Florence-2 model from HuggingFace (CPU only)"""
    
    try:
        from transformers import AutoProcessor, AutoModelForCausalLM
    except ImportError:
        print("❌ transformers not installed. Install with:")
        print("   pip install transformers")
        sys.exit(1)
    
    print(f"Loading Florence-2: {model_name} on CPU")
    
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    
    # Use float32 and eager attention for compatibility
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float32,  # Use float32 for stability
        trust_remote_code=True,
        attn_implementation="eager"  # Fix for _supports_sdpa error
    )
    model = model.to("cpu")
    model.eval()
    
    print("✅ Model loaded successfully (CPU mode)")
    
    return model, processor


def parse_florence_output(text: str, image_size: Tuple[int, int]) -> Dict:
    """Parse Florence-2 detection output to boxes"""
    
    boxes = []
    labels = []
    scores = []
    
    # Florence outputs boxes in format: <loc_X1><loc_Y1><loc_X2><loc_Y2> label
    # where X,Y are normalized values 0-999
    
    import re
    
    # Pattern for location tokens
    loc_pattern = r'<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>'
    
    # Find all bounding boxes
    for match in re.finditer(loc_pattern, text):
        x1 = int(match.group(1)) / 1000 * image_size[0]
        y1 = int(match.group(2)) / 1000 * image_size[1]
        x2 = int(match.group(3)) / 1000 * image_size[0]
        y2 = int(match.group(4)) / 1000 * image_size[1]
        
        boxes.append([x1, y1, x2, y2])
        labels.append("object")  # Florence doesn't always give labels
        scores.append(1.0)  # Florence doesn't give confidence scores
    
    return {
        'boxes': np.array(boxes) if boxes else np.zeros((0, 4)),
        'labels': labels,
        'scores': np.array(scores) if scores else np.array([]),
    }


@torch.no_grad()
def detect_objects(model, processor, image: Image.Image, 
                   task: str = "object_detection",
                   text_prompt: str = None,
                   device: str = 'cuda'):
    """Run detection on a single image"""
    
    # Build prompt
    if task == "open_vocabulary" and text_prompt:
        prompt = f"<OPEN_VOCABULARY_DETECTION> {text_prompt}"
    else:
        prompt = TASK_PROMPTS.get(task, "<OD>")
    
    # Process
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    
    # Move to device (float32 for stability)
    input_ids = inputs["input_ids"].to(device)
    pixel_values = inputs["pixel_values"].to(device=device, dtype=torch.float32)
    
    # Generate with greedy decoding (avoid beam search compatibility issues)
    generated_ids = model.generate(
        input_ids=input_ids,
        pixel_values=pixel_values,
        max_new_tokens=1024,
        do_sample=False,
        num_beams=1,  # Greedy decoding
    )
    
    # Decode
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    
    # Parse output
    result = parse_florence_output(generated_text, image.size)
    
    return result


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


def compute_map(predictions: List[Dict], targets: List[Dict], iou_threshold: float = 0.5) -> Dict:
    """Compute mAP at given IoU threshold"""
    
    all_tp = []
    all_fp = []
    all_scores = []
    n_gt = 0
    
    for pred, target in zip(predictions, targets):
        pred_boxes = pred['boxes']
        pred_scores = pred['scores'] if len(pred['scores']) > 0 else np.ones(len(pred_boxes))
        gt_boxes = target['boxes']
        
        n_gt += len(gt_boxes)
        matched = set()
        
        order = np.argsort(-pred_scores) if len(pred_scores) > 0 else np.arange(len(pred_boxes))
        
        for i in order:
            all_scores.append(pred_scores[i] if i < len(pred_scores) else 1.0)
            
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
                     task: str = "object_detection",
                     text_prompt: str = None,
                     device: str = 'cuda'):
    """Evaluate on entire dataset"""
    
    predictions = []
    targets = []
    
    for i in tqdm(range(len(dataset)), desc="Evaluating"):
        image, target = dataset[i]
        
        result = detect_objects(
            model, processor, image,
            task=task,
            text_prompt=text_prompt,
            device=device
        )
        
        predictions.append(result)
        targets.append(target)
    
    # Compute metrics
    map50 = compute_map(predictions, targets, iou_threshold=0.5)
    
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
    print("🔍 Florence-2 Zero-Shot Evaluation")
    print("="*60)
    
    device = config['device'] if torch.cuda.is_available() else 'cpu'
    
    # Load model
    model, processor = load_florence2(config['model'], device)
    
    results = {
        'model': config['model'],
        'task': config['task'],
        'prompt': config.get('text_prompt'),
        'datasets': {},
    }
    
    # Evaluate on TACO
    if config.get('taco_images'):
        print(f"\n📍 Evaluating on TACO")
        print(f"   Task: {config['task']}")
        
        taco_dataset = DetectionDataset(
            config['taco_images'],
            config['taco_labels']
        )
        
        metrics, _ = evaluate_dataset(
            model, processor, taco_dataset,
            task=config['task'],
            text_prompt=config.get('text_prompt'),
            device=device
        )
        
        results['datasets']['taco'] = metrics
        print(f"   mAP@0.5: {metrics['mAP50']:.4f}")
        print(f"   mAP@0.5:0.95: {metrics['mAP50-95']:.4f}")
    
    # Evaluate on ICRA19
    if config.get('icra19_images'):
        print(f"\n📍 Evaluating on Trash-ICRA19")
        
        icra19_dataset = DetectionDataset(
            config['icra19_images'],
            config['icra19_labels']
        )
        
        metrics, _ = evaluate_dataset(
            model, processor, icra19_dataset,
            task=config['task'],
            text_prompt=config.get('text_prompt'),
            device=device
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
    results_path = output_dir / f"florence2_results_{timestamp}.json"
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {results_path}")
    
    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate Florence-2")
    
    parser.add_argument('--model', type=str, default='base-ft',
                        choices=['base', 'large', 'base-ft', 'large-ft'],
                        help='Model variant')
    parser.add_argument('--task', type=str, default='object_detection',
                        choices=['object_detection', 'open_vocabulary', 
                                'dense_region_caption', 'region_proposal'],
                        help='Detection task')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Text prompt for open vocabulary detection')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--output-dir', type=str, default='runs/florence2')
    
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['taco', 'icra19', 'all'])
    
    args = parser.parse_args()
    
    # Setup paths
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    
    config = {
        'model': MODELS[args.model],
        'task': args.task,
        'text_prompt': args.prompt,
        'device': args.device,
        'output_dir': args.output_dir,
    }
    
    if args.dataset in ['taco', 'all']:
        config['taco_images'] = str(data_dir / 'taco_yolo' / 'val' / 'images')
        config['taco_labels'] = str(data_dir / 'taco_yolo' / 'val' / 'labels')
    
    if args.dataset in ['icra19', 'all']:
        config['icra19_images'] = str(data_dir / 'icra19_yolo' / 'test' / 'images')
        config['icra19_labels'] = str(data_dir / 'icra19_yolo' / 'test' / 'labels')
    
    run_evaluation(config)


if __name__ == "__main__":
    main()
