#!/usr/bin/env python3
"""
RetinaNet Training Script for Object Detection

Train RetinaNet on TACO dataset and evaluate on both TACO (in-domain) 
and Trash-ICRA19 (cross-domain).

Uses torchvision's implementation of RetinaNet with ResNet-50 FPN backbone.

Usage:
    python train_retinanet.py --epochs 50 --batch 4
    python train_retinanet.py --eval --weights checkpoints/best.pth
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
import torchvision.transforms as T
from PIL import Image
import numpy as np
from functools import partial

# ============================================================================
# Dataset Class (Same as Faster R-CNN)
# ============================================================================

class YOLODataset(Dataset):
    """Dataset for YOLO format annotations"""
    
    def __init__(self, images_dir: str, labels_dir: str, transforms=None, 
                 img_size: int = 640, class_names: List[str] = None):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transforms = transforms
        self.img_size = img_size
        self.class_names = class_names or ['trash', 'bio']
        
        # Get all image files
        self.image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            self.image_files.extend(self.images_dir.glob(ext))
        self.image_files = sorted(self.image_files)
        
        print(f"Found {len(self.image_files)} images in {images_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size
        
        # Load labels
        label_path = self.labels_dir / f"{img_path.stem}.txt"
        boxes = []
        labels = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # Convert YOLO format to absolute coordinates
                        x1 = (x_center - width/2) * orig_w
                        y1 = (y_center - height/2) * orig_h
                        x2 = (x_center + width/2) * orig_w
                        y2 = (y_center + height/2) * orig_h
                        
                        # Clamp to image bounds
                        x1 = max(0, min(x1, orig_w))
                        y1 = max(0, min(y1, orig_h))
                        x2 = max(0, min(x2, orig_w))
                        y2 = max(0, min(y2, orig_h))
                        
                        if x2 > x1 and y2 > y1:
                            boxes.append([x1, y1, x2, y2])
                            labels.append(class_id + 1)  # +1 because 0 is background
        
        # Handle empty annotations
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        target = {
            "boxes": boxes,
            "labels": labels,
            "area": area,
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
            "image_id": torch.tensor([idx]),
        }
        
        if self.transforms:
            image = self.transforms(image)
        else:
            image = T.ToTensor()(image)
        
        return image, target


def get_transform(train: bool):
    """Get transforms for training/evaluation"""
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def collate_fn(batch):
    """Custom collate function for variable-size targets"""
    return tuple(zip(*batch))


# ============================================================================
# Model
# ============================================================================

def get_model(num_classes: int):
    """Get RetinaNet model with custom number of classes"""
    
    # Load pretrained model
    model = retinanet_resnet50_fpn_v2(weights=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT)
    
    # Get anchor generator info
    num_anchors = model.head.classification_head.num_anchors
    
    # Replace classification head
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=256,
        num_anchors=num_anchors,
        num_classes=num_classes,
        prior_probability=0.01,
    )
    
    return model


# ============================================================================
# Training Functions
# ============================================================================

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    loss_dict_accum = defaultdict(float)
    num_batches = 0
    
    for i, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Skip batch if any target has no boxes
        if any(t["boxes"].shape[0] == 0 for t in targets):
            continue
        
        try:
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            if not torch.isfinite(losses):
                print(f"  Warning: Non-finite loss at iteration {i}")
                continue
            
            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += losses.item()
            for k, v in loss_dict.items():
                loss_dict_accum[k] += v.item()
            num_batches += 1
            
        except Exception as e:
            print(f"  Warning: Error at iteration {i}: {e}")
            continue
        
        if (i + 1) % print_freq == 0:
            avg_loss = total_loss / max(num_batches, 1)
            print(f"  Epoch [{epoch}] Iter [{i+1}/{len(data_loader)}] Loss: {avg_loss:.4f}")
    
    avg_loss = total_loss / max(num_batches, 1)
    for k in loss_dict_accum:
        loss_dict_accum[k] /= max(num_batches, 1)
    
    return avg_loss, dict(loss_dict_accum)


@torch.no_grad()
def evaluate(model, data_loader, device):
    """Evaluate model and compute mAP"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        
        predictions = model(images)
        
        for pred, target in zip(predictions, targets):
            all_predictions.append({
                'boxes': pred['boxes'].cpu(),
                'labels': pred['labels'].cpu(),
                'scores': pred['scores'].cpu(),
            })
            all_targets.append({
                'boxes': target['boxes'],
                'labels': target['labels'],
            })
    
    metrics = compute_map(all_predictions, all_targets)
    
    return metrics


def compute_map(predictions: List[Dict], targets: List[Dict], 
                iou_thresholds: List[float] = None) -> Dict:
    """Compute mAP at different IoU thresholds"""
    
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    
    def compute_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0
    
    aps_per_threshold = []
    
    for iou_thresh in iou_thresholds:
        tp_list = []
        fp_list = []
        scores_list = []
        n_gt = 0
        
        for pred, target in zip(predictions, targets):
            pred_boxes = pred['boxes'].numpy()
            pred_scores = pred['scores'].numpy()
            gt_boxes = target['boxes'].numpy()
            
            n_gt += len(gt_boxes)
            matched_gt = set()
            
            order = np.argsort(-pred_scores)
            
            for i in order:
                if pred_scores[i] < 0.01:
                    continue
                    
                scores_list.append(pred_scores[i])
                
                best_iou = 0
                best_gt = -1
                
                for j, gt_box in enumerate(gt_boxes):
                    if j in matched_gt:
                        continue
                    iou = compute_iou(pred_boxes[i], gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = j
                
                if best_iou >= iou_thresh and best_gt not in matched_gt:
                    tp_list.append(1)
                    fp_list.append(0)
                    matched_gt.add(best_gt)
                else:
                    tp_list.append(0)
                    fp_list.append(1)
        
        if n_gt == 0 or len(tp_list) == 0:
            aps_per_threshold.append(0.0)
            continue
        
        order = np.argsort(-np.array(scores_list))
        tp = np.array(tp_list)[order]
        fp = np.array(fp_list)[order]
        
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        recall = tp_cumsum / n_gt
        
        ap = 0
        for r in np.arange(0, 1.1, 0.1):
            prec_at_recall = precision[recall >= r]
            ap += np.max(prec_at_recall) / 11 if len(prec_at_recall) > 0 else 0
        
        aps_per_threshold.append(ap)
    
    return {
        'mAP50': aps_per_threshold[0] if len(aps_per_threshold) > 0 else 0,
        'mAP50-95': np.mean(aps_per_threshold) if aps_per_threshold else 0,
        'precision': float(np.sum(tp_list)) / (np.sum(tp_list) + np.sum(fp_list)) if (np.sum(tp_list) + np.sum(fp_list)) > 0 else 0,
        'recall': float(np.sum(tp_list)) / n_gt if n_gt > 0 else 0,
    }


def train_retinanet(config: dict):
    """Main training function"""
    
    print("\n" + "="*60)
    print("🚀 Training RetinaNet")
    print("="*60)
    
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(config['output_dir']) / f"retinanet_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create datasets
    train_dataset = YOLODataset(
        config['train_images'],
        config['train_labels'],
        transforms=get_transform(train=True),
        class_names=config['class_names']
    )
    
    val_dataset = YOLODataset(
        config['val_images'],
        config['val_labels'],
        transforms=get_transform(train=False),
        class_names=config['class_names']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch'],
        shuffle=True,
        num_workers=config['workers'],
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch'],
        shuffle=False,
        num_workers=config['workers'],
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    # Model (num_classes doesn't include background for RetinaNet)
    num_classes = len(config['class_names'])
    model = get_model(num_classes)
    model.to(device)
    
    print(f"Model: RetinaNet ResNet50-FPN-v2")
    print(f"Classes: {config['class_names']} ({num_classes})")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=config['lr'], weight_decay=config['weight_decay'])
    
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs'], eta_min=config['lr'] * 0.01
    )
    
    best_map = 0
    history = {'train_loss': [], 'val_map50': [], 'val_map50_95': []}
    
    for epoch in range(1, config['epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['epochs']}")
        print("-" * 40)
        
        train_loss, loss_dict = train_one_epoch(model, optimizer, train_loader, device, epoch)
        lr_scheduler.step()
        
        print(f"  Train Loss: {train_loss:.4f}")
        for k, v in loss_dict.items():
            print(f"    {k}: {v:.4f}")
        
        history['train_loss'].append(train_loss)
        
        if epoch % config.get('val_freq', 5) == 0 or epoch == config['epochs']:
            metrics = evaluate(model, val_loader, device)
            
            print(f"  Val mAP@0.5: {metrics['mAP50']:.4f}")
            print(f"  Val mAP@0.5:0.95: {metrics['mAP50-95']:.4f}")
            
            history['val_map50'].append(metrics['mAP50'])
            history['val_map50_95'].append(metrics['mAP50-95'])
            
            if metrics['mAP50'] > best_map:
                best_map = metrics['mAP50']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': metrics,
                }, output_dir / 'best.pth')
                print(f"  ✅ New best model saved!")
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, output_dir / 'last.pth')
    
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n✅ Training complete!")
    print(f"Best mAP@0.5: {best_map:.4f}")
    print(f"Output directory: {output_dir}")
    
    return output_dir, best_map


def cross_domain_evaluation(model_path: str, config: dict):
    """Evaluate on both in-domain and cross-domain datasets"""
    
    print("\n" + "="*60)
    print("🔄 Cross-Domain Evaluation")
    print("="*60)
    
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    
    num_classes = len(config['class_names'])
    model = get_model(num_classes)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    results = {}
    
    if config.get('val_images'):
        print("\n📍 In-Domain: Validation set")
        val_dataset = YOLODataset(
            config['val_images'],
            config['val_labels'],
            transforms=get_transform(train=False),
            class_names=config['class_names']
        )
        val_loader = DataLoader(val_dataset, batch_size=config['batch'], 
                               shuffle=False, num_workers=config['workers'],
                               collate_fn=collate_fn)
        results['in_domain'] = evaluate(model, val_loader, device)
        print(f"  mAP@0.5: {results['in_domain']['mAP50']:.4f}")
    
    if config.get('cross_domain_images'):
        print("\n📍 Cross-Domain: Test set")
        cross_dataset = YOLODataset(
            config['cross_domain_images'],
            config['cross_domain_labels'],
            transforms=get_transform(train=False),
            class_names=config['class_names']
        )
        cross_loader = DataLoader(cross_dataset, batch_size=config['batch'],
                                 shuffle=False, num_workers=config['workers'],
                                 collate_fn=collate_fn)
        results['cross_domain'] = evaluate(model, cross_loader, device)
        print(f"  mAP@0.5: {results['cross_domain']['mAP50']:.4f}")
    
    if 'in_domain' in results and 'cross_domain' in results:
        gap = results['in_domain']['mAP50'] - results['cross_domain']['mAP50']
        relative_gap = (gap / results['in_domain']['mAP50']) * 100 if results['in_domain']['mAP50'] > 0 else 0
        
        print("\n" + "="*60)
        print("📊 Generalization Analysis")
        print("="*60)
        print(f"In-domain mAP@0.5:    {results['in_domain']['mAP50']:.4f}")
        print(f"Cross-domain mAP@0.5: {results['cross_domain']['mAP50']:.4f}")
        print(f"Absolute Drop:        {gap:.4f}")
        print(f"Relative Drop:        {relative_gap:.1f}%")
        
        results['generalization'] = {
            'absolute_drop': gap,
            'relative_drop_percent': relative_gap,
        }
    
    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train RetinaNet for object detection")
    
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--output-dir', type=str, default='runs/retinanet')
    
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--weights', type=str, default=None)
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    
    config = {
        'epochs': args.epochs,
        'batch': args.batch,
        'lr': args.lr,
        'weight_decay': 0.0005,
        'workers': args.workers,
        'device': args.device,
        'output_dir': args.output_dir,
        'class_names': ['trash', 'bio'],
        
        'train_images': str(data_dir / 'taco_yolo' / 'train' / 'images'),
        'train_labels': str(data_dir / 'taco_yolo' / 'train' / 'labels'),
        'val_images': str(data_dir / 'taco_yolo' / 'val' / 'images'),
        'val_labels': str(data_dir / 'taco_yolo' / 'val' / 'labels'),
        
        'cross_domain_images': str(data_dir / 'icra19_yolo' / 'test' / 'images'),
        'cross_domain_labels': str(data_dir / 'icra19_yolo' / 'test' / 'labels'),
    }
    
    if not args.eval:
        args.train = True
    
    if args.train:
        output_dir, best_map = train_retinanet(config)
        
        best_weights = output_dir / 'best.pth'
        if best_weights.exists():
            results = cross_domain_evaluation(str(best_weights), config)
            
            with open(output_dir / 'cross_domain_results.json', 'w') as f:
                json.dump(results, f, indent=2)
    
    elif args.eval:
        if not args.weights:
            print("❌ Please provide --weights path")
            sys.exit(1)
        
        results = cross_domain_evaluation(args.weights, config)


if __name__ == "__main__":
    main()
