#!/usr/bin/env python3
"""
Mask R-CNN Training Script for Waste Segmentation

Train Mask R-CNN on TACO dataset and evaluate on both TACO and BePLi.

Usage:
    python train_maskrcnn.py --epochs 50 --batch 2
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.metrics import jaccard_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import pandas as pd
from PIL import Image
import json
from datetime import datetime
import numpy as np
from pathlib import Path
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CONFIG = {
    'model_name': 'maskrcnn',
    'epochs': 50,
    'batch_size': 1,  # Reduced batch size for Mask R-CNN memory issues
    'learning_rate': 1e-4,
    'img_size': (256, 256),
    'num_classes': 2  # Background + waste
}

# ============================================================================
# Dataset Class
# ============================================================================

class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform

        # Get all image files
        self.image_files = sorted(list(self.images_dir.glob('*.jpg')))
        self.mask_files = sorted(list(self.masks_dir.glob('*.png')))

        # Ensure matching pairs
        assert len(self.image_files) == len(self.mask_files), "Mismatch between images and masks"

        print(f"📊 Dataset: {len(self.image_files)} image-mask pairs")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        image = transforms.ToTensor()(image)

        # Load mask
        mask_path = self.mask_files[idx]
        mask = Image.open(mask_path).convert('L')  # Grayscale
        mask = np.array(mask)

        # Convert to COCO format (binary mask)
        mask = (mask > 127).astype(np.uint8)  # Threshold to binary

        # Create bounding box from mask
        if mask.sum() > 0:
            rows, cols = np.where(mask > 0)
            x1, y1 = cols.min(), rows.min()
            x2, y2 = cols.max(), rows.max()
            boxes = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)
            labels = torch.tensor([1], dtype=torch.int64)  # Class 1 for waste
            masks = torch.tensor(mask, dtype=torch.uint8).unsqueeze(0)
            area = torch.tensor([(x2 - x1) * (y2 - y1)], dtype=torch.float32)
            iscrowd = torch.tensor([0], dtype=torch.int64)
        else:
            # Empty mask
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, mask.shape[0], mask.shape[1]), dtype=torch.uint8)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': torch.tensor([idx], dtype=torch.int64),
            'area': area,
            'iscrowd': iscrowd
        }

        return image, target

# ============================================================================
# Model Setup
# ============================================================================

def create_model(num_classes):
    # Load pre-trained Mask R-CNN
    model = maskrcnn_resnet50_fpn(pretrained=True)

    # Replace the classifier with a new one for our number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Replace the mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model

# ============================================================================
# Training Function
# ============================================================================

def train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs, device):
    best_iou = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Training phase
        model.train()
        running_loss = 0.0

        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            try:
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                running_loss += losses.item()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"⚠️  OOM during training, skipping batch")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

            # Clear cache after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        epoch_loss = running_loss / len(train_loader)
        print(f'Train Loss: {epoch_loss:.4f}')

        # Validation phase
        model.eval()
        val_iou = 0.0
        val_count = 0

        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                outputs = model(images)

                for output, target in zip(outputs, targets):
                    if len(output['masks']) > 0 and len(target['masks']) > 0:
                        pred_mask = (output['masks'][0] > 0.5).cpu().numpy().squeeze().astype(int)
                        true_mask = target['masks'][0].cpu().numpy().squeeze().astype(int)

                        iou = jaccard_score(true_mask.flatten(), pred_mask.flatten(), average='binary')
                        val_iou += iou
                        val_count += 1

        if val_count > 0:
            val_iou = val_iou / val_count
            print(f'Val IoU: {val_iou:.4f}')
        else:
            val_iou = 0.0
            print('Val IoU: N/A (no valid predictions)')

        scheduler.step()

        # Save best model
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), 'results/maskrcnn_best.pth')

    return model

# ============================================================================
# Evaluation Function
# ============================================================================

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_masks = []

    with torch.no_grad():
        for images, targets in test_loader:
            # Process one image at a time to save memory
            for img, target in zip(images, targets):
                img = img.unsqueeze(0).to(device)  # Add batch dimension
                target = {k: v.to(device) for k, v in target.items()}

                try:
                    outputs = model(img)

                    if len(outputs[0]['masks']) > 0 and len(target['masks']) > 0:
                        pred_mask = (outputs[0]['masks'][0] > 0.5).cpu().numpy().squeeze().astype(int)
                        true_mask = target['masks'][0].cpu().numpy().squeeze().astype(int)

                        all_preds.extend(pred_mask.flatten())
                        all_masks.extend(true_mask.flatten())

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"⚠️  OOM during evaluation, skipping image")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e

                # Clear cache after each image
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    if len(all_preds) == 0:
        return {'iou': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    # Calculate metrics
    iou = jaccard_score(all_masks, all_preds, average='binary')
    precision = precision_score(all_masks, all_preds, average='binary')
    recall = recall_score(all_masks, all_preds, average='binary')
    f1 = f1_score(all_masks, all_preds, average='binary')

    return {
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# ============================================================================
# Evaluation Function
# ============================================================================

def evaluate_model(model, test_loader, device):
    model.eval()
    total_iou = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    num_samples = 0

    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Evaluating"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)

            for output, target in zip(outputs, targets):
                # For Mask R-CNN, combine all predicted instance masks into semantic mask
                if len(output['masks']) > 0 and len(target['masks']) > 0:
                    # Combine all predicted masks (take max confidence for overlapping regions)
                    pred_masks = output['masks']  # Shape: [num_instances, 1, H, W]
                    pred_scores = output['scores']  # Shape: [num_instances]

                    # Create semantic prediction mask by taking the highest scoring instance
                    pred_semantic = torch.zeros_like(pred_masks[0, 0], dtype=torch.float32, device=device)
                    for mask, score in zip(pred_masks, pred_scores):
                        pred_semantic = torch.where(mask[0] > 0.5, torch.max(pred_semantic, mask[0] * score), pred_semantic)

                    pred_mask = (pred_semantic > 0.5).cpu().numpy().astype(int)

                    # Ground truth is already semantic (single mask per image)
                    true_mask = target['masks'][0].cpu().numpy().astype(int)

                    # Ensure same shape
                    if pred_mask.shape != true_mask.shape:
                        pred_mask = pred_mask.squeeze()
                        true_mask = true_mask.squeeze()

                    # Flatten for metric calculation
                    pred_flat = pred_mask.flatten()
                    true_flat = true_mask.flatten()

                    # Calculate IoU
                    intersection = (pred_flat * true_flat).sum()
                    union = pred_flat.sum() + true_flat.sum() - intersection
                    iou = (intersection / (union + 1e-6)).item()

                    # Calculate precision, recall, F1
                    tp = (pred_flat * true_flat).sum().item()
                    fp = (pred_flat * (1 - true_flat)).sum().item()
                    fn = ((1 - pred_flat) * true_flat).sum().item()

                    precision = tp / (tp + fp + 1e-6)
                    recall = tp / (tp + fn + 1e-6)
                    f1 = 2 * precision * recall / (precision + recall + 1e-6)

                    total_iou += iou
                    total_precision += precision
                    total_recall += recall
                    total_f1 += f1
                    num_samples += 1

    if num_samples == 0:
        return {'iou': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    return {
        'iou': total_iou / num_samples,
        'precision': total_precision / num_samples,
        'recall': total_recall / num_samples,
        'f1': total_f1 / num_samples
    }

# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Mask R-CNN for waste segmentation')
    parser.add_argument('--epochs', type=int, default=DEFAULT_CONFIG['epochs'], help='Number of epochs')
    parser.add_argument('--batch', type=int, default=DEFAULT_CONFIG['batch_size'], help='Batch size')
    parser.add_argument('--lr', type=float, default=DEFAULT_CONFIG['learning_rate'], help='Learning rate')
    parser.add_argument('--gpu', type=int, default=6, help='GPU device ID (default: 6, using available GPU)')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage')
    args = parser.parse_args()

    # Device selection (simplified for single GPU)
    if args.cpu or not torch.cuda.is_available():
        device = torch.device('cpu')
        print('Using device: cpu')
    else:
        # When CUDA_VISIBLE_DEVICES is set, device numbering starts from 0
        device = torch.device('cuda:0')
        print(f'Using device: {device} ({torch.cuda.get_device_name(0)})')
        torch.cuda.set_device(device)

    # Data transforms (Mask R-CNN handles its own preprocessing)
    data_transforms = transforms.Compose([
        transforms.Resize(DEFAULT_CONFIG['img_size']),
        transforms.ToTensor()  # No normalization - Mask R-CNN handles this internally
    ])

    # Create datasets
    train_dataset = SegmentationDataset(
        '/u/student/2024/cs24mtech11024/Capstone/stage_1/segmentation/data/taco/train/images',
        '/u/student/2024/cs24mtech11024/Capstone/stage_1/segmentation/data/taco/train/masks',
        data_transforms
    )

    val_dataset = SegmentationDataset(
        '/u/student/2024/cs24mtech11024/Capstone/stage_1/segmentation/data/taco/val/images',
        '/u/student/2024/cs24mtech11024/Capstone/stage_1/segmentation/data/taco/val/masks',
        data_transforms
    )

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=2, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False, num_workers=2, collate_fn=lambda x: tuple(zip(*x)))

    # Model
    model = create_model(DEFAULT_CONFIG['num_classes'])
    model = model.to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Train
    print("Starting training...")
    trained_model = train_model(model, train_loader, val_loader, optimizer, scheduler, args.epochs, device)

    # Evaluate on TACO
    print("Evaluating on TACO...")
    taco_metrics = evaluate_model(trained_model, val_loader, device)

    # Evaluate on DWSD
    print("Evaluating on DWSD...")
    dwsd_dataset = SegmentationDataset(
        '/u/student/2024/cs24mtech11024/Capstone/stage_1/segmentation/data/dwsd/test/images',
        '/u/student/2024/cs24mtech11024/Capstone/stage_1/segmentation/data/dwsd/test/masks',
        data_transforms
    )
    dwsd_loader = DataLoader(dwsd_dataset, batch_size=args.batch, shuffle=False, num_workers=2, collate_fn=lambda x: tuple(zip(*x)))
    dwsd_metrics = evaluate_model(trained_model, dwsd_loader, device)

    # Save results
    results = {
        'model': 'Mask R-CNN',
        'taco_iou': taco_metrics['iou'],
        'dwsd_iou': dwsd_metrics['iou'],
        'taco_metrics': taco_metrics,
        'dwsd_metrics': dwsd_metrics,
        'training_config': {
            'epochs': args.epochs,
            'batch_size': args.batch,
            'learning_rate': args.lr
        }
    }

    os.makedirs('results', exist_ok=True)
    with open('results/maskrcnn_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("Training complete! Results saved to results/maskrcnn_results.json")
    print(".4f")
    print(".4f")
    print(".4f")

if __name__ == '__main__':
    main()