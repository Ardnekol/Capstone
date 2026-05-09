#!/usr/bin/env python3
"""
DeepLabV3+ Training Script for Waste Segmentation

Train DeepLabV3+ on TACO dataset and evaluate on both TACO and BePLi.

Usage:
    python train_deeplabv3.py --epochs 100 --batch 8
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp
from torchvision import transforms
from sklearn.metrics import jaccard_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from PIL import Image
import json
from datetime import datetime
import numpy as np
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CONFIG = {
    'model_name': 'deeplabv3plus',
    'encoder_name': 'resnet101',
    'encoder_weights': 'imagenet',
    'epochs': 100,
    'batch_size': 8,
    'learning_rate': 1e-4,
    'img_size': (256, 256),
    'num_classes': 1  # Binary segmentation
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

        # Load mask
        mask_path = self.mask_files[idx]
        mask = Image.open(mask_path).convert('L')  # Grayscale

        # Apply transforms to image only
        if self.transform:
            image = self.transform(image)

        # Apply basic transforms to mask (resize and ToTensor only)
        mask_transform = transforms.Compose([
            transforms.Resize(DEFAULT_CONFIG['img_size']),
            transforms.ToTensor()
        ])
        mask = mask_transform(mask)

        # Ensure mask is binary (0 or 1)
        mask = (mask > 0.5).float()

        return image, mask

# ============================================================================
# Model Setup
# ============================================================================

def create_model():
    model = smp.DeepLabV3Plus(
        encoder_name=DEFAULT_CONFIG['encoder_name'],
        encoder_weights=DEFAULT_CONFIG['encoder_weights'],
        in_channels=3,
        classes=DEFAULT_CONFIG['num_classes']
    )
    return model

# ============================================================================
# Training Function
# ============================================================================

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    best_iou = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Training phase
        model.train()
        running_loss = 0.0

        for inputs, masks in train_loader:
            inputs = inputs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Train Loss: {epoch_loss:.4f}')

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_iou = 0.0

        with torch.no_grad():
            for inputs, masks in val_loader:
                inputs = inputs.to(device)
                masks = masks.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * inputs.size(0)

                # Calculate IoU
                preds = (torch.sigmoid(outputs) > 0.5).float()
                iou = jaccard_score(
                    masks.cpu().numpy().flatten(),
                    preds.cpu().numpy().flatten(),
                    average='binary'
                )
                val_iou += iou * inputs.size(0)

        val_loss = val_loss / len(val_loader.dataset)
        val_iou = val_iou / len(val_loader.dataset)

        print(f'Val Loss: {val_loss:.4f} IoU: {val_iou:.4f}')

        scheduler.step()

        # Save best model
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), 'results/deeplabv3plus_best.pth')

    return model

# ============================================================================
# Evaluation Function
# ============================================================================

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_masks = []

    with torch.no_grad():
        for inputs, masks in test_loader:
            inputs = inputs.to(device)
            masks = masks.to(device)

            outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).float()

            all_preds.extend(preds.cpu().numpy().flatten())
            all_masks.extend(masks.cpu().numpy().flatten())

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
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train DeepLabV3+ for waste segmentation')
    parser.add_argument('--epochs', type=int, default=DEFAULT_CONFIG['epochs'], help='Number of epochs')
    parser.add_argument('--batch', type=int, default=DEFAULT_CONFIG['batch_size'], help='Batch size')
    parser.add_argument('--lr', type=float, default=DEFAULT_CONFIG['learning_rate'], help='Learning rate')
    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Data transforms
    data_transforms = transforms.Compose([
        transforms.Resize(DEFAULT_CONFIG['img_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False, num_workers=4)

    # Model
    model = create_model()
    model = model.to(device)

    # Loss and optimizer
    criterion = smp.losses.DiceLoss(mode='binary')
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Train
    print("Starting training...")
    trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, args.epochs, device)

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
    dwsd_loader = DataLoader(dwsd_dataset, batch_size=args.batch, shuffle=False, num_workers=4)
    dwsd_metrics = evaluate_model(trained_model, dwsd_loader, device)

    # Save results
    results = {
        'model': 'DeepLabV3+',
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
    with open('results/deeplabv3plus_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("Training complete! Results saved to results/deeplabv3plus_results.json")
    print(".4f")
    print(".4f")
    print(".4f")

if __name__ == '__main__':
    main()