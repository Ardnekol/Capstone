#!/usr/bin/env python3
"""
SAM (Segment Anything Model) Evaluation Script for Waste Segmentation

Evaluate SAM zero-shot on TACO and DWSD datasets.

Usage:
    python eval_sam.py
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
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
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CONFIG = {
    'model_name': 'sam',
    'sam_checkpoint': 'sam_vit_h_4b8939.pth',  # Will be downloaded
    'model_type': 'vit_h',
    'img_size': (256, 256)
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
        image_np = np.array(image)

        # Load mask
        mask_path = self.mask_files[idx]
        mask = Image.open(mask_path).convert('L')  # Grayscale
        mask_np = np.array(mask)

        return image_np, mask_np, str(img_path)

# ============================================================================
# SAM Evaluation Function
# ============================================================================

def evaluate_sam_on_dataset(images_dir, masks_dir, mask_generator):
    """Evaluate SAM on a dataset using automatic mask generation."""
    dataset = SegmentationDataset(images_dir, masks_dir)

    all_preds = []
    all_masks = []

    print(f"🔍 Evaluating SAM on {len(dataset)} images...")

    for i, (image_np, true_mask, img_path) in enumerate(dataset):
        if i % 10 == 0:
            print(f"Processing image {i+1}/{len(dataset)}")

        # Generate masks with SAM
        masks = mask_generator.generate(image_np)

        if len(masks) == 0:
            # No masks generated
            pred_mask = np.zeros_like(true_mask)
        else:
            # Use the mask with highest confidence
            best_mask = max(masks, key=lambda x: x['predicted_iou'])
            pred_mask = best_mask['segmentation'].astype(np.uint8) * 255

            # Resize prediction to match ground truth if needed
            if pred_mask.shape != true_mask.shape:
                pred_mask = np.array(Image.fromarray(pred_mask).resize(
                    (true_mask.shape[1], true_mask.shape[0]), Image.NEAREST))

        # Threshold and flatten
        pred_binary = (pred_mask > 127).astype(np.uint8).flatten()
        true_binary = (true_mask > 127).astype(np.uint8).flatten()

        all_preds.extend(pred_binary)
        all_masks.extend(true_binary)

    # Calculate metrics
    iou = jaccard_score(all_masks, all_preds, average='binary')
    precision = precision_score(all_masks, all_preds, average='binary', zero_division=0)
    recall = recall_score(all_masks, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_masks, all_preds, average='binary', zero_division=0)

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
    parser = argparse.ArgumentParser(description='Evaluate SAM for waste segmentation')
    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load SAM model
    print("🔄 Loading SAM model...")
    sam_checkpoint = DEFAULT_CONFIG['sam_checkpoint']

    # Check if checkpoint exists, if not download it
    if not os.path.exists(sam_checkpoint):
        print("📥 SAM checkpoint not found. Please download sam_vit_h_4b8939.pth from:")
        print("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
        print("And place it in the current directory.")
        return

    sam = sam_model_registry[DEFAULT_CONFIG['model_type']](checkpoint=sam_checkpoint)
    sam.to(device=device)

    # Create mask generator
    mask_generator = SamAutomaticMaskGenerator(sam)

    # Evaluate on TACO
    print("Evaluating on TACO...")
    taco_metrics = evaluate_sam_on_dataset(
        '/u/student/2024/cs24mtech11024/Capstone/stage_1/segmentation/data/taco/val/images',
        '/u/student/2024/cs24mtech11024/Capstone/stage_1/segmentation/data/taco/val/masks',
        mask_generator
    )

    # Evaluate on DWSD
    print("Evaluating on DWSD...")
    dwsd_metrics = evaluate_sam_on_dataset(
        '/u/student/2024/cs24mtech11024/Capstone/stage_1/segmentation/data/dwsd/test/images',
        '/u/student/2024/cs24mtech11024/Capstone/stage_1/segmentation/data/dwsd/test/masks',
        mask_generator
    )

    # Save results
    results = {
        'model': 'SAM (ViT-H)',
        'taco_iou': taco_metrics['iou'],
        'dwsd_iou': dwsd_metrics['iou'],
        'taco_metrics': taco_metrics,
        'dwsd_metrics': dwsd_metrics,
        'config': DEFAULT_CONFIG
    }

    os.makedirs('results', exist_ok=True)
    with open('results/sam_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("Evaluation complete! Results saved to results/sam_results.json")
    print(".4f")
    print(".4f")
    print(".4f")

if __name__ == '__main__':
    main()