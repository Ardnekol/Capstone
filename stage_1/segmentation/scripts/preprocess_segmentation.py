#!/usr/bin/env python3
"""
Segmentation Dataset Preprocessing Script

Prepare TACO and BePLi datasets for segmentation training and evaluation.

Usage:
    python preprocess_segmentation.py
"""

import os
import json
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import argparse

def preprocess_taco_dataset():
    """Preprocess TACO dataset for segmentation."""
    print("🔄 Preprocessing TACO dataset...")

    taco_base = Path("/u/student/2024/cs24mtech11024/Capstone/stage_1/datasets/segmentation/taco_masks/TACO/data")
    output_dir = Path("/u/student/2024/cs24mtech11024/Capstone/stage_1/segmentation/data/taco")

    # Create output directories
    for split in ['train', 'val', 'test']:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'masks').mkdir(parents=True, exist_ok=True)

    # Load COCO annotations
    ann_file = taco_base / "annotations.json"
    if not ann_file.exists():
        print("❌ TACO annotations.json not found")
        return

    coco = COCO(str(ann_file))

    # Get all images
    img_ids = coco.getImgIds()
    print(f"📊 Found {len(img_ids)} images in TACO dataset")

    # Simple split: 70% train, 20% val, 10% test
    np.random.seed(42)
    np.random.shuffle(img_ids)
    n_train = int(0.7 * len(img_ids))
    n_val = int(0.2 * len(img_ids))

    splits = {
        'train': img_ids[:n_train],
        'val': img_ids[n_train:n_train+n_val],
        'test': img_ids[n_train+n_val:]
    }

    for split_name, split_img_ids in splits.items():
        print(f"📁 Processing {split_name} split ({len(split_img_ids)} images)...")

        for img_id in split_img_ids:
            img_info = coco.loadImgs(img_id)[0]
            img_path = taco_base / img_info['file_name']

            if not img_path.exists():
                continue

            # Copy image
            shutil.copy(img_path, output_dir / split_name / 'images' / f"{img_id:06d}.jpg")

            # Create mask
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)

            # Create binary mask (255 for waste, 0 for background)
            mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)

            for ann in anns:
                if 'segmentation' in ann and ann['segmentation']:
                    # Convert COCO RLE to mask
                    if isinstance(ann['segmentation'], list):
                        rle = maskUtils.frPyObjects(ann['segmentation'], img_info['height'], img_info['width'])
                    else:
                        rle = ann['segmentation']

                    if isinstance(rle, list) and len(rle) > 0:
                        rle = maskUtils.merge(rle)
                        mask += maskUtils.decode(rle).astype(np.uint8)

            # Save mask
            mask = np.clip(mask, 0, 1) * 255  # Ensure binary
            mask_img = Image.fromarray(mask.astype(np.uint8))
            mask_img.save(output_dir / split_name / 'masks' / f"{img_id:06d}.png")

    print("✅ TACO preprocessing complete!")

def preprocess_bepli_dataset():
    """Preprocess BePLi dataset for segmentation."""
    print("🔄 Preprocessing BePLi dataset...")

    bepli_base = Path("/u/student/2024/cs24mtech11024/Capstone/stage_1/datasets/segmentation/bepli/plastic_coco")
    output_dir = Path("/u/student/2024/cs24mtech11024/Capstone/stage_1/segmentation/data/bepli")

    # Create output directories
    for split in ['train', 'val', 'test']:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'masks').mkdir(parents=True, exist_ok=True)

    # Process each split
    for split in ['train', 'val', 'test']:
        ann_file = bepli_base / "annotation" / f"{split}.json"
        if not ann_file.exists():
            print(f"⚠️  {split}.json not found, skipping...")
            continue

        coco = COCO(str(ann_file))
        img_ids = coco.getImgIds()

        print(f"📁 Processing {split} split ({len(img_ids)} images)...")

        for img_id in img_ids:
            img_info = coco.loadImgs(img_id)[0]
            img_path = bepli_base / "images" / split / img_info['file_name']

            if not img_path.exists():
                continue

            # Copy image
            shutil.copy(img_path, output_dir / split / 'images' / f"{img_id:06d}.jpg")

            # Create mask
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)

            # Create binary mask (255 for plastic, 0 for background)
            mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)

            for ann in anns:
                if 'segmentation' in ann and ann['segmentation']:
                    # Convert COCO RLE to mask
                    if isinstance(ann['segmentation'], list):
                        rle = maskUtils.frPyObjects(ann['segmentation'], img_info['height'], img_info['width'])
                    else:
                        rle = ann['segmentation']

                    if isinstance(rle, list) and len(rle) > 0:
                        rle = maskUtils.merge(rle)
                        mask += maskUtils.decode(rle).astype(np.uint8)

            # Save mask
            mask = np.clip(mask, 0, 1) * 255  # Ensure binary
            mask_img = Image.fromarray(mask.astype(np.uint8))
            mask_img.save(output_dir / split / 'masks' / f"{img_id:06d}.png")

    print("✅ BePLi preprocessing complete!")

def preprocess_dwsd_dataset():
    """Preprocess DWSD dataset for segmentation."""
    print("🔄 Preprocessing DWSD dataset...")

    dwsd_base = Path("/u/student/2024/cs24mtech11024/Capstone/stage_1/datasets/segmentation/Dense Waste Segmentation Dataset/DSWD")
    output_dir = Path("/u/student/2024/cs24mtech11024/Capstone/stage_1/segmentation/data/dwsd")

    # Create output directories
    for split in ['train', 'test']:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'masks').mkdir(parents=True, exist_ok=True)

    # Process train and test splits
    for split in ['Train', 'Test']:
        split_lower = split.lower()
        img_dir = dwsd_base / split / 'Image'
        mask_dir = dwsd_base / split / 'Mask'

        if not img_dir.exists() or not mask_dir.exists():
            print(f"⚠️  {split} directories not found, skipping...")
            continue

        # Get all image files
        img_files = sorted(list(img_dir.glob('*.png')))
        print(f"📁 Processing {split_lower} split ({len(img_files)} images)...")

        for i, img_path in enumerate(img_files):
            # Get corresponding mask
            img_name = img_path.stem  # e.g., 'img_3156'
            mask_name = f"mask_{img_name.split('_')[1]}.png"  # e.g., 'mask_3156.png'
            mask_path = mask_dir / mask_name

            if not mask_path.exists():
                print(f"⚠️  Mask not found for {img_name}, skipping...")
                continue

            # Copy image
            shutil.copy(img_path, output_dir / split_lower / 'images' / f"{i:06d}.jpg")

            # Load and convert mask to binary
            mask = Image.open(mask_path).convert('L')  # Grayscale
            mask_array = np.array(mask)

            # Convert multi-class to binary: any non-zero pixel is waste
            binary_mask = (mask_array > 0).astype(np.uint8) * 255

            # Save binary mask
            mask_img = Image.fromarray(binary_mask)
            mask_img.save(output_dir / split_lower / 'masks' / f"{i:06d}.png")

    print("✅ DWSD preprocessing complete!")

def main():
    parser = argparse.ArgumentParser(description='Preprocess segmentation datasets')
    parser.add_argument('--dataset', choices=['taco', 'bepli', 'dwsd', 'all'], default='all',
                       help='Dataset to preprocess')
    args = parser.parse_args()

    if args.dataset in ['taco', 'all']:
        preprocess_taco_dataset()

    if args.dataset in ['bepli', 'all']:
        preprocess_bepli_dataset()

    if args.dataset in ['dwsd', 'all']:
        preprocess_dwsd_dataset()

    print("🎉 All preprocessing complete!")
    print("\n📊 Dataset Statistics:")
    print("- TACO: Urban waste segmentation (train/val/test splits)")
    print("- BePLi: Beach plastic segmentation (train/val/test splits)")
    print("- DWSD: Campus waste segmentation (train/test splits)")
    print("\n📁 Output format: images/ and masks/ subdirectories for each split")

if __name__ == '__main__':
    main()