#!/usr/bin/env python3
"""
Convert and prepare datasets for object detection training.

This script:
1. Converts TACO to YOLO format (train/val split)
2. Converts Trash-ICRA19 to YOLO format
3. Creates unified class mapping for cross-domain evaluation

Usage:
    python convert_annotations.py --all
    python convert_annotations.py --taco
    python convert_annotations.py --icra19
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from dataset_utils import TACODataset, TrashICRA19Dataset


def convert_taco(base_dir: Path, output_dir: Path, val_ratio: float = 0.2):
    """Convert TACO dataset to YOLO format with train/val split"""
    
    print("\n" + "="*60)
    print("Converting TACO to YOLO format")
    print("="*60)
    
    taco_path = base_dir / "taco" / "TACO"
    
    # Load with unified classes for cross-domain eval
    taco = TACODataset(
        root_dir=taco_path,
        unified_classes=True
    )
    
    # Split train/val
    train_ids, val_ids = taco.split_train_val(val_ratio=val_ratio)
    
    print(f"\nSplit: {len(train_ids)} train, {len(val_ids)} val")
    
    # Export train
    train_dir = output_dir / "taco_yolo" / "train"
    taco.export_to_yolo(train_dir, train_ids)
    
    # Export val
    val_dir = output_dir / "taco_yolo" / "val"
    taco.export_to_yolo(val_dir, val_ids)
    
    # Create combined data.yaml
    yaml_content = f"""
# TACO Dataset - YOLO Format
# Unified classes for cross-domain evaluation

path: {(output_dir / "taco_yolo").absolute()}
train: train/images
val: val/images

nc: {len(taco.class_names)}
names: {taco.class_names}
"""
    
    with open(output_dir / "taco_yolo" / "data.yaml", 'w') as f:
        f.write(yaml_content)
    
    print(f"\n✅ TACO converted to: {output_dir / 'taco_yolo'}")


def convert_icra19(base_dir: Path, output_dir: Path):
    """Convert Trash-ICRA19 dataset to YOLO format"""
    
    print("\n" + "="*60)
    print("Converting Trash-ICRA19 to YOLO format")
    print("="*60)
    
    icra_path = base_dir / "trash_icra19" / "trash_ICRA19"
    
    for split in ['train', 'val', 'test']:
        print(f"\nProcessing {split} split...")
        
        icra = TrashICRA19Dataset(
            root_dir=icra_path,
            split=split,
            unified_classes=True
        )
        
        split_dir = output_dir / "icra19_yolo" / split
        icra.export_to_yolo(split_dir)
    
    # Create combined data.yaml
    yaml_content = f"""
# Trash-ICRA19 Dataset - YOLO Format
# Unified classes for cross-domain evaluation

path: {(output_dir / "icra19_yolo").absolute()}
train: train/images
val: val/images
test: test/images

nc: 2
names: ['trash', 'bio']
"""
    
    with open(output_dir / "icra19_yolo" / "data.yaml", 'w') as f:
        f.write(yaml_content)
    
    print(f"\n✅ Trash-ICRA19 converted to: {output_dir / 'icra19_yolo'}")


def main():
    parser = argparse.ArgumentParser(description="Convert detection datasets")
    parser.add_argument('--all', action='store_true', help='Convert all datasets')
    parser.add_argument('--taco', action='store_true', help='Convert TACO only')
    parser.add_argument('--icra19', action='store_true', help='Convert Trash-ICRA19 only')
    parser.add_argument('--val-ratio', type=float, default=0.2, help='Validation ratio for TACO')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    
    args = parser.parse_args()
    
    # Default paths
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent / "datasets" / "detection"
    output_dir = Path(args.output_dir) if args.output_dir else script_dir / "data"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Input datasets: {base_dir}")
    print(f"Output directory: {output_dir}")
    
    if args.all or (not args.taco and not args.icra19):
        convert_taco(base_dir, output_dir, args.val_ratio)
        convert_icra19(base_dir, output_dir)
    else:
        if args.taco:
            convert_taco(base_dir, output_dir, args.val_ratio)
        if args.icra19:
            convert_icra19(base_dir, output_dir)
    
    print("\n" + "="*60)
    print("✅ Conversion complete!")
    print("="*60)
    print(f"\nOutput structure:")
    print(f"  {output_dir}/")
    print(f"  ├── taco_yolo/")
    print(f"  │   ├── train/images/")
    print(f"  │   ├── train/labels/")
    print(f"  │   ├── val/images/")
    print(f"  │   ├── val/labels/")
    print(f"  │   └── data.yaml")
    print(f"  └── icra19_yolo/")
    print(f"      ├── train/images/")
    print(f"      ├── val/images/")
    print(f"      ├── test/images/")
    print(f"      └── data.yaml")


if __name__ == "__main__":
    main()
