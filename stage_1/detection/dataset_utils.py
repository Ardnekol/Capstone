#!/usr/bin/env python3
"""
Dataset Utilities for Object Detection Experiments

Handles loading and converting between:
- TACO (COCO format)
- Trash-ICRA19 (PASCAL VOC XML / YOLO TXT format)

Provides unified data loaders for training and evaluation.
"""

import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import shutil

import numpy as np
from PIL import Image

# ============================================================================
# Class Mappings
# ============================================================================

# TACO 60 categories → 7 super categories
TACO_SUPERCATEGORY_MAP = {
    # Plastic
    "Plastic bag & wrapper": "plastic",
    "Other plastic wrapper": "plastic",
    "Single-use carrier bag": "plastic",
    "Polypropylene bag": "plastic",
    "Plastic bottle": "plastic",
    "Other plastic bottle": "plastic",
    "Clear plastic bottle": "plastic",
    "Drink bottle": "plastic",
    "Plastic bottle cap": "plastic",
    "Plastic glooves": "plastic",
    "Plastic utensils": "plastic",
    "Plastic straw": "plastic",
    "Straw": "plastic",
    "Styrofoam piece": "plastic",
    "Disposable plastic cup": "plastic",
    "Foam cup": "plastic",
    "Other plastic cup": "plastic",
    "Cup": "plastic",
    "Lid": "plastic",
    "Disposable food container": "plastic",
    "Other plastic container": "plastic",
    "Plastic film": "plastic",
    "Six pack rings": "plastic",
    "Spread tub": "plastic",
    "Tupperware": "plastic",
    "Squeezable tube": "plastic",
    "Garbage bag": "plastic",
    "Blister pack": "plastic",
    "Carded blister pack": "plastic",
    "Other plastic": "plastic",
    
    # Metal
    "Aluminium foil": "metal",
    "Aluminium blister pack": "metal",
    "Can": "metal",
    "Food Can": "metal",
    "Drink can": "metal",
    "Aerosol": "metal",
    "Metal bottle cap": "metal",
    "Metal lid": "metal",
    "Pop tab": "metal",
    "Scrap metal": "metal",
    "Other metal": "metal",
    
    # Glass
    "Glass bottle": "glass",
    "Broken glass": "glass",
    "Glass cup": "glass",
    "Glass jar": "glass",
    "Other glass": "glass",
    
    # Paper/Cardboard
    "Carton": "paper",
    "Other carton": "paper",
    "Egg carton": "paper",
    "Drink carton": "paper",
    "Corrugated carton": "paper",
    "Meal carton": "paper",
    "Pizza box": "paper",
    "Paper": "paper",
    "Paper bag": "paper",
    "Tissues": "paper",
    "Wrapping paper": "paper",
    "Magazine paper": "paper",
    "Normal paper": "paper",
    "Toilet tube": "paper",
    "Paper cup": "paper",
    "Paper straw": "paper",
    "Other paper": "paper",
    
    # Organic
    "Food waste": "organic",
    
    # Textile
    "Rope & strings": "textile",
    "Shoe": "textile",
    "Squeezable tube": "textile",
    
    # Other
    "Battery": "other",
    "Cigarette": "other",
    "Lighter": "other",
    "Unlabeled litter": "other",
}

# Trash-ICRA19 class mapping
ICRA19_CLASS_MAP = {
    "plastic": "trash",
    "metal": "trash",
    "wood": "trash",
    "rubber": "trash",
    "paper": "trash",
    "cloth": "trash",
    "fishing": "trash",
    "papper": "trash",  # typo in dataset
    "platstic": "trash",  # typo in dataset
    "bio": "bio",
    "rov": None,  # Ignore
    "timestamp": None,  # Ignore
    "unknown": None,  # Ignore
}

# Unified classes for cross-domain evaluation
UNIFIED_CLASSES = ["trash", "bio"]  # Binary: trash detection
UNIFIED_CLASS_TO_ID = {cls: i for i, cls in enumerate(UNIFIED_CLASSES)}

# For TACO: map supercategories to unified
TACO_TO_UNIFIED = {
    "plastic": "trash",
    "metal": "trash",
    "glass": "trash",
    "paper": "trash",
    "organic": "trash",
    "textile": "trash",
    "other": "trash",
}


# ============================================================================
# TACO Dataset Loader (COCO Format)
# ============================================================================

class TACODataset:
    """Load and process TACO dataset in COCO format"""
    
    def __init__(self, 
                 root_dir: str,
                 annotation_file: str = "data/annotations.json",
                 use_supercategories: bool = True,
                 unified_classes: bool = False):
        """
        Args:
            root_dir: Path to TACO folder (containing 'data' subfolder)
            annotation_file: Path to annotations.json relative to root
            use_supercategories: Map 60 classes to 7 super categories
            unified_classes: Map to unified classes for cross-domain eval
        """
        self.root_dir = Path(root_dir)
        self.use_supercategories = use_supercategories
        self.unified_classes = unified_classes
        
        # Load annotations
        ann_path = self.root_dir / annotation_file
        with open(ann_path, 'r') as f:
            self.coco = json.load(f)
        
        # Build lookup dictionaries
        self.images = {img['id']: img for img in self.coco['images']}
        self.categories = {cat['id']: cat for cat in self.coco['categories']}
        
        # Group annotations by image
        self.img_to_anns = defaultdict(list)
        for ann in self.coco['annotations']:
            self.img_to_anns[ann['image_id']].append(ann)
        
        # Build class mappings
        self._build_class_mapping()
        
        print(f"Loaded TACO dataset:")
        print(f"  Images: {len(self.images)}")
        print(f"  Annotations: {len(self.coco['annotations'])}")
        print(f"  Original categories: {len(self.categories)}")
        print(f"  Mapped classes: {len(self.class_names)}")
    
    def _build_class_mapping(self):
        """Build class name to ID mapping"""
        if self.unified_classes:
            self.class_names = UNIFIED_CLASSES
            self.class_to_id = UNIFIED_CLASS_TO_ID
        elif self.use_supercategories:
            # Get unique supercategories
            supercats = set()
            for cat in self.categories.values():
                if cat['name'] in TACO_SUPERCATEGORY_MAP:
                    supercats.add(TACO_SUPERCATEGORY_MAP[cat['name']])
                else:
                    supercats.add('other')
            self.class_names = sorted(list(supercats))
            self.class_to_id = {cls: i for i, cls in enumerate(self.class_names)}
        else:
            self.class_names = [cat['name'] for cat in self.categories.values()]
            self.class_to_id = {cat['name']: cat['id'] for cat in self.categories.values()}
    
    def get_image_path(self, image_id: int) -> Path:
        """Get full path to image file"""
        img_info = self.images[image_id]
        # TACO stores images in batch folders
        return self.root_dir / "data" / img_info['file_name']
    
    def get_annotations(self, image_id: int) -> List[Dict]:
        """Get annotations for an image with mapped class IDs"""
        anns = self.img_to_anns.get(image_id, [])
        result = []
        
        for ann in anns:
            cat = self.categories[ann['category_id']]
            cat_name = cat['name']
            
            # Map to supercategory or unified class
            if self.use_supercategories:
                mapped_name = TACO_SUPERCATEGORY_MAP.get(cat_name, 'other')
            else:
                mapped_name = cat_name
            
            if self.unified_classes:
                mapped_name = TACO_TO_UNIFIED.get(mapped_name, 'trash')
            
            if mapped_name not in self.class_to_id:
                continue
            
            # COCO bbox format: [x, y, width, height]
            bbox = ann['bbox']
            
            result.append({
                'bbox': bbox,  # [x, y, w, h]
                'bbox_xyxy': [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]],
                'class_id': self.class_to_id[mapped_name],
                'class_name': mapped_name,
                'area': ann.get('area', bbox[2] * bbox[3]),
                'segmentation': ann.get('segmentation', None),
            })
        
        return result
    
    def get_all_image_ids(self) -> List[int]:
        """Get all image IDs"""
        return list(self.images.keys())
    
    def split_train_val(self, val_ratio: float = 0.2, seed: int = 42) -> Tuple[List[int], List[int]]:
        """Split dataset into train/val"""
        np.random.seed(seed)
        image_ids = self.get_all_image_ids()
        np.random.shuffle(image_ids)
        
        val_size = int(len(image_ids) * val_ratio)
        val_ids = image_ids[:val_size]
        train_ids = image_ids[val_size:]
        
        return train_ids, val_ids
    
    def export_to_yolo(self, output_dir: str, image_ids: Optional[List[int]] = None):
        """Export dataset to YOLO format"""
        output_dir = Path(output_dir)
        images_dir = output_dir / "images"
        labels_dir = output_dir / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        if image_ids is None:
            image_ids = self.get_all_image_ids()
        
        for img_id in image_ids:
            img_info = self.images[img_id]
            img_path = self.get_image_path(img_id)
            
            if not img_path.exists():
                continue
            
            # Copy image
            new_img_name = f"{img_id:06d}.jpg"
            shutil.copy(img_path, images_dir / new_img_name)
            
            # Create label file
            anns = self.get_annotations(img_id)
            label_path = labels_dir / f"{img_id:06d}.txt"
            
            img_w, img_h = img_info['width'], img_info['height']
            
            with open(label_path, 'w') as f:
                for ann in anns:
                    bbox = ann['bbox']  # [x, y, w, h]
                    # Convert to YOLO format: [class_id, x_center, y_center, width, height] (normalized)
                    x_center = (bbox[0] + bbox[2] / 2) / img_w
                    y_center = (bbox[1] + bbox[3] / 2) / img_h
                    w = bbox[2] / img_w
                    h = bbox[3] / img_h
                    f.write(f"{ann['class_id']} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
        
        # Create data.yaml
        yaml_content = f"""
path: {output_dir.absolute()}
train: images
val: images
test: images

names:
"""
        for i, name in enumerate(self.class_names):
            yaml_content += f"  {i}: {name}\n"
        
        with open(output_dir / "data.yaml", 'w') as f:
            f.write(yaml_content)
        
        print(f"Exported {len(image_ids)} images to YOLO format at {output_dir}")


# ============================================================================
# Trash-ICRA19 Dataset Loader (PASCAL VOC / YOLO Format)
# ============================================================================

class TrashICRA19Dataset:
    """Load and process Trash-ICRA19 dataset"""
    
    def __init__(self,
                 root_dir: str,
                 split: str = "test",
                 unified_classes: bool = True):
        """
        Args:
            root_dir: Path to trash_ICRA19 folder
            split: 'train', 'val', or 'test'
            unified_classes: Map to unified classes for cross-domain eval
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.unified_classes = unified_classes
        self.data_dir = self.root_dir / "dataset" / split
        
        # Get all images
        self.image_files = sorted(list(self.data_dir.glob("*.jpg")))
        
        # Build class mapping
        if unified_classes:
            self.class_names = UNIFIED_CLASSES
            self.class_to_id = UNIFIED_CLASS_TO_ID
        else:
            self.class_names = ["trash", "bio"]
            self.class_to_id = {"trash": 0, "bio": 1}
        
        print(f"Loaded Trash-ICRA19 {split} split:")
        print(f"  Images: {len(self.image_files)}")
        print(f"  Classes: {self.class_names}")
    
    def get_image_path(self, idx: int) -> Path:
        """Get image path by index"""
        return self.image_files[idx]
    
    def parse_xml_annotation(self, xml_path: Path) -> List[Dict]:
        """Parse PASCAL VOC XML annotation"""
        if not xml_path.exists():
            return []
        
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        size = root.find('size')
        img_w = int(size.find('width').text)
        img_h = int(size.find('height').text)
        
        annotations = []
        for obj in root.findall('object'):
            class_name = obj.find('name').text.lower()
            
            # Map class
            mapped_class = ICRA19_CLASS_MAP.get(class_name)
            if mapped_class is None:
                continue  # Skip ignored classes
            
            if mapped_class not in self.class_to_id:
                continue
            
            bbox = obj.find('bndbox')
            xmin = int(float(bbox.find('xmin').text))
            ymin = int(float(bbox.find('ymin').text))
            xmax = int(float(bbox.find('xmax').text))
            ymax = int(float(bbox.find('ymax').text))
            
            annotations.append({
                'bbox': [xmin, ymin, xmax - xmin, ymax - ymin],  # [x, y, w, h]
                'bbox_xyxy': [xmin, ymin, xmax, ymax],
                'class_id': self.class_to_id[mapped_class],
                'class_name': mapped_class,
                'img_width': img_w,
                'img_height': img_h,
            })
        
        return annotations
    
    def get_annotations(self, idx: int) -> List[Dict]:
        """Get annotations for an image by index"""
        img_path = self.image_files[idx]
        xml_path = img_path.with_suffix('.xml')
        return self.parse_xml_annotation(xml_path)
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[Path, List[Dict]]:
        """Get image path and annotations"""
        return self.get_image_path(idx), self.get_annotations(idx)
    
    def export_to_yolo(self, output_dir: str):
        """Export dataset to YOLO format"""
        output_dir = Path(output_dir)
        images_dir = output_dir / "images"
        labels_dir = output_dir / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        for idx in range(len(self)):
            img_path, anns = self[idx]
            
            if not anns:
                continue
            
            # Copy image
            new_img_name = img_path.name
            shutil.copy(img_path, images_dir / new_img_name)
            
            # Create label file
            label_path = labels_dir / img_path.with_suffix('.txt').name
            img_w, img_h = anns[0]['img_width'], anns[0]['img_height']
            
            with open(label_path, 'w') as f:
                for ann in anns:
                    bbox = ann['bbox']
                    x_center = (bbox[0] + bbox[2] / 2) / img_w
                    y_center = (bbox[1] + bbox[3] / 2) / img_h
                    w = bbox[2] / img_w
                    h = bbox[3] / img_h
                    f.write(f"{ann['class_id']} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
        
        # Create data.yaml
        yaml_content = f"""
path: {output_dir.absolute()}
train: images
val: images
test: images

names:
"""
        for i, name in enumerate(self.class_names):
            yaml_content += f"  {i}: {name}\n"
        
        with open(output_dir / "data.yaml", 'w') as f:
            f.write(yaml_content)
        
        print(f"Exported {len(self)} images to YOLO format at {output_dir}")


# ============================================================================
# Utility Functions
# ============================================================================

def get_dataset_stats(dataset) -> Dict:
    """Get statistics for a dataset"""
    class_counts = defaultdict(int)
    total_boxes = 0
    
    if hasattr(dataset, 'get_all_image_ids'):
        # TACO
        for img_id in dataset.get_all_image_ids():
            anns = dataset.get_annotations(img_id)
            for ann in anns:
                class_counts[ann['class_name']] += 1
                total_boxes += 1
    else:
        # ICRA19
        for idx in range(len(dataset)):
            _, anns = dataset[idx]
            for ann in anns:
                class_counts[ann['class_name']] += 1
                total_boxes += 1
    
    return {
        'total_boxes': total_boxes,
        'class_counts': dict(class_counts),
    }


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Dataset utilities")
    parser.add_argument('--dataset', choices=['taco', 'icra19', 'both'], default='both')
    parser.add_argument('--stats', action='store_true', help='Print dataset statistics')
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent.parent / "datasets" / "detection"
    
    if args.dataset in ['taco', 'both']:
        print("\n" + "="*60)
        print("TACO Dataset")
        print("="*60)
        taco = TACODataset(
            root_dir=base_dir / "taco" / "TACO",
            unified_classes=True
        )
        if args.stats:
            stats = get_dataset_stats(taco)
            print(f"Total boxes: {stats['total_boxes']}")
            print(f"Class distribution: {stats['class_counts']}")
    
    if args.dataset in ['icra19', 'both']:
        print("\n" + "="*60)
        print("Trash-ICRA19 Dataset")
        print("="*60)
        for split in ['train', 'val', 'test']:
            icra = TrashICRA19Dataset(
                root_dir=base_dir / "trash_icra19" / "trash_ICRA19",
                split=split,
                unified_classes=True
            )
            if args.stats:
                stats = get_dataset_stats(icra)
                print(f"  {split}: {stats['total_boxes']} boxes, {stats['class_counts']}")
