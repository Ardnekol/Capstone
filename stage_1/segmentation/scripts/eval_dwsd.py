#!/usr/bin/env python3
"""
DWSD Evaluation Script for Pre-trained Models

Load trained models and evaluate them on DWSD dataset.

Usage:
    python eval_dwsd.py
"""

import os
import json
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import segmentation_models_pytorch as smp
from sklearn.metrics import jaccard_score, precision_score, recall_score, f1_score
from PIL import Image
import numpy as np
from pathlib import Path
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    'img_size': (256, 256),
    'batch_size': 8
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
            transforms.Resize(CONFIG['img_size']),
            transforms.ToTensor()
        ])
        mask = mask_transform(mask)

        # Ensure mask is binary (0 or 1)
        mask = (mask > 0.5).float()

        return image, mask

# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_model(model, dataloader, device):
    """Evaluate model on dataset."""
    model.eval()
    ious, precisions, recalls, f1s = [], [], [], []

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            outputs = model(images)

            # For segmentation models
            if hasattr(outputs, 'sigmoid'):
                preds = (outputs.sigmoid() > 0.5).float()
            else:
                preds = (outputs > 0.5).float()

            # Flatten for metric calculation
            preds_flat = preds.view(-1).cpu().numpy()
            masks_flat = masks.view(-1).cpu().numpy()

            # Calculate metrics
            iou = jaccard_score(masks_flat, preds_flat, average='binary', zero_division=0)
            precision = precision_score(masks_flat, preds_flat, average='binary', zero_division=0)
            recall = recall_score(masks_flat, preds_flat, average='binary', zero_division=0)
            f1 = f1_score(masks_flat, preds_flat, average='binary', zero_division=0)

            ious.append(iou)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

    return {
        'iou': np.mean(ious),
        'precision': np.mean(precisions),
        'recall': np.mean(recalls),
        'f1': np.mean(f1s)
    }

def evaluate_maskrcnn_model(model, dataloader, device):
    """Evaluate Mask R-CNN model."""
    model.eval()
    ious, precisions, recalls, f1s = [], [], [], []

    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass
            outputs = model(images)

            for i, output in enumerate(outputs):
                if len(output['masks']) == 0:
                    # No predictions
                    ious.append(0)
                    precisions.append(0)
                    recalls.append(0)
                    f1s.append(0)
                    continue

                # Take the mask with highest score
                mask_pred = (output['masks'][0] > 0.5).float().cpu().numpy().squeeze()
                mask_true = targets[i]['masks'][0].cpu().numpy().squeeze()

                # Resize predictions to match ground truth if needed
                if mask_pred.shape != mask_true.shape:
                    from PIL import Image as PILImage
                    mask_pred = PILImage.fromarray(mask_pred.astype(np.uint8))
                    mask_pred = mask_pred.resize(mask_true.shape[::-1], PILImage.NEAREST)
                    mask_pred = np.array(mask_pred)

                # Flatten
                pred_flat = mask_pred.flatten()
                true_flat = mask_true.flatten()

                iou = jaccard_score(true_flat, pred_flat, average='binary', zero_division=0)
                precision = precision_score(true_flat, pred_flat, average='binary', zero_division=0)
                recall = recall_score(true_flat, pred_flat, average='binary', zero_division=0)
                f1 = f1_score(true_flat, pred_flat, average='binary', zero_division=0)

                ious.append(iou)
                precisions.append(precision)
                recalls.append(recall)
                f1s.append(f1)

    return {
        'iou': np.mean(ious),
        'precision': np.mean(precisions),
        'recall': np.mean(recalls),
        'f1': np.mean(f1s)
    }

def evaluate_sam_on_dataset(images_dir, masks_dir, mask_generator):
    """Evaluate SAM on dataset."""
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)

    image_files = sorted(list(images_dir.glob('*.jpg')))
    mask_files = sorted(list(masks_dir.glob('*.png')))

    ious, precisions, recalls, f1s = [], [], [], []

    for img_path, mask_path in zip(image_files, mask_files):
        # Load image
        image = np.array(Image.open(img_path).convert('RGB'))

        # Load ground truth mask
        mask_true = np.array(Image.open(mask_path).convert('L'))
        mask_true = (mask_true > 127).astype(np.uint8)  # Binary

        # Generate SAM masks
        masks = mask_generator.generate(image)

        if not masks:
            ious.append(0)
            precisions.append(0)
            recalls.append(0)
            f1s.append(0)
            continue

        # Find best matching mask (highest IoU with ground truth)
        best_iou = 0
        best_mask = None

        for sam_mask in masks:
            mask_pred = sam_mask['segmentation'].astype(np.uint8)

            # Resize if needed
            if mask_pred.shape != mask_true.shape:
                from PIL import Image as PILImage
                mask_pred = PILImage.fromarray(mask_pred)
                mask_pred = mask_pred.resize(mask_true.shape[::-1], PILImage.NEAREST)
                mask_pred = np.array(mask_pred)

            pred_flat = mask_pred.flatten()
            true_flat = mask_true.flatten()

            iou = jaccard_score(true_flat, pred_flat, average='binary', zero_division=0)
            if iou > best_iou:
                best_iou = iou
                best_mask = mask_pred

        if best_mask is not None:
            pred_flat = best_mask.flatten()
            true_flat = mask_true.flatten()

            precision = precision_score(true_flat, pred_flat, average='binary', zero_division=0)
            recall = recall_score(true_flat, pred_flat, average='binary', zero_division=0)
            f1 = f1_score(true_flat, pred_flat, average='binary', zero_division=0)

            ious.append(best_iou)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
        else:
            ious.append(0)
            precisions.append(0)
            recalls.append(0)
            f1s.append(0)

    return {
        'iou': np.mean(ious),
        'precision': np.mean(precisions),
        'recall': np.mean(recalls),
        'f1': np.mean(f1s)
    }

# ============================================================================
# Main Function
# ============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Data transforms
    data_transforms = transforms.Compose([
        transforms.Resize(CONFIG['img_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # DWSD dataset
    dwsd_dataset = SegmentationDataset(
        '/u/student/2024/cs24mtech11024/Capstone/stage_1/segmentation/data/dwsd/test/images',
        '/u/student/2024/cs24mtech11024/Capstone/stage_1/segmentation/data/dwsd/test/masks',
        data_transforms
    )
    dwsd_loader = DataLoader(dwsd_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4)

    # Evaluate U-Net
    print("Evaluating U-Net on DWSD...")
    unet_model = smp.Unet(
        encoder_name='resnet34',
        encoder_weights=None,  # No pretrained weights for inference
        in_channels=3,
        classes=1
    )
    unet_model.load_state_dict(torch.load('results/unet_best.pth', map_location=device))
    unet_model = unet_model.to(device)
    unet_metrics = evaluate_model(unet_model, dwsd_loader, device)

    # Update U-Net results
    with open('results/unet_results.json', 'r') as f:
        unet_results = json.load(f)
    unet_results['dwsd_iou'] = unet_metrics['iou']
    unet_results['dwsd_metrics'] = unet_metrics
    with open('results/unet_results.json', 'w') as f:
        json.dump(unet_results, f, indent=2)

    # Evaluate DeepLabV3+
    print("Evaluating DeepLabV3+ on DWSD...")
    deeplab_model = smp.DeepLabV3Plus(
        encoder_name='resnet101',
        encoder_weights=None,
        in_channels=3,
        classes=1
    )
    deeplab_model.load_state_dict(torch.load('results/deeplabv3plus_best.pth', map_location=device))
    deeplab_model = deeplab_model.to(device)
    deeplab_metrics = evaluate_model(deeplab_model, dwsd_loader, device)

    # Update DeepLabV3+ results
    with open('results/deeplabv3plus_results.json', 'r') as f:
        deeplab_results = json.load(f)
    deeplab_results['dwsd_iou'] = deeplab_metrics['iou']
    deeplab_results['dwsd_metrics'] = deeplab_metrics
    with open('results/deeplabv3plus_results.json', 'w') as f:
        json.dump(deeplab_results, f, indent=2)

    # Evaluate Mask R-CNN
    print("Evaluating Mask R-CNN on DWSD...")
    def create_maskrcnn_model():
        model = maskrcnn_resnet50_fpn(pretrained=False)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, 2)
        return model

    maskrcnn_model = create_maskrcnn_model()
    maskrcnn_model.load_state_dict(torch.load('results/maskrcnn_best.pth', map_location=device))
    maskrcnn_model = maskrcnn_model.to(device)

    # Create Mask R-CNN dataset (different format)
    class MaskRCNNDataset(Dataset):
        def __init__(self, images_dir, masks_dir, transforms=None):
            self.images_dir = Path(images_dir)
            self.masks_dir = Path(masks_dir)
            self.transforms = transforms
            self.image_files = sorted(list(self.images_dir.glob('*.jpg')))
            self.mask_files = sorted(list(self.masks_dir.glob('*.png')))

        def __len__(self):
            return len(self.image_files)

        def __getitem__(self, idx):
            img_path = self.image_files[idx]
            mask_path = self.mask_files[idx]

            img = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')
            mask = np.array(mask)
            mask = (mask > 127).astype(np.uint8)

            # Convert to tensor format expected by Mask R-CNN
            if self.transforms:
                img = self.transforms(img)

            # Create target dict
            obj_ids = np.unique(mask)
            obj_ids = obj_ids[1:]  # Remove background

            masks = []
            for obj_id in obj_ids:
                masks.append(mask == obj_id)

            if len(masks) == 0:
                # No objects, create dummy mask
                masks = np.zeros((1,) + mask.shape, dtype=np.uint8)
                masks[0] = mask > 0  # Binary mask

            masks = np.array(masks)
            num_objs = len(masks)

            boxes = []
            for i in range(num_objs):
                pos = np.where(masks[i])
                if len(pos[0]) > 0:
                    xmin = np.min(pos[1])
                    xmax = np.max(pos[1])
                    ymin = np.min(pos[0])
                    ymax = np.max(pos[0])
                    boxes.append([xmin, ymin, xmax, ymax])

            if len(boxes) == 0:
                boxes = [[0, 0, 1, 1]]  # Dummy box
                masks = np.zeros((1, mask.shape[0], mask.shape[1]), dtype=np.uint8)
                masks[0] = mask > 0

            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.ones((len(boxes),), dtype=torch.int64)
            masks = torch.as_tensor(masks, dtype=torch.uint8)
            image_id = torch.tensor([idx])
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

            target = {
                'boxes': boxes,
                'labels': labels,
                'masks': masks,
                'image_id': image_id,
                'area': area,
                'iscrowd': iscrowd
            }

            return img, target

    maskrcnn_dataset = MaskRCNNDataset(
        '/u/student/2024/cs24mtech11024/Capstone/stage_1/segmentation/data/dwsd/test/images',
        '/u/student/2024/cs24mtech11024/Capstone/stage_1/segmentation/data/dwsd/test/masks',
        data_transforms
    )
    maskrcnn_loader = DataLoader(maskrcnn_dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=lambda x: tuple(zip(*x)))
    maskrcnn_metrics = evaluate_maskrcnn_model(maskrcnn_model, maskrcnn_loader, device)

    # Update Mask R-CNN results
    with open('results/maskrcnn_results.json', 'r') as f:
        maskrcnn_results = json.load(f)
    maskrcnn_results['dwsd_iou'] = maskrcnn_metrics['iou']
    maskrcnn_results['dwsd_metrics'] = maskrcnn_metrics
    with open('results/maskrcnn_results.json', 'w') as f:
        json.dump(maskrcnn_results, f, indent=2)

    # Evaluate SAM
    print("Evaluating SAM on DWSD...")
    sam_checkpoint = 'sam_vit_h_4b8939.pth'
    sam = sam_model_registry['vit_h'](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    sam_metrics = evaluate_sam_on_dataset(
        '/u/student/2024/cs24mtech11024/Capstone/stage_1/segmentation/data/dwsd/test/images',
        '/u/student/2024/cs24mtech11024/Capstone/stage_1/segmentation/data/dwsd/test/masks',
        mask_generator
    )

    # Update SAM results
    with open('results/sam_results.json', 'r') as f:
        sam_results = json.load(f)
    sam_results['dwsd_iou'] = sam_metrics['iou']
    sam_results['dwsd_metrics'] = sam_metrics
    with open('results/sam_results.json', 'w') as f:
        json.dump(sam_results, f, indent=2)

    print("DWSD evaluation complete!")
    print(f"U-Net DWSD IoU: {unet_metrics['iou']:.4f}")
    print(f"DeepLabV3+ DWSD IoU: {deeplab_metrics['iou']:.4f}")
    print(f"Mask R-CNN DWSD IoU: {maskrcnn_metrics['iou']:.4f}")
    print(f"SAM DWSD IoU: {sam_metrics['iou']:.4f}")

if __name__ == '__main__':
    main()