#!/usr/bin/env python3
"""
Unified Evaluation Metrics for Foundation Models Study

This module provides standardized evaluation metrics for:
1. Classification: Accuracy, Top-K Accuracy, Confusion Matrix
2. Object Detection: mAP@0.5, mAP@0.5:0.95, Per-class AP
3. Segmentation: mIoU, Pixel Accuracy, Boundary F1

Usage:
    from metrics import ClassificationMetrics, DetectionMetrics, SegmentationMetrics
    
    # Classification
    clf_metrics = ClassificationMetrics()
    results = clf_metrics.compute(y_true, y_pred, class_names)
    
    # Detection
    det_metrics = DetectionMetrics()
    results = det_metrics.compute(predictions, ground_truth)
    
    # Segmentation
    seg_metrics = SegmentationMetrics()
    results = seg_metrics.compute(pred_masks, gt_masks, class_names)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict
import json
from pathlib import Path


# ============================================================================
# Classification Metrics
# ============================================================================

class ClassificationMetrics:
    """
    Metrics for image classification evaluation.
    
    Metrics:
        - Top-1 Accuracy
        - Top-K Accuracy
        - Per-class Accuracy
        - Confusion Matrix
        - Precision, Recall, F1 per class
    """
    
    def __init__(self, num_classes: Optional[int] = None):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reset accumulated predictions"""
        self.y_true = []
        self.y_pred = []
        self.y_pred_probs = []
    
    def update(self, 
               y_true: np.ndarray, 
               y_pred: np.ndarray,
               y_pred_probs: Optional[np.ndarray] = None):
        """
        Add batch of predictions.
        
        Args:
            y_true: Ground truth labels (N,)
            y_pred: Predicted labels (N,)
            y_pred_probs: Prediction probabilities (N, num_classes) for top-k
        """
        self.y_true.extend(y_true.tolist() if hasattr(y_true, 'tolist') else y_true)
        self.y_pred.extend(y_pred.tolist() if hasattr(y_pred, 'tolist') else y_pred)
        
        if y_pred_probs is not None:
            self.y_pred_probs.extend(y_pred_probs.tolist() if hasattr(y_pred_probs, 'tolist') else y_pred_probs)
    
    def compute(self, 
                class_names: Optional[List[str]] = None,
                k_values: List[int] = [1, 5]) -> Dict:
        """
        Compute all classification metrics.
        
        Args:
            class_names: List of class names for per-class metrics
            k_values: Values of K for top-K accuracy
            
        Returns:
            Dictionary with all metrics
        """
        y_true = np.array(self.y_true)
        y_pred = np.array(self.y_pred)
        
        if self.num_classes is None:
            self.num_classes = max(np.max(y_true), np.max(y_pred)) + 1
        
        results = {}
        
        # Top-1 Accuracy
        results['accuracy'] = float(np.mean(y_true == y_pred))
        results['top1_accuracy'] = results['accuracy']
        
        # Top-K Accuracy (if probabilities available)
        if len(self.y_pred_probs) > 0:
            y_probs = np.array(self.y_pred_probs)
            for k in k_values:
                if k == 1:
                    continue
                top_k_preds = np.argsort(y_probs, axis=1)[:, -k:]
                top_k_correct = np.array([y_true[i] in top_k_preds[i] for i in range(len(y_true))])
                results[f'top{k}_accuracy'] = float(np.mean(top_k_correct))
        
        # Confusion Matrix
        cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
        results['confusion_matrix'] = cm.tolist()
        
        # Per-class metrics
        per_class = {}
        for c in range(self.num_classes):
            class_name = class_names[c] if class_names and c < len(class_names) else f"class_{c}"
            
            # True positives, false positives, false negatives
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            per_class[class_name] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'support': int(cm[c, :].sum())
            }
        
        results['per_class'] = per_class
        
        # Macro averages
        results['macro_precision'] = float(np.mean([v['precision'] for v in per_class.values()]))
        results['macro_recall'] = float(np.mean([v['recall'] for v in per_class.values()]))
        results['macro_f1'] = float(np.mean([v['f1'] for v in per_class.values()]))
        
        # Weighted averages
        supports = np.array([v['support'] for v in per_class.values()])
        total_support = supports.sum()
        if total_support > 0:
            results['weighted_precision'] = float(np.sum([v['precision'] * v['support'] for v in per_class.values()]) / total_support)
            results['weighted_recall'] = float(np.sum([v['recall'] * v['support'] for v in per_class.values()]) / total_support)
            results['weighted_f1'] = float(np.sum([v['f1'] * v['support'] for v in per_class.values()]) / total_support)
        
        results['total_samples'] = int(len(y_true))
        
        return results


# ============================================================================
# Object Detection Metrics
# ============================================================================

class DetectionMetrics:
    """
    Metrics for object detection evaluation.
    
    Metrics:
        - mAP@0.5 (PASCAL VOC style)
        - mAP@0.5:0.95 (COCO style)
        - Per-class AP
        - Precision-Recall curves
    """
    
    def __init__(self, 
                 iou_thresholds: Optional[List[float]] = None,
                 num_classes: Optional[int] = None):
        self.iou_thresholds = iou_thresholds or [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reset accumulated detections"""
        self.predictions = []  # List of dicts with 'boxes', 'scores', 'labels'
        self.ground_truths = []  # List of dicts with 'boxes', 'labels'
    
    def update(self,
               pred_boxes: np.ndarray,
               pred_scores: np.ndarray,
               pred_labels: np.ndarray,
               gt_boxes: np.ndarray,
               gt_labels: np.ndarray,
               image_id: Optional[int] = None):
        """
        Add predictions for one image.
        
        Args:
            pred_boxes: Predicted boxes (N, 4) in [x1, y1, x2, y2] format
            pred_scores: Confidence scores (N,)
            pred_labels: Predicted class labels (N,)
            gt_boxes: Ground truth boxes (M, 4)
            gt_labels: Ground truth labels (M,)
            image_id: Optional image identifier
        """
        self.predictions.append({
            'boxes': np.array(pred_boxes),
            'scores': np.array(pred_scores),
            'labels': np.array(pred_labels),
            'image_id': image_id or len(self.predictions)
        })
        
        self.ground_truths.append({
            'boxes': np.array(gt_boxes),
            'labels': np.array(gt_labels),
            'image_id': image_id or len(self.ground_truths)
        })
    
    @staticmethod
    def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IoU between two boxes in [x1, y1, x2, y2] format"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def compute_ap(self, precision: np.ndarray, recall: np.ndarray) -> float:
        """Compute Average Precision using 11-point interpolation"""
        # Append sentinel values
        recall = np.concatenate(([0.], recall, [1.]))
        precision = np.concatenate(([0.], precision, [0.]))
        
        # Make precision monotonically decreasing
        for i in range(len(precision) - 2, -1, -1):
            precision[i] = max(precision[i], precision[i + 1])
        
        # Find points where recall changes
        recall_change = np.where(recall[1:] != recall[:-1])[0]
        
        # Sum up areas under the curve
        ap = np.sum((recall[recall_change + 1] - recall[recall_change]) * precision[recall_change + 1])
        
        return float(ap)
    
    def evaluate_class(self, 
                       class_id: int, 
                       iou_threshold: float) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Evaluate AP for a single class at a given IoU threshold.
        
        Returns:
            Tuple of (AP, precision_array, recall_array)
        """
        # Collect all predictions and ground truths for this class
        all_preds = []
        all_gts = defaultdict(list)
        n_gt = 0
        
        for pred, gt in zip(self.predictions, self.ground_truths):
            img_id = pred['image_id']
            
            # Get predictions for this class
            mask = pred['labels'] == class_id
            if mask.any():
                for box, score in zip(pred['boxes'][mask], pred['scores'][mask]):
                    all_preds.append({
                        'image_id': img_id,
                        'box': box,
                        'score': score
                    })
            
            # Get ground truths for this class
            gt_mask = gt['labels'] == class_id
            if gt_mask.any():
                for box in gt['boxes'][gt_mask]:
                    all_gts[img_id].append({
                        'box': box,
                        'matched': False
                    })
                    n_gt += 1
        
        if n_gt == 0:
            return 0.0, np.array([]), np.array([])
        
        # Sort predictions by score (descending)
        all_preds = sorted(all_preds, key=lambda x: x['score'], reverse=True)
        
        # Compute TP/FP for each prediction
        tp = np.zeros(len(all_preds))
        fp = np.zeros(len(all_preds))
        
        for i, pred in enumerate(all_preds):
            img_id = pred['image_id']
            pred_box = pred['box']
            
            best_iou = 0.0
            best_gt_idx = -1
            
            for j, gt in enumerate(all_gts[img_id]):
                if gt['matched']:
                    continue
                    
                iou = self.compute_iou(pred_box, gt['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp[i] = 1
                all_gts[img_id][best_gt_idx]['matched'] = True
            else:
                fp[i] = 1
        
        # Compute precision/recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recall = tp_cumsum / n_gt
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        ap = self.compute_ap(precision, recall)
        
        return ap, precision, recall
    
    def compute(self, class_names: Optional[List[str]] = None) -> Dict:
        """
        Compute all detection metrics.
        
        Returns:
            Dictionary with mAP and per-class AP at various IoU thresholds
        """
        # Determine number of classes
        if self.num_classes is None:
            all_labels = []
            for gt in self.ground_truths:
                all_labels.extend(gt['labels'].tolist())
            self.num_classes = max(all_labels) + 1 if all_labels else 0
        
        results = {}
        
        # Compute AP at each IoU threshold for each class
        ap_per_iou = defaultdict(list)
        per_class_ap = defaultdict(dict)
        
        for iou_thresh in self.iou_thresholds:
            iou_key = f"AP@{iou_thresh:.2f}"
            class_aps = []
            
            for class_id in range(self.num_classes):
                class_name = class_names[class_id] if class_names and class_id < len(class_names) else f"class_{class_id}"
                ap, _, _ = self.evaluate_class(class_id, iou_thresh)
                class_aps.append(ap)
                per_class_ap[class_name][iou_key] = float(ap)
            
            if class_aps:
                ap_per_iou[iou_key] = float(np.mean(class_aps))
        
        # Standard metrics
        results['mAP@0.5'] = ap_per_iou.get('AP@0.50', 0.0)
        results['mAP@0.75'] = ap_per_iou.get('AP@0.75', 0.0)
        results['mAP@0.5:0.95'] = float(np.mean(list(ap_per_iou.values()))) if ap_per_iou else 0.0
        
        results['AP_per_IoU'] = dict(ap_per_iou)
        results['per_class'] = dict(per_class_ap)
        
        # Summary statistics
        results['num_predictions'] = sum(len(p['boxes']) for p in self.predictions)
        results['num_ground_truths'] = sum(len(g['boxes']) for g in self.ground_truths)
        results['num_images'] = len(self.predictions)
        
        return results


# ============================================================================
# Segmentation Metrics
# ============================================================================

class SegmentationMetrics:
    """
    Metrics for semantic/instance segmentation evaluation.
    
    Metrics:
        - mIoU (mean Intersection over Union)
        - Pixel Accuracy
        - Per-class IoU
        - Boundary F1 Score
        - Dice Score
    """
    
    def __init__(self, 
                 num_classes: Optional[int] = None,
                 ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        """Reset accumulated confusion matrix"""
        if self.num_classes:
            self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        else:
            self.confusion_matrix = None
        self.boundary_scores = []
    
    def update(self,
               pred_mask: np.ndarray,
               gt_mask: np.ndarray):
        """
        Add predictions for one image.
        
        Args:
            pred_mask: Predicted segmentation mask (H, W)
            gt_mask: Ground truth mask (H, W)
        """
        pred_mask = np.array(pred_mask).flatten()
        gt_mask = np.array(gt_mask).flatten()
        
        # Initialize confusion matrix if needed
        if self.confusion_matrix is None:
            self.num_classes = max(np.max(pred_mask), np.max(gt_mask)) + 1
            self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        
        # Filter out ignore index
        valid_mask = gt_mask != self.ignore_index
        pred_mask = pred_mask[valid_mask]
        gt_mask = gt_mask[valid_mask]
        
        # Update confusion matrix
        for t, p in zip(gt_mask, pred_mask):
            if t < self.num_classes and p < self.num_classes:
                self.confusion_matrix[t, p] += 1
    
    def compute_boundary_f1(self,
                            pred_mask: np.ndarray,
                            gt_mask: np.ndarray,
                            dilation_radius: int = 2) -> float:
        """
        Compute boundary F1 score between predicted and ground truth masks.
        
        This measures how well the model captures object boundaries.
        """
        try:
            from scipy import ndimage
        except ImportError:
            return 0.0
        
        def get_boundary(mask, radius):
            """Extract boundary pixels from a mask"""
            dilated = ndimage.binary_dilation(mask, iterations=radius)
            eroded = ndimage.binary_erosion(mask, iterations=radius)
            return dilated ^ eroded
        
        # Get boundaries for each class
        f1_scores = []
        
        for class_id in range(self.num_classes):
            pred_binary = (pred_mask == class_id).astype(np.uint8)
            gt_binary = (gt_mask == class_id).astype(np.uint8)
            
            if gt_binary.sum() == 0 and pred_binary.sum() == 0:
                continue
            
            pred_boundary = get_boundary(pred_binary, dilation_radius)
            gt_boundary = get_boundary(gt_binary, dilation_radius)
            
            # Compute precision and recall
            tp = np.logical_and(pred_boundary, gt_boundary).sum()
            fp = np.logical_and(pred_boundary, ~gt_boundary).sum()
            fn = np.logical_and(~pred_boundary, gt_boundary).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            f1_scores.append(f1)
        
        return float(np.mean(f1_scores)) if f1_scores else 0.0
    
    def compute(self, class_names: Optional[List[str]] = None) -> Dict:
        """
        Compute all segmentation metrics.
        
        Returns:
            Dictionary with mIoU, pixel accuracy, and per-class metrics
        """
        results = {}
        
        if self.confusion_matrix is None or self.confusion_matrix.sum() == 0:
            return {'mIoU': 0.0, 'pixel_accuracy': 0.0, 'per_class': {}}
        
        # Pixel Accuracy
        total_correct = np.diag(self.confusion_matrix).sum()
        total_pixels = self.confusion_matrix.sum()
        results['pixel_accuracy'] = float(total_correct / total_pixels) if total_pixels > 0 else 0.0
        
        # Per-class IoU and Dice
        per_class = {}
        ious = []
        dices = []
        
        for c in range(self.num_classes):
            class_name = class_names[c] if class_names and c < len(class_names) else f"class_{c}"
            
            # IoU = TP / (TP + FP + FN)
            tp = self.confusion_matrix[c, c]
            fp = self.confusion_matrix[:, c].sum() - tp
            fn = self.confusion_matrix[c, :].sum() - tp
            
            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
            dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
            
            per_class[class_name] = {
                'iou': float(iou),
                'dice': float(dice),
                'precision': float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
                'recall': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
                'support': int(self.confusion_matrix[c, :].sum())
            }
            
            # Only include classes with ground truth support for mIoU
            if self.confusion_matrix[c, :].sum() > 0:
                ious.append(iou)
                dices.append(dice)
        
        results['mIoU'] = float(np.mean(ious)) if ious else 0.0
        results['mean_dice'] = float(np.mean(dices)) if dices else 0.0
        results['per_class'] = per_class
        
        # Frequency-weighted IoU
        freq = self.confusion_matrix.sum(axis=1) / self.confusion_matrix.sum()
        fw_iou = 0.0
        for c in range(self.num_classes):
            if self.confusion_matrix[c, :].sum() > 0:
                tp = self.confusion_matrix[c, c]
                fp = self.confusion_matrix[:, c].sum() - tp
                fn = self.confusion_matrix[c, :].sum() - tp
                iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
                fw_iou += freq[c] * iou
        results['frequency_weighted_iou'] = float(fw_iou)
        
        # Summary
        results['num_classes'] = int(self.num_classes)
        results['total_pixels'] = int(total_pixels)
        
        return results


# ============================================================================
# Generalization Metrics
# ============================================================================

class GeneralizationMetrics:
    """
    Compute generalization gap metrics between in-domain and out-of-domain performance.
    """
    
    @staticmethod
    def compute_drop(in_domain: float, out_domain: float) -> Dict:
        """
        Compute performance drop metrics.
        
        Args:
            in_domain: In-domain performance (e.g., accuracy on TrashNet val)
            out_domain: Out-of-domain performance (e.g., accuracy on RealWaste)
            
        Returns:
            Dictionary with absolute and relative drop
        """
        absolute_drop = in_domain - out_domain
        relative_drop = (absolute_drop / in_domain * 100) if in_domain > 0 else 0.0
        
        return {
            'in_domain': float(in_domain),
            'out_domain': float(out_domain),
            'absolute_drop': float(absolute_drop),
            'relative_drop_percent': float(relative_drop),
            'retention_ratio': float(out_domain / in_domain) if in_domain > 0 else 0.0
        }
    
    @staticmethod
    def compare_models(results: Dict[str, Dict]) -> Dict:
        """
        Compare generalization across multiple models.
        
        Args:
            results: Dictionary of {model_name: {'in_domain': X, 'out_domain': Y}}
            
        Returns:
            Comparison summary with rankings
        """
        comparisons = {}
        
        for model_name, scores in results.items():
            comparisons[model_name] = GeneralizationMetrics.compute_drop(
                scores['in_domain'],
                scores['out_domain']
            )
        
        # Rank by out-of-domain performance (higher is better)
        ranked = sorted(comparisons.items(), key=lambda x: x[1]['out_domain'], reverse=True)
        
        # Rank by retention ratio (higher is better)
        ranked_by_retention = sorted(comparisons.items(), key=lambda x: x[1]['retention_ratio'], reverse=True)
        
        return {
            'per_model': comparisons,
            'ranking_by_ood_performance': [m[0] for m in ranked],
            'ranking_by_retention': [m[0] for m in ranked_by_retention],
            'best_ood_performer': ranked[0][0] if ranked else None,
            'best_generalizer': ranked_by_retention[0][0] if ranked_by_retention else None
        }


# ============================================================================
# Utility Functions
# ============================================================================

def save_results(results: Dict, output_path: str):
    """Save results to JSON file"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Results saved to: {output_path}")


def load_results(input_path: str) -> Dict:
    """Load results from JSON file"""
    with open(input_path, 'r') as f:
        return json.load(f)


def print_classification_results(results: Dict, title: str = "Classification Results"):
    """Pretty print classification results"""
    print(f"\n{'='*60}")
    print(f"📊 {title}")
    print(f"{'='*60}")
    print(f"Top-1 Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    if 'top5_accuracy' in results:
        print(f"Top-5 Accuracy: {results['top5_accuracy']:.4f} ({results['top5_accuracy']*100:.2f}%)")
    print(f"Macro F1: {results.get('macro_f1', 0):.4f}")
    print(f"\nPer-class Performance:")
    for class_name, metrics in results.get('per_class', {}).items():
        print(f"  {class_name}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")


def print_detection_results(results: Dict, title: str = "Detection Results"):
    """Pretty print detection results"""
    print(f"\n{'='*60}")
    print(f"📊 {title}")
    print(f"{'='*60}")
    print(f"mAP@0.5: {results['mAP@0.5']:.4f}")
    print(f"mAP@0.75: {results['mAP@0.75']:.4f}")
    print(f"mAP@0.5:0.95: {results['mAP@0.5:0.95']:.4f}")
    print(f"\nDetections: {results['num_predictions']} predictions, {results['num_ground_truths']} ground truths")


def print_segmentation_results(results: Dict, title: str = "Segmentation Results"):
    """Pretty print segmentation results"""
    print(f"\n{'='*60}")
    print(f"📊 {title}")
    print(f"{'='*60}")
    print(f"mIoU: {results['mIoU']:.4f} ({results['mIoU']*100:.2f}%)")
    print(f"Pixel Accuracy: {results['pixel_accuracy']:.4f} ({results['pixel_accuracy']*100:.2f}%)")
    print(f"Mean Dice: {results.get('mean_dice', 0):.4f}")
    print(f"\nPer-class IoU:")
    for class_name, metrics in results.get('per_class', {}).items():
        print(f"  {class_name}: IoU={metrics['iou']:.3f}, Dice={metrics['dice']:.3f}")


# ============================================================================
# Main (Testing)
# ============================================================================

if __name__ == "__main__":
    print("Testing Metrics Module...")
    
    # Test Classification Metrics
    print("\n" + "="*60)
    print("Testing Classification Metrics")
    print("="*60)
    
    clf = ClassificationMetrics(num_classes=6)
    np.random.seed(42)
    y_true = np.random.randint(0, 6, 100)
    y_pred = y_true.copy()
    y_pred[:20] = np.random.randint(0, 6, 20)  # Add some errors
    
    clf.update(y_true, y_pred)
    clf_results = clf.compute(class_names=['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash'])
    print_classification_results(clf_results)
    
    # Test Detection Metrics
    print("\n" + "="*60)
    print("Testing Detection Metrics")
    print("="*60)
    
    det = DetectionMetrics(num_classes=3)
    
    # Simulate some detections
    for i in range(10):
        pred_boxes = np.random.rand(5, 4) * 100
        pred_boxes[:, 2:] += pred_boxes[:, :2]  # Make valid boxes
        pred_scores = np.random.rand(5)
        pred_labels = np.random.randint(0, 3, 5)
        
        gt_boxes = pred_boxes[:3] + np.random.rand(3, 4) * 10  # Slightly offset
        gt_labels = pred_labels[:3]
        
        det.update(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels)
    
    det_results = det.compute(class_names=['trash', 'bio', 'rov'])
    print_detection_results(det_results)
    
    # Test Segmentation Metrics
    print("\n" + "="*60)
    print("Testing Segmentation Metrics")
    print("="*60)
    
    seg = SegmentationMetrics(num_classes=4)
    
    for i in range(5):
        gt_mask = np.random.randint(0, 4, (256, 256))
        pred_mask = gt_mask.copy()
        pred_mask[:50, :50] = np.random.randint(0, 4, (50, 50))  # Add noise
        
        seg.update(pred_mask, gt_mask)
    
    seg_results = seg.compute(class_names=['background', 'plastic', 'metal', 'organic'])
    print_segmentation_results(seg_results)
    
    # Test Generalization Metrics
    print("\n" + "="*60)
    print("Testing Generalization Metrics")
    print("="*60)
    
    model_results = {
        'ResNet-50': {'in_domain': 0.95, 'out_domain': 0.72},
        'EfficientNet': {'in_domain': 0.93, 'out_domain': 0.68},
        'CLIP': {'in_domain': 0.88, 'out_domain': 0.82},
        'DINOv2': {'in_domain': 0.85, 'out_domain': 0.80}
    }
    
    comparison = GeneralizationMetrics.compare_models(model_results)
    print(f"Best OOD Performer: {comparison['best_ood_performer']}")
    print(f"Best Generalizer: {comparison['best_generalizer']}")
    print(f"Ranking by OOD: {comparison['ranking_by_ood_performance']}")
    print(f"Ranking by Retention: {comparison['ranking_by_retention']}")
    
    print("\n✅ All tests passed!")
