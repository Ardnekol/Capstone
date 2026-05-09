#!/usr/bin/env python3
"""Evaluate the unified Florence-2 multi-task model on all three tasks.

Runs classification, object detection, and segmentation evaluation using
a single Florence-2 model (base or LoRA fine-tuned) across in-domain and
cross-domain datasets.

Evaluation datasets:
  Classification: TrashNet (in-domain), RealWaste (cross-domain)
  Detection:      TACO (in-domain), Trash-ICRA19 (cross-domain)
  Segmentation:   TACO masks (in-domain), DWSD (cross-domain)

Metrics:
  Classification: accuracy, macro F1, per-class F1
  Detection:      mAP@0.5, precision, recall (via IoU matching)
  Segmentation:   mIoU, pixel accuracy
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

warnings.filterwarnings("ignore")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _stage2_root() -> Path:
    return _repo_root() / "stage_2"


def _resolve_path_arg(path_str: Optional[str]) -> Optional[Path]:
    if path_str is None:
        return None
    p = Path(path_str).expanduser()
    if p.is_absolute():
        return p.resolve()
    capstone = _repo_root().resolve()
    workspace = capstone.parent
    candidates = [Path.cwd() / p, capstone / p, workspace / p]
    if len(p.parts) > 0 and p.parts[0] == "Capstone":
        candidates.insert(0, workspace / p)
    for c in candidates:
        if c.exists():
            return c.resolve()
    return candidates[0].resolve()


# ──────────────────────────────────────────────────────────
# HF Cache setup
# ──────────────────────────────────────────────────────────

def configure_hf_cache(cache_dir: Path) -> None:
    cache_dir = cache_dir.expanduser().resolve()
    (cache_dir / "hub").mkdir(parents=True, exist_ok=True)
    (cache_dir / "modules").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_dir / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir / "hub"))
    os.environ.setdefault("HF_MODULES_CACHE", str(cache_dir / "modules"))


def default_cache_dir() -> Path:
    for key in ("SLURM_TMPDIR", "TMPDIR", "TMP"):
        val = os.environ.get(key)
        if val:
            return Path(val) / f"hf_cache_uid{os.getuid()}"
    return Path("/tmp") / f"hf_cache_uid{os.getuid()}"


# ──────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────

MODEL_ALIASES = {
    "base": "microsoft/Florence-2-base",
    "large": "microsoft/Florence-2-large",
    "base-ft": "microsoft/Florence-2-base-ft",
    "large-ft": "microsoft/Florence-2-large-ft",
}


def load_florence2(model_id: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    model_id = MODEL_ALIASES.get(model_id, model_id)
    adapter_dir = Path(model_id).expanduser()
    is_local_adapter = (
        adapter_dir.exists()
        and adapter_dir.is_dir()
        and (adapter_dir / "adapter_config.json").exists()
    )

    if is_local_adapter:
        adapter_cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
        base_id = adapter_cfg.get("base_model_name_or_path", "microsoft/Florence-2-large-ft")
        print(f"Loading LoRA adapter from {adapter_dir} (base: {base_id})")

        try:
            processor = AutoProcessor.from_pretrained(str(adapter_dir), trust_remote_code=True)
        except Exception:
            processor = AutoProcessor.from_pretrained(base_id, trust_remote_code=True)

        base_model = AutoModelForCausalLM.from_pretrained(
            base_id, trust_remote_code=True,
            torch_dtype=torch.float32, attn_implementation="eager",
        )

        from peft import PeftModel
        model = PeftModel.from_pretrained(base_model, str(adapter_dir))
        if hasattr(model, "merge_and_unload"):
            model = model.merge_and_unload()
    else:
        print(f"Loading model: {model_id}")
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True,
            torch_dtype=torch.float32, attn_implementation="eager",
        )

    model.eval()
    model.to(device)
    return model, processor


def run_florence_task(model, processor, image: Image.Image, task_prompt: str,
                     device: str, max_new_tokens: Optional[int] = None) -> Dict:
    import torch
    if max_new_tokens is None:
        if task_prompt.startswith("<CAPTION>"):
            max_new_tokens = 64
        elif task_prompt.startswith("<OD>"):
            max_new_tokens = 256
        elif task_prompt.startswith("<REFERRING_EXPRESSION_SEGMENTATION>"):
            max_new_tokens = 256
        else:
            max_new_tokens = 256

    inputs = processor(text=task_prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_new_tokens,
            num_beams=3,
            early_stopping=True,
        )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    result = processor.post_process_generation(
        generated_text, task=task_prompt.split(">")[0] + ">",
        image_size=(image.width, image.height),
    )
    return result


# ──────────────────────────────────────────────────────────
# Classification evaluation
# ──────────────────────────────────────────────────────────

REALWASTE_TO_TRASHNET = {
    "cardboard": "cardboard",
    "glass": "glass",
    "metal": "metal",
    "paper": "paper",
    "plastic": "plastic",
    "miscellaneous trash": "trash",
    "food organics": None,  # no TrashNet equivalent
    "textile trash": None,
    "vegetation": None,
}


def eval_classification(model, processor, dataset_root: Path, device: str,
                        max_images: int = 0, label_map: Optional[Dict] = None,
                        dataset_name: str = "dataset") -> Dict:
    """Evaluate classification using <CAPTION> task."""
    classes = sorted([
        d.name for d in dataset_root.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])
    print(f"\n  [{dataset_name}] Classes: {classes}")

    y_true, y_pred = [], []
    total, processed = 0, 0

    # When max_images is set, sample roughly evenly across class folders.
    # This avoids bias from alphabetical folder traversal taking only early classes.
    per_class_cap = 0
    if max_images and len(classes) > 0:
        per_class_cap = int(np.ceil(max_images / len(classes)))

    for cls_dir in sorted(dataset_root.iterdir()):
        if not cls_dir.is_dir() or cls_dir.name.startswith("."):
            continue
        gt_label = cls_dir.name.lower().strip()

        # Apply label mapping (e.g., RealWaste -> TrashNet classes)
        if label_map:
            mapped = label_map.get(gt_label)
            if mapped is None:
                continue  # skip unmapped classes
            gt_label = mapped

        images = sorted([
            f for f in cls_dir.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        ])

        class_processed = 0

        for img_path in images:
            total += 1
            if max_images and processed >= max_images:
                break
            if per_class_cap and class_processed >= per_class_cap:
                break
            try:
                img = Image.open(img_path).convert("RGB")
                result = run_florence_task(model, processor, img, "<CAPTION>", device, max_new_tokens=64)
                pred_text = ""
                for key in result:
                    val = result[key]
                    if isinstance(val, str):
                        pred_text = val.strip().lower()
                        break

                # Match prediction to known classes
                known = {"cardboard", "glass", "metal", "paper", "plastic", "trash"}
                pred_label = "unknown"
                for k in known:
                    if k in pred_text:
                        pred_label = k
                        break

                y_true.append(gt_label)
                y_pred.append(pred_label)
                processed += 1
                class_processed += 1

                if processed % 50 == 0:
                    print(f"    Processed {processed} images...")
            except Exception as e:
                print(f"    Error on {img_path.name}: {e}")
                continue

        if max_images and processed >= max_images:
            break

    # Compute metrics
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / len(y_true) if y_true else 0.0

    # Per-class F1
    all_labels = sorted(set(y_true + y_pred))
    per_class: Dict[str, Dict] = {}
    for label in all_labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[label] = {"precision": prec, "recall": rec, "f1": f1}

    macro_f1 = np.mean([v["f1"] for v in per_class.values()]) if per_class else 0.0

    result = {
        "dataset": dataset_name,
        "total_images": total,
        "evaluated": processed,
        "accuracy": round(accuracy, 4),
        "macro_f1": round(float(macro_f1), 4),
        "per_class": {k: {m: round(v, 4) for m, v in metrics.items()} for k, metrics in per_class.items()},
        "predictions": [{"true": t, "pred": p} for t, p in zip(y_true, y_pred)],
    }

    print(f"  [{dataset_name}] Accuracy: {accuracy:.4f}, Macro F1: {macro_f1:.4f}")
    for label, metrics in sorted(per_class.items()):
        print(f"    {label}: P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1']:.3f}")

    return result


# ──────────────────────────────────────────────────────────
# Detection evaluation
# ──────────────────────────────────────────────────────────

def _compute_iou_bbox(box1: List[float], box2: List[float]) -> float:
    """IoU between two [x1,y1,x2,y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def _parse_voc_xml(xml_path: Path, img_w: int, img_h: int) -> List[Dict]:
    """Parse PASCAL VOC XML annotation."""
    import xml.etree.ElementTree as ET
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    boxes = []
    for obj in root.findall("object"):
        name = obj.find("name")
        if name is None:
            continue
        label = name.text.strip().lower()
        if label in ("rov", "timestamp"):
            continue
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
        x1 = float(bndbox.find("xmin").text)
        y1 = float(bndbox.find("ymin").text)
        x2 = float(bndbox.find("xmax").text)
        y2 = float(bndbox.find("ymax").text)
        boxes.append({"label": label, "bbox": [x1, y1, x2, y2]})
    return boxes


def _compute_ap(precisions: List[float], recalls: List[float]) -> float:
    """Compute AP using all-points interpolation."""
    if not precisions or not recalls:
        return 0.0
    mrec = [0.0] + list(recalls) + [1.0]
    mpre = [0.0] + list(precisions) + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap


def eval_detection_taco(model, processor, taco_root: Path, device: str,
                        max_images: int = 0) -> Dict:
    """Evaluate detection on TACO test split using COCO annotations."""
    ann_path = taco_root / "annotations.json"
    if not ann_path.exists():
        print(f"  [TACO] annotations.json not found at {ann_path}")
        return {"dataset": "taco_detection", "error": "annotations not found"}

    data = json.loads(ann_path.read_text())
    categories = {int(c["id"]): c["name"].strip().lower() for c in data["categories"]}
    img_info = {int(im["id"]): im for im in data["images"]}

    ann_by_img = defaultdict(list)
    for ann in data["annotations"]:
        ann_by_img[int(ann["image_id"])].append(ann)

    # Use last 20% as test split
    img_ids = sorted(img_info.keys())
    test_start = int(len(img_ids) * 0.8)
    test_ids = img_ids[test_start:]

    all_tp, all_fp, all_fn = 0, 0, 0
    processed = 0
    iou_threshold = 0.5

    for img_id in test_ids:
        if max_images and processed >= max_images:
            break

        im = img_info[img_id]
        file_name = str(im.get("file_name", "")).lstrip("/")
        img_path = taco_root / file_name
        if not img_path.exists():
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        # Ground truth
        gt_boxes = []
        for ann in ann_by_img.get(img_id, []):
            if int(ann.get("iscrowd", 0)) == 1:
                continue
            bbox = ann.get("bbox", [])
            if len(bbox) != 4:
                continue
            x, y, w, h = map(float, bbox)
            gt_boxes.append({"label": categories.get(int(ann["category_id"]), "unknown"),
                             "bbox": [x, y, x + w, y + h]})

        # Prediction
        result = run_florence_task(model, processor, img, "<OD>", device)
        pred_boxes = []
        od_key = [k for k in result if "od" in k.lower() or "detect" in k.lower() or "bbox" in k.lower()]
        if od_key:
            od_result = result[od_key[0]]
            if isinstance(od_result, dict):
                bboxes = od_result.get("bboxes", [])
                labels = od_result.get("labels", [])
                for bbox, label in zip(bboxes, labels):
                    pred_boxes.append({"label": label.strip().lower(), "bbox": list(bbox)})

        # Match predictions to ground truth
        matched_gt = set()
        tp, fp = 0, 0
        for pred in pred_boxes:
            best_iou, best_idx = 0.0, -1
            for gi, gt in enumerate(gt_boxes):
                if gi in matched_gt:
                    continue
                iou = _compute_iou_bbox(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = gi
            if best_iou >= iou_threshold and best_idx >= 0:
                tp += 1
                matched_gt.add(best_idx)
            else:
                fp += 1

        fn = len(gt_boxes) - len(matched_gt)
        all_tp += tp
        all_fp += fp
        all_fn += fn
        processed += 1

        if processed % 20 == 0:
            print(f"    Processed {processed} images...")

    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    result = {
        "dataset": "taco_detection",
        "evaluated": processed,
        "iou_threshold": iou_threshold,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": all_tp, "fp": all_fp, "fn": all_fn,
    }
    print(f"  [TACO Det] P={precision:.4f} R={recall:.4f} F1={f1:.4f} (TP={all_tp} FP={all_fp} FN={all_fn})")
    return result


def eval_detection_icra19(model, processor, icra_root: Path, device: str,
                          max_images: int = 0) -> Dict:
    """Evaluate detection on ICRA19 test split (VOC XML annotations)."""
    test_dir = icra_root / "dataset" / "test"
    if not test_dir.exists():
        # Try val
        test_dir = icra_root / "dataset" / "val"
    if not test_dir.exists():
        return {"dataset": "icra19_detection", "error": f"test dir not found: {test_dir}"}

    image_files = sorted([
        f for f in test_dir.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png")
    ])

    all_tp, all_fp, all_fn = 0, 0, 0
    processed = 0
    iou_threshold = 0.5

    for img_path in image_files:
        if max_images and processed >= max_images:
            break

        xml_path = img_path.with_suffix(".xml")
        if not xml_path.exists():
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        gt_boxes = _parse_voc_xml(xml_path, img.width, img.height)
        if not gt_boxes:
            continue

        result = run_florence_task(model, processor, img, "<OD>", device)
        pred_boxes = []
        od_key = [k for k in result if "od" in k.lower() or "detect" in k.lower() or "bbox" in k.lower()]
        if od_key:
            od_result = result[od_key[0]]
            if isinstance(od_result, dict):
                bboxes = od_result.get("bboxes", [])
                labels = od_result.get("labels", [])
                for bbox, label in zip(bboxes, labels):
                    pred_boxes.append({"label": label.strip().lower(), "bbox": list(bbox)})

        matched_gt = set()
        tp, fp = 0, 0
        for pred in pred_boxes:
            best_iou, best_idx = 0.0, -1
            for gi, gt in enumerate(gt_boxes):
                if gi in matched_gt:
                    continue
                iou = _compute_iou_bbox(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = gi
            if best_iou >= iou_threshold and best_idx >= 0:
                tp += 1
                matched_gt.add(best_idx)
            else:
                fp += 1

        fn = len(gt_boxes) - len(matched_gt)
        all_tp += tp
        all_fp += fp
        all_fn += fn
        processed += 1

        if processed % 50 == 0:
            print(f"    Processed {processed} images...")

    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    result = {
        "dataset": "icra19_detection",
        "evaluated": processed,
        "iou_threshold": iou_threshold,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": all_tp, "fp": all_fp, "fn": all_fn,
    }
    print(f"  [ICRA19 Det] P={precision:.4f} R={recall:.4f} F1={f1:.4f} (TP={all_tp} FP={all_fp} FN={all_fn})")
    return result


# ──────────────────────────────────────────────────────────
# Segmentation evaluation
# ──────────────────────────────────────────────────────────

def _florence_seg_to_mask(result: Dict, image_size: Tuple[int, int]) -> Optional[np.ndarray]:
    """Extract binary mask from Florence-2 segmentation result."""
    w, h = image_size
    mask = np.zeros((h, w), dtype=np.uint8)

    # Look for polygon data in result
    for key in result:
        val = result[key]
        if isinstance(val, dict) and "polygons" in val:
            polygons = val["polygons"]
            if polygons:
                try:
                    from PIL import ImageDraw
                    pil_mask = Image.new("L", (w, h), 0)
                    draw = ImageDraw.Draw(pil_mask)
                    for poly_group in polygons:
                        for poly in poly_group:
                            if len(poly) >= 6:
                                coords = [(poly[i], poly[i + 1]) for i in range(0, len(poly) - 1, 2)]
                                draw.polygon(coords, fill=255)
                    mask = np.array(pil_mask)
                except Exception:
                    pass
    return mask


def eval_segmentation(model, processor, images_dir: Path, masks_dir: Path,
                      device: str, max_images: int = 0,
                      seg_phrase: str = "waste",
                      dataset_name: str = "dataset") -> Dict:
    """Evaluate segmentation using <REFERRING_EXPRESSION_SEGMENTATION> task."""

    image_files = sorted([
        f for f in images_dir.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
    ])

    total_iou, total_pixel_acc = 0.0, 0.0
    processed = 0

    for img_path in image_files:
        if max_images and processed >= max_images:
            break

        # Find corresponding mask
        mask_path = None
        for ext in (".png", ".jpg", ".bmp"):
            candidate = masks_dir / (img_path.stem + ext)
            if candidate.exists():
                mask_path = candidate
                break
        if mask_path is None:
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            gt_mask = np.array(Image.open(mask_path).convert("L"))
            gt_binary = (gt_mask > 127).astype(np.uint8)
        except Exception:
            continue

        task_prompt = f"<REFERRING_EXPRESSION_SEGMENTATION>{seg_phrase}"
        result = run_florence_task(model, processor, img, task_prompt, device)
        pred_mask = _florence_seg_to_mask(result, (img.width, img.height))

        if pred_mask is None:
            pred_mask = np.zeros_like(gt_binary)

        pred_binary = (pred_mask > 127).astype(np.uint8)

        # Resize if needed
        if pred_binary.shape != gt_binary.shape:
            pred_pil = Image.fromarray(pred_binary * 255)
            pred_pil = pred_pil.resize((gt_binary.shape[1], gt_binary.shape[0]), Image.NEAREST)
            pred_binary = (np.array(pred_pil) > 127).astype(np.uint8)

        # IoU
        intersection = np.sum(pred_binary & gt_binary)
        union = np.sum(pred_binary | gt_binary)
        iou = float(intersection) / float(union) if union > 0 else 0.0
        total_iou += iou

        # Pixel accuracy
        correct = np.sum(pred_binary == gt_binary)
        pixel_acc = float(correct) / float(gt_binary.size) if gt_binary.size > 0 else 0.0
        total_pixel_acc += pixel_acc

        processed += 1
        if processed % 20 == 0:
            print(f"    Processed {processed} images...")

    mean_iou = total_iou / processed if processed > 0 else 0.0
    mean_pixel_acc = total_pixel_acc / processed if processed > 0 else 0.0

    result = {
        "dataset": dataset_name,
        "evaluated": processed,
        "mean_iou": round(mean_iou, 4),
        "pixel_accuracy": round(mean_pixel_acc, 4),
    }
    print(f"  [{dataset_name}] mIoU={mean_iou:.4f}, PixelAcc={mean_pixel_acc:.4f} ({processed} images)")
    return result


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

def main() -> None:
    capstone = _repo_root()

    ap = argparse.ArgumentParser(description="Evaluate unified Florence-2 on all tasks")
    ap.add_argument(
        "--model-id", type=str, default="large-ft",
        help="Model alias (base/large/base-ft/large-ft) or path to LoRA adapter dir",
    )
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--cache-dir", type=str, default=None)
    ap.add_argument(
        "--max-images", type=int, default=0,
        help="Max images per dataset (0 = all). Useful for quick testing.",
    )
    ap.add_argument(
        "--tasks", type=str, default="classification,detection,segmentation",
        help="Comma-separated tasks to evaluate",
    )
    ap.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for results (default: stage_2/eval_results/<timestamp>)",
    )

    # Dataset paths
    ap.add_argument("--trashnet-root", type=str,
                    default=str(capstone / "datasets/classification/trashnet/dataset-preprocessed"))
    ap.add_argument("--realwaste-root", type=str,
                    default=str(capstone / "datasets/classification/realwaste/realwaste-main/RealWaste"))
    ap.add_argument("--taco-root", type=str,
                    default=str(capstone / "datasets/detection/taco/TACO/data"))
    ap.add_argument("--icra19-root", type=str,
                    default=str(capstone / "datasets/detection/trash_icra19/trash_ICRA19"))
    ap.add_argument("--taco-seg-images", type=str,
                    default=str(capstone / "stage_1/segmentation/data/taco/test/images"))
    ap.add_argument("--taco-seg-masks", type=str,
                    default=str(capstone / "stage_1/segmentation/data/taco/test/masks"))
    ap.add_argument("--dwsd-images", type=str,
                    default=str(capstone / "datasets/segmentation/Dense Waste Segmentation Dataset/DSWD/Test/Image"))
    ap.add_argument("--dwsd-masks", type=str,
                    default=str(capstone / "datasets/segmentation/Dense Waste Segmentation Dataset/DSWD/Test/Annotation"))

    args = ap.parse_args()

    cache_dir = _resolve_path_arg(args.cache_dir) if args.cache_dir else default_cache_dir()
    configure_hf_cache(cache_dir)

    import torch
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
        print("CUDA not available, using CPU")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (
        _resolve_path_arg(args.output_dir)
        if args.output_dir
        else _stage2_root() / "eval_results" / f"unified_{ts}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = [t.strip() for t in args.tasks.split(",")]

    print(f"Loading model: {args.model_id}")
    model, processor = load_florence2(args.model_id, args.device)

    all_results: Dict[str, Any] = {
        "model_id": args.model_id,
        "timestamp": ts,
        "device": args.device,
        "max_images": args.max_images,
    }

    # ── Classification ──
    if "classification" in tasks:
        print("\n" + "=" * 60)
        print("CLASSIFICATION EVALUATION")
        print("=" * 60)

        trashnet_root = _resolve_path_arg(args.trashnet_root)
        if trashnet_root and trashnet_root.exists():
            all_results["classification_trashnet"] = eval_classification(
                model, processor, trashnet_root, args.device,
                max_images=args.max_images, dataset_name="TrashNet (in-domain)",
            )

        realwaste_root = _resolve_path_arg(args.realwaste_root)
        if realwaste_root and realwaste_root.exists():
            all_results["classification_realwaste"] = eval_classification(
                model, processor, realwaste_root, args.device,
                max_images=args.max_images, label_map=REALWASTE_TO_TRASHNET,
                dataset_name="RealWaste (cross-domain)",
            )

    # ── Detection ──
    if "detection" in tasks:
        print("\n" + "=" * 60)
        print("DETECTION EVALUATION")
        print("=" * 60)

        taco_root = _resolve_path_arg(args.taco_root)
        if taco_root and taco_root.exists():
            all_results["detection_taco"] = eval_detection_taco(
                model, processor, taco_root, args.device,
                max_images=args.max_images,
            )

        icra19_root = _resolve_path_arg(args.icra19_root)
        if icra19_root and icra19_root.exists():
            all_results["detection_icra19"] = eval_detection_icra19(
                model, processor, icra19_root, args.device,
                max_images=args.max_images,
            )

    # ── Segmentation ──
    if "segmentation" in tasks:
        print("\n" + "=" * 60)
        print("SEGMENTATION EVALUATION")
        print("=" * 60)

        taco_seg_img = _resolve_path_arg(args.taco_seg_images)
        taco_seg_mask = _resolve_path_arg(args.taco_seg_masks)
        if taco_seg_img and taco_seg_img.exists() and taco_seg_mask and taco_seg_mask.exists():
            all_results["segmentation_taco"] = eval_segmentation(
                model, processor, taco_seg_img, taco_seg_mask, args.device,
                max_images=args.max_images, seg_phrase="waste",
                dataset_name="TACO Seg (in-domain)",
            )

        dwsd_img = _resolve_path_arg(args.dwsd_images)
        dwsd_mask = _resolve_path_arg(args.dwsd_masks)
        if dwsd_img and dwsd_img.exists() and dwsd_mask and dwsd_mask.exists():
            all_results["segmentation_dwsd"] = eval_segmentation(
                model, processor, dwsd_img, dwsd_mask, args.device,
                max_images=args.max_images, seg_phrase="waste",
                dataset_name="DWSD (cross-domain)",
            )

    # ── Save results ──
    results_path = out_dir / "results.json"
    results_path.write_text(json.dumps(all_results, indent=2))
    print(f"\n{'=' * 60}")
    print(f"Results saved to: {results_path}")

    # ── Summary table ──
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Model: {args.model_id}")
    print(f"{'─' * 60}")

    for key, val in all_results.items():
        if not isinstance(val, dict) or "dataset" not in val:
            continue
        ds = val["dataset"]
        if "accuracy" in val:
            print(f"  {ds}: Acc={val['accuracy']:.4f}, F1={val['macro_f1']:.4f}")
        elif "precision" in val and "recall" in val:
            print(f"  {ds}: P={val['precision']:.4f}, R={val['recall']:.4f}, F1={val['f1']:.4f}")
        elif "mean_iou" in val:
            print(f"  {ds}: mIoU={val['mean_iou']:.4f}, PixAcc={val['pixel_accuracy']:.4f}")

    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
