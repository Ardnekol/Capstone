#!/usr/bin/env python3
"""Florence-2 inference and evaluation utilities for CCTV garbage detection."""

import os
import re
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

LOC_BINS = 1000
IOU_THRESH = 0.5
GARBAGE_KEYWORDS = {
    "garbage", "trash", "waste", "litter", "rubbish", "debris",
    "refuse", "junk", "dirt", "mess", "bag", "plastic", "bottle",
}


def configure_hf_cache():
    cache = Path(os.environ.get("SLURM_TMPDIR", "/tmp")) / f"hf_cache_uid{os.getuid()}"
    cache.mkdir(parents=True, exist_ok=True)
    for var, sub in [("HF_HOME", ""), ("HF_HUB_CACHE", "hub"), ("TRANSFORMERS_CACHE", "hub")]:
        os.environ.setdefault(var, str(cache / sub) if sub else str(cache))


def load_florence2(model_id: str = "microsoft/Florence-2-large", adapter_path: Optional[str] = None, device: str = "cuda"):
    configure_hf_cache()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        attn_implementation="eager",
    ).to(device)

    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        print(f"Loaded LoRA adapter from: {adapter_path}")

    model.eval()
    return model, processor


def parse_florence2_od_output(text: str, img_w: int, img_h: int):
    """Parse Florence-2 OD output into list of (label, x1, y1, x2, y2) in pixel coords."""
    pattern = re.compile(
        r"([a-zA-Z][a-zA-Z0-9 _-]*?)"
        r"(?:<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>)+"
    )
    boxes = []
    for m in re.finditer(
        r"([a-zA-Z][a-zA-Z0-9 _-]*?)<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>",
        text
    ):
        label = m.group(1).strip().lower()
        lx1, ly1, lx2, ly2 = int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))
        x1 = lx1 / (LOC_BINS - 1) * img_w
        y1 = ly1 / (LOC_BINS - 1) * img_h
        x2 = lx2 / (LOC_BINS - 1) * img_w
        y2 = ly2 / (LOC_BINS - 1) * img_h
        boxes.append((label, x1, y1, x2, y2))
    return boxes


def is_garbage_label(label: str) -> bool:
    label = label.lower()
    return any(kw in label for kw in GARBAGE_KEYWORDS)


def run_florence2_od(model, processor, image: Image.Image, device: str = "cuda", is_finetuned: bool = False):
    """Run Florence-2 OD on a single PIL image. Returns raw output text."""
    prompt = "<OD>"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    # cast pixel_values to match model dtype
    dtype = next(model.parameters()).dtype
    inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
        )
    text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    return text


def compute_iou(box_a, box_b):
    """IoU between two [x1,y1,x2,y2] boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_predictions(pred_boxes, gt_boxes, iou_thresh=IOU_THRESH):
    """Greedy matching of pred_boxes to gt_boxes. Returns TP, FP, FN."""
    matched_gt = set()
    tp = 0
    for pb in pred_boxes:
        best_iou = 0.0
        best_j   = -1
        for j, gb in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            iou = compute_iou(pb, gb)
            if iou > best_iou:
                best_iou = iou
                best_j   = j
        if best_iou >= iou_thresh:
            tp += 1
            matched_gt.add(best_j)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return tp, fp, fn


def compute_metrics(total_tp, total_fp, total_fn):
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": total_tp, "fp": total_fp, "fn": total_fn}
