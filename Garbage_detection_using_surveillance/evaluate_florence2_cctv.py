#!/usr/bin/env python3
"""Evaluate Florence-2 (zero-shot or fine-tuned) on the CCTV garbage test set."""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from PIL import Image
from tqdm import tqdm

PROJECT_DIR = Path(__file__).parent
DATA_DIR    = PROJECT_DIR / "data" / "annotations"
IMG_SRC     = Path("/u/student/2024/cs24mtech11024/Capstone/datasets/Garbage Detection using CCTV.coco/train")


def get_device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_gt_boxes(coco, img_id, img_w, img_h):
    boxes = []
    for ann in coco["annotations"]:
        if ann["image_id"] != img_id:
            continue
        x, y, w, h = ann["bbox"]
        boxes.append((x, y, x + w, y + h))
    return boxes


def evaluate(model_id: str, adapter_path: str | None, split: str, conf_thresh: float, vis_dir: Path | None):
    from florence_cctv_utils import (
        load_florence2, run_florence2_od, parse_florence2_od_output,
        is_garbage_label, match_predictions, compute_metrics,
    )

    coco_path = DATA_DIR / f"{split}.coco.json"
    if not coco_path.exists():
        raise FileNotFoundError(f"Run split_cctv_coco.py first: {coco_path}")
    with open(coco_path) as f:
        coco = json.load(f)

    is_finetuned = adapter_path is not None
    tag = "finetuned" if is_finetuned else "zeroshot"
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_DIR / "eval_results" / f"florence2_{tag}_{split}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Loading Florence-2: {model_id}  adapter={adapter_path}  device={device}")
    model, processor = load_florence2(model_id, adapter_path, device)

    img_index = {img["id"]: img for img in coco["images"]}
    img_to_anns: dict[int, list] = {}
    for ann in coco["annotations"]:
        img_to_anns.setdefault(ann["image_id"], []).append(ann)

    total_tp = total_fp = total_fn = 0
    per_image = []

    for img_info in tqdm(coco["images"], desc=f"Florence-2 {tag}"):
        img_id  = img_info["id"]
        img_w   = img_info["width"]
        img_h   = img_info["height"]
        fname   = img_info["file_name"]
        img_path = IMG_SRC / fname

        if not img_path.exists():
            continue

        image = Image.open(img_path).convert("RGB")
        raw   = run_florence2_od(model, processor, image, device, is_finetuned)
        all_boxes = parse_florence2_od_output(raw, img_w, img_h)

        if is_finetuned:
            pred_boxes = [(b[1], b[2], b[3], b[4]) for b in all_boxes]
        else:
            pred_boxes = [(b[1], b[2], b[3], b[4]) for b in all_boxes if is_garbage_label(b[0])]

        gt_anns  = img_to_anns.get(img_id, [])
        gt_boxes = [(float(a["bbox"][0]), float(a["bbox"][1]),
                     float(a["bbox"][0]) + float(a["bbox"][2]),
                     float(a["bbox"][1]) + float(a["bbox"][3]))
                    for a in gt_anns]

        tp, fp, fn = match_predictions(pred_boxes, gt_boxes)
        total_tp += tp; total_fp += fp; total_fn += fn

        per_image.append({"file": fname, "tp": tp, "fp": fp, "fn": fn,
                           "n_pred": len(pred_boxes), "n_gt": len(gt_boxes)})

        if vis_dir and (tp > 0 or fp > 0):
            _save_vis(image, pred_boxes, gt_boxes, vis_dir / Path(fname).stem)

    metrics = compute_metrics(total_tp, total_fp, total_fn)
    metrics["model"]        = model_id
    metrics["adapter"]      = adapter_path
    metrics["split"]        = split
    metrics["evaluated"]    = len(per_image)
    metrics["tag"]          = tag

    print(f"\n{'='*50}")
    print(f"Florence-2 {tag.upper()} — {split}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1        : {metrics['f1']:.4f}")
    print(f"  TP={total_tp}  FP={total_fp}  FN={total_fn}")
    print(f"{'='*50}")

    with open(out_dir / "results.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "per_image_results.json", "w") as f:
        json.dump(per_image, f, indent=2)

    print(f"Results saved to: {out_dir}")
    return metrics


def _save_vis(image, pred_boxes, gt_boxes, out_path: Path):
    try:
        from PIL import ImageDraw
        img = image.copy()
        draw = ImageDraw.Draw(img)
        for b in gt_boxes:
            draw.rectangle(list(b), outline="green", width=2)
        for b in pred_boxes:
            draw.rectangle(list(b), outline="red", width=2)
        img.save(str(out_path) + ".jpg")
    except Exception:
        pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id",  default="microsoft/Florence-2-large")
    p.add_argument("--adapter",   default=None, help="Path to LoRA adapter (fine-tuned only)")
    p.add_argument("--split",     default="test", choices=["train", "val", "test"])
    p.add_argument("--conf",      type=float, default=0.0)
    p.add_argument("--visualize", action="store_true")
    args = p.parse_args()

    vis_dir = None
    if args.visualize:
        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        vis_dir = PROJECT_DIR / "eval_results" / f"vis_{ts}"
        vis_dir.mkdir(parents=True, exist_ok=True)

    evaluate(args.model_id, args.adapter, args.split, args.conf, vis_dir)


if __name__ == "__main__":
    main()
