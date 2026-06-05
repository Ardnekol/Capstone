#!/usr/bin/env python3
"""Florence-2 OD JSONL prep v2 — adds hard-negative records and a 'kind' field.

Improvements over v1:
  - Emits negative records: random background crops with NO garbage boxes,
    written with empty `suffix` and `kind: "neg"`. The v2 trainer teaches the
    model to emit no boxes on these, directly attacking the FP=705 problem
    seen in stage-1 (model hallucinates "garbage" everywhere).
  - Tags each positive record with `kind: "pos"` so the trainer can balance
    pos/neg sampling.
  - Skips degenerate boxes (w or h <= 1 px after rounding).
"""

import argparse
import json
import random
from pathlib import Path

from PIL import Image

PROJECT_DIR = Path(__file__).parent
DATA_DIR    = PROJECT_DIR / "data" / "annotations"
OUT_DIR     = PROJECT_DIR / "data" / "finetune_data"
IMG_SRC     = Path("/u/student/2024/cs24mtech11024/Capstone/datasets/Garbage Detection using CCTV.coco/train")

LOC_BINS = 1000
NEG_CROP_SIZE = 384       # square crop side in pixels (before Florence resize)
MAX_NEG_TRIES = 20        # rejection-sampling attempts per negative
NEG_IOU_MAX   = 0.02      # negative crop must barely touch any GT box


def normalize_box(box, img_w, img_h):
    x, y, w, h = [float(v) for v in box]
    x1 = round(x / img_w * (LOC_BINS - 1))
    y1 = round(y / img_h * (LOC_BINS - 1))
    x2 = round((x + w) / img_w * (LOC_BINS - 1))
    y2 = round((y + h) / img_h * (LOC_BINS - 1))
    return (
        max(0, min(LOC_BINS - 1, x1)),
        max(0, min(LOC_BINS - 1, y1)),
        max(0, min(LOC_BINS - 1, x2)),
        max(0, min(LOC_BINS - 1, y2)),
    )


def _box_iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    return inter / area_a  # fraction of crop covered by GT


def sample_negative_crop(img_w, img_h, gt_xyxy, rng):
    """Return (x1,y1,x2,y2) for a square crop that barely overlaps any GT, or None."""
    side = min(NEG_CROP_SIZE, img_w, img_h)
    if side < 64:
        return None
    for _ in range(MAX_NEG_TRIES):
        x1 = rng.randint(0, img_w - side)
        y1 = rng.randint(0, img_h - side)
        x2, y2 = x1 + side, y1 + side
        cand = (x1, y1, x2, y2)
        if all(_box_iou_xyxy(cand, g) <= NEG_IOU_MAX for g in gt_xyxy):
            return cand
    return None


def build_jsonl(split: str, neg_ratio: float, save_neg_crops: bool, seed: int):
    coco_path = DATA_DIR / f"{split}.coco.json"
    if not coco_path.exists():
        raise FileNotFoundError(f"Run split_cctv_coco.py first: {coco_path}")

    with open(coco_path) as f:
        coco = json.load(f)

    img_to_anns: dict[int, list] = {}
    for ann in coco["annotations"]:
        img_to_anns.setdefault(ann["image_id"], []).append(ann)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    neg_dir = OUT_DIR / f"neg_crops_{split}"
    if save_neg_crops:
        neg_dir.mkdir(parents=True, exist_ok=True)

    out_path = OUT_DIR / f"cctv_od_{split}_v2.jsonl"

    rng = random.Random(seed)
    n_pos = n_neg = n_skip = 0

    with open(out_path, "w") as f:
        for img_info in coco["images"]:
            img_id = img_info["id"]
            anns   = img_to_anns.get(img_id, [])
            if not anns:
                n_skip += 1
                continue

            img_path = IMG_SRC / img_info["file_name"]
            if not img_path.exists():
                n_skip += 1
                continue

            img_w, img_h = img_info["width"], img_info["height"]

            # Positive record
            suffix_parts = []
            gt_xyxy_px = []
            for ann in anns:
                x1, y1, x2, y2 = normalize_box(ann["bbox"], img_w, img_h)
                if (x2 - x1) <= 1 or (y2 - y1) <= 1:
                    continue
                suffix_parts.append(f"garbage<loc_{x1}><loc_{y1}><loc_{x2}><loc_{y2}>")
                bx, by, bw, bh = ann["bbox"]
                gt_xyxy_px.append((bx, by, bx + bw, by + bh))

            if not suffix_parts:
                n_skip += 1
                continue

            f.write(json.dumps({
                "image_path": str(img_path),
                "prefix": "<OD>",
                "suffix": "".join(suffix_parts),
                "kind": "pos",
            }) + "\n")
            n_pos += 1

            # Negative crop(s) — only on training split, rate-limited by neg_ratio
            if neg_ratio <= 0:
                continue
            if rng.random() > neg_ratio:
                continue
            crop_box = sample_negative_crop(img_w, img_h, gt_xyxy_px, rng)
            if crop_box is None:
                continue

            if save_neg_crops:
                try:
                    im = Image.open(img_path).convert("RGB")
                    neg = im.crop(crop_box)
                    neg_name = f"{Path(img_info['file_name']).stem}_neg.jpg"
                    neg_path = neg_dir / neg_name
                    neg.save(neg_path, quality=92)
                    neg_image_path = str(neg_path)
                except Exception:
                    continue
                f.write(json.dumps({
                    "image_path": neg_image_path,
                    "prefix": "<OD>",
                    "suffix": "",
                    "kind": "neg",
                }) + "\n")
            else:
                # In-memory negative: store source path + crop coords; trainer must crop on the fly
                f.write(json.dumps({
                    "image_path": str(img_path),
                    "prefix": "<OD>",
                    "suffix": "",
                    "kind": "neg",
                    "crop_xyxy": list(crop_box),
                }) + "\n")
            n_neg += 1

    summary = {"split": split, "pos": n_pos, "neg": n_neg, "skipped": n_skip,
               "neg_ratio_arg": neg_ratio}
    with open(OUT_DIR / f"cctv_od_{split}_v2.summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"{split}: pos={n_pos}  neg={n_neg}  skipped={n_skip}  → {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    p.add_argument("--neg-ratio", type=float, default=0.5,
                   help="Per-image probability of also emitting a negative crop (train only)")
    p.add_argument("--save-neg-crops", action="store_true",
                   help="Pre-save negative crops to disk instead of cropping on-the-fly")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    for s in splits:
        # Only sample negatives for train split — eval splits stay clean
        ratio = args.neg_ratio if s == "train" else 0.0
        build_jsonl(s, ratio, args.save_neg_crops, args.seed)


if __name__ == "__main__":
    main()
