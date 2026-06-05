#!/usr/bin/env python3
"""Florence-2 evaluation with size-conditional tile inference for CCTV garbage.

Why tiling: Florence-2 processes images at a fixed ~768 px. CCTV images range
from 275x183 to 3000x4000 — large frames lose small objects after resize.
For images with min(W, H) >= --tile-min-side, this script splits into an
overlapping 2x2 grid, runs OD per tile, maps predictions back to full-image
coords, and merges with class-agnostic NMS.

Small images are evaluated whole (tiling would over-zoom and produce garbage).
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from PIL import Image
from tqdm import tqdm

PROJECT_DIR = Path(__file__).parent
DATA_DIR    = PROJECT_DIR / "data" / "annotations"
IMG_SRC     = Path("/u/student/2024/cs24mtech11024/Capstone/datasets/Garbage Detection using CCTV.coco/train")


def _tile_boxes(w: int, h: int, overlap: float) -> List[Tuple[int, int, int, int]]:
    """2x2 grid with `overlap` fraction of tile size as overlap between tiles."""
    tw = int(w * (0.5 + overlap / 2))
    th = int(h * (0.5 + overlap / 2))
    xs = [0, w - tw]
    ys = [0, h - th]
    return [(x, y, x + tw, y + th) for y in ys for x in xs]


def _nms(boxes: List[Tuple[float, float, float, float]], iou_thresh: float
         ) -> List[Tuple[float, float, float, float]]:
    """Class-agnostic NMS (greedy, no scores — score-less Florence-2 OD output).
    Sorts by box area as a tiebreak proxy (larger boxes kept first)."""
    if not boxes:
        return []
    order = sorted(range(len(boxes)),
                   key=lambda i: -((boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])))
    kept: List[int] = []
    suppressed = set()
    for i in order:
        if i in suppressed:
            continue
        kept.append(i)
        for j in order:
            if j == i or j in suppressed:
                continue
            if _iou(boxes[i], boxes[j]) >= iou_thresh:
                suppressed.add(j)
    return [boxes[i] for i in kept]


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def evaluate(model_id: str, adapter_path: str | None, split: str,
             tile_min_side: int, tile_overlap: float, nms_iou: float,
             vis_dir: Path | None):
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
    tag = "finetuned_tiled" if is_finetuned else "zeroshot_tiled"
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_DIR / "eval_results" / f"florence2_{tag}_{split}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Florence-2: {model_id}  adapter={adapter_path}  device={device}")
    model, processor = load_florence2(model_id, adapter_path, device)

    img_to_anns: dict[int, list] = {}
    for ann in coco["annotations"]:
        img_to_anns.setdefault(ann["image_id"], []).append(ann)

    total_tp = total_fp = total_fn = 0
    per_image = []
    n_tiled = n_whole = 0

    for img_info in tqdm(coco["images"], desc=f"Florence-2 {tag}"):
        img_id = img_info["id"]
        img_w  = img_info["width"]
        img_h  = img_info["height"]
        fname  = img_info["file_name"]
        img_path = IMG_SRC / fname
        if not img_path.exists():
            continue

        image = Image.open(img_path).convert("RGB")
        use_tiles = min(img_w, img_h) >= tile_min_side

        if use_tiles:
            n_tiled += 1
            tile_boxes = _tile_boxes(img_w, img_h, tile_overlap)
            all_preds: List[Tuple[float, float, float, float]] = []
            for (tx1, ty1, tx2, ty2) in tile_boxes:
                tile = image.crop((tx1, ty1, tx2, ty2))
                tw, th = tile.size
                raw = run_florence2_od(model, processor, tile, device, is_finetuned)
                tile_preds = parse_florence2_od_output(raw, tw, th)
                if not is_finetuned:
                    tile_preds = [b for b in tile_preds if is_garbage_label(b[0])]
                for (_label, lx1, ly1, lx2, ly2) in tile_preds:
                    all_preds.append((lx1 + tx1, ly1 + ty1, lx2 + tx1, ly2 + ty1))
            pred_boxes = _nms(all_preds, nms_iou)
        else:
            n_whole += 1
            raw = run_florence2_od(model, processor, image, device, is_finetuned)
            tile_preds = parse_florence2_od_output(raw, img_w, img_h)
            if not is_finetuned:
                tile_preds = [b for b in tile_preds if is_garbage_label(b[0])]
            pred_boxes = [(b[1], b[2], b[3], b[4]) for b in tile_preds]

        gt_anns = img_to_anns.get(img_id, [])
        gt_boxes = [(float(a["bbox"][0]), float(a["bbox"][1]),
                     float(a["bbox"][0]) + float(a["bbox"][2]),
                     float(a["bbox"][1]) + float(a["bbox"][3]))
                    for a in gt_anns]

        tp, fp, fn = match_predictions(pred_boxes, gt_boxes)
        total_tp += tp; total_fp += fp; total_fn += fn
        per_image.append({"file": fname, "tp": tp, "fp": fp, "fn": fn,
                          "n_pred": len(pred_boxes), "n_gt": len(gt_boxes),
                          "tiled": use_tiles})

        if vis_dir and (tp > 0 or fp > 0):
            _save_vis(image, pred_boxes, gt_boxes, vis_dir / Path(fname).stem)

    metrics = compute_metrics(total_tp, total_fp, total_fn)
    metrics.update({
        "model": model_id, "adapter": adapter_path, "split": split, "tag": tag,
        "evaluated": len(per_image),
        "tiled_images": n_tiled, "whole_images": n_whole,
        "tile_min_side": tile_min_side, "tile_overlap": tile_overlap,
        "nms_iou": nms_iou,
    })

    print(f"\n{'='*50}")
    print(f"Florence-2 {tag.upper()} — {split}")
    print(f"  Tiled {n_tiled} / whole {n_whole} (tile_min_side={tile_min_side})")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1        : {metrics['f1']:.4f}")
    print(f"  TP={total_tp}  FP={total_fp}  FN={total_fn}")
    print(f"{'='*50}")

    (out_dir / "results.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "per_image_results.json").write_text(json.dumps(per_image, indent=2))
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
    p.add_argument("--model-id",      default="microsoft/Florence-2-large")
    p.add_argument("--adapter",       default=None, help="Path to LoRA adapter (fine-tuned only)")
    p.add_argument("--split",         default="test", choices=["train", "val", "test"])
    p.add_argument("--tile-min-side", type=int, default=800,
                   help="Only tile images with min(W,H) >= this; smaller images run whole.")
    p.add_argument("--tile-overlap",  type=float, default=0.2,
                   help="Fraction overlap between adjacent tiles in a 2x2 grid.")
    p.add_argument("--nms-iou",       type=float, default=0.5,
                   help="IoU threshold for merging tile predictions.")
    p.add_argument("--visualize",     action="store_true")
    args = p.parse_args()

    vis_dir = None
    if args.visualize:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        vis_dir = PROJECT_DIR / "eval_results" / f"vis_tiled_{ts}"
        vis_dir.mkdir(parents=True, exist_ok=True)

    evaluate(args.model_id, args.adapter, args.split,
             args.tile_min_side, args.tile_overlap, args.nms_iou, vis_dir)


if __name__ == "__main__":
    main()
