#!/usr/bin/env python3
"""Prepare a Florence-2 fine-tuning JSONL for TACO (COCO-format) object detection.

This creates training examples of the form:
  - prefix prompt: "<OD>" (Florence task token)
  - suffix target: "<label><loc_x1><loc_y1><loc_x2><loc_y2>..." repeated

The `<loc_#>` bins follow Florence's official processor quantization:
  - 1000 bins for width and height
  - floor quantization, clamped to [0, 999]

Output JSONL schema (one line per image):
  {"image_path": "/abs/path.jpg", "prefix": "<OD>", "suffix": "..."}

Note: This script does not download any model weights.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Tuple

from PIL import Image


TASK_OD = "<OD>"


def _repo_root() -> Path:
    # Capstone/stage_2/*.py -> Capstone
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


def _sanitize_label(label: str) -> str:
    # Florence post-processing regex expects [a-zA-Z0-9 ]+ for cat_name.
    # Keep it simple and stable for fine-tuning.
    label = label.strip().lower().replace("_", " ")
    label = re.sub(r"[^a-z0-9 ]+", " ", label)
    label = " ".join(label.split())
    return label


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _quantize_coord_floor(value: float, size: int, bins: int = 1000) -> int:
    # Mirrors Florence2 BoxQuantizer(mode='floor').
    # size_per_bin = size / bins
    # q = floor(value / size_per_bin)
    if size <= 0:
        return 0
    size_per_bin = float(size) / float(bins)
    if size_per_bin <= 0:
        return 0
    q = int(math.floor(float(value) / size_per_bin))
    if q < 0:
        return 0
    if q > bins - 1:
        return bins - 1
    return q


def _quantize_xyxy_to_loc_tokens(
    xyxy: Tuple[float, float, float, float], image_size: Tuple[int, int]
) -> str:
    w, h = image_size
    x1, y1, x2, y2 = xyxy
    x1 = _clamp(x1, 0.0, float(w - 1))
    x2 = _clamp(x2, 0.0, float(w - 1))
    y1 = _clamp(y1, 0.0, float(h - 1))
    y2 = _clamp(y2, 0.0, float(h - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    qx1 = _quantize_coord_floor(x1, w)
    qy1 = _quantize_coord_floor(y1, h)
    qx2 = _quantize_coord_floor(x2, w)
    qy2 = _quantize_coord_floor(y2, h)
    return f"<loc_{qx1}><loc_{qy1}><loc_{qx2}><loc_{qy2}>"


def _coco_bbox_xywh_to_xyxy(b: List[float]) -> Optional[Tuple[float, float, float, float]]:
    if not isinstance(b, list) or len(b) != 4:
        return None
    try:
        x, y, w, h = map(float, b)
    except Exception:
        return None
    if not all(math.isfinite(v) for v in (x, y, w, h)):
        return None
    if w <= 1e-6 or h <= 1e-6:
        return None
    return (x, y, x + w, y + h)


def _group_suffix(
    items: Iterable[Tuple[str, Tuple[float, float, float, float], float]],
    image_size: Tuple[int, int],
) -> str:
    # Group by label to allow multiple boxes for same label.
    # Sort by total area per label to keep high-signal early.
    by_label: DefaultDict[str, List[Tuple[Tuple[float, float, float, float], float]]] = defaultdict(list)
    for label, xyxy, area in items:
        by_label[label].append((xyxy, area))

    labels_sorted = sorted(by_label.keys(), key=lambda k: sum(a for _, a in by_label[k]), reverse=True)
    parts: List[str] = []
    for label in labels_sorted:
        boxes = sorted(by_label[label], key=lambda t: t[1], reverse=True)
        tok = label
        for xyxy, _ in boxes:
            tok += _quantize_xyxy_to_loc_tokens(xyxy, image_size)
        parts.append(tok)
    return "".join(parts)


def main() -> None:
    capstone = _repo_root()
    default_taco = capstone / "datasets/detection/taco/TACO/data"

    ap = argparse.ArgumentParser(description="Prepare TACO COCO annotations into Florence-2 OD JSONL")
    ap.add_argument("--taco-root", type=str, default=str(default_taco), help="TACO data root containing annotations.json")
    ap.add_argument("--annotations", type=str, default=None, help="Path to COCO annotations.json (defaults to <taco-root>/annotations.json)")
    ap.add_argument(
        "--out",
        type=str,
        default=str(_stage2_root() / "finetune_data" / "taco_od_train.jsonl"),
        help="Output JSONL path (created under stage_2 by default)",
    )
    ap.add_argument("--max-images", type=int, default=0, help="Optional cap on number of images (0 = no cap)")
    ap.add_argument("--min-box-area", type=float, default=1.0, help="Skip boxes with area < min-box-area pixels")
    ap.add_argument(
        "--categories",
        type=str,
        default="",
        help="Optional comma-separated category names to include (after sanitization); empty = all",
    )

    args = ap.parse_args()

    taco_root = _resolve_path_arg(args.taco_root)
    if taco_root is None or not taco_root.exists():
        raise SystemExit(f"TACO root not found: {taco_root}")

    ann_path = _resolve_path_arg(args.annotations) if args.annotations else (taco_root / "annotations.json").resolve()
    if ann_path is None or not ann_path.exists():
        raise SystemExit(f"annotations.json not found: {ann_path}")

    out_path = _resolve_path_arg(args.out)
    if out_path is None:
        raise SystemExit("Invalid --out")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    include_cats = set(_sanitize_label(x) for x in args.categories.split(",") if x.strip())

    data = json.loads(ann_path.read_text())
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])

    cat_id_to_name: Dict[int, str] = {}
    for c in categories:
        try:
            cid = int(c.get("id"))
        except Exception:
            continue
        nm = _sanitize_label(str(c.get("name", "")))
        if not nm:
            continue
        cat_id_to_name[cid] = nm

    img_id_to_info: Dict[int, Dict] = {}
    for im in images:
        try:
            iid = int(im.get("id"))
        except Exception:
            continue
        img_id_to_info[iid] = im

    ann_by_img: DefaultDict[int, List[Dict]] = defaultdict(list)
    for ann in annotations:
        try:
            iid = int(ann.get("image_id"))
        except Exception:
            continue
        ann_by_img[iid].append(ann)

    lines: List[str] = []
    kept = 0
    for idx, (img_id, im) in enumerate(sorted(img_id_to_info.items(), key=lambda kv: kv[0])):
        if args.max_images and kept >= int(args.max_images):
            break

        file_name = str(im.get("file_name", "")).lstrip("/")
        if not file_name:
            continue
        img_path = (taco_root / file_name).resolve()
        if not img_path.exists():
            # Some TACO mirrors may store images under taco_root/batch_*/
            # If file_name already has batch_ prefix, this should work.
            continue

        # Use actual image size for quantization when possible.
        try:
            with Image.open(img_path) as img:
                w, h = img.size
        except Exception:
            w = int(im.get("width", 0) or 0)
            h = int(im.get("height", 0) or 0)
        if w <= 0 or h <= 0:
            continue

        items: List[Tuple[str, Tuple[float, float, float, float], float]] = []
        for ann in ann_by_img.get(img_id, []):
            if int(ann.get("iscrowd", 0) or 0) == 1:
                continue
            bbox = _coco_bbox_xywh_to_xyxy(ann.get("bbox"))
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            # Basic clamp
            x1 = _clamp(x1, 0.0, float(w - 1))
            y1 = _clamp(y1, 0.0, float(h - 1))
            x2 = _clamp(x2, 0.0, float(w - 1))
            y2 = _clamp(y2, 0.0, float(h - 1))
            if x2 <= x1 + 1e-3 or y2 <= y1 + 1e-3:
                continue
            area = float((x2 - x1) * (y2 - y1))
            if area < float(args.min_box_area):
                continue

            try:
                cid = int(ann.get("category_id"))
            except Exception:
                continue
            label = cat_id_to_name.get(cid, "")
            if not label:
                continue
            if include_cats and label not in include_cats:
                continue
            items.append((label, (x1, y1, x2, y2), area))

        if not items:
            continue

        suffix = _group_suffix(items, (w, h))
        if not suffix:
            continue

        rec = {"image_path": str(img_path), "prefix": TASK_OD, "suffix": suffix}
        lines.append(json.dumps(rec))
        kept += 1

    out_path.write_text("\n".join(lines) + ("\n" if lines else ""))
    print(f"Wrote {kept} training examples -> {out_path}")


if __name__ == "__main__":
    main()
