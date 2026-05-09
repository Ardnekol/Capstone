#!/usr/bin/env python3
"""Prepare a Florence-2 fine-tuning JSONL for TACO segmentation.

This creates training examples of the form:
  - prefix prompt: "<REFERRING_EXPRESSION_SEGMENTATION>"
  - suffix target: "<loc_x1><loc_y1><loc_x2><loc_y2>...<loc_xn><loc_yn>"

Each annotation's polygon is quantized to Florence-2 location tokens (1000 bins).
One JSONL record per (image, annotation) pair.

Output JSONL schema (one line per image-annotation pair):
  {"image_path": "/abs/path.jpg", "prefix": "<REFERRING_EXPRESSION_SEGMENTATION>label", "suffix": "<loc_...>..."}

Note: The referring expression segmentation task in Florence-2 takes the
text description as part of the prefix, and the polygon coordinates as suffix.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Tuple

from PIL import Image


TASK_SEG = "<REFERRING_EXPRESSION_SEGMENTATION>"


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


def _sanitize_label(label: str) -> str:
    label = label.strip().lower().replace("_", " ")
    label = re.sub(r"[^a-z0-9 ]+", " ", label)
    label = " ".join(label.split())
    return label


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _quantize_coord_floor(value: float, size: int, bins: int = 1000) -> int:
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


def _polygon_to_loc_tokens(
    polygon: List[float], image_size: Tuple[int, int]
) -> Optional[str]:
    """Convert a flat COCO polygon [x1,y1,x2,y2,...] to Florence loc tokens."""
    w, h = image_size
    if len(polygon) < 6:  # Need at least 3 points
        return None

    tokens: List[str] = []
    for i in range(0, len(polygon) - 1, 2):
        x = _clamp(float(polygon[i]), 0.0, float(w - 1))
        y = _clamp(float(polygon[i + 1]), 0.0, float(h - 1))
        qx = _quantize_coord_floor(x, w)
        qy = _quantize_coord_floor(y, h)
        tokens.append(f"<loc_{qx}><loc_{qy}>")

    if not tokens:
        return None
    return "".join(tokens)


def main() -> None:
    capstone = _repo_root()
    default_taco = capstone / "datasets/detection/taco/TACO/data"

    ap = argparse.ArgumentParser(
        description="Prepare TACO COCO polygon annotations into Florence-2 segmentation JSONL"
    )
    ap.add_argument(
        "--taco-root", type=str, default=str(default_taco),
        help="TACO data root containing annotations.json",
    )
    ap.add_argument(
        "--annotations", type=str, default=None,
        help="Path to COCO annotations.json (defaults to <taco-root>/annotations.json)",
    )
    ap.add_argument(
        "--out", type=str,
        default=str(_stage2_root() / "finetune_data" / "taco_seg_train.jsonl"),
        help="Output JSONL path",
    )
    ap.add_argument(
        "--max-images", type=int, default=0,
        help="Optional cap on number of images (0 = no cap)",
    )
    ap.add_argument(
        "--min-polygon-points", type=int, default=3,
        help="Skip polygons with fewer than this many points",
    )
    ap.add_argument(
        "--min-box-area", type=float, default=1.0,
        help="Skip annotations with bbox area < this (pixels)",
    )
    ap.add_argument(
        "--categories", type=str, default="",
        help="Comma-separated category names to include (after sanitization); empty = all",
    )

    args = ap.parse_args()

    taco_root = _resolve_path_arg(args.taco_root)
    if taco_root is None or not taco_root.exists():
        raise SystemExit(f"TACO root not found: {taco_root}")

    ann_path = (
        _resolve_path_arg(args.annotations)
        if args.annotations
        else (taco_root / "annotations.json").resolve()
    )
    if ann_path is None or not ann_path.exists():
        raise SystemExit(f"annotations.json not found: {ann_path}")

    out_path = _resolve_path_arg(args.out)
    if out_path is None:
        raise SystemExit("Invalid --out")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    include_cats = set(
        _sanitize_label(x) for x in args.categories.split(",") if x.strip()
    )

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
    images_kept = 0
    records_total = 0

    for img_id, im in sorted(img_id_to_info.items(), key=lambda kv: kv[0]):
        if args.max_images and images_kept >= int(args.max_images):
            break

        file_name = str(im.get("file_name", "")).lstrip("/")
        if not file_name:
            continue
        img_path = (taco_root / file_name).resolve()
        if not img_path.exists():
            continue

        try:
            with Image.open(img_path) as img:
                w, h = img.size
        except Exception:
            w = int(im.get("width", 0) or 0)
            h = int(im.get("height", 0) or 0)
        if w <= 0 or h <= 0:
            continue

        image_has_records = False

        # Group annotations by label for this image
        label_polygons: DefaultDict[str, List[str]] = defaultdict(list)

        for ann in ann_by_img.get(img_id, []):
            if int(ann.get("iscrowd", 0) or 0) == 1:
                continue

            # Check bbox area
            bbox = ann.get("bbox")
            if bbox and len(bbox) == 4:
                bx, by, bw, bh = map(float, bbox)
                if bw * bh < float(args.min_box_area):
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

            seg = ann.get("segmentation")
            if not seg or not isinstance(seg, list):
                continue

            # COCO segmentation can be a list of polygons (multiple parts)
            for polygon in seg:
                if not isinstance(polygon, list):
                    continue
                if len(polygon) < args.min_polygon_points * 2:
                    continue

                loc_str = _polygon_to_loc_tokens(polygon, (w, h))
                if loc_str:
                    label_polygons[label].append(loc_str)

        # Create one record per label per image (group all polygons of same label)
        for label, poly_tokens_list in sorted(label_polygons.items()):
            # Florence-2 RES format: prefix includes the phrase,
            # suffix is the polygon tokens
            suffix = "".join(poly_tokens_list)
            rec = {
                "image_path": str(img_path),
                "prefix": f"{TASK_SEG}{label}",
                "suffix": suffix,
            }
            lines.append(json.dumps(rec))
            records_total += 1
            image_has_records = True

        if image_has_records:
            images_kept += 1

    out_path.write_text("\n".join(lines) + ("\n" if lines else ""))
    print(f"Wrote {records_total} segmentation records from {images_kept} images -> {out_path}")


if __name__ == "__main__":
    main()
