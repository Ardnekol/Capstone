#!/usr/bin/env python3
"""Build a shared-image tri-task proxy dataset from TACO.

The raw TACO dataset already provides object detection and polygon segmentation.
This script adds a weak image-level classification label by mapping each TACO
category to one of the six waste classes used in Stage 1:

  cardboard, glass, metal, paper, plastic, trash

Each output record contains all three tasks for the same image:
  - classification: one dominant waste class per image
  - detection: all mapped bounding boxes
  - segmentation: all mapped polygons converted to Florence-2 location tokens

This is a proxy dataset. It is not a native multi-label benchmark, but it does
give one image with all three outputs attached.
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


WASTE_CLASSES = ("cardboard", "glass", "metal", "paper", "plastic", "trash")


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
    return " ".join(label.split())


def _map_to_waste_class(raw_label: str) -> str:
    label = _sanitize_label(raw_label)

    if not label:
        return "trash"

    if "glass" in label:
        return "glass"

    if any(
        key in label
        for key in (
            "aluminium",
            "aerosol",
            "battery",
            "can",
            "metal",
            "pop tab",
            "scrap metal",
        )
    ):
        return "metal"

    if any(
        key in label
        for key in (
            "cardboard",
            "carton",
            "corrugated",
            "egg carton",
            "drink carton",
            "meal carton",
            "other carton",
            "pizza box",
        )
    ):
        return "cardboard"

    if any(
        key in label
        for key in (
            "paper",
            "tissue",
            "magazine",
            "toilet tube",
            "paper cup",
            "paper bag",
            "paper straw",
        )
    ):
        return "paper"

    if any(
        key in label
        for key in (
            "plastic",
            "polypropylene",
            "foam",
            "styrofoam",
            "wrapper",
            "film",
            "bag",
            "cup",
            "container",
            "straw",
            "tube",
            "gloves",
            "utensils",
            "lid",
            "cap",
            "bottle",
        )
    ):
        return "plastic"

    if any(
        key in label
        for key in (
            "food waste",
            "unlabeled litter",
            "cigarette",
            "garbage bag",
            "shoe",
            "rope & strings",
        )
    ):
        return "trash"

    return "trash"


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


def _polygon_to_loc_tokens(polygon: List[float], image_size: Tuple[int, int]) -> Optional[str]:
    w, h = image_size
    if len(polygon) < 6:
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


def _annotation_area(ann: Dict) -> float:
    bbox = ann.get("bbox")
    if isinstance(bbox, list) and len(bbox) == 4:
        try:
            _, _, bw, bh = map(float, bbox)
            return max(0.0, bw * bh)
        except Exception:
            return 0.0
    return 0.0


def main() -> None:
    capstone = _repo_root()
    default_taco = capstone / "datasets/detection/taco/TACO/data"

    ap = argparse.ArgumentParser(description="Prepare a shared-image tri-task proxy dataset from TACO")
    ap.add_argument("--taco-root", type=str, default=str(default_taco), help="TACO data root containing annotations.json")
    ap.add_argument(
        "--annotations",
        type=str,
        default=None,
        help="Path to COCO annotations.json (defaults to <taco-root>/annotations.json)",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=str(_stage2_root() / "finetune_data" / "taco_tritask_proxy.jsonl"),
        help="Output JSONL path with one shared-image record per image",
    )
    ap.add_argument(
        "--stats-out",
        type=str,
        default=str(_stage2_root() / "finetune_data" / "taco_tritask_proxy_stats.json"),
        help="Output stats JSON path",
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

    stats_out = _resolve_path_arg(args.stats_out)
    if stats_out is None:
        raise SystemExit("Invalid --stats-out")
    stats_out.parent.mkdir(parents=True, exist_ok=True)

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
        if nm:
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

    records: List[str] = []
    class_counts: DefaultDict[str, int] = defaultdict(int)
    images_kept = 0
    anns_kept = 0
    seg_parts_kept = 0

    for img_id, im in sorted(img_id_to_info.items(), key=lambda kv: kv[0]):
        file_name = str(im.get("file_name", "")).lstrip("/")
        if not file_name:
            continue
        img_path = (taco_root / file_name).resolve()
        if not img_path.exists():
            continue

        try:
            with Image.open(img_path) as img:
                width, height = img.size
        except Exception:
            width = int(im.get("width", 0) or 0)
            height = int(im.get("height", 0) or 0)
        if width <= 0 or height <= 0:
            continue

        detections: List[Dict] = []
        segmentations: List[Dict] = []
        class_area: DefaultDict[str, float] = defaultdict(float)
        class_sources: DefaultDict[str, List[str]] = defaultdict(list)

        for ann in ann_by_img.get(img_id, []):
            if int(ann.get("iscrowd", 0) or 0) == 1:
                continue

            try:
                cid = int(ann.get("category_id"))
            except Exception:
                continue

            raw_label = cat_id_to_name.get(cid, "")
            mapped_label = _map_to_waste_class(raw_label)

            bbox = ann.get("bbox")
            if isinstance(bbox, list) and len(bbox) == 4:
                try:
                    x, y, bw, bh = map(float, bbox)
                except Exception:
                    x = y = bw = bh = 0.0
                area = _annotation_area(ann)
                detections.append(
                    {
                        "label": mapped_label,
                        "raw_label": raw_label,
                        "bbox": [x, y, bw, bh],
                        "area": area,
                    }
                )
                class_area[mapped_label] += area
                if raw_label:
                    class_sources[mapped_label].append(raw_label)
                anns_kept += 1

            seg = ann.get("segmentation")
            if not seg or not isinstance(seg, list):
                continue

            for polygon in seg:
                if not isinstance(polygon, list):
                    continue
                loc_tokens = _polygon_to_loc_tokens(polygon, (width, height))
                if not loc_tokens:
                    continue
                segmentations.append(
                    {
                        "label": mapped_label,
                        "raw_label": raw_label,
                        "polygon": polygon,
                        "loc_tokens": loc_tokens,
                    }
                )
                seg_parts_kept += 1

        if not detections and not segmentations:
            continue

        if class_area:
            classification_label = max(class_area.items(), key=lambda kv: (kv[1], kv[0]))[0]
        else:
            classification_label = "trash"

        class_counts[classification_label] += 1
        images_kept += 1

        record = {
            "image_path": str(img_path),
            "image_size": {"width": width, "height": height},
            "source_dataset": "TACO",
            "weak_classification": True,
            "classification": {
                "label": classification_label,
                "candidate_labels": sorted(set(class_sources.get(classification_label, []))),
            },
            "detection": detections,
            "segmentation": segmentations,
        }
        records.append(json.dumps(record, ensure_ascii=False))

    out_path.write_text("\n".join(records) + ("\n" if records else ""))
    stats = {
        "images_total": len(img_id_to_info),
        "images_kept": images_kept,
        "annotations_kept": anns_kept,
        "segmentation_parts_kept": seg_parts_kept,
        "class_counts": {k: class_counts.get(k, 0) for k in WASTE_CLASSES},
        "output": str(out_path),
    }
    stats_out.write_text(json.dumps(stats, indent=2) + "\n")

    print(f"Wrote {images_kept} shared-image tri-task records from {ann_path} -> {out_path}")
    print(f"Stats saved to {stats_out}")
    print(f"Classification distribution: {stats['class_counts']}")


if __name__ == "__main__":
    main()