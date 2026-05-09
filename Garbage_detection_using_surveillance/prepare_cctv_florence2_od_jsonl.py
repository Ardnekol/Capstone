#!/usr/bin/env python3
"""Prepare Florence-2 OD training JSONL from CCTV COCO split annotations."""

import argparse
import json
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
DATA_DIR    = PROJECT_DIR / "data" / "annotations"
OUT_DIR     = PROJECT_DIR / "data" / "finetune_data"
IMG_SRC     = Path("/u/student/2024/cs24mtech11024/Capstone/datasets/Garbage Detection using CCTV.coco/train")

LOC_BINS = 1000  # Florence-2 uses 0..999 loc tokens


def normalize_box(box, img_w, img_h):
    """COCO [x,y,w,h] → Florence-2 [x1,y1,x2,y2] in [0,999]."""
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


def build_jsonl(split: str):
    coco_path = DATA_DIR / f"{split}.coco.json"
    if not coco_path.exists():
        raise FileNotFoundError(f"Run split_cctv_coco.py first: {coco_path}")

    with open(coco_path) as f:
        coco = json.load(f)

    img_to_anns: dict[int, list] = {}
    for ann in coco["annotations"]:
        img_to_anns.setdefault(ann["image_id"], []).append(ann)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"cctv_od_{split}.jsonl"

    written = skipped = 0
    with open(out_path, "w") as f:
        for img_info in coco["images"]:
            img_id = img_info["id"]
            anns   = img_to_anns.get(img_id, [])
            if not anns:
                skipped += 1
                continue

            img_path = IMG_SRC / img_info["file_name"]
            if not img_path.exists():
                skipped += 1
                continue

            img_w = img_info["width"]
            img_h = img_info["height"]

            suffix_parts = []
            for ann in anns:
                x1, y1, x2, y2 = normalize_box(ann["bbox"], img_w, img_h)
                suffix_parts.append(
                    f"garbage<loc_{x1}><loc_{y1}><loc_{x2}><loc_{y2}>"
                )

            record = {
                "image_path": str(img_path),
                "prefix": "<OD>",
                "suffix": "".join(suffix_parts),
            }
            f.write(json.dumps(record) + "\n")
            written += 1

    summary = {"split": split, "written": written, "skipped": skipped}
    with open(OUT_DIR / f"cctv_od_{split}.summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"{split}: {written} records written, {skipped} skipped → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    args = parser.parse_args()

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    for s in splits:
        build_jsonl(s)


if __name__ == "__main__":
    main()
