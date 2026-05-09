#!/usr/bin/env python3
"""Convert COCO split annotations to YOLO format and copy images."""

import json
import shutil
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
DATA_DIR    = PROJECT_DIR / "data" / "annotations"
IMG_SRC     = Path("/u/student/2024/cs24mtech11024/Capstone/datasets/Garbage Detection using CCTV.coco/train")
YOLO_DIR    = PROJECT_DIR / "yolo_data"


def coco_bbox_to_yolo(bbox, img_w, img_h):
    x, y, w, h = [float(v) for v in bbox]
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    return cx, cy, nw, nh


def convert_split(split_name: str):
    coco_path = DATA_DIR / f"{split_name}.coco.json"
    if not coco_path.exists():
        print(f"  Skipping {split_name}: {coco_path} not found (run split_cctv_coco.py first)")
        return

    with open(coco_path) as f:
        coco = json.load(f)

    img_dir   = YOLO_DIR / "images" / split_name
    label_dir = YOLO_DIR / "labels" / split_name
    img_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    # Build annotation index
    img_to_anns: dict[int, list] = {}
    for ann in coco["annotations"]:
        img_to_anns.setdefault(ann["image_id"], []).append(ann)

    cat_id_to_yolo = {c["id"]: 0 for c in coco["categories"]}  # single class → 0

    copied = 0
    for img_info in coco["images"]:
        fname  = img_info["file_name"]
        img_w  = img_info["width"]
        img_h  = img_info["height"]
        img_id = img_info["id"]

        src = IMG_SRC / fname
        dst = img_dir / fname
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            copied += 1

        anns = img_to_anns.get(img_id, [])
        stem = Path(fname).stem
        label_path = label_dir / f"{stem}.txt"
        with open(label_path, "w") as lf:
            for ann in anns:
                cls = cat_id_to_yolo.get(ann["category_id"], 0)
                cx, cy, nw, nh = coco_bbox_to_yolo(ann["bbox"], img_w, img_h)
                lf.write(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

    print(f"  {split_name}: {len(coco['images'])} images ({copied} copied), {len(coco['annotations'])} labels")


def main():
    print("Converting COCO → YOLO format...")
    for split in ("train", "val", "test"):
        convert_split(split)
    print("Done.")


if __name__ == "__main__":
    main()
