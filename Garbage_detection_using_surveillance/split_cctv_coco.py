#!/usr/bin/env python3
"""Split the CCTV garbage COCO dataset into train/val/test (80/10/10)."""

import json
import random
from pathlib import Path

DATASET_DIR = Path("/u/student/2024/cs24mtech11024/Capstone/datasets/Garbage Detection using CCTV.coco")
ANNOTATIONS_FILE = DATASET_DIR / "_annotations.coco.json"
OUT_DIR = Path(__file__).parent / "data" / "annotations"

TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
# TEST = remainder

SEED = 42


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(ANNOTATIONS_FILE) as f:
        coco = json.load(f)

    # Keep only the real object category (garbage, id=1)
    keep_cat_ids = {c["id"] for c in coco["categories"] if c["name"] == "garbage"}
    categories   = [c for c in coco["categories"] if c["id"] in keep_cat_ids]

    # Filter annotations to kept categories
    annotations = [a for a in coco["annotations"] if a["category_id"] in keep_cat_ids]
    annotated_ids = {a["image_id"] for a in annotations}

    # Only keep images that have at least one garbage annotation
    images = [img for img in coco["images"] if img["id"] in annotated_ids]

    random.seed(SEED)
    random.shuffle(images)

    n = len(images)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    train_imgs = images[:n_train]
    val_imgs   = images[n_train : n_train + n_val]
    test_imgs  = images[n_train + n_val :]

    def build_split(imgs):
        img_ids = {img["id"] for img in imgs}
        anns    = [a for a in annotations if a["image_id"] in img_ids]
        return {
            "info":        coco.get("info", {}),
            "licenses":    coco.get("licenses", []),
            "categories":  categories,
            "images":      imgs,
            "annotations": anns,
        }

    splits = {
        "train": build_split(train_imgs),
        "val":   build_split(val_imgs),
        "test":  build_split(test_imgs),
    }

    for name, data in splits.items():
        out_path = OUT_DIR / f"{name}.coco.json"
        with open(out_path, "w") as f:
            json.dump(data, f)
        print(f"{name:5s}: {len(data['images']):4d} images, {len(data['annotations']):5d} annotations → {out_path}")

    summary = {s: {"images": len(d["images"]), "annotations": len(d["annotations"])} for s, d in splits.items()}
    with open(OUT_DIR / "split_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nDone. Split summary saved.")


if __name__ == "__main__":
    main()
