#!/usr/bin/env python3
"""
Merge Dataset 2 (YOLO format) into existing yolo_data directory.
Dataset 2: /u/student/2024/cs24mtech11024/Capstone/datasets/Garbage Detection using CCTV_2
Both datasets have nc=1, class=garbage — directly compatible.
"""

import shutil
from pathlib import Path

DS2_DIR  = Path("/u/student/2024/cs24mtech11024/Capstone/datasets/Garbage Detection using CCTV_2")
YOLO_DIR = Path(__file__).parent / "yolo_data"

SPLITS = [
    ("train/images", "train/labels", "train"),
    ("valid/images", "valid/labels", "val"),
]


def merge_split(src_img_dir: Path, src_lbl_dir: Path, dst_split: str):
    dst_img = YOLO_DIR / "images" / dst_split
    dst_lbl = YOLO_DIR / "labels" / dst_split
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    copied_imgs = copied_lbls = skipped = 0

    for img_path in src_img_dir.glob("*"):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue

        # Avoid filename collision — prefix with ds2_
        dst_name = f"ds2_{img_path.name}"
        dst_img_path = dst_img / dst_name

        if dst_img_path.exists():
            skipped += 1
            continue

        shutil.copy2(img_path, dst_img_path)
        copied_imgs += 1

        # Copy matching label
        lbl_path = src_lbl_dir / (img_path.stem + ".txt")
        if lbl_path.exists():
            shutil.copy2(lbl_path, dst_lbl / (Path(dst_name).stem + ".txt"))
            copied_lbls += 1

    print(f"  {dst_split:6s}: {copied_imgs} images, {copied_lbls} labels copied  ({skipped} skipped)")


def main():
    print("Merging Dataset 2 into yolo_data...")
    print(f"Source: {DS2_DIR}")
    print(f"Target: {YOLO_DIR}\n")

    for src_img_rel, src_lbl_rel, dst_split in SPLITS:
        src_img = DS2_DIR / src_img_rel
        src_lbl = DS2_DIR / src_lbl_rel
        if not src_img.exists():
            print(f"  Skipping {dst_split} — {src_img} not found")
            continue
        merge_split(src_img, src_lbl, dst_split)

    # Final counts
    print("\nFinal dataset size:")
    for split in ("train", "val", "test"):
        n = len(list((YOLO_DIR / "images" / split).glob("*"))) if (YOLO_DIR / "images" / split).exists() else 0
        print(f"  {split:6s}: {n} images")

    print("\nDone. Now run: bash train_yolo.sh")


if __name__ == "__main__":
    main()
