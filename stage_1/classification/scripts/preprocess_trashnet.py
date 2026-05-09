#!/usr/bin/env python3
"""
Preprocess TrashNet dataset: resize, clean, and organize images for classification.
"""
import os
from PIL import Image

SRC_DIR = '/u/student/2024/cs24mtech11024/Capstone/stage_1/datasets/classification/trashnet/dataset-original/dataset-original/'
DST_DIR = '/u/student/2024/cs24mtech11024/Capstone/stage_1/datasets/classification/trashnet/dataset-preprocessed/'
IMG_SIZE = (224, 224)
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

os.makedirs(DST_DIR, exist_ok=True)
for cls in CLASSES:
    src_cls = os.path.join(SRC_DIR, cls)
    dst_cls = os.path.join(DST_DIR, cls)
    print(f"Looking in: {os.path.abspath(src_cls)}")
    os.makedirs(dst_cls, exist_ok=True)
    if not os.path.exists(src_cls):
        print(f"[ERROR] Source class directory does not exist: {os.path.abspath(src_cls)}")
        continue
    try:
        img_count = 0
        for img_name in os.listdir(src_cls):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            src_img = os.path.join(src_cls, img_name)
            dst_img = os.path.join(dst_cls, img_name)
            try:
                img = Image.open(src_img).convert('RGB').resize(IMG_SIZE)
                img.save(dst_img)
                img_count += 1
            except Exception as e:
                print(f"Error processing {src_img}: {e}")
        print(f"Processed {img_count} images for class '{cls}'")
    except Exception as e:
        print(f"Error accessing directory {src_cls}: {e}")
print("TrashNet preprocessing complete. Output in:", DST_DIR)