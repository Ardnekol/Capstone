"""
Run finetuned YOLO (CCTV garbage) on 100 random images from AllGVPImages/before
and save each image with predicted bounding boxes drawn on it.
"""
import random
import shutil
from pathlib import Path

from ultralytics import YOLO

SRC_DIR = Path("/u/student/2024/cs24mtech11024/Capstone/datasets/AllGVPImages/before")
WEIGHTS = Path(
    "/u/student/2024/cs24mtech11024/Capstone/Garbage_detection_using_surveillance"
    "/runs/detect/cctv_garbage_yolo_v42/weights/best.pt"
)
OUT_DIR = Path(
    "/u/student/2024/cs24mtech11024/Capstone/Garbage_detection_using_surveillance"
    "/allgvp_before_predictions"
)
N_SAMPLES = 100
SEED = 42
CONF = 0.25
IMGSZ = 640

OUT_DIR.mkdir(parents=True, exist_ok=True)

exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
all_imgs = sorted(p for p in SRC_DIR.iterdir() if p.suffix.lower() in exts)
print(f"Found {len(all_imgs)} images in {SRC_DIR}")

random.seed(SEED)
sampled = random.sample(all_imgs, min(N_SAMPLES, len(all_imgs)))
print(f"Sampled {len(sampled)} images (seed={SEED})")

sample_list_file = OUT_DIR / "_sampled_filenames.txt"
sample_list_file.write_text("\n".join(p.name for p in sampled) + "\n")

model = YOLO(str(WEIGHTS))
print(f"Loaded model: {WEIGHTS.name}")

results = model.predict(
    source=[str(p) for p in sampled],
    conf=CONF,
    imgsz=IMGSZ,
    device="cpu",
    save=True,
    save_txt=True,
    save_conf=True,
    project=str(OUT_DIR),
    name="yolo_v42_pred",
    exist_ok=True,
    verbose=False,
)

pred_dir = OUT_DIR / "yolo_v42_pred"
labels_subdir = pred_dir / "labels"
if labels_subdir.is_dir():
    for txt in labels_subdir.glob("*.txt"):
        shutil.move(str(txt), str(pred_dir / txt.name))
    labels_subdir.rmdir()

n_with_det = sum(1 for r in results if r.boxes is not None and len(r.boxes) > 0)
total_boxes = sum(len(r.boxes) for r in results if r.boxes is not None)
print(f"Done. {n_with_det}/{len(results)} images had detections; "
      f"{total_boxes} boxes total.")
print(f"Annotated images + labels saved under: {pred_dir}")
