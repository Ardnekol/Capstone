#!/usr/bin/env python3
"""Cross-domain DETECTION comparison on ZeroWaste-f and WaRP-D.

Three regimes, ONE consistent metric (class-agnostic, IoU>=0.5 greedy matching →
Precision / Recall / F1 — identical to stage_2/evaluate_unified_model.py). This
deliberately avoids the mAP-vs-F1 mismatch: every model is scored the same way.

  1. Task-Specific : YOLOv8m (Stage 1 checkpoint)
  2. Foundation    : Grounding DINO (zero-shot, text-prompted)
  3. Unified FT     : Florence-2-large-ft + unified multitask LoRA (<OD>)

Datasets (test splits only — never trained on):
  - ZeroWaste-f : COCO json  (datasets/zerowaste-f/splits_final_deblurred/test)
  - WaRP-D      : YOLO txt    (datasets/WARP/Warp-D/test)

Each regime is wrapped in try/except on its imports — if a dependency is missing
the regime is SKIPPED with a clear message and the rest still run.

Usage:
  python3 eval_detection.py \
      --dataset zerowaste --max-images 0 \
      --yolo ../../stage_1/detection/runs/detect/yolov8m_20251225_182218/weights/best.pt \
      --gdino IDEA-Research/grounding-dino-base \
      --lora ../finetuned/florence2_unified_multitask_lora \
      --output-dir ../eval_results/detection_zerowaste
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

warnings.filterwarnings("ignore")

WASTE_PROMPT = "trash . garbage . litter . waste . plastic . cardboard . metal . bottle"
IOU_THR = 0.5


# ──────────────────────────────────────────────────────────────────────────────
# Metric (class-agnostic greedy matching) — mirrors evaluate_unified_model.py
# ──────────────────────────────────────────────────────────────────────────────
def _iou(b1: List[float], b2: List[float]) -> float:
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def score_image(pred_boxes: List[List[float]], gt_boxes: List[List[float]]) -> Tuple[int, int, int]:
    """Return (tp, fp, fn) for one image, class-agnostic, IoU>=0.5 greedy."""
    matched = set()
    tp = 0
    for pb in pred_boxes:
        best_iou, best_idx = 0.0, -1
        for gi, gb in enumerate(gt_boxes):
            if gi in matched:
                continue
            iou = _iou(pb, gb)
            if iou > best_iou:
                best_iou, best_idx = iou, gi
        if best_iou >= IOU_THR and best_idx >= 0:
            tp += 1
            matched.add(best_idx)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - len(matched)
    return tp, fp, fn


def prf(tp: int, fp: int, fn: int) -> Dict:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn}


# ──────────────────────────────────────────────────────────────────────────────
# Ground-truth loaders → list of (image_path, [ [x1,y1,x2,y2], ... ])
# ──────────────────────────────────────────────────────────────────────────────
def load_zerowaste(root: Path) -> List[Tuple[Path, List[List[float]]]]:
    test = root / "splits_final_deblurred" / "test"
    data = json.loads((test / "labels.json").read_text())
    from collections import defaultdict
    img_info = {int(im["id"]): im for im in data["images"]}
    ann_by_img = defaultdict(list)
    for a in data["annotations"]:
        ann_by_img[int(a["image_id"])].append(a)
    out = []
    for iid, im in img_info.items():
        p = test / "data" / im["file_name"]
        if not p.exists():
            continue
        boxes = []
        for a in ann_by_img.get(iid, []):
            if int(a.get("iscrowd", 0)) == 1:
                continue
            x, y, w, h = map(float, a["bbox"])
            boxes.append([x, y, x + w, y + h])
        out.append((p, boxes))
    return out


def load_warpd(root: Path) -> List[Tuple[Path, List[List[float]]]]:
    img_dir, lbl_dir = root / "test" / "images", root / "test" / "labels"
    out = []
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        lbl = lbl_dir / (p.stem + ".txt")
        if not lbl.exists():
            continue
        with Image.open(p) as im:
            W, H = im.size
        boxes = []
        for line in lbl.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            _, cx, cy, w, h = map(float, parts[:5])
            x1, y1 = (cx - w / 2) * W, (cy - h / 2) * H
            x2, y2 = (cx + w / 2) * W, (cy + h / 2) * H
            boxes.append([x1, y1, x2, y2])
        out.append((p, boxes))
    return out


def load_icra19(root: Path) -> List[Tuple[Path, List[List[float]]]]:
    """Trash-ICRA19 test split, PASCAL VOC XML. (rov/timestamp labels skipped.)"""
    import xml.etree.ElementTree as ET
    test = root / "trash_ICRA19" / "dataset" / "test"
    if not test.exists():
        test = root / "dataset" / "test"
    out = []
    for p in sorted(test.iterdir()):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        xml = p.with_suffix(".xml")
        if not xml.exists():
            continue
        boxes = []
        for obj in ET.parse(str(xml)).getroot().findall("object"):
            name = obj.find("name")
            if name is not None and name.text.strip().lower() in ("rov", "timestamp"):
                continue
            bb = obj.find("bndbox")
            if bb is None:
                continue
            boxes.append([float(bb.find("xmin").text), float(bb.find("ymin").text),
                          float(bb.find("xmax").text), float(bb.find("ymax").text)])
        if boxes:
            out.append((p, boxes))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Regime predictors → each yields {img_path: [boxes]}
# ──────────────────────────────────────────────────────────────────────────────
def predict_yolo(weights: str, items, device: str) -> Dict[str, List[List[float]]]:
    from ultralytics import YOLO
    model = YOLO(weights)
    preds = {}
    dev = 0 if device.startswith("cuda") else "cpu"
    for i, (p, _) in enumerate(items):
        r = model.predict(str(p), device=dev, verbose=False, conf=0.25)[0]
        boxes = r.boxes.xyxy.cpu().numpy().tolist() if r.boxes is not None else []
        preds[str(p)] = boxes
        if (i + 1) % 100 == 0:
            print(f"    [YOLO] {i + 1}/{len(items)}")
    return preds


def predict_fasterrcnn(weights: str, items, device: str, score_thr: float = 0.25) -> Dict[str, List[List[float]]]:
    import torch
    import torchvision.transforms.functional as TF
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck
    ncls = sd["roi_heads.box_predictor.cls_score.weight"].shape[0]  # auto-infer
    model = fasterrcnn_resnet50_fpn_v2(weights=None)
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, ncls)
    model.load_state_dict(sd)
    model = model.to(device).eval()
    preds = {}
    with torch.no_grad():
        for i, (p, _) in enumerate(items):
            img = Image.open(p).convert("RGB")
            x = TF.to_tensor(img).to(device)
            out = model([x])[0]
            keep = out["scores"] > score_thr
            preds[str(p)] = out["boxes"][keep].cpu().numpy().tolist()
            if (i + 1) % 100 == 0:
                print(f"    [FasterRCNN] {i + 1}/{len(items)}")
    return preds


def _gdino_post(processor, out, input_ids, target_size):
    """Robust across transformers versions: 4.40.x uses `box_threshold`,
    4.5x renamed it to `threshold`."""
    kw = dict(input_ids=input_ids, text_threshold=0.25, target_sizes=[target_size])
    try:
        return processor.post_process_grounded_object_detection(out, box_threshold=0.25, **kw)[0]
    except TypeError:
        return processor.post_process_grounded_object_detection(out, threshold=0.25, **kw)[0]


def predict_gdino(model_name: str, items, device: str, prompt: str) -> Dict[str, List[List[float]]]:
    import torch
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name).to(device).eval()
    preds = {}
    for i, (p, _) in enumerate(items):
        img = Image.open(p).convert("RGB")
        inputs = processor(images=img, text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
        res = _gdino_post(processor, out, inputs["input_ids"], img.size[::-1])
        preds[str(p)] = res["boxes"].cpu().numpy().tolist()
        if (i + 1) % 100 == 0:
            print(f"    [G-DINO] {i + 1}/{len(items)}")
    return preds


def predict_florence(lora: str, items, device: str, method: str = "od") -> Dict[str, List[List[float]]]:
    from florence_common import load_florence2, run_boxes
    model, processor = load_florence2(lora, device)
    preds = {}
    for i, (p, _) in enumerate(items):
        img = Image.open(p).convert("RGB")
        preds[str(p)] = run_boxes(model, processor, img, device, method)
        if (i + 1) % 50 == 0:
            print(f"    [Florence/{method}] {i + 1}/{len(items)}")
    return preds


# ──────────────────────────────────────────────────────────────────────────────
def aggregate(items, preds) -> Dict:
    tp = fp = fn = 0
    for p, gt in items:
        pb = preds.get(str(p), [])
        a, b, c = score_image(pb, gt)
        tp += a; fp += b; fn += c
    res = prf(tp, fp, fn)
    res["evaluated"] = len(items)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="3-regime cross-domain detection")
    ap.add_argument("--dataset", choices=["zerowaste", "warpd", "icra19"], required=True)
    ap.add_argument("--zerowaste-root", default="../../datasets/zerowaste-f")
    ap.add_argument("--warpd-root", default="../../datasets/WARP/Warp-D")
    ap.add_argument("--icra19-root", default="../../datasets/detection/trash_icra19")
    ap.add_argument("--yolo", default="../../stage_1/detection/runs/detect/yolov8m_20251225_182218/weights/best.pt")
    ap.add_argument("--fasterrcnn", default="../../stage_1/detection/runs/fasterrcnn/fasterrcnn_20251225_235400/best.pth")
    ap.add_argument("--gdino", default="IDEA-Research/grounding-dino-base")
    ap.add_argument("--lora", default="../finetuned/florence2_unified_multitask_lora")
    ap.add_argument("--prompt", default=WASTE_PROMPT)
    ap.add_argument("--det-method", default="od",
                    choices=["od", "region_proposal", "dense_region_caption"],
                    help="Florence detection head (region_proposal = class-agnostic, better recall)")
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip", default="")
    args = ap.parse_args()

    import torch
    device = args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    if args.dataset == "zerowaste":
        items = load_zerowaste(Path(args.zerowaste_root).expanduser().resolve())
        ds_label = "ZeroWaste-f"
    elif args.dataset == "icra19":
        items = load_icra19(Path(args.icra19_root).expanduser().resolve())
        ds_label = "Trash-ICRA19"
    else:
        items = load_warpd(Path(args.warpd_root).expanduser().resolve())
        ds_label = "WaRP-D"
    if args.max_images > 0:
        items = items[:args.max_images]
    out_dir = Path(args.output_dir or f"../eval_results/detection_{args.dataset}").expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{ds_label}: {len(items)} test images | device={device}\n")

    results: Dict[str, Dict] = {}
    regimes = [
        ("yolov8", "task-specific", lambda: predict_yolo(args.yolo, items, device)),
        ("faster_rcnn", "task-specific", lambda: predict_fasterrcnn(args.fasterrcnn, items, device)),
        ("grounding_dino", "foundation", lambda: predict_gdino(args.gdino, items, device, args.prompt)),
        ("florence2_ft", "unified-ft", lambda: predict_florence(args.lora, items, device, args.det_method)),
    ]
    for key, regime, fn in regimes:
        if key in skip:
            continue
        print(f"==== {regime}: {key} ====")
        try:
            preds = fn()
            results[key] = aggregate(items, preds)
            results[key]["regime"] = regime
            r = results[key]
            print(f"  {key}: P={r['precision']} R={r['recall']} F1={r['f1']}\n")
        except Exception as e:
            print(f"  [SKIP] {key}: {type(e).__name__}: {str(e)[:160]}\n")

    results["_meta"] = {"florence_det_method": args.det_method, "n_images": len(items)}
    (out_dir / f"detection_{args.dataset}_results.json").write_text(json.dumps(results, indent=2))

    name_map = {"yolov8": "YOLOv8m", "faster_rcnn": "Faster R-CNN",
                "grounding_dino": "Grounding DINO", "florence2_ft": "Florence-2 + LoRA"}
    lab = {"task-specific": "Task-Specific", "foundation": "Foundation", "unified-ft": "Unified FT"}
    lines = [f"# {ds_label} Cross-Domain Detection (class-agnostic IoU>=0.5)\n",
             f"Test images: **{len(items)}**\n",
             "| Model | Regime | Precision | Recall | F1 |", "|---|---|---:|---:|---:|"]
    for k in ["yolov8", "faster_rcnn", "grounding_dino", "florence2_ft"]:
        if k in results:
            r = results[k]
            lines.append(f"| {name_map[k]} | {lab[r['regime']]} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} |")
    (out_dir / f"detection_{args.dataset}_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {out_dir}/detection_{args.dataset}_results.json")


if __name__ == "__main__":
    main()
