#!/usr/bin/env python3
"""Evaluate a trained YOLO model on the CCTV garbage test set."""

import argparse
import json
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
DATA_DIR    = PROJECT_DIR / "data" / "annotations"
IMG_SRC     = Path("/u/student/2024/cs24mtech11024/Capstone/datasets/Garbage Detection using CCTV.coco/train")


def evaluate(model_path: str, split: str, conf: float, iou: float):
    from ultralytics import YOLO
    from florence_cctv_utils import match_predictions, compute_metrics

    model = YOLO(model_path)

    # --- YOLO built-in val on the CCTV splits ---
    data_yaml = PROJECT_DIR / "data.yaml"
    print(f"\nRunning YOLO built-in validation ({split} split)...")
    val_results = model.val(
        data=str(data_yaml),
        split=split,
        conf=conf,
        iou=iou,
        verbose=False,
    )

    map50     = float(val_results.box.map50)
    map50_95  = float(val_results.box.map)
    precision = float(val_results.box.mp)
    recall    = float(val_results.box.mr)
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics = {
        "model":       model_path,
        "split":       split,
        "conf":        conf,
        "iou":         iou,
        "mAP@0.5":     map50,
        "mAP@0.5:0.95":map50_95,
        "precision":   precision,
        "recall":      recall,
        "f1":          f1,
    }

    print(f"\n{'='*50}")
    print(f"YOLO Evaluation — {split}")
    print(f"  mAP@0.5     : {map50:.4f}")
    print(f"  mAP@0.5:0.95: {map50_95:.4f}")
    print(f"  Precision   : {precision:.4f}")
    print(f"  Recall      : {recall:.4f}")
    print(f"  F1          : {f1:.4f}")
    print(f"{'='*50}")

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_DIR / "eval_results" / "yolo"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"yolo_eval_{split}_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Also update the canonical summary (overwrite for latest best)
    with open(out_dir / "yolo_eval_summary.json", "w") as f:
        json.dump({"model": model_path, "validation": metrics, "test": metrics}, f, indent=2)

    print(f"Results saved to: {out_path}")
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",  required=True, help="Path to best.pt")
    p.add_argument("--split",  default="test", choices=["val", "test"])
    p.add_argument("--conf",   type=float, default=0.15,
                   help="Confidence threshold (0.15 = higher recall)")
    p.add_argument("--iou",    type=float, default=0.5)
    args = p.parse_args()
    evaluate(args.model, args.split, args.conf, args.iou)


if __name__ == "__main__":
    main()
