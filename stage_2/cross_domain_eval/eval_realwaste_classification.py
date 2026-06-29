#!/usr/bin/env python3
"""Matched-protocol RealWaste classification recompute (3-regime).

Recomputes specialists (ViT/ResNet/EffNet) + CLIP + Florence-2 on the RealWaste
cross-domain test set, using the SAME code path as the WaRP-C / new-dataset
evaluation, so every cross-domain classification number is on one protocol.

RealWaste 9 classes -> TrashNet 6 (food organics / textile / vegetation dropped).

Usage:
  python3 eval_realwaste_classification.py \
      --root ../../datasets/classification/realwaste/realwaste-main/RealWaste \
      --lora ../finetuned/florence2_unified_multitask_lora \
      --output-dir ../eval_results/realwaste_classification_matched
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from collections import Counter

from eval_warpc_classification import (
    TRASHNET_CLASSES, eval_specialist, eval_clip, eval_florence, compute_metrics,
)

REALWASTE_TO_TRASHNET = {
    "cardboard": "cardboard", "glass": "glass", "metal": "metal",
    "paper": "paper", "plastic": "plastic", "miscellaneous trash": "trash",
    "food organics": None, "textile trash": None, "vegetation": None,
}


def list_samples(root: Path, max_per_class: int):
    samples = []
    for cls_dir in sorted(root.iterdir()):
        if not cls_dir.is_dir() or cls_dir.name.startswith("."):
            continue
        mapped = REALWASTE_TO_TRASHNET.get(cls_dir.name.lower().strip())
        if mapped is None:
            continue
        imgs = sorted(f for f in cls_dir.rglob("*")
                      if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        if max_per_class > 0:
            imgs = imgs[:max_per_class]
        samples.extend((p, mapped) for p in imgs)
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../../datasets/classification/realwaste/realwaste-main/RealWaste")
    ap.add_argument("--lora", default="../finetuned/florence2_unified_multitask_lora")
    ap.add_argument("--ckpt-dir", default="../../stage_1/classification/results")
    ap.add_argument("--max-per-class", type=int, default=0)
    ap.add_argument("--output-dir", default="../eval_results/realwaste_classification_matched")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip", default="")
    args = ap.parse_args()

    import torch
    device = args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    root = Path(args.root).expanduser().resolve()
    ckpt_dir = Path(args.ckpt_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve(); out_dir.mkdir(parents=True, exist_ok=True)

    samples = list_samples(root, args.max_per_class)
    print(f"RealWaste samples (mapped): {len(samples)} | by-target {dict(Counter(t for _,t in samples))}\n")

    results = {}
    for name in ["vit_base", "resnet50", "efficientnetb0"]:
        if name in skip:
            continue
        ck = ckpt_dir / f"{name}_best.pth"
        if not ck.exists():
            print(f"[skip] {name}: missing {ck}"); continue
        print(f"==== {name} ====")
        try:
            results[name] = eval_specialist(name, ck, samples, device); results[name]["regime"] = "task-specific"
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
    if "clip" not in skip:
        print("==== CLIP ====")
        try:
            results["clip"] = eval_clip(samples, device); results["clip"]["regime"] = "foundation"
        except Exception as e:
            print(f"[FAIL] clip: {e}")
    if "florence" not in skip:
        print("==== Florence-2 + LoRA ====")
        try:
            results["florence2_ft"] = eval_florence(args.lora, samples, device); results["florence2_ft"]["regime"] = "unified-ft"
        except Exception as e:
            print(f"[FAIL] florence: {e}")

    (out_dir / "realwaste_classification_results.json").write_text(json.dumps(results, indent=2))
    nm = {"vit_base": "ViT-Base", "resnet50": "ResNet-50", "efficientnetb0": "EfficientNet-B0",
          "clip": "CLIP ViT-B/16", "florence2_ft": "Florence-2 + LoRA"}
    lab = {"task-specific": "Task-Specific", "foundation": "Foundation", "unified-ft": "Unified FT"}
    lines = ["# RealWaste Cross-Domain Classification (matched 3-regime)\n",
             f"Test images: **{len(samples)}**\n",
             "| Model | Regime | Accuracy | Macro-F1 |", "|---|---|---:|---:|"]
    for k in ["vit_base", "resnet50", "efficientnetb0", "clip", "florence2_ft"]:
        if k in results:
            r = results[k]
            lines.append(f"| {nm[k]} | {lab.get(r.get('regime',''),'')} | {r['accuracy']*100:.2f}% | {r['macro_f1']:.4f} |")
    (out_dir / "realwaste_classification_summary.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
