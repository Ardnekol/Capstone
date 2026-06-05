#!/usr/bin/env python3
"""Diagnostic: why is Florence detection recall low on ZeroWaste-f?

Compares Florence-2-FT prompts on a sample of ZeroWaste test images, reporting
mean boxes/image and class-agnostic P/R/F1 @IoU0.5. Picks the most appropriate
detection head for a class-agnostic localization metric. NOT a final result —
whatever wins here must be applied to ALL detection datasets and re-reported.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

from florence_common import load_florence2, _run
from eval_detection import load_zerowaste, score_image, prf


def boxes_from(result: dict) -> list:
    for k in result:
        v = result[k]
        if isinstance(v, dict) and "bboxes" in v:
            return [list(b) for b in v["bboxes"]]
    return []


PROMPTS = {
    "<OD> (current)":            ("<OD>", 256),
    "<OD> max1024":              ("<OD>", 1024),
    "<REGION_PROPOSAL>":         ("<REGION_PROPOSAL>", 1024),
    "<DENSE_REGION_CAPTION>":    ("<DENSE_REGION_CAPTION>", 1024),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", default="../finetuned/florence2_unified_multitask_lora")
    ap.add_argument("--root", default="../../datasets/zerowaste-f")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    items = load_zerowaste(Path(args.root).expanduser().resolve())[: args.n]
    gt_total = sum(len(g) for _, g in items)
    print(f"Sample: {len(items)} imgs, {gt_total} GT boxes ({gt_total/len(items):.1f}/img)\n")

    model, processor = load_florence2(args.lora, args.device)
    for name, (prompt, mnt) in PROMPTS.items():
        tp = fp = fn = 0
        nboxes = 0
        for p, gt in items:
            img = Image.open(p).convert("RGB")
            res = _run(model, processor, img, prompt, args.device, mnt)
            pb = boxes_from(res)
            nboxes += len(pb)
            a, b, c = score_image(pb, gt)
            tp += a; fp += b; fn += c
        m = prf(tp, fp, fn)
        print(f"{name:28s} boxes/img={nboxes/len(items):5.1f}  "
              f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")


if __name__ == "__main__":
    main()
