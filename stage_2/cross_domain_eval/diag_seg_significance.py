#!/usr/bin/env python3
"""Significance test: Mask R-CNN vs Florence-cascade on ZeroWaste-f segmentation.

Is the 0.169 vs 0.160 gap a real difference or statistical noise? Computes paired
per-image IoU for both models on the SAME images, then:
  - Wilcoxon signed-rank test (paired, non-parametric)
  - bootstrap 95% CI of the mean per-image IoU difference

If the CI for (MaskRCNN - Florence) includes 0 → not significant → report a TIE.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

from eval_segmentation import list_pairs, eval_maskrcnn, eval_florence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../../datasets/zerowaste-f")
    ap.add_argument("--maskrcnn", default="../../stage_1/segmentation/results/maskrcnn_best.pth")
    ap.add_argument("--lora", default="../finetuned/florence2_unified_multitask_lora")
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    pairs = list_pairs(Path(args.root).expanduser().resolve(), args.max_images)
    print(f"ZeroWaste-f pairs: {len(pairs)}\n")

    print("Running Mask R-CNN ...")
    mr = eval_maskrcnn(args.maskrcnn, pairs, args.device)
    print("Running Florence-2 cascade ...")
    fl = eval_florence(args.lora, pairs, args.device, "cascade")

    a = np.array(mr["per_image_iou"], dtype=float)   # Mask R-CNN
    b = np.array(fl["per_image_iou"], dtype=float)    # Florence cascade
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    d = a - b

    print("\n================ SIGNIFICANCE ================")
    print(f"n images          : {n}")
    print(f"Mask R-CNN  mIoU  : {a.mean():.4f}")
    print(f"Florence    mIoU  : {b.mean():.4f}")
    print(f"mean diff (MR-Flo): {d.mean():.4f}")

    try:
        from scipy.stats import wilcoxon
        stat, p = wilcoxon(a, b)
        print(f"Wilcoxon signed-rank p = {p:.4g}")
    except Exception as e:
        p = None
        print(f"(wilcoxon unavailable: {e})")

    rng = np.random.default_rng(0)
    boot = np.array([rng.choice(d, size=n, replace=True).mean() for _ in range(10000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"bootstrap 95% CI of mean diff: [{lo:.4f}, {hi:.4f}]")

    print("\n---------------- VERDICT ----------------")
    tie_ci = (lo <= 0 <= hi)
    tie_p = (p is not None and p > 0.05)
    if tie_ci or tie_p:
        print("NOT statistically significant → report as a TIE / on-par.")
        print("Honest claim: 6 cross-domain wins + 1 statistical tie.")
    else:
        print("Statistically significant → narrow but real loss; keep 6/7.")


if __name__ == "__main__":
    main()
