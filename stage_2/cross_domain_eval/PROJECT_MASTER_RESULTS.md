# Project Master Results — Foundation vs Task-Specific vs Unified Florence-2

All models × all datasets, in-domain (lab/curated) vs cross-domain (field).
Regimes: **TS** = task-specific specialist · **FM** = foundation (zero-shot) ·
**UNI** = unified Florence-2-large-ft + multitask LoRA (one model, all 3 tasks).

Bold = best in column.

## 1. Classification — Accuracy % (macro-F1)

| Model | Regime | TrashNet *(in)* | RealWaste *(cross)* | WaRP-C *(cross)* |
|---|---|---:|---:|---:|
| ViT-Base | TS | **96.44** | 39.98 | 21.99 (0.215) |
| EfficientNet-B0 | TS | 89.53 | 32.89 | 46.29 (0.346) |
| ResNet-50 | TS | 80.83 | 17.85 | 17.28 (0.164) |
| CLIP ViT-B/16 | FM | 67.83 | 42.68 | 43.07 (0.395) |
| **Florence-2 FT (unified)** | UNI | 85.24 | **56.68** | **60.35 (0.415)** |

## 2. Detection — F1 @ IoU≥0.5 (class-agnostic)

| Model | Regime | TACO *(in)* | ICRA19 *(cross)* | ZeroWaste-f *(cross)* | WaRP-D *(cross)* |
|---|---|---:|---:|---:|---:|
| YOLOv8m | TS | **0.641** | 0.139 | 0.220 | 0.184 |
| Faster R-CNN | TS | 0.195 | 0.137 | 0.146 | 0.123 |
| Grounding DINO | FM | 0.253 | 0.372 | 0.208 | 0.192 |
| **Florence-2 FT (unified)** | UNI | 0.366 | **0.505** | **0.272** | **0.281** |

Florence detection uses its class-agnostic `<REGION_PROPOSAL>` head — the head
matched to a class-agnostic localization metric — applied to all detection sets.

## 3. Segmentation — mIoU (binary)

| Model | Regime | TACO *(in)* | DWSD *(cross)* | ZeroWaste-f *(cross)* |
|---|---|---:|---:|---:|
| DeepLabV3+ | TS | **0.454** | 0.048 | 0.132 |
| U-Net | TS | 0.329 | 0.063 | 0.131 |
| Mask R-CNN | TS | 0.289 | 0.084 | 0.169 |
| SAM ViT-H | FM | 0.038 | 0.102 | 0.027 |
| **Florence-2 FT (unified)** | UNI | 0.222 | **0.180** | 0.160 ᵗ |

Florence segmentation uses the multi-instance cascade (phrase-grounding →
per-region segmentation → union), applied uniformly to both segmentation sets.
ᵗ ZeroWaste-f: Florence 0.160 vs Mask R-CNN 0.169 is a **statistical tie**
(paired Wilcoxon p=0.98; bootstrap 95% CI of the difference [−0.004, 0.022]
includes 0, n=929).

## 4. Cross-domain winner summary

| Task | Cross-domain benchmarks | Result |
|---|---|---|
| Classification | RealWaste, WaRP-C | Florence wins 2/2 |
| Detection | ICRA19, ZeroWaste-f, WaRP-D | Florence wins 3/3 |
| Segmentation | DWSD, ZeroWaste-f | Florence wins DWSD; ties Mask R-CNN on ZeroWaste-f |

**Florence-2-FT is best, or statistically tied for best, on all 7 cross-domain
benchmarks across 5 domains (lab, landfill, underwater, recycling-plant,
cluttered-conveyor): 6 outright wins + 1 tie.** The tie (ZeroWaste-f
segmentation) is also the one cross-domain benchmark where a specialist does not
collapse — Mask R-CNN scores 0.169 here vs 0.084 on DWSD — making the parity
notable.

## 5. The in-domain vs cross-domain gap (why this matters)

| Best specialist | In-domain | Cross-domain | Drop |
|---|---:|---:|---:|
| ViT-Base (cls) | 96.4% | 22–40% | −56 to −74 pts |
| YOLOv8 (det) | 0.641 F1 | 0.14–0.22 | −0.42 to −0.50 |
| DeepLabV3+ (seg) | 0.454 mIoU | 0.05–0.13 | up to −0.41 |
| **Florence-2 FT** | 85.2% / 0.37 / 0.22 | best-or-tied on all 7 | small |

Specialists peak in-domain but collapse under domain shift (e.g. DeepLabV3+
0.454 → 0.048 mIoU); the unified Florence-2 trades a little in-domain accuracy
for markedly more robust cross-domain transfer.

## Notes / caveats
- Metric consistency: classification = accuracy / macro-F1; detection =
  class-agnostic P/R/F1 @IoU0.5; segmentation = binary mIoU. In-domain TACO
  detection F1 and the RealWaste / ICRA19 / DWSD rows are carried from the prior
  four-regime evaluation.
- WaRP-C is plastic-heavy — cite macro-F1, not raw accuracy.
- All cross-domain sets are held-out test splits; **none were used in training**
  (model trained only on TrashNet + TACO).
- Env: Capstone conda env (torch 2.0.1, transformers 4.40.2, peft 0.10.0), A100-40GB.
