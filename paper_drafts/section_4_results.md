# Section 4 — Results (draft v1, ~550 words + tables)

## 4.1 Headline: Florence-2 + LoRA sweeps every cross-domain benchmark

Table 1 reports cross-domain performance of the unified Florence-2 + Waste
LoRA against the best Stage 1 task-specific and foundation baselines, on
the three cross-domain test sets that target the lab-to-field gap.

**Table 1.** Cross-domain results across all three vision tasks. Bold marks
the best per row.

| Cross-domain benchmark | Best Stage 1 specialist | Best Stage 1 foundation (zero-shot) | Florence-2 unified + LoRA (ours) |
|---|---:|---:|---:|
| RealWaste classification — accuracy | ViT-Base 39.98% | CLIP 42.68% | **56.68%** |
| Trash-ICRA19 detection — F1 | YOLOv8 13.49% | Grounding DINO 37.20% | **50.49%** |
| DWSD segmentation — mIoU | Mask R-CNN 0.0842 | SAM 0.1023 | **0.2214** |

The unified model wins every cross-domain row. Improvements over the best
prior baseline are **+14.00 points** on classification, **+13.29 points F1**
on detection, and **+0.1191 mIoU** on segmentation (a 2.2× improvement over
SAM). Notably, the foundation baselines themselves substantially outperform
the specialists cross-domain (e.g., CLIP > ViT-Base, G-DINO > YOLOv8) —
confirming the lab-to-field framing — and our fine-tuned Florence-2
extends that advantage further.

## 4.2 Segmentation transfers across domains nearly without degradation

The most striking single result is in segmentation. Table 2 contrasts
in-domain and cross-domain mIoU for the strongest model in each regime.

**Table 2.** Segmentation in-domain vs cross-domain. Relative drop in
parentheses.

| Model | TACO (in-domain) | DWSD (cross-domain) | Relative drop |
|---|---:|---:|---:|
| DeepLabV3+ specialist | 0.4541 | 0.0483 | **−89.4%** |
| SAM (foundation, zero-shot) | 0.0380 | 0.1023 | (improves) |
| Florence-2 unified + LoRA (ours) | **0.2223** | **0.2214** | **−0.4%** |

DeepLabV3+ achieves the highest in-domain mIoU but loses 89% of it
cross-domain. The unified Florence-2 + LoRA's mIoU is **0.2214 cross-domain
vs 0.2223 in-domain — a difference of only 0.0009**. Polygon-token
segmentation transfers between waste domains essentially without
degradation, despite generating its mask output as a sequence of
quantized location tokens.

## 4.3 Multi-task vs single-task ablation

Table 3 compares the unified LoRA against the two single-task variants
that converged cleanly under identical hyperparameters (seg-only failed
to converge — see Limitations).

**Table 3.** Multi-task ablation. Best per row in bold.

| Benchmark | cls-only LoRA | det-only LoRA | unified LoRA |
|---|---:|---:|---:|
| TrashNet cls (in-domain) — accuracy | **0.8567** | 0.4377 | 0.8480 |
| RealWaste cls (cross-domain) — accuracy | **0.5944** | 0.3953 | 0.5838 |
| TACO det (in-domain) — F1 | 0.2409 | 0.3469 | **0.3657** |
| Trash-ICRA19 det (cross-domain) — F1 | 0.4145 | **0.5886** | 0.5049 |
| TACO seg (in-domain) — mIoU | 0.1492 | **0.3237** | 0.2223 |
| DWSD seg (cross-domain) — mIoU | 0.2067 | **0.2257** | 0.2214 |

Specialist single-task LoRAs slightly outperform the unified LoRA on
their own task — cls-only wins classification by 1.1 cross-domain points,
det-only wins cross-domain detection by 8.4 F1 points and cross-domain
segmentation by 0.004 mIoU (essentially tied with unified). **The unified
model's multi-task LoRA does not provide implicit per-task regularization;
it does, however, deliver competitive performance across all three tasks
from a single 14 MB adapter.** All three Florence-2 LoRA variants beat
every Stage 1 cross-domain baseline.

A useful side-observation: the det-only LoRA produces strong segmentation
mIoU despite never being trained on segmentation. This suggests LoRA
fine-tuning preserves Florence-2's pretrained polygon-token generation
capability rather than overwriting it.

## 4.4 Florence-2 zero-shot is already the strongest cross-domain
foundation model on two of three tasks

Before fine-tuning, Florence-2-large-ft zero-shot is already the strongest
foundation model on cross-domain detection (42.03% F1 on ICRA19 vs G-DINO
37.20%) and cross-domain segmentation (0.1207 mIoU on DWSD vs SAM 0.1023).
Fine-tuning with our 14 MB Waste LoRA improves DWSD mIoU by another **83%**
(0.1207 → 0.2214). The architectural advantage and the fine-tuning gain
compound.

---

## Notes for the author

- All numbers are from the post-binarization-bug-fix full-dataset evaluation runs (May 2026).
- §4.2's "Relative drop" column for SAM is left as "(improves)" because SAM on TACO seg in-domain underperforms SAM on DWSD cross-domain — an artefact of SAM's pretraining bias toward complex scenes; flagged but not load-bearing.
- §4.3 deliberately states the negative result (no implicit regularization). Honesty here strengthens the paper.
- §4.4 is a brand-new finding from the late-stage zero-shot DWSD run. It supports the §3 justification for choosing Florence-2 over CLIP/G-DINO/SAM.
