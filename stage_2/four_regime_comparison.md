# Four-Regime Comparison: Stage 1 vs Stage 2

This table compares four experiment regimes:

1. **Stage 1 Task-Specific**: separate supervised model per task.
2. **Stage 1 Foundation**: zero-shot foundation model per task.
3. **Stage 2 Zero-Shot**: one base Florence-2 model, no waste fine-tuning.
4. **Stage 2 Fine-Tuned**: one unified Florence-2 model with multi-task LoRA.

## Main Comparison Table

| Dataset / Task | Metric | Stage 1 Task-Specific | Stage 1 Foundation | Stage 2 Zero-Shot Florence-2 | Stage 2 Fine-Tuned Florence-2 | Best Result |
|---|---|---:|---:|---:|---:|---|
| TrashNet classification, in-domain | Accuracy | **96.44%** (ViT-Base) | 67.83% (CLIP) | 26.43% | 85.24% | Stage 1 task-specific |
| RealWaste classification, cross-domain | Accuracy | 39.98% (ViT-Base) | 42.68% (CLIP) | 30.36% | **56.68%** | Stage 2 fine-tuned |
| TACO detection, in-domain | F1 | **64.10%** (YOLOv8) | 25.30% (Grounding DINO) | 24.53% | 36.57% | Stage 1 task-specific |
| Trash-ICRA19 detection, cross-domain | F1 | 13.90% (YOLOv8) | 37.20% (Grounding DINO) | 42.03% | **50.49%** | Stage 2 fine-tuned |
| TACO segmentation, in-domain | mIoU | **0.4541** (DeepLabV3+) | 0.0380 (SAM) | 0.1773 | 0.2223 | Stage 1 task-specific |
| DWSD segmentation, cross-domain | mIoU | 0.0842 (Mask R-CNN) | 0.1023 (SAM) | 0.1207 | **0.2214** | Stage 2 fine-tuned |

## Winner Summary

| Regime | Wins | Where It Wins |
|---|---:|---|
| Stage 1 Task-Specific | 3 | TrashNet classification, TACO detection, TACO segmentation (all in-domain) |
| Stage 1 Foundation | 0 | No overall wins |
| Stage 2 Zero-Shot Florence-2 | 0 | No overall wins, but strong on ICRA19 detection |
| Stage 2 Fine-Tuned Florence-2 | 3 | RealWaste classification, Trash-ICRA19 detection, **DWSD segmentation** (all cross-domain) |

**Tied 3–3 between Stage 1 task-specific (sweeps in-domain) and Stage 2 fine-tuned Florence-2 (sweeps cross-domain).** The split is clean: specialists win when train and test domains match; the unified fine-tuned foundation model wins when they don't.

## Direct Insights

- **Best in controlled/in-domain settings**: Stage 1 task-specific models. ViT-Base, YOLOv8, and DeepLabV3+ are strongest when train and test domains match.
- **Best cross-domain classification**: Stage 2 fine-tuned Florence-2. It improves RealWaste accuracy to **56.68%**, beating CLIP's **42.68%**.
- **Best cross-domain detection**: Stage 2 fine-tuned Florence-2. It reaches **50.49% F1** on Trash-ICRA19, above Grounding DINO's **37.20%**.
- **Best cross-domain segmentation**: Stage 2 fine-tuned Florence-2 on DWSD with **0.2214 mIoU** — 2.2× SAM (0.1023) and 2.6× Mask R-CNN (0.0842). Strikingly, in-domain and cross-domain segmentation mIoU are nearly identical (0.2223 vs 0.2214), indicating that the LoRA-fine-tuned Florence-2's polygon-token segmentation transfers across waste domains almost without degradation.
- **Stage 2 zero-shot Florence-2** is not the best overall, but it is useful as a baseline. It is competitive cross-domain even without fine-tuning: it beats Grounding DINO on ICRA19 detection (**42.03% F1** vs 37.20%) and beats SAM on DWSD segmentation (**0.1207 mIoU** vs 0.1023). This means **Florence-2 is the strongest cross-domain foundation model on two of the three tasks even before any waste-specific training**.
- **Stage 2 fine-tuning helps every Stage 2 metric** compared with zero-shot Florence-2 — the largest absolute gain is on in-domain classification (+58.81 pts), the largest cross-domain gain is on DWSD segmentation (+0.1007, an 83% relative improvement).

## Stage 2 Fine-Tuned Improvement Over Zero-Shot

| Dataset / Task | Metric | Zero-Shot | Fine-Tuned | Improvement |
|---|---|---:|---:|---:|
| TrashNet classification | Accuracy | 26.43% | 85.24% | +58.81 pts |
| RealWaste classification | Accuracy | 30.36% | 56.68% | +26.32 pts |
| TACO detection | F1 | 24.53% | 36.57% | +12.04 pts |
| Trash-ICRA19 detection | F1 | 42.03% | 50.49% | +8.46 pts |
| TACO segmentation, in-domain | mIoU | 0.1773 | 0.2223 | +0.0450 |
| DWSD segmentation, cross-domain | mIoU | 0.1207 | 0.2214 | +0.1007 (+83.4%) |

## Final Answer

If the goal is **highest in-domain accuracy**, use **Stage 1 task-specific models**.

If the goal is **best cross-domain generalization with one model**, use **Stage 2 fine-tuned Florence-2**.

If the goal is a concise project conclusion: **Stage 1 proves specialists win in familiar domains, while Stage 2 fine-tuned Florence-2 sweeps all three cross-domain benchmarks (classification, detection, segmentation) using a single 14 MB LoRA adapter — the cleanest evidence yet that foundation models with task-specific fine-tuning are the right choice for real-world waste-management deployment.**
