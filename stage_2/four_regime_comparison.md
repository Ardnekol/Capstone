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
| TACO segmentation, in-domain | mIoU | **0.4541** (DeepLabV3+) | 0.0380 (SAM) | 0.1773 | 0.2126 | Stage 1 task-specific |
| DWSD segmentation, cross-domain | mIoU | 0.0842 (Mask R-CNN) | **0.1023** (SAM) | Not evaluated | Not evaluated | Stage 1 foundation |

## Winner Summary

| Regime | Wins | Where It Wins |
|---|---:|---|
| Stage 1 Task-Specific | 3 | TrashNet classification, TACO detection, TACO segmentation |
| Stage 1 Foundation | 1 | DWSD cross-domain segmentation |
| Stage 2 Zero-Shot Florence-2 | 0 | No overall wins, but strong on ICRA19 detection |
| Stage 2 Fine-Tuned Florence-2 | 2 | RealWaste classification, Trash-ICRA19 detection |

Note: if only the five datasets evaluated in the current Stage 2 full-dataset runs are counted, Stage 1 task-specific wins three rows and Stage 2 fine-tuned Florence-2 wins two rows.

## Direct Insights

- **Best in controlled/in-domain settings**: Stage 1 task-specific models. ViT-Base, YOLOv8, and DeepLabV3+ are strongest when train and test domains match.
- **Best cross-domain classification**: Stage 2 fine-tuned Florence-2. It improves RealWaste accuracy to **56.68%**, beating CLIP's **42.68%**.
- **Best cross-domain detection**: Stage 2 fine-tuned Florence-2. It reaches **50.49% F1** on Trash-ICRA19, above Grounding DINO's **37.20%**.
- **Best cross-domain segmentation among evaluated Stage 1 models**: SAM on DWSD with **0.1023 mIoU**. Stage 2 DWSD segmentation was not present in the saved full-dataset results.
- **Stage 2 zero-shot Florence-2** is not the best overall, but it is useful as a baseline. Its strongest result is cross-domain detection on Trash-ICRA19 with **42.03% F1**.
- **Stage 2 fine-tuning helps every Stage 2 metric** compared with zero-shot Florence-2.

## Stage 2 Fine-Tuned Improvement Over Zero-Shot

| Dataset / Task | Metric | Zero-Shot | Fine-Tuned | Improvement |
|---|---|---:|---:|---:|
| TrashNet classification | Accuracy | 26.43% | 85.24% | +58.81 pts |
| RealWaste classification | Accuracy | 30.36% | 56.68% | +26.32 pts |
| TACO detection | F1 | 24.53% | 36.57% | +12.04 pts |
| Trash-ICRA19 detection | F1 | 42.03% | 50.49% | +8.46 pts |
| TACO segmentation | mIoU | 0.1773 | 0.2126 | +0.0353 |

## Final Answer

If the goal is **highest in-domain accuracy**, use **Stage 1 task-specific models**.

If the goal is **best cross-domain generalization with one model**, use **Stage 2 fine-tuned Florence-2**.

If the goal is a concise project conclusion: **Stage 1 proves specialists win in familiar domains, while Stage 2 fine-tuned Florence-2 gives the best unified cross-domain result across classification and detection.**
