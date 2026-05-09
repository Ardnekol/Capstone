# Unified vs Stage 1 Baseline Matrix

| Task | Baseline Best | Baseline Metric | Florence-2 Unified | Delta |
|------|---------------|----------------|--------------------|-------|
| Classification (in-domain) | ViT-Base | 96.44% Accuracy | 85.24% Accuracy | -11.20 pts |
| Classification (cross-domain) | CLIP | 42.68% Accuracy | 56.68% Accuracy | +14.00 pts |
| Detection (in-domain) | YOLOv8 | 64.10% F1 | 36.57% F1 | -27.53 pts |
| Detection (cross-domain) | Grounding DINO | 37.20% F1 | 50.49% F1 | +13.29 pts |
| Segmentation (in-domain) | DeepLabV3+ | 0.4541 mIoU | 0.2126 mIoU | -0.2415 |

Notes:
- Delta = Unified - Baseline.
- Positive delta means unified model is better on that metric.
- Detection comparison uses F1 (not mAP) for apples-to-apples with unified evaluator output.
