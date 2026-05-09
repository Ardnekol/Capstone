# Unified vs Stage 1 Baseline Matrix

| Task | Baseline Best | Baseline Metric | Florence-2 Zero-Shot | Delta |
|------|---------------|----------------|--------------------|-------|
| Classification (in-domain) | ViT-Base | 96.44% Accuracy | 26.43% Accuracy | -70.01 pts |
| Classification (cross-domain) | CLIP | 42.68% Accuracy | 30.36% Accuracy | -12.32 pts |
| Detection (in-domain) | YOLOv8 | 64.10% F1 | 24.53% F1 | -39.57 pts |
| Detection (cross-domain) | Grounding DINO | 37.20% F1 | 42.03% F1 | +4.83 pts |
| Segmentation (in-domain) | DeepLabV3+ | 0.4541 mIoU | 0.1773 mIoU | -0.2768 |

Notes:
- Delta = Unified - Baseline.
- Positive delta means unified model is better on that metric.
- Detection comparison uses F1 (not mAP) for apples-to-apples with unified evaluator output.
