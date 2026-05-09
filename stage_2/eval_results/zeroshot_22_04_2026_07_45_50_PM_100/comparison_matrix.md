# Unified vs Stage 1 Baseline Matrix

| Task | Baseline Best | Baseline Metric | Florence-2 Zero-Shot | Delta |
|------|---------------|----------------|--------------------|-------|
| Classification (in-domain) | ViT-Base | 96.44% Accuracy | 18.00% Accuracy | -78.44 pts |
| Classification (cross-domain) | CLIP | 42.68% Accuracy | 34.72% Accuracy | -7.96 pts |
| Detection (in-domain) | YOLOv8 | 64.10% F1 | 14.87% F1 | -49.23 pts |
| Detection (cross-domain) | Grounding DINO | 37.20% F1 | 65.86% F1 | +28.66 pts |
| Segmentation (in-domain) | DeepLabV3+ | 0.4541 mIoU | 0.2381 mIoU | -0.2160 |

Notes:
- Delta = Unified - Baseline.
- Positive delta means unified model is better on that metric.
- Detection comparison uses F1 (not mAP) for apples-to-apples with unified evaluator output.
