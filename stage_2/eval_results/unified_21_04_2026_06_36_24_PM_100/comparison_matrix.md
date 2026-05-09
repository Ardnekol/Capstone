# Unified vs Stage 1 Baseline Matrix

| Task | Baseline Best | Baseline Metric | Florence-2 Unified | Delta |
|------|---------------|----------------|--------------------|-------|
| Classification (in-domain) | ViT-Base | 96.44% Accuracy | 80.00% Accuracy | -16.44 pts |
| Classification (cross-domain) | CLIP | 42.68% Accuracy | 56.94% Accuracy | +14.26 pts |
| Detection (in-domain) | YOLOv8 | 64.10% F1 | 21.70% F1 | -42.40 pts |
| Detection (cross-domain) | Grounding DINO | 37.20% F1 | 63.25% F1 | +26.05 pts |
| Segmentation (in-domain) | DeepLabV3+ | 0.4541 mIoU | 0.2329 mIoU | -0.2212 |

Notes:
- Delta = Unified - Baseline.
- Positive delta means unified model is better on that metric.
- Detection comparison uses F1 (not mAP) for apples-to-apples with unified evaluator output.
