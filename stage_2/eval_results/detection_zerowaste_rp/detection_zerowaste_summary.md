# ZeroWaste-f Cross-Domain Detection (3-regime, class-agnostic IoU>=0.5)

Test images: **929**

| Model | Regime | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| YOLOv8m | Task-Specific | 0.1789 | 0.2864 | 0.2202 |
| Grounding DINO | Foundation | 0.1681 | 0.2744 | 0.2084 |
| Florence-2 + LoRA | Unified FT | 0.2320 | 0.3272 | 0.2715 |
