# WaRP-D Cross-Domain Detection (3-regime, class-agnostic IoU>=0.5)

Test images: **522**

| Model | Regime | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| YOLOv8m | Task-Specific | 0.1243 | 0.3507 | 0.1835 |
| Grounding DINO | Foundation | 0.1335 | 0.3385 | 0.1915 |
| Florence-2 + LoRA | Unified FT | 0.1959 | 0.4965 | 0.2810 |
