# Trash-ICRA19 Cross-Domain Detection (class-agnostic IoU>=0.5)

Test images: **1120**

| Model | Regime | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| YOLOv8m | Task-Specific | 0.3825 | 0.2361 | 0.2920 |
| Faster R-CNN | Task-Specific | 0.2105 | 0.1596 | 0.1815 |
| Grounding DINO | Foundation | 0.4492 | 0.1913 | 0.2684 |
| Florence-2 + LoRA | Unified FT | 0.3294 | 0.4996 | 0.3970 |
