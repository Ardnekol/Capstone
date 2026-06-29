# DWSD Cross-Domain Segmentation (binary mIoU) — Florence: cascade

Test pairs: **144**

| Model | Regime | mIoU | Pixel Acc |
|---|---|---:|---:|
| DeepLabV3+ | Task-Specific | 0.1515 | 0.8612 |
| U-Net | Task-Specific | 0.1264 | 0.8594 |
| Mask R-CNN | Task-Specific | 0.1564 | 0.8553 |
| SAM ViT-H | Foundation | 0.1494 | 0.5749 |
| Florence-2 + LoRA | Unified FT | 0.1799 | 0.6366 |
