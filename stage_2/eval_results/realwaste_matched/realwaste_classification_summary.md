# RealWaste Cross-Domain Classification (matched 3-regime)

Test images: **3587**

| Model | Regime | Accuracy | Macro-F1 |
|---|---|---:|---:|
| ViT-Base | Task-Specific | 50.29% | 0.4554 |
| ResNet-50 | Task-Specific | 23.42% | 0.1825 |
| EfficientNet-B0 | Task-Specific | 44.35% | 0.3986 |
| CLIP ViT-B/16 | Foundation | 58.02% | 0.5208 |
| Florence-2 + LoRA | Unified FT | 58.29% | 0.5686 |
