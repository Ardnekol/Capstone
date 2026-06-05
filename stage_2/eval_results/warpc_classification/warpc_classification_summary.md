# WaRP-C Cross-Domain Classification (3-regime)

Test images (mapped to TrashNet space): **1551**

Mapping: `{'bottle': 'plastic', 'cans': 'metal', 'cardboard': 'cardboard', 'detergent': 'plastic', 'canister': 'plastic'}`

| Model | Regime | Accuracy | Macro-F1 |
|---|---|---:|---:|
| ViT-Base | Task-Specific | 21.99% | 0.2152 |
| ResNet-50 | Task-Specific | 17.28% | 0.1635 |
| EfficientNet-B0 | Task-Specific | 46.29% | 0.3461 |
| CLIP ViT-B/16 | Foundation | 43.07% | 0.3949 |
| Florence-2 + LoRA | Unified FT | 60.35% | 0.4148 |
