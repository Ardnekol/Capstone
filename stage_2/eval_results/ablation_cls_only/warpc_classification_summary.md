# WaRP-C Cross-Domain Classification (3-regime)

Test images (mapped to TrashNet space): **1551**

Mapping: `{'bottle': 'plastic', 'cans': 'metal', 'cardboard': 'cardboard', 'detergent': 'plastic', 'canister': 'plastic'}`

| Model | Regime | Accuracy | Macro-F1 |
|---|---|---:|---:|
| Florence-2 + LoRA | Unified FT | 39.97% | 0.3307 |
