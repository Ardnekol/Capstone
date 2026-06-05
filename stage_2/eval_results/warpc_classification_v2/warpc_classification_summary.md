# WaRP-C Cross-Domain Classification (3-regime)

Test images (mapped to TrashNet space): **230**

Mapping: `{'bottle': 'plastic', 'cans': 'metal', 'cardboard': 'cardboard', 'detergent': 'plastic', 'canister': 'plastic'}`

| Model | Regime | Accuracy | Macro-F1 |
|---|---|---:|---:|
| Florence-2 + LoRA | Unified FT | 41.74% | 0.3964 |
