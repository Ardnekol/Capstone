# Deployment cost: one unified model vs specialist stack

| | Models | Peak VRAM (MB) | Cls (ms) | Det (ms) | Seg (ms) | On-disk |
|---|---|---:|---:|---:|---:|---:|
| Unified Florence-2 | 1 | 1796 | 262 | 326 | 219 | 14 MB adapter |
| Specialist stack | 3 | 768 | 5 | 14 | 10 | 552 MB weights |
