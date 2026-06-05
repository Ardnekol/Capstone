# E1 — Multi-Task Ablation Comparison

Each column is a Florence-2-large-ft LoRA trained on a single task (or all). Same hyperparams across all three runs.


| Benchmark / Metric | cls-only | det-only | unified | Best |
|---|---|---|---|---|
| TrashNet cls (in-domain) — accuracy | 0.8567 | 0.4377 | 0.8480 | **cls-only** |
| TrashNet cls (in-domain) — macro F1 | 0.7191 | 0.3830 | 0.6941 | **cls-only** |
| RealWaste cls (cross-domain) — accuracy | 0.5944 | 0.3953 | 0.5838 | **cls-only** |
| RealWaste cls (cross-domain) — macro F1 | 0.5293 | 0.3452 | 0.4881 | **cls-only** |
| TACO det (in-domain) — F1 | 0.2409 | 0.3469 | 0.3657 | **unified** |
| TACO det (in-domain) — precision | 0.3805 | 0.5062 | 0.5288 | **unified** |
| TACO det (in-domain) — recall | 0.1762 | 0.2638 | 0.2795 | **unified** |
| ICRA19 det (cross-domain) — F1 | 0.4145 | 0.5886 | 0.5049 | **det-only** |
| TACO seg (in-domain) — mIoU | 0.1492 | 0.3237 | 0.2223 | **det-only** |
| TACO seg (in-domain) — pixel acc | 0.7295 | 0.8868 | 0.9180 | **unified** |
| DWSD seg (cross-domain) — mIoU | 0.2067 | 0.2257 | 0.2214 | **det-only** |
| DWSD seg (cross-domain) — pixel acc | 0.5373 | 0.5824 | 0.6791 | **unified** |

## Notes

- Best column per row is bolded.
- Dashes mean that LoRA's eval didn't produce that metric (missing dir, eval skipped, or DWSD seg not run for that LoRA).
- DWSD seg rows use eval_results/dwsd_<lora>/results.json (unified uses dwsd_unified_v3).
- Cross-domain rows (RealWaste cls, ICRA19 det, DWSD seg) are what reviewers care about for the lab-to-field gap claim.
