# Paper Drafts — Workshop Submission

Drafts for the workshop paper "Bridging the Lab-to-Field Gap in Waste Vision
with a Unified Foundation Model" (working title), based on the final
post-bug-fix experimental results from May 2026.

## Files

| File | Purpose |
|---|---|
| `abstract.md` | Draft abstract (~200 words) + alternative short version (~150 words). Includes notes on which claims are load-bearing. |
| `section_5_analysis.md` | Revised §5 (Analysis) with four subsections: cross-domain generalization, multi-task vs single-task ablation, deployment rationale, seg-only failure mode. |

## Status (as of post-DWSD evaluation)

- All cross-domain numbers are final. Florence-2 + LoRA sweeps all three cross-domain benchmarks.
- Multi-task ablation done: cls-only, det-only, unified. Seg-only excluded (mode-collapsed; documented as a finding).
- DWSD cross-domain segmentation: 0.2214 mIoU for unified, beating SAM's 0.1023 by 2.16×.

## What's NOT yet drafted

- Introduction / motivation
- Related work
- §3 (Method) — model, LoRA setup, training data, hyperparameters
- §4 (Results) — main tables
- §6 (Limitations) and §7 (Conclusion)

These can be drafted from the existing `stage_2/STAGE2_REPORT.md` and `stage_2/four_regime_comparison.md`, which are already consistent with the new findings.

## Workshop target

- **Primary:** NeurIPS 2026 Climate Change AI Workshop (Sept–Oct deadline)
- **Backup:** ICVGIP 2026 (Aug deadline)
