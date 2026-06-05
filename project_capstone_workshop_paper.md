---
name: Capstone Workshop Paper Plan
description: Plan for a workshop paper based on the Capstone thesis — framing, target venues, gap-closing experiments
type: project
originSessionId: 5969d37a-3057-46d7-9675-3d4f23035975
---
User is pursuing a workshop paper from their Capstone thesis (see `project_capstone_thesis.md`).

**Working title:** "Bridging the Lab-to-Field Gap in Waste Vision with a Unified Foundation Model"

**Pitch:** Specialist vision models trained on lab waste datasets lose 60-80% performance in the field; a single LoRA-tuned Florence-2 handling classification/detection/segmentation generalizes better than any specialist and is cheaper to deploy. Workshop-tier story uses ~90% of existing Stage 1 + Stage 2 work.

**Structure (6-8 pages):** intro → related work → benchmark (Stage 1 compressed) → method (Stage 2 unified Florence-2 + LoRA) → results → analysis (failure modes, latency, multi-task contribution) → limitations.

**Gap-closing experiments to add (~2-3 weeks):**
1. Multi-task ablation — train 3 single-task Florence-2 LoRAs vs unified one (must-have, ~1 day training)
2. Add ZeroWaste or TrashCan as third cross-domain test set (~2 days)
3. Latency/memory table: Florence-2 FPS+VRAM vs Stage 1 specialists (~half day)
4. Qualitative failure grid: 12-16 images with captions (~half day)
5. Optional: confidence/abstention recovery curve (~1 day)

**Target venues (deadlines reckoned from 2026-05-19):**
- Primary: **NeurIPS 2026 Climate Change AI Workshop** (~Sept-Oct 2026 deadline) — best fit for sustainability framing
- Backup: **ICVGIP 2026** (~Aug 2026 deadline) — applied vision, friendly reviewing
- Later option: **WACV 2027 Workshops** (~Oct 2026) or **CVPR 2027 Workshops** (~Mar 2027)
- Skipped: ECCV 2026 workshops (already past)

**Why:** User wanted to publish their MTech thesis work; full-conference novelty bar is too high, but workshop is achievable with existing material plus light additional experiments. Deployment angle (25 MB LoRA adapters, one model vs 15) is the reviewer-friendly hook.

**How to apply:** When user works on the paper, prioritize the 5 add-on experiments in order — #1 (multi-task ablation) is non-negotiable for defensibility, #2 strengthens generalization claim, #3 sells deployment, #4 is qualitative polish, #5 is bonus. Verify workshop deadlines via official CFP before committing — these are typical-year estimates, not confirmed for 2026.
