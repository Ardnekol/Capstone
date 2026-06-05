# Workshop Paper — Next Steps

**Working title:** Bridging the Lab-to-Field Gap in Waste Vision with a Unified Foundation Model

**Pitch:** Specialist vision models trained on lab waste datasets lose 60–80% of their performance in the field; a single LoRA-tuned Florence-2 handling classification, detection, and segmentation generalizes better than any specialist and is cheaper to deploy.

**Target venues (verify CFPs before committing):**
- **Primary:** NeurIPS 2026 Climate Change AI Workshop (~Sept–Oct 2026 deadline)
- **Backup:** ICVGIP 2026 (~Aug 2026 deadline)
- **Later option:** WACV 2027 / CVPR 2027 sustainability or applications workshops

---

## Critical Rules (do not break)

- **Never fine-tune on RealWaste, Trash-ICRA19, or any cross-domain test set.** That destroys the whole story.
- **Pick a side for DWSD** — either training data or test data, not both.
- **Train splits only** when adding ZeroWaste, TrashCan, etc. Keep their test splits clean for separate evaluation.

---

## Paper Structure (6–8 pages)

1. Intro — waste sorting matters, lab-to-field gap is the obstacle
2. Related work — domain generalization, VLMs, waste-vision datasets
3. Benchmark (Stage 1, compressed) — 6 models × 3 tasks × in/cross-domain
4. Method (Stage 2) — unified JSONL, LoRA on Florence-2, polygon quantization
5. Results — main comparison + per-class breakdown + cross-domain win
6. Analysis — failure modes, latency, multi-task contribution
7. **Deployment & Demo (new)** — Playground architecture + side-by-side comparison
8. Limitations & future work

---

## Experiments — Prioritized

### Tier 1: must-have for defensibility

- [ ] **E1. Multi-task ablation** — train 3 single-task Florence-2 LoRAs (cls-only, det-only, seg-only) and compare to the unified model on all 5 benchmarks. Proves multi-task training contributes, not just convenience. *Effort: ~1 day training + ~half day eval.*
- [x] **E2. Third cross-domain test set** — DONE (2026-06-01). Added ZeroWaste-f + WaRP (C/D) as new cross-domain benchmarks across classification, detection, and segmentation. 3-regime eval in `stage_2/cross_domain_eval/` (uniform class-agnostic metric). Result: Florence-2-FT best on **6/7** cross-domain benchmarks (all specialist cells now filled). Detection scored with Florence's class-agnostic `<REGION_PROPOSAL>` head (applied uniformly): ZeroWaste-f F1 0.189→0.272 (now > YOLO 0.220), WaRP-D 0.281 (still wins). Result framed as **best-or-tied on all 7** (6 wins + 1 tie). The 7th, **ZeroWaste-f segmentation**, is a **statistical tie**: Florence multi-instance cascade 0.160 vs Mask R-CNN 0.169, paired Wilcoxon p=0.98, bootstrap 95% CI of diff includes 0 (n=929). Cascade applied uniformly (DWSD cascade 0.180, still wins). Significance test: `diag_seg_significance.py`. Follow-up: re-run ICRA19 with `<REGION_PROPOSAL>` for full consistency (already wins at 0.505 with `<OD>`). See `cross_domain_eval/PROJECT_MASTER_RESULTS.md`.
- [ ] **E3. Latency / memory table** — measure FPS and VRAM for Florence-2 unified vs Stage 1 specialists on the same GPU. Sells the deployment angle. *Effort: ~half day.*
- [ ] **E4. Qualitative failure grid** — 12–16 images showing wins and losses with brief captions. *Effort: ~half day.*

### Tier 2: high-leverage additions

- [ ] **E5. Field-data mini-study ("IITH-Waste-50")** — collect 50–100 phone photos of real waste from campus bins / mess / hostel. Run through Playground. Report numbers + qualitative grid. Releasing this dataset is a real workshop contribution. *Effort: 1 afternoon collection + 1 day eval.*
- [ ] **E6. Playground compare mode** — model selector in UI with side-by-side **Florence-2-large (base)** vs **Florence-2-FT** outputs on the same image. Becomes the hero figure of the paper. *Effort: ~1 day backend + ~half day frontend.*
- [ ] **E7. Demo video** — 2–3 min screencap of Playground running cls/det/seg on cross-domain images. Supplementary material. *Effort: ~2 hours after E6.*

### Tier 3: optional / time-permitting

- [ ] **E8. Confidence / abstention recovery curve** — use Florence-2 token log-probs as uncertainty; show abstaining on bottom-X% restores near-in-domain accuracy. *Effort: ~1 day.*
- [ ] **E9. FT v2 on broader data** — train second LoRA variant adding ZeroWaste-train + TrashCan-train (only if datasets already downloaded). Compare FT v1 vs v2 across all benchmarks. *Effort: ~1 day training + ~1 day eval.* Only do if E1–E5 are complete.

---

## Playground Status & Sub-Checklist (for E6)

### Already done ✅
- FastAPI backend with `/api/infer`, `/api/health`, `/api/tasks`, `/api/models`
- Multi-model architecture in place: `model_id` param on `/api/infer`, `FlorenceModel._models` dict caches multiple loaded models
- 4 hub variants registered (Florence-2 large / large-ft / base / base-ft)
- 17 tasks across 6 groups including custom multi-instance segmentation cascade
- flash_attn workaround for clean loads, GPU memory reporting in `/api/health`
- Full React + TS frontend: Header, Sidebar, HistoryDrawer, ImageDropzone, TaskPanel, ResultPanel, AnnotatedImage, BboxOverlay, JsonViewer, ExportButton, ImageCompare, Lightbox, StructuredOutput
- `useInference` hook already takes `modelId` — plumbed end-to-end

### Sub-tasks remaining (ordered by paper-value-per-hour)

- [ ] **P1. PEFT loading for the unified FT LoRA** (~3-4 hr) — unlocks everything else
  - Add a `local_lora` entry type to `AVAILABLE_MODELS` with metadata `{id, label, base_model, adapter_path}`
  - Adapter path: `~/Capstone/stage_2/finetuned/florence2_unified_multitask_lora/`
  - Extend `FlorenceModel.load()` to detect LoRA entries and use `PeftModel.from_pretrained(base, adapter_path)`
  - Smoke test via `/api/infer` on a TrashNet sample
- [ ] **P2. Sample image gallery** (~2 hr) — makes the demo reproducible
  - Bundle ~10-15 curated images in `frontend/public/samples/` (mix of TrashNet, RealWaste, ICRA19, and IITH-Waste-50 once collected)
  - Add a "Sample Images" panel beside ImageDropzone for one-click loading
- [ ] **P3. `/api/compare` endpoint + UI** (~4-5 hr) — produces the hero figure
  - New endpoint `POST /api/compare` accepting `(image, task, model_a, model_b, text_input)`, returns `{model_a: {...}, model_b: {...}}`
  - Audit existing `ImageCompare.tsx` — repurpose if it's input-vs-output, or add a new `ModelCompareView` component
  - Add model selector (radio: Base / FT / Compare) in TaskPanel or Header
  - Two-column compare layout reusing `AnnotatedImage` and `JsonViewer`
- [ ] **P4. Demo video** (~2 hr, after P1-P3) — supplementary material
  - 2-3 min screencap walking through cls / det / seg on cross-domain images using Compare mode
- [ ] **P5. Latency measurement helper** (~1 hr) — feeds E3 in the main experiment list
  - Script that hits `/api/infer` with N sample images for each model+task combo, logs `processing_time_ms`, dumps CSV for the paper's latency table

### Compare-mode UI sketch

**UI:**
```
Model: [● Compare]  [○ Base only]  [○ FT only]
┌──────────────────────┬─────────────────────────────┐
│  Florence-2-large    │  Florence-2-FT (waste)      │
│  [annotated image]   │  [annotated image]          │
│  [JSON output]       │  [JSON output]              │
└──────────────────────┴─────────────────────────────┘
```

**Backend changes:**
- Add `model_id` param to `/api/infer`
- Load both models at startup (~6 GB VRAM total in float16 — A100 handles it)
- New endpoint `/api/compare` that runs both in parallel and returns both outputs

**Frontend changes:**
- Model selector component (radio or dropdown)
- Two-column compare layout reusing existing `AnnotatedImage` and `JsonViewer`

---

## Writing Plan

1. Finish Tier 1 experiments first (E1–E4) → results section becomes complete.
2. Do E5 + E6 + E7 → deployment section + hero figure + supplementary video.
3. Draft abstract + intro once Tier 1 numbers are in hand.
4. Full draft → internal review → revisions → submission.

**Rough timeline (target NeurIPS CCAI Sept-Oct 2026 deadline):**
- Weeks 1–2: E1–E4
- Week 3: E5 (data collection) + E6 (Playground compare)
- Week 4: E7 (video) + start writing
- Weeks 5–6: full draft
- Week 7: revisions + final submission

---

## Open Questions to Resolve

- [ ] Confirm exact NeurIPS CCAI 2026 deadline when CFP drops
- [ ] Decide DWSD: training data or test data?
- [ ] Decide whether to do E9 (FT v2) — depends on time after Tier 1+2
- [ ] Co-authors and acknowledgement list





2 selectable models (what I designed)
florence2-large (= #1)
florence2-large-waste-ft (= #2 + #3, your contribution)
Pros: clean demo, simple hero figure, ~3.2 GB VRAM
Cons: mixes two effects — "general FT" + "waste FT" — into one bucket. A picky reviewer could ask "how much improvement comes from #2 alone vs your LoRA?"
