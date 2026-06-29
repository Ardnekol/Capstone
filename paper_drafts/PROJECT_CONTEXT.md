# Project Context — "AI for Waste Management" (IEEE DSAA paper)

> Attach this file (and the paper `waste_vision_ieee.tex`) to your Claude Project.
> It gives an assistant everything needed to edit the paper accurately without
> re-deriving context or inventing numbers.

---

## 1. What this project is

An MTech capstone studying the **lab-to-field gap** in waste vision: models
trained on clean, curated datasets fail under real-world conditions (lighting,
clutter, occlusion). It compares **task-specific specialists vs. foundation
models** across three tasks — **classification, detection, segmentation** — then
fine-tunes **one unified Florence-2 model with LoRA** that performs all three.

- **Target venue:** IEEE DSAA (IEEE conference format, `\documentclass[conference]{IEEEtran}`).
- **Main paper file:** `waste_vision_ieee.tex` (self-contained: pgfplots figures,
  inline `thebibliography`; the only external images are 3 optional Playground
  screenshots, currently placeholder boxes).
- **Authors:** Lokendra Mandloi (cs24mtech11024) and Prof. Srijith P. K., Dept. of
  Computer Science and Engineering, IIT Hyderabad.

## 2. Headline result — state it EXACTLY this way

A single LoRA-fine-tuned Florence-2 is **best, or statistically tied for best, on
all 7 cross-domain benchmarks** across 5 domains: **6 outright wins + 1 tie**.

> **Never** write "7/7 wins", "wins all 7", or "beats every model". The 7th
> (ZeroWaste-f segmentation) is a **statistical tie**, not a win.

## 3. The numbers (use verbatim; do not invent or alter)

**Classification — accuracy %** (TrashNet in-domain / RealWaste cross / WaRP-C cross)

| Model | TrashNet | RealWaste | WaRP-C |
|---|--:|--:|--:|
| ViT-Base | 96.44 | 39.98 | 21.99 |
| EfficientNet-B0 | 89.53 | 32.89 | 46.29 |
| ResNet-50 | 80.83 | 17.85 | 17.28 |
| CLIP | 67.83 | 42.68 | 43.07 |
| **Florence-2 FT** | 85.24 | 56.68 | 60.35 |

**Detection — F1 @ IoU 0.5, class-agnostic** (TACO in / ICRA19 / ZeroWaste-f / WaRP-D)

| Model | TACO | ICRA19 | ZeroWaste-f | WaRP-D |
|---|--:|--:|--:|--:|
| YOLOv8m | 0.641 | 0.139 | 0.220 | 0.184 |
| Faster R-CNN | 0.195 | 0.137 | 0.146 | 0.123 |
| Grounding DINO | 0.253 | 0.372 | 0.208 | 0.192 |
| **Florence-2 FT** | 0.366 | 0.505 | 0.272 | 0.281 |

**Segmentation — binary mIoU** (TACO in / DWSD / ZeroWaste-f)

| Model | TACO | DWSD | ZeroWaste-f |
|---|--:|--:|--:|
| DeepLabV3+ | 0.454 | 0.048 | 0.132 |
| U-Net | 0.329 | 0.063 | 0.131 |
| Mask R-CNN | 0.289 | 0.084 | 0.169 |
| SAM | 0.038 | 0.102 | 0.027 |
| **Florence-2 FT** | 0.222 | 0.180 | 0.160 |

**The tie:** ZeroWaste-f segmentation — Florence **0.160** vs Mask R-CNN **0.169**
is a statistical tie (paired Wilcoxon **p = 0.98**; bootstrap 95% CI of the
per-image IoU difference includes 0; **n = 929**).

## 4. Method (keep these facts consistent)

- **Florence-2-large + LoRA** (rank 16, alpha 32, dropout 0.05, 3 epochs) on the
  attention projection layers.
- **Trained ONLY on TrashNet (classification) + TACO (detection & segmentation).**
  It is **never** trained on any cross-domain test set. This is the integrity
  backbone of the paper — do not contradict it.
- Task prompts: `<CAPTION>` (classification), `<OD>` / `<REGION_PROPOSAL>`
  (detection), referring-expression / multi-instance cascade (segmentation).
- Detection scored with the class-agnostic region-proposal head; segmentation
  with the grounding -> per-region -> union cascade. One method per task, applied
  uniformly to every dataset of that task.
- Environment: NVIDIA A100-40GB; torch 2.0.1, transformers 4.40.2, peft 0.10.0.

## 5. Datasets

- **Training (curated):** TrashNet, TACO.
- **Cross-domain test (field):** RealWaste, Trash-ICRA19, DWSD, ZeroWaste-f, WaRP
  (WaRP-C for classification, WaRP-D for detection).
- **Five domains:** landfill, underwater, recycling-plant, cluttered-conveyor,
  dense-scene. Test splits only are used; never for training.

## 6. Metric definitions (state when quoting numbers)

- Classification = **accuracy** (also macro-F1). WaRP-C is plastic-heavy, so prefer
  **macro-F1** for fairness when discussing WaRP-C.
- Detection = **class-agnostic precision / recall / F1 at IoU 0.5**.
- Segmentation = **binary mean IoU (mIoU)**.

## 7. Key qualitative findings (safe to use)

- Specialists peak in-domain but lose **58–80%** of their quality cross-domain.
- Foundation models generalize better but cap lower in-domain (e.g., CLIP 67.8%).
- Florence-2 cross-domain **retention is 69–96%** vs specialists' **20–32%**
  (retention = mean cross-domain score / in-domain score).
- One model + a lightweight adapter replaces a stack of specialists (deployment
  advantage), demonstrated in a "Playground" web app (FastAPI + React).

## 8. Paper structure (current order)

Introduction -> Related Work -> Methodology (with a pipeline figure) ->
Experimental Setup -> Results (3 tables + classification-gap and retention
figures) -> Analysis and Discussion -> Demo and Deployment (Playground; 3 figures,
currently placeholders, placed after the references) -> Limitations ->
Conclusion -> References (21 entries).

## 9. Writing style (match this)

- Simple words, **short crisp sentences**, active voice ("We benchmark… We
  fine-tune… The result is…").
- IEEE rules: **no math or symbols in the title or abstract**; write "percent",
  not `%`; avoid em-dashes in the abstract.
- Honest framing everywhere: "best or statistically tied for best", never
  "beats all".

## 10. Do NOT do

- Do not claim 7/7 wins, or that Florence beats Mask R-CNN on ZeroWaste-f
  segmentation (it is a tie).
- Do not say the model was trained on any field / cross-domain dataset.
- Do not drop Mask R-CNN or any baseline to make results look better.
- Do not invent numbers — use the tables in Section 3 verbatim.

## 11. Known limitations / open items

- Florence's segmentation writes masks as coordinate tokens (not true pixel
  masks), so it loses detail on dense scenes — this is why a dedicated instance
  segmenter ties it on ZeroWaste-f.
- Some cross-domain baselines (RealWaste, Trash-ICRA19, DWSD) are reused from an
  earlier matched evaluation rather than recomputed in this run.
- Results are from single training runs (no multi-seed / error bars).
- The 3 Playground demo figures are placeholders; swap in `class.png`,
  `det.png`, `seg.png` when exported.
- DSAA may be double-blind — check the CFP; if so, anonymize the author block.
