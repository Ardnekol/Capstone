# AI for Waste Management: A Unified Foundation Model

Foundation models vs. task-specific models for waste vision, across
**classification, detection, and segmentation** — and a single **unified
Florence-2** model that does all three. This repository studies the
*lab-to-field gap*: models trained on clean, curated datasets degrade sharply in
the field, where lighting, clutter, and occlusion differ from training data.

**Headline result:** a single LoRA-fine-tuned Florence-2 is **best, or
statistically tied for best, on all 7 cross-domain benchmarks** across 5 distinct
waste domains — using one model and a lightweight adapter instead of many
task-specific pipelines.

---

## Repository structure

| Path | What it is |
|---|---|
| [`stage_1/`](stage_1/) | **Benchmark** — specialist vs. foundation models, in-domain vs. cross-domain, per task |
| [`stage_2/`](stage_2/) | **Unified model** — Florence-2 + LoRA for all three tasks, training and evaluation |
| [`stage_2/cross_domain_eval/`](stage_2/cross_domain_eval/) | 3-regime cross-domain evaluation on new datasets + consolidated results |
| [`Playground/`](Playground/) | **Interactive demo** — one web app serving the unified model for every task |
| [`paper_drafts/`](paper_drafts/) | IEEE conference paper (`waste_vision_ieee.tex`) |

> Large assets (datasets, model weights, `node_modules`) are git-ignored. See
> *Datasets* and *Setup* below to obtain them.

---

## Stage 1 — Benchmark: specialist vs. foundation

Each task is evaluated **in-domain** (curated training distribution) and
**cross-domain** (a held-out dataset from a different domain, never seen in
training). This quantifies the lab-to-field gap.

| Task | Task-specific models | Foundation model | In-domain → Cross-domain |
|---|---|---|---|
| Classification | ViT-Base, ResNet-50, EfficientNet-B0 | CLIP | specialists collapse, foundation robust |
| Detection | YOLOv8m, Faster R-CNN | Grounding DINO | same pattern |
| Segmentation | DeepLabV3+, U-Net, Mask R-CNN | SAM | same pattern |

**Finding:** specialists peak in-domain but lose **58–80%** of their quality
under domain shift; zero-shot foundation models generalize better but cap lower.
Code/reports under [`stage_1/`](stage_1/) (`classification/`, `detection/`,
`segmentation/`, `MINI_REPORT.md`).

## Stage 2 — Unified Florence-2 + LoRA

A single **Florence-2-large** model is fine-tuned with **LoRA** to perform all
three tasks through task prompts:

| Task | Prompt | Output |
|---|---|---|
| Classification | `<CAPTION>` | material label |
| Detection | `<OD>` / `<REGION_PROPOSAL>` | bounding boxes |
| Segmentation | referring expression / multi-instance cascade | masks |

- **Training data:** TrashNet (classification) + TACO (detection & segmentation),
  combined into one multi-task corpus. LoRA on attention projections (rank 16).
- **Crucially**, none of the cross-domain test sets are used in training.
- Key scripts: `finetune_florence2_od_lora.py`, `prepare_unified_multitask_jsonl.py`,
  `evaluate_unified_model.py`, `train_unified.sh`.
- Reports: `STAGE2_REPORT.md`, and the consolidated
  [`cross_domain_eval/PROJECT_MASTER_RESULTS.md`](stage_2/cross_domain_eval/PROJECT_MASTER_RESULTS.md).

### Results (cross-domain)

**Classification — accuracy %**

| Model | TrashNet *(in)* | RealWaste *(cross)* | WaRP-C *(cross)* |
|---|--:|--:|--:|
| ViT-Base (specialist) | **96.4** | 40.0 | 22.0 |
| CLIP (foundation) | 67.8 | 42.7 | 43.1 |
| **Florence-2 FT (unified)** | 85.2 | **56.7** | **60.4** |

**Detection — F1 @ IoU 0.5 (class-agnostic)**

| Model | TACO *(in)* | ICRA19 | ZeroWaste-f | WaRP-D |
|---|--:|--:|--:|--:|
| YOLOv8m (specialist) | **0.64** | 0.14 | 0.22 | 0.18 |
| Grounding DINO (foundation) | 0.25 | 0.37 | 0.21 | 0.19 |
| **Florence-2 FT (unified)** | 0.37 | **0.51** | **0.27** | **0.28** |

**Segmentation — binary mIoU**

| Model | TACO *(in)* | DWSD | ZeroWaste-f |
|---|--:|--:|--:|
| DeepLabV3+ (specialist) | **0.45** | 0.05 | 0.13 |
| Mask R-CNN (specialist) | 0.29 | 0.08 | 0.17 |
| SAM (foundation) | 0.04 | 0.10 | 0.03 |
| **Florence-2 FT (unified)** | 0.22 | **0.18** | 0.16 † |

† ZeroWaste-f: Florence 0.160 vs. Mask R-CNN 0.169 is a **statistical tie**
(paired Wilcoxon p=0.98, n=929).

→ **Florence-2 FT is best or tied-for-best on all 7 cross-domain benchmarks**
(2/2 classification, 3/3 detection, 2/2 segmentation incl. one tie).

### Run the cross-domain evaluation
```bash
cd stage_2/cross_domain_eval
bash run_all.sh          # all tasks, auto-selects the freest GPU
```
See [`cross_domain_eval/README.md`](stage_2/cross_domain_eval/README.md) for
per-task commands and the environment checker (`check_env.py`).

## Playground — interactive demo

A web app (FastAPI backend + React frontend) that serves the **single unified
model** for all tasks through one inference endpoint — upload an image, pick a
task (captioning/classification, detection, grounding, segmentation, OCR, or a
multi-instance cascade), and the same model and adapter produce the output, with
no model switching.

```bash
cd Playground
# backend
python backend/main.py
# frontend
cd frontend && npm install && npm run dev
```
See [`Playground/PLAN.md`](Playground/PLAN.md) and `Playground/how to run.txt`.

---

## Datasets

Training (curated): **TrashNet**, **TACO**.
Cross-domain test (field): **RealWaste**, **Trash-ICRA19**, **DWSD**,
**ZeroWaste-f**, **WaRP**.

Datasets are git-ignored due to size; download them into `datasets/` (test splits
only are used for cross-domain evaluation — never for training).

## Setup

Experiments run on an **NVIDIA A100-40GB**. The complete Python stack is the
`Capstone` conda environment (Python 3.11, torch 2.0.1+cu117, transformers 4.40.2,
peft 0.10.0, ultralytics, segmentation-models-pytorch, segment-anything,
open_clip).

```bash
# on the GPU node
srun -p cse-gpu-all --nodelist=dgx-a100-02 --pty bash
export PATH="$HOME/.conda/envs/Capstone/bin:$PATH"
```

## Paper

An IEEE conference paper is in [`paper_drafts/waste_vision_ieee.tex`](paper_drafts/waste_vision_ieee.tex)
(self-contained; compiles on Overleaf). Title: *AI for Waste Management: A Unified
Foundation Model across Classification, Detection, and Segmentation.*

## Authors

**Lokendra Mandloi** (cs24mtech11024) · Guided by **Prof. Srijith P. K.**
Dept. of Computer Science and Engineering, IIT Hyderabad.
