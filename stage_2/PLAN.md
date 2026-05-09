# Stage 2 Plan: Unified Florence-2 Multi-Task Model for Waste Management

## Goal

Fine-tune **one single Florence-2 model** (via LoRA) that performs **all three tasks**:
- **Classification** — What type of waste is this?
- **Object Detection** — Where is the waste in the image?
- **Segmentation** — Pixel-level mask of the waste

One model, one LoRA adapter, three tasks.

---

## Current State

| Asset | Status | Details |
|-------|--------|---------|
| `taco_od_train.jsonl` | ✅ Done | 1,492 records, `<OD>` prefix, TACO bounding boxes |
| `trashnet_caption_train.jsonl` | ✅ Done | 2,274 records, `<CAPTION>` prefix, class labels |
| `trashnet_caption_val.jsonl` | ✅ Done | 253 records, validation split |
| `florence2_taco_od_lora/` | ✅ Done | OD-only LoRA (Florence-2-large-ft base) |
| `florence2_trashnet_caption_lora/` | ✅ Done | Caption-only LoRA (Florence-2-base) |
| Segmentation training data | ❌ Missing | No JSONL for `<REFERRING_EXPRESSION_SEGMENTATION>` |
| Unified multi-task JSONL | ❌ Missing | No combined dataset |
| Unified LoRA checkpoint | ❌ Missing | No single adapter for all 3 tasks |
| Unified evaluation script | ❌ Missing | No script to evaluate all tasks & compare with Stage 1 |

---

## Data Available

### TACO (Detection + Segmentation)
- **1,500 images** with 4,784 annotations
- Each annotation has: bounding box (COCO xywh) + **polygon segmentation** + category (60 classes)
- Location: `datasets/detection/taco/TACO/data/`
- Annotations: `annotations.json` (COCO format with polygon segmentation)
- Also: pre-split binary masks in `stage_1/segmentation/data/taco/` (1,050 train images)

### TrashNet (Classification only)
- **2,527 images** across 6 classes: cardboard, glass, metal, paper, plastic, trash
- Location: `datasets/classification/trashnet/dataset-preprocessed/`
- No bounding boxes or masks (whole-image classification)

---

## Plan

### Step 1: Prepare Segmentation JSONL from TACO

**New script**: `prepare_taco_florence2_seg_jsonl.py`

TACO annotations already have polygon segmentation data in COCO format.
Florence-2's `<REFERRING_EXPRESSION_SEGMENTATION>` task takes a text phrase as input
and produces polygon coordinates. For fine-tuning, the suffix format is:

```
<REFERRING_EXPRESSION_SEGMENTATION> → suffix: "label<loc_x1><loc_y1><loc_x2><loc_y2>...<loc_xn><loc_yn>"
```

Where the polygon points are quantized to 1000 bins (same as OD).

The script will:
- Read TACO COCO annotations (same source as OD script)
- Convert polygon segmentation coordinates to Florence-2 location tokens
- Group by label per image
- Output: `finetune_data/taco_seg_train.jsonl`

### Step 2: Combine All Task JSONLs into Unified Training Data

**New script**: `prepare_unified_multitask_jsonl.py`

Merges all three task JSONLs into one shuffled training file:
- `taco_od_train.jsonl` (1,492 records, `<OD>`)
- `trashnet_caption_train.jsonl` (2,274 records, `<CAPTION>`)
- `taco_seg_train.jsonl` (~1,050+ records, `<REFERRING_EXPRESSION_SEGMENTATION>`)

Output: `finetune_data/unified_multitask_train.jsonl`

Optional: balance task proportions via oversampling the smaller tasks.

Also creates a small validation split: `finetune_data/unified_multitask_val.jsonl`

### Step 3: Fine-Tune Unified LoRA

Use the **existing** `finetune_florence2_od_lora.py` — it already supports any (prefix, suffix) pair.

```bash
python finetune_florence2_od_lora.py \
  --model-id microsoft/Florence-2-large-ft \
  --train-jsonl finetune_data/unified_multitask_train.jsonl \
  --eval-jsonl  finetune_data/unified_multitask_val.jsonl \
  --output-dir  finetuned/florence2_unified_multitask_lora \
  --num-train-epochs 3 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-4 \
  --lora-r 16 --lora-alpha 32
```

The existing `JsonlODDataset` + `FlorenceCollator` don't care what the prefix is —
they just pass `prefix` and `suffix` to the processor. The model learns to dispatch
based on the task prefix token.

**No changes needed** to the fine-tuning script. It already works for multi-task.

### Step 4: Evaluate Unified Model on All Tasks

**New script**: `evaluate_unified_model.py`

Runs the unified LoRA model on all evaluation datasets:

| Task | In-Domain Dataset | Cross-Domain Dataset |
|------|-------------------|---------------------|
| Classification | TrashNet (val split) | RealWaste |
| Detection | TACO (test split) | Trash-ICRA19 |
| Segmentation | TACO Seg (test split) | BePLi / DWSD |

Computes metrics:
- Classification: accuracy, macro F1, per-class F1
- Detection: mAP@0.5, mAP@0.5:0.95, precision, recall
- Segmentation: mIoU, pixel accuracy

Compares against Stage 1 baselines and generates a summary report.

### Step 5: Train Script (convenience wrapper)

**New script**: `train_unified.sh`

One-command wrapper to run the full pipeline:
1. Prepare segmentation JSONL
2. Combine into unified JSONL
3. Fine-tune unified LoRA
4. Evaluate on all datasets

---

## File Inventory (New Files)

```
stage_2/
├── PLAN.md                                      ← This file
├── prepare_taco_florence2_seg_jsonl.py           ← Step 1 (NEW)
├── prepare_unified_multitask_jsonl.py            ← Step 2 (NEW)
├── evaluate_unified_model.py                    ← Step 4 (NEW)
├── train_unified.sh                             ← Step 5 (NEW)
├── finetune_florence2_od_lora.py                 ← Step 3 (EXISTING, no changes)
├── finetune_data/
│   ├── taco_od_train.jsonl                      ← Existing
│   ├── trashnet_caption_train.jsonl             ← Existing
│   ├── trashnet_caption_val.jsonl               ← Existing
│   ├── taco_seg_train.jsonl                     ← Step 1 output (NEW)
│   ├── unified_multitask_train.jsonl            ← Step 2 output (NEW)
│   └── unified_multitask_val.jsonl              ← Step 2 output (NEW)
└── finetuned/
    ├── florence2_taco_od_lora/                   ← Existing (OD only)
    ├── florence2_trashnet_caption_lora/          ← Existing (caption only)
    └── florence2_unified_multitask_lora/         ← Step 3 output (NEW)
```

---

## Key Design Decisions

1. **Base model**: `microsoft/Florence-2-large-ft` — best pre-trained variant
2. **LoRA rank**: 16 (same as existing, proven to work)
3. **Multi-task training**: Simple JSONL interleaving — Florence-2 dispatches on prefix token
4. **No segmentation ground-truth masks needed for Florence-2**: It uses polygon format natively
5. **TACO provides data for 2 of 3 tasks** (OD + segmentation), TrashNet provides classification
