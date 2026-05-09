# Object Detection Experiments

## Overview

This directory contains all code for the **Object Detection** task comparing:
- **Task-Specific Models**: YOLOv8, Faster R-CNN, RetinaNet
- **Foundation Models**: Grounding-DINO, Florence-2

## Dataset Summary

### Training: TACO (Trash Annotations in Context)
- **Images**: 1,500
- **Annotations**: 4,784 bounding boxes
- **Categories**: 60 fine-grained → mapped to 7 super-categories
- **Format**: COCO JSON
- **Location**: `datasets/detection/taco/TACO/`

### Testing: Trash-ICRA19 (Underwater Debris)
- **Images**: 7,684 (train: 5,720, val: 820, test: 1,144)
- **Categories**: 
  - `plastic` (main trash class)
  - `bio` (biological objects)
  - `rov` (robot - ignore)
  - `timestamp` (ignore - watermark)
- **Format**: PASCAL VOC XML + YOLO TXT
- **Location**: `datasets/detection/trash_icra19/trash_ICRA19/dataset/`

## Class Mapping

### TACO → Unified Classes (7 categories)
| Super Category | TACO Categories |
|----------------|-----------------|
| **plastic** | Plastic bottle, Plastic bag, Plastic container, Straw, Cup, Lid, etc. |
| **paper** | Paper, Carton, Magazine, Newspaper, etc. |
| **metal** | Can, Aluminium foil, Metal bottle cap, Aerosol, etc. |
| **glass** | Glass bottle, Broken glass, etc. |
| **organic** | Food waste, etc. |
| **textile** | Shoe, Rope, etc. |
| **other** | Battery, Cigarette, etc. |

### Trash-ICRA19 → Unified Classes
| Original | Mapped | Notes |
|----------|--------|-------|
| plastic | **trash** | Main detection target |
| bio | **bio** | Biological (plants, animals) |
| rov | IGNORE | Robot in frame |
| timestamp | IGNORE | Watermark |
| metal | **trash** | Merge with plastic |
| wood | **trash** | Merge with plastic |

**For cross-domain evaluation**: We use **binary detection** (trash vs background) or map both datasets to common categories.

## Experiment Plan

### Phase 1: Train on TACO
```
TACO (1,500 images) → Train/Val split (80/20)
                    → Train task-specific models
                    → Evaluate on TACO val (in-domain)
```

### Phase 2: Cross-Domain Evaluation
```
Trained models → Evaluate on Trash-ICRA19 test (out-of-domain)
              → Compare generalization gap
```

### Phase 3: Foundation Model Evaluation
```
Grounding-DINO → Zero-shot on TACO val
              → Zero-shot on Trash-ICRA19 test
              → Compare with fine-tuned models
```

## Files

```
detection/
├── README.md                    # This file
├── config.yaml                  # Detection experiment config
├── dataset_utils.py             # Data loading utilities
├── convert_annotations.py       # Convert between formats
├── train_yolov8.py             # YOLOv8 training
├── train_fasterrcnn.py         # Faster R-CNN training
├── train_retinanet.py          # RetinaNet training
├── eval_grounding_dino.py      # Grounding-DINO zero-shot
├── eval_florence2.py           # Florence-2 zero-shot
├── evaluate.py                 # Unified evaluation script
└── results/                    # Output results
```

## Quick Start

```bash
# 1. Prepare datasets (convert annotations)
python convert_annotations.py

# 2. Train YOLOv8
python train_yolov8.py --epochs 50

# 3. Evaluate on both datasets
python evaluate.py --model yolov8 --dataset both

# 4. Run foundation model
python eval_grounding_dino.py --dataset both
```

## Expected Results

| Model | Type | TACO (val) mAP@0.5 | ICRA19 (test) mAP@0.5 | Δ Drop |
|-------|------|--------------------|-----------------------|--------|
| YOLOv8-M | Task-Specific | ~0.60 | ~0.35 | ~0.25 |
| Faster R-CNN | Task-Specific | ~0.55 | ~0.30 | ~0.25 |
| RetinaNet | Task-Specific | ~0.52 | ~0.28 | ~0.24 |
| Grounding-DINO | Foundation | ~0.45 | ~0.40 | ~0.05 |
| Florence-2 | Foundation | ~0.42 | ~0.38 | ~0.04 |

**Hypothesis**: Foundation models show smaller performance drop under domain shift.
