# Foundation Models vs Task-Specific Models: A Systematic Generalization Study

## Research Objective

**A systematic, task-wise evaluation of foundation models versus classical task-specific models under cross-dataset distribution shift for classification, detection, and segmentation in waste management domain.**

---

## 1. Research Hypothesis

> Foundation models exhibit superior cross-domain generalization compared to task-specific models when evaluated under distribution shift, while task-specific models achieve higher in-domain performance.

### Sub-hypotheses:
1. **H1 (Classification)**: Foundation models (CLIP, DINOv2) show smaller accuracy degradation than task-specific models (ResNet, EfficientNet) when tested on out-of-distribution waste images.
2. **H2 (Detection)**: Foundation models (Grounding-DINO, Florence-2) maintain higher mAP scores under domain shift compared to task-specific detectors (YOLO, Faster R-CNN).
3. **H3 (Segmentation)**: Foundation models (SAM, Florence-2) produce more consistent IoU scores across domains than task-specific segmentation networks (U-Net, DeepLabV3+).

---

## 2. Dataset Selection (Open-Source Only)

### Selection Criteria:
- ✅ Fully open-source with permissive licenses
- ✅ Strong domain shift potential between train/test
- ✅ Widely accepted in academic literature
- ✅ High-quality annotations

---

### 📊 TASK 1: CLASSIFICATION

| Role | Dataset | Description | License | Classes | Images |
|------|---------|-------------|---------|---------|--------|
| **Train** | TrashNet | Lab-isolated waste objects on white background | MIT | 6 | ~2,500 |
| **Test** | RealWaste | Real-world waste images with cluttered backgrounds | CC BY 4.0 | 9 | ~4,752 |

**Domain Shift**: 
- TrashNet: Clean, controlled lighting, single objects
- RealWaste: Noisy backgrounds, variable lighting, occlusion

**Class Mapping** (for cross-dataset evaluation):
| TrashNet | RealWaste |
|----------|-----------|
| glass | Glass |
| paper | Paper |
| cardboard | Cardboard |
| plastic | Plastic |
| metal | Metal |
| trash | Miscellaneous Trash |

---

### 📊 TASK 2: OBJECT DETECTION

| Role | Dataset | Description | License | Classes | Images |
|------|---------|-------------|---------|---------|--------|
| **Train** | TACO | Trash Annotations in Context (urban litter) | CC BY 4.0 | 60 → mapped to ~10 | ~1,500 |
| **Test** | Trash-ICRA19 | Underwater marine debris | Academic | 3 (bio, ROV, trash) | ~5,700 |

**Domain Shift**:
- TACO: Urban/terrestrial, outdoor lighting, varied backgrounds
- Trash-ICRA19: Underwater, blue-green color cast, marine environment

**Why This Shift is Strong**:
- Complete environmental change (land → water)
- Color distribution shift (natural → underwater)
- Object appearance change (same items look different underwater)

---

### 📊 TASK 3: SEGMENTATION

| Role | Dataset | Description | License | Classes | Images |
|------|---------|-------------|---------|---------|--------|
| **Train** | TACO (Segmentation) | Pixel-level waste annotations | CC BY 4.0 | 60 → mapped | ~1,500 |
| **Test** | BePLi v1 | Beach plastic litter segmentation | CC BY 4.0 | 4 | ~3,647 |

**Domain Shift**:
- TACO: Urban streets, parks, indoor
- BePLi: Beach/coastal, sand backgrounds, weathered plastics

---

## 3. Model Selection

### Design Principle
For each task, we compare:
- **2-3 Task-Specific Models**: Traditional architectures trained from scratch or fine-tuned
- **2 Foundation Models**: Pre-trained on large-scale data, evaluated with minimal/no fine-tuning

---

### 🧠 CLASSIFICATION MODELS

| Type | Model | Backbone | Pre-training | Fine-tuning Strategy |
|------|-------|----------|--------------|---------------------|
| Task-Specific | ResNet-50 | CNN | ImageNet-1K | Full fine-tune on TrashNet |
| Task-Specific | EfficientNet-B0 | CNN | ImageNet-1K | Full fine-tune on TrashNet |
| Task-Specific | ViT-Base | Transformer | ImageNet-21K | Full fine-tune on TrashNet |
| Foundation | CLIP (ViT-B/16) | Transformer | 400M image-text pairs | Zero-shot / Linear probe |
| Foundation | DINOv2 (ViT-B/14) | Transformer | LVD-142M | Linear probe only |

---

### 🧠 OBJECT DETECTION MODELS

| Type | Model | Backbone | Pre-training | Fine-tuning Strategy |
|------|-------|----------|--------------|---------------------|
| Task-Specific | YOLOv8-M | CSPDarknet | COCO | Fine-tune on TACO |
| Task-Specific | Faster R-CNN | ResNet-50-FPN | COCO | Fine-tune on TACO |
| Task-Specific | RetinaNet | ResNet-50-FPN | COCO | Fine-tune on TACO |
| Foundation | Grounding-DINO | Swin-T | O365 + GoldG | Zero-shot with text prompts |
| Foundation | Florence-2-Large | DaViT | FLD-5B | Zero-shot / Fine-tune |

---

### 🧠 SEGMENTATION MODELS

| Type | Model | Backbone | Pre-training | Fine-tuning Strategy |
|------|-------|----------|--------------|---------------------|
| Task-Specific | U-Net | ResNet-34 | ImageNet | Fine-tune on TACO |
| Task-Specific | DeepLabV3+ | ResNet-101 | ImageNet + Cityscapes | Fine-tune on TACO |
| Task-Specific | Mask R-CNN | ResNet-50-FPN | COCO | Fine-tune on TACO |
| Foundation | SAM (ViT-H) | ViT-H | SA-1B (1B masks) | Zero-shot (auto-mask) |
| Foundation | Florence-2-Large | DaViT | FLD-5B | Zero-shot / Fine-tune |

---

## 4. Evaluation Metrics

### 📏 Classification Metrics

```python
# Primary Metrics
- Top-1 Accuracy: Correct predictions / Total samples
- Top-5 Accuracy: Target in top 5 predictions / Total samples
- Per-class Accuracy: Accuracy breakdown by class

# Generalization Metrics
- Accuracy Drop (Δ): Acc(train_domain) - Acc(test_domain)
- Relative Drop (%): (Δ / Acc(train_domain)) × 100
```

### 📏 Object Detection Metrics

```python
# Primary Metrics
- mAP@0.5: Mean Average Precision at IoU threshold 0.5
- mAP@0.5:0.95: Mean AP across IoU thresholds [0.5, 0.55, ..., 0.95]
- AP per class: Per-category average precision

# Generalization Metrics
- mAP Drop (Δ): mAP(train_domain) - mAP(test_domain)
- Localization Accuracy: Measures bounding box precision
```

### 📏 Segmentation Metrics

```python
# Primary Metrics
- mIoU: Mean Intersection over Union across all classes
- Pixel Accuracy: Correctly classified pixels / Total pixels
- Boundary F1: F1-score at object boundaries

# Generalization Metrics
- mIoU Drop (Δ): mIoU(train_domain) - mIoU(test_domain)
- Per-class IoU: IoU breakdown by class
```

---

## 5. Experimental Protocol

### 5.1 Training Protocol

```yaml
Classification:
  epochs: 100
  optimizer: AdamW
  learning_rate: 1e-4 (CNN), 1e-5 (Transformer)
  weight_decay: 0.01
  scheduler: CosineAnnealingLR
  augmentation: RandomHorizontalFlip, RandomRotation, ColorJitter
  batch_size: 32

Detection:
  epochs: 50
  optimizer: SGD (momentum=0.9)
  learning_rate: 0.01 → 0.001 (step decay)
  weight_decay: 0.0005
  augmentation: Mosaic, MixUp, RandomScale
  batch_size: 16

Segmentation:
  epochs: 100
  optimizer: AdamW
  learning_rate: 1e-4
  weight_decay: 0.01
  augmentation: RandomCrop, RandomFlip, PhotometricDistortion
  batch_size: 8
```

### 5.2 Evaluation Protocol

1. **In-Domain Evaluation**: Test on held-out split of training dataset
2. **Cross-Domain Evaluation**: Test on completely different dataset (no fine-tuning)
3. **Few-Shot Adaptation** (optional): Fine-tune on K samples from test domain

### 5.3 Statistical Rigor

- Run each experiment **3 times** with different random seeds
- Report **mean ± std** for all metrics
- Use **paired t-test** for significance (p < 0.05)

---

## 6. Complete Experiment Matrix

### 6.1 Classification Experiments

| ID | Model | Type | Train | Test (In-Domain) | Test (Cross-Domain) |
|----|-------|------|-------|------------------|---------------------|
| C1 | ResNet-50 | Task-Specific | TrashNet | TrashNet (val) | RealWaste |
| C2 | EfficientNet-B0 | Task-Specific | TrashNet | TrashNet (val) | RealWaste |
| C3 | ViT-Base | Task-Specific | TrashNet | TrashNet (val) | RealWaste |
| C4 | CLIP (ViT-B/16) | Foundation | TrashNet (linear) | TrashNet (val) | RealWaste |
| C5 | DINOv2 (ViT-B/14) | Foundation | TrashNet (linear) | TrashNet (val) | RealWaste |
| C6 | CLIP (Zero-shot) | Foundation | - | TrashNet | RealWaste |
| C7 | DINOv2 (k-NN) | Foundation | - | TrashNet | RealWaste |

### 6.2 Detection Experiments

| ID | Model | Type | Train | Test (In-Domain) | Test (Cross-Domain) |
|----|-------|------|-------|------------------|---------------------|
| D1 | YOLOv8-M | Task-Specific | TACO | TACO (val) | Trash-ICRA19 |
| D2 | Faster R-CNN | Task-Specific | TACO | TACO (val) | Trash-ICRA19 |
| D3 | RetinaNet | Task-Specific | TACO | TACO (val) | Trash-ICRA19 |
| D4 | Grounding-DINO | Foundation | - (zero-shot) | TACO | Trash-ICRA19 |
| D5 | Florence-2 | Foundation | TACO | TACO (val) | Trash-ICRA19 |

### 6.3 Segmentation Experiments

| ID | Model | Type | Train | Test (In-Domain) | Test (Cross-Domain) |
|----|-------|------|-------|------------------|---------------------|
| S1 | U-Net | Task-Specific | TACO | TACO (val) | BePLi v1 |
| S2 | DeepLabV3+ | Task-Specific | TACO | TACO (val) | BePLi v1 |
| S3 | Mask R-CNN | Task-Specific | TACO | TACO (val) | BePLi v1 |
| S4 | SAM (ViT-H) | Foundation | - (zero-shot) | TACO | BePLi v1 |
| S5 | Florence-2 | Foundation | TACO | TACO (val) | BePLi v1 |

---

## 7. Expected Results Table Template

### Classification Results

| Model | Type | TrashNet (val) | RealWaste | Δ Accuracy | Relative Drop (%) |
|-------|------|----------------|-----------|------------|-------------------|
| ViT-Base | Task-Specific | **96.44%** | 39.98% | -56.46% | -58.53% |
| EfficientNet-B0 | Task-Specific | 89.53% | 32.89% | -56.63% | -63.29% |
| ResNet-50 | Task-Specific | 80.83% | 17.85% | -63.05% | -77.98% |
| CLIP | Foundation | 67.83% | **42.68%** | -25.15% | -37.06% |

### Detection Results

| Model | Type | TACO (val) mAP@0.5 | Trash-ICRA19 mAP@0.5 | Δ mAP | Relative Drop (%) |
|-------|------|--------------------|-----------------------|-------|-------------------|
| YOLOv8-M | Task-Specific | X.XX ± X.XX | X.XX ± X.XX | X.XX | X.X% |
| Faster R-CNN | Task-Specific | X.XX ± X.XX | X.XX ± X.XX | X.XX | X.X% |
| RetinaNet | Task-Specific | X.XX ± X.XX | X.XX ± X.XX | X.XX | X.X% |
| Grounding-DINO | Foundation | X.XX ± X.XX | X.XX ± X.XX | X.XX | X.X% |
| Florence-2 | Foundation | X.XX ± X.XX | X.XX ± X.XX | X.XX | X.X% |

### Segmentation Results

| Model | Type | TACO (val) mIoU | BePLi mIoU | Δ mIoU | Relative Drop (%) |
|-------|------|-----------------|------------|--------|-------------------|
| U-Net | Task-Specific | X.XX ± X.XX | X.XX ± X.XX | X.XX | X.X% |
| DeepLabV3+ | Task-Specific | X.XX ± X.XX | X.XX ± X.XX | X.XX | X.X% |
| Mask R-CNN | Task-Specific | X.XX ± X.XX | X.XX ± X.XX | X.XX | X.X% |
| SAM | Foundation | X.XX ± X.XX | X.XX ± X.XX | X.XX | X.X% |
| Florence-2 | Foundation | X.XX ± X.XX | X.XX ± X.XX | X.XX | X.X% |

---

## 8. Additional Analyses (Thesis Extensions)

### 8.1 Ablation Studies

1. **Effect of Training Data Size**: Train with 25%, 50%, 75%, 100% of data
2. **Effect of Fine-tuning Depth**: Linear probe vs. partial fine-tune vs. full fine-tune
3. **Effect of Augmentation**: With/without domain-specific augmentations

### 8.2 Qualitative Analysis

1. **Failure Case Analysis**: Visualize where models fail under domain shift
2. **Attention Map Visualization**: Compare what foundation vs. task-specific models attend to
3. **Confidence Calibration**: Are foundation models better calibrated?

### 8.3 Computational Analysis

| Model | Parameters (M) | FLOPs (G) | Inference Time (ms) | GPU Memory (GB) | Training Time (A100) |
|-------|----------------|-----------|---------------------|-----------------|---------------------|
| ResNet-50 | 25.6 | 4.1 | ~15 | 2.1 | ~2 hours |
| EfficientNet-B0 | 5.3 | 2.2 | ~8 | 1.2 | ~1.5 hours |
| ViT-Base | 86.6 | 17.6 | ~25 | 3.8 | ~3 hours |
| CLIP ViT-B/16 | 149.6 | 17.5 | ~30 | 4.2 | N/A (zero-shot) |

---

## 9. Dataset Download Links

### Classification
- **TrashNet**: https://huggingface.co/datasets/garythung/trashnet
- **RealWaste**: https://archive.ics.uci.edu/dataset/908/realwaste

### Detection
- **TACO**: https://zenodo.org/records/3587843 (DOI: 10.5281/zenodo.3587843)
- **Trash-ICRA19**: https://conservancy.umn.edu/items/c34b2945-4052-48fa-b7e7-ce0fba2fe649 (DOI: 10.13020/x0qn-y082)

### Segmentation
- **TACO (with masks)**: Same as above
- **BePLi v1**: https://www.seanoe.org/data/00811/92297/ (DOI: 10.17882/92297)

---

## 10. Thesis Contribution Statement

> This thesis presents a systematic, task-wise evaluation of foundation models versus classical task-specific models under cross-dataset distribution shift. Using controlled experiments across classification, detection, and segmentation tasks in the waste management domain, we quantify the generalization gap and provide empirical evidence for when foundation models outperform specialized architectures.

### Key Contributions:
1. **Unified Evaluation Framework**: First systematic comparison across all three core vision tasks
2. **Generalization Metrics**: Novel use of relative performance drop as generalization indicator
3. **Domain Shift Analysis**: Quantitative analysis of foundation model robustness
4. **Practical Guidelines**: Recommendations for model selection in waste management applications

---

## 11. Timeline (Suggested)

| Phase | Duration | Tasks |
|-------|----------|-------|
| **Phase 1: Setup** | 2 weeks | Dataset download, preprocessing, codebase setup |
| **Phase 2: Classification** | 3 weeks | Train all classification models, evaluate, analyze |
| **Phase 3: Detection** | 4 weeks | Train all detection models, evaluate, analyze |
| **Phase 4: Segmentation** | 4 weeks | Train all segmentation models, evaluate, analyze |
| **Phase 5: Analysis** | 2 weeks | Cross-task analysis, ablations, visualizations |
| **Phase 6: Writing** | 3 weeks | Thesis writing, revision |

**Total: ~18 weeks**

---

## 12. File Structure (Recommended)

```
Capstone/
├── stage_1/                    # Dataset preparation
│   ├── datasets/
│   │   ├── classification/
│   │   │   ├── trashnet/
│   │   │   └── realwaste/
│   │   ├── detection/
│   │   │   ├── taco/
│   │   │   └── trash_icra19/
│   │   └── segmentation/
│   │       ├── taco_masks/
│   │       └── bepli/
│   └── scripts/
│       ├── download_all_datasets.py
│       └── preprocess_datasets.py
│
├── stage_2/                    # Model training
│   ├── classification/
│   │   ├── train_resnet.py
│   │   ├── train_efficientnet.py
│   │   ├── train_vit.py
│   │   ├── train_clip.py
│   │   └── train_dinov2.py
│   ├── detection/
│   │   ├── train_yolov8.py
│   │   ├── train_fasterrcnn.py
│   │   ├── train_retinanet.py
│   │   ├── eval_grounding_dino.py
│   │   └── train_florence2.py
│   └── segmentation/
│       ├── train_unet.py
│       ├── train_deeplabv3.py
│       ├── train_maskrcnn.py
│       ├── eval_sam.py
│       └── train_florence2.py
│
├── stage_3/                    # Evaluation
│   ├── evaluate_classification.py
│   ├── evaluate_detection.py
│   ├── evaluate_segmentation.py
│   └── metrics/
│       ├── classification_metrics.py
│       ├── detection_metrics.py
│       └── segmentation_metrics.py
│
├── stage_4/                    # Analysis & Visualization
│   ├── analyze_results.py
│   ├── visualize_attention.py
│   ├── plot_comparison.py
│   └── generate_tables.py
│
├── configs/                    # Configuration files
│   ├── classification_config.yaml
│   ├── detection_config.yaml
│   └── segmentation_config.yaml
│
├── results/                    # Output results
│   ├── classification/
│   ├── detection/
│   └── segmentation/
│
└── notebooks/                  # Jupyter notebooks for exploration
    ├── EDA_datasets.ipynb
    ├── classification_analysis.ipynb
    ├── detection_analysis.ipynb
    └── segmentation_analysis.ipynb
```

---

*Document Version: 1.0*
*Last Updated: December 2024*
*Author: CS24MTECH11024*
