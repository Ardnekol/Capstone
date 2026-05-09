# Waste Classification Experiments

## Overview

This directory contains all code for the **Waste Classification** task comparing:
- **Task-Specific Models**: ResNet-50, EfficientNet-B0
- **Foundation Models**: CLIP (ViT-B/16)

## Dataset Summary

### Training: TrashNet
- **Images**: ~2,500 across 6 classes
- **Categories**: cardboard, glass, metal, paper, plastic, trash
- **Format**: JPEG images in class folders
- **Location**: `datasets/classification/trashnet/dataset-original/dataset-original/`

### Testing: RealWaste
- **Images**: ~4,700 across 8 classes
- **Categories**: Cardboard, Food Organics, Glass, Metal, Miscellaneous Trash, Paper, Plastic, Textile Trash, Vegetation
- **Format**: JPEG images in class folders
- **Location**: `datasets/classification/realwaste/realwaste-main/RealWaste/`

## Class Mapping

### Unified Classification Classes
| Category | Description |
|----------|-------------|
| **cardboard** | Cardboard materials |
| **glass** | Glass bottles, containers |
| **metal** | Metal cans, foil, caps |
| **paper** | Paper, magazines, newspapers |
| **plastic** | Plastic bottles, bags, containers |
| **trash** | Miscellaneous waste |

## Pipeline Structure

1. **Preprocessing**: Resize images to 224x224, organize in train/val/test splits
2. **Training**: Train each model on TrashNet
3. **Evaluation**: Evaluate on both TrashNet (in-domain) and RealWaste (cross-domain)
4. **Reporting**: Generate comparison tables and plots

## Quick Start

```bash
# Full pipeline
bash quick_start.sh

# Quick test (5 epochs)
bash quick_start.sh --quick

# Run individual model
bash run_one_model.sh resnet50
bash run_one_model.sh efficientnet
bash run_one_model.sh clip
```

## Results

All results are saved in the `results/` folder with metrics, confusion matrices, and comparison reports.