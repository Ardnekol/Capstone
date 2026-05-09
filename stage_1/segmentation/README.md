# Waste Segmentation Pipeline

This directory contains a complete segmentation pipeline for waste management research, comparing task-specific models against foundation models under domain shift scenarios.

## Overview

The pipeline evaluates four segmentation models:
- **U-Net** (ResNet-34 encoder) - Task-specific semantic segmentation
- **DeepLabV3+** (ResNet-101 encoder) - Task-specific semantic segmentation
- **Mask R-CNN** (ResNet-50-FPN) - Task-specific instance segmentation
- **SAM (ViT-H)** - Foundation model for zero-shot segmentation

## Datasets

- **Training**: TACO dataset (urban waste, ~1,500 images)
- **Evaluation**: TACO (in-domain) + BePLi (cross-domain - beach plastic, ~3,647 images)

## Directory Structure

```
segmentation/
├── scripts/                 # All Python scripts
│   ├── preprocess_segmentation.py    # Dataset preprocessing
│   ├── train_unet.py                 # U-Net training
│   ├── train_deeplabv3plus.py        # DeepLabV3+ training
│   ├── train_maskrcnn.py             # Mask R-CNN training
│   ├── eval_sam.py                   # SAM evaluation
│   └── evaluate_segmentation.py      # Results analysis
├── data/                    # Processed datasets
│   ├── taco/               # TACO dataset (processed)
│   └── bepli/              # BePLi dataset (processed)
├── results/                 # Model outputs and reports
│   ├── unet_results.json
│   ├── deeplabv3plus_results.json
│   ├── maskrcnn_results.json
│   ├── sam_results.json
│   ├── SEGMENTATION_REPORT.md
│   └── segmentation_iou_comparison.png
├── requirements.txt         # Python dependencies
└── run_segmentation_pipeline.py    # Main pipeline runner
```

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Datasets**:
   - TACO: Download from [official repository](https://github.com/pedropro/TACO)
   - BePLi: Download from [official repository](https://github.com/earthlab-be/BePLi)

3. **Update Dataset Paths**:
   Edit the paths in `scripts/preprocess_segmentation.py` to point to your dataset locations.

## Usage

### Run Complete Pipeline
```bash
python run_segmentation_pipeline.py
```

This will:
1. Preprocess TACO and BePLi datasets
2. Train U-Net, DeepLabV3+, and Mask R-CNN models
3. Evaluate SAM (zero-shot)
4. Generate comprehensive comparison report

### Run Individual Components

**Preprocess Data**:
```bash
python scripts/preprocess_segmentation.py
```

**Train Models**:
```bash
python scripts/train_unet.py
python scripts/train_deeplabv3plus.py
python scripts/train_maskrcnn.py
```

**Evaluate SAM**:
```bash
python scripts/eval_sam.py
```

**Generate Report**:
```bash
python scripts/evaluate_segmentation.py
```

## Configuration

All training scripts use the following default hyperparameters:
- **Batch Size**: 8
- **Learning Rate**: 1e-4 (AdamW optimizer)
- **Epochs**: 50 (with early stopping)
- **Loss**: Dice Loss for semantic segmentation, COCO loss for Mask R-CNN
- **Image Size**: 512x512

Modify the scripts directly to adjust these parameters.

## Results

After running the pipeline, results are saved in `results/`:
- **JSON files**: Detailed metrics for each model
- **SEGMENTATION_REPORT.md**: Comprehensive analysis and comparison
- **segmentation_iou_comparison.png**: IoU comparison plot

## Key Metrics

- **IoU (Intersection over Union)**: Primary segmentation accuracy metric
- **Precision/Recall/F1-Score**: Additional evaluation metrics
- **Cross-domain performance**: TACO → BePLi domain shift analysis

## Expected Performance

Typical results (may vary based on training):
- Task-specific models: 0.7-0.9 IoU on TACO, 0.4-0.7 IoU on BePLi
- SAM (zero-shot): 0.3-0.6 IoU across both datasets

## Hardware Requirements

- **GPU**: Recommended (A100, V100, or RTX series)
- **RAM**: 16GB+ for training, 8GB+ for evaluation
- **Storage**: ~50GB for datasets and models

## Troubleshooting

1. **CUDA Out of Memory**: Reduce batch size in training scripts
2. **Dataset Path Issues**: Ensure absolute paths are correct in preprocessing script
3. **Import Errors**: Install all requirements from `requirements.txt`
4. **Slow Training**: Use mixed precision training (enable in scripts)

## Citation

If you use this pipeline in your research, please cite the original datasets and models:

- TACO: https://github.com/pedropro/TACO
- BePLi: https://github.com/earthlab-be/BePLi
- SAM: https://segment-anything.com/
- Segmentation Models: https://github.com/qubvel/segmentation_models.pytorch