# Segmentation Model Comparison Report

## Overview

This report compares the performance of different segmentation models on waste segmentation tasks.

- **Task-Specific Models**: U-Net, DeepLabV3+, Mask R-CNN
- **Foundation Model**: SAM (ViT-H) for zero-shot segmentation
- **Training Dataset**: TACO (urban waste)
- **Evaluation Datasets**: TACO (in-domain), DWSD (cross-domain - campus waste)

## Training Methodology Note

> **Fine-Tuned Models (Task-Specific):** U-Net (ResNet-34 backbone), DeepLabV3+ (ResNet-101 backbone), and Mask R-CNN were **fine-tuned** on the TACO segmentation dataset using supervised learning with ImageNet pre-trained weights. Training configuration: 50 epochs, batch size 8, learning rate 1e-4.
>
> **Zero-Shot Models (Foundation):** SAM (Segment Anything Model, ViT-H) was evaluated in **automatic mask generation mode** without any fine-tuning or point/box prompts on the waste datasets. The model used pre-trained weights from the sam_vit_h_4b8939.pth checkpoint. Note: SAM's lower in-domain performance reflects zero-shot evaluation without task-specific adaptation.

## IoU Comparison

| Model       |   TACO IoU |   DWSD IoU |
|:------------|-----------:|-----------:|
| U-Net       |     0.3293 |     0.063  |
| DeepLabV3+  |     0.4541 |     0.0483 |
| Mask R-CNN  |     0.2885 |     0.0842 |
| SAM (ViT-H) |     0.038  |     0.1023 |

## Detailed Metrics

| Model       | Dataset   |    IoU |   Precision |   Recall |   F1-Score |
|:------------|:----------|-------:|------------:|---------:|-----------:|
| U-Net       | TACO      | 0.3293 |      0.5297 |   0.3716 |     0.3981 |
| U-Net       | DWSD      | 0.063  |      0.3482 |   0.0647 |     0.0936 |
| DeepLabV3+  | TACO      | 0.4541 |      0.8185 |   0.505  |     0.6246 |
| DeepLabV3+  | DWSD      | 0.0483 |      0.7867 |   0.0489 |     0.0921 |
| Mask R-CNN  | TACO      | 0.2885 |      0.405  |   0.3087 |     0.3338 |
| Mask R-CNN  | DWSD      | 0.0842 |      0.2237 |   0.0873 |     0.106  |
| SAM (ViT-H) | TACO      | 0.038  |      0.041  |   0.3421 |     0.0732 |
| SAM (ViT-H) | DWSD      | 0.1023 |      0.1297 |   0.3266 |     0.1856 |

## Analysis

- **Best on TACO**: DeepLabV3+ (0.4541)
- **Best on DWSD**: SAM (ViT-H) (0.1023)

### Cross-Domain Performance

#### TACO → DWSD (Urban → Campus)
- **U-Net**: 0.2663 IoU drop (80.9% relative)
- **DeepLabV3+**: 0.4058 IoU drop (89.4% relative)
- **Mask R-CNN**: 0.2043 IoU drop (70.8% relative)
- **SAM (ViT-H)**: -0.0643 IoU drop (-169.1% relative)

## Conclusion

This analysis provides insights into model performance across different waste segmentation scenarios.
Task-specific models generally perform better on in-domain data, while foundation models offer
competitive zero-shot performance without fine-tuning.

The DWSD dataset enables meaningful urban-urban domain shift analysis, providing a complementary
evaluation scenario to the extreme urban-beach shift observed with BePLi.

