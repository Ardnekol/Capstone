# Stage 2 Zero-Shot Full-Dataset Result Report

## Run Metadata

| Field | Value |
|---|---|
| Model | `microsoft/Florence-2-large-ft` |
| Regime | Zero-shot Florence-2, no LoRA adapter |
| Run timestamp | `20260423_004555` |
| Evaluation date | April 23, 2026 |
| Device | `cuda:0` |
| Max images | `0` meaning full available dataset per evaluator |
| Source results | `eval_results/zeroshot_23_04_2026_12_45_51_AM_0/results.json` |
| Comparison matrix | `eval_results/zeroshot_23_04_2026_12_45_51_AM_0/comparison_matrix.md` |

This report summarizes the full-dataset Stage 2 zero-shot evaluation for the base Florence-2 model. The model was evaluated directly with task prompts and without waste-specific fine-tuning.

## Dataset Coverage

| Task | Dataset | Domain | Evaluated Samples |
|---|---:|---|---:|
| Classification | TrashNet | In-domain | 2,527 / 2,527 images |
| Classification | RealWaste | Cross-domain | 3,587 / 3,587 mapped images |
| Detection | TACO test split | In-domain | 300 images |
| Detection | Trash-ICRA19 | Cross-domain | 1,120 images |
| Segmentation | TACO segmentation test split | In-domain | 150 images |

Note: RealWaste evaluation uses the TrashNet-compatible mapped classes. Unmapped RealWaste classes are skipped by the evaluator.

## Overall Metrics

| Task | Dataset | Metric 1 | Metric 2 | Metric 3 |
|---|---|---:|---:|---:|
| Classification | TrashNet | Accuracy 0.2643 | Macro F1 0.2862 | - |
| Classification | RealWaste | Accuracy 0.3036 | Macro F1 0.3456 | - |
| Detection | TACO | Precision 0.3523 | Recall 0.1882 | F1 0.2453 |
| Detection | Trash-ICRA19 | Precision 0.3984 | Recall 0.4448 | F1 0.4203 |
| Segmentation | TACO | mIoU 0.1773 | Pixel Accuracy 0.7902 | - |

## Comparison Against Stage 1 Best Baselines

| Task | Stage 1 Best Baseline | Stage 1 Metric | Florence-2 Zero-Shot | Delta |
|---|---|---:|---:|---:|
| Classification, in-domain | ViT-Base | 96.44% accuracy | 26.43% accuracy | -70.01 pts |
| Classification, cross-domain | CLIP | 42.68% accuracy | 30.36% accuracy | -12.32 pts |
| Detection, in-domain | YOLOv8 | 64.10% F1 | 24.53% F1 | -39.57 pts |
| Detection, cross-domain | Grounding DINO | 37.20% F1 | 42.03% F1 | +4.83 pts |
| Segmentation, in-domain | DeepLabV3+ | 0.4541 mIoU | 0.1773 mIoU | -0.2768 |

## Classification Results

### TrashNet, In-Domain

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Cardboard | 0.9829 | 0.2854 | 0.4423 |
| Glass | 0.8804 | 0.3234 | 0.4730 |
| Metal | 0.5714 | 0.1073 | 0.1807 |
| Paper | 0.6681 | 0.2609 | 0.3753 |
| Plastic | 0.8000 | 0.3983 | 0.5319 |
| Trash | 0.0000 | 0.0000 | 0.0000 |
| Unknown | 0.0000 | 0.0000 | 0.0000 |

### RealWaste, Cross-Domain

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Cardboard | 1.0000 | 0.1692 | 0.2894 |
| Glass | 0.8981 | 0.5667 | 0.6949 |
| Metal | 0.5540 | 0.0975 | 0.1658 |
| Paper | 0.6741 | 0.6660 | 0.6700 |
| Plastic | 0.8571 | 0.3713 | 0.5182 |
| Trash | 0.8750 | 0.0424 | 0.0809 |
| Unknown | 0.0000 | 0.0000 | 0.0000 |

### Classification Insight

Zero-shot Florence-2 is weak as a closed-set waste classifier. The model often produces captions that do not cleanly map to the six TrashNet labels, creating many `unknown` predictions. Precision is high for some classes, especially cardboard, glass, and plastic, but recall is low. This means that when the model says a class name, it is often correct, but it misses many true examples.

The RealWaste result is slightly better than TrashNet in accuracy and macro F1, which is unusual for an in-domain versus cross-domain comparison. The likely reason is that Florence-2's pretraining favors natural images and scene-level captions more than isolated lab-style TrashNet images.

## Detection Results

| Dataset | Evaluated | IoU Threshold | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TACO | 300 | 0.5 | 204 | 375 | 880 | 0.3523 | 0.1882 | 0.2453 |
| Trash-ICRA19 | 1,120 | 0.5 | 616 | 930 | 769 | 0.3984 | 0.4448 | 0.4203 |

### Detection Insight

Detection is the strongest zero-shot result. Florence-2 performs poorly on in-domain TACO compared with YOLOv8, but it performs better on cross-domain Trash-ICRA19 than the Stage 1 Grounding DINO F1 baseline. This supports the main foundation-model argument: even without task-specific training, a large vision-language model can retain useful object-localization ability under domain shift.

The ICRA19 improvement comes mainly from recall: 0.4448 on ICRA19 versus 0.1882 on TACO. The model finds more objects in the underwater domain than in the urban TACO split. The current evaluator measures IoU-based localization F1, so this should be described as cross-domain object-localization strength rather than class-specific mAP.

## Segmentation Results

| Dataset | Evaluated | mIoU | Pixel Accuracy |
|---|---:|---:|---:|
| TACO segmentation | 150 | 0.1773 | 0.7902 |

### Segmentation Insight

Zero-shot Florence-2 segmentation is below the best Stage 1 specialist segmentation model. The high pixel accuracy with low mIoU indicates that much of the background is handled correctly, but object masks are not accurate enough. This is expected because Florence-2 generates polygon coordinates as text tokens rather than using a dedicated dense mask decoder.

## Key Takeaways

1. Zero-shot Florence-2 cannot replace task-specific or fine-tuned models for closed-set waste classification.
2. The best zero-shot signal is cross-domain detection, where Florence-2 reaches 0.4203 F1 on Trash-ICRA19 and beats the Stage 1 cross-domain detection F1 baseline by 4.83 points.
3. In-domain specialist models still dominate in peak performance: ViT-Base for TrashNet classification, YOLOv8 for TACO detection, and DeepLabV3+ for TACO segmentation.
4. Segmentation remains the weakest Stage 2 zero-shot task because polygon-token generation is not as precise as specialist segmentation architectures.
5. These results justify the next Stage 2 step: LoRA fine-tuning on the unified multi-task waste dataset.

## Final Conclusion

The full-dataset zero-shot run establishes a strong baseline for Stage 2. It shows that base Florence-2 has broad cross-domain detection ability, but it is not sufficient as a complete waste-management model without fine-tuning. The result supports the project narrative: foundation models generalize better under domain shift, but task-specific adaptation is still needed for reliable deployment across classification, detection, and segmentation.
