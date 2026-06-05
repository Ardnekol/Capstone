# Stage 2 Fine-Tuned Full-Dataset Result Report

## Run Metadata

| Field | Value |
|---|---|
| Model | `finetuned/florence2_unified_multitask_lora` |
| Base model | `microsoft/Florence-2-large-ft` |
| Regime | Fine-tuned unified Florence-2 with LoRA |
| Run timestamp | `20260417_164757` |
| Evaluation date | April 17, 2026 |
| Device | `cuda:0` |
| Max images | `0` meaning full available dataset per evaluator |
| Source results | `eval_results/unified_20260417_164757/results.json` |
| Comparison matrix | `eval_results/unified_20260417_164757/comparison_matrix.md` |

This report summarizes the full-dataset Stage 2 fine-tuned evaluation. The model is a single Florence-2 model with one unified multi-task LoRA adapter for classification, detection, and segmentation.

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
| Classification | TrashNet | Accuracy 0.8524 | Macro F1 0.7008 | - |
| Classification | RealWaste | Accuracy 0.5668 | Macro F1 0.4769 | - |
| Detection | TACO | Precision 0.5288 | Recall 0.2795 | F1 0.3657 |
| Detection | Trash-ICRA19 | Precision 0.5254 | Recall 0.4859 | F1 0.5049 |
| Segmentation | TACO | mIoU 0.2223 | Pixel Accuracy 0.9180 | - |
| Segmentation | DWSD | mIoU 0.2214 | Pixel Accuracy 0.6791 | - |

## Comparison Against Stage 1 Best Baselines

| Task | Stage 1 Best Baseline | Stage 1 Metric | Fine-Tuned Florence-2 | Delta |
|---|---|---:|---:|---:|
| Classification, in-domain | ViT-Base | 96.44% accuracy | 85.24% accuracy | -11.20 pts |
| Classification, cross-domain | CLIP | 42.68% accuracy | 56.68% accuracy | +14.00 pts |
| Detection, in-domain | YOLOv8 | 64.10% F1 | 36.57% F1 | -27.53 pts |
| Detection, cross-domain | Grounding DINO | 37.20% F1 | 50.49% F1 | +13.29 pts |
| Segmentation, in-domain | DeepLabV3+ | 0.4541 mIoU | 0.2223 mIoU | -0.2318 |
| Segmentation, cross-domain | SAM | 0.1023 mIoU | 0.2214 mIoU | +0.1191 |

## Fine-Tuned vs Zero-Shot Florence-2

| Task | Dataset | Zero-Shot | Fine-Tuned | Improvement |
|---|---|---:|---:|---:|
| Classification | TrashNet accuracy | 0.2643 | 0.8524 | +0.5881 |
| Classification | RealWaste accuracy | 0.3036 | 0.5668 | +0.2632 |
| Detection | TACO F1 | 0.2453 | 0.3657 | +0.1204 |
| Detection | Trash-ICRA19 F1 | 0.4203 | 0.5049 | +0.0846 |
| Segmentation | TACO mIoU (in-domain) | 0.1773 | 0.2223 | +0.0450 |
| Segmentation | DWSD mIoU (cross-domain) | 0.1207 | 0.2214 | +0.1007 (+83.4%) |

Fine-tuning improves every measured task. The largest gains are in classification, where the model learns the closed-set waste labels instead of producing generic captions or `unknown` outputs.

## Classification Results

### TrashNet, In-Domain

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Cardboard | 0.9436 | 0.9132 | 0.9281 |
| Glass | 0.9127 | 0.8762 | 0.8941 |
| Metal | 0.8608 | 0.8146 | 0.8371 |
| Paper | 0.9081 | 0.8822 | 0.8950 |
| Plastic | 0.8417 | 0.9046 | 0.8720 |
| Trash | 0.6310 | 0.3869 | 0.4796 |
| Unknown | 0.0000 | 0.0000 | 0.0000 |

### RealWaste, Cross-Domain

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Cardboard | 0.9748 | 0.2516 | 0.4000 |
| Glass | 0.8676 | 0.7643 | 0.8127 |
| Metal | 0.8302 | 0.5076 | 0.6300 |
| Paper | 0.6320 | 0.9240 | 0.7506 |
| Plastic | 0.6697 | 0.7904 | 0.7251 |
| Trash | 0.8333 | 0.0101 | 0.0200 |
| Unknown | 0.0000 | 0.0000 | 0.0000 |

### Classification Insight

Fine-tuning makes Florence-2 much stronger as a closed-set waste classifier. TrashNet accuracy rises to 85.24%, and RealWaste cross-domain accuracy rises to 56.68%. The model still trails ViT-Base on in-domain TrashNet, but it beats the Stage 1 CLIP cross-domain baseline by 14.00 points.

The strongest RealWaste classes are glass, paper, plastic, and metal. The weakest class remains `trash`, which is expected because it is an ambiguous catch-all class with visually diverse contents.

## Detection Results

| Dataset | Evaluated | IoU Threshold | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TACO | 300 | 0.5 | 303 | 270 | 781 | 0.5288 | 0.2795 | 0.3657 |
| Trash-ICRA19 | 1,120 | 0.5 | 673 | 608 | 712 | 0.5254 | 0.4859 | 0.5049 |

### Detection Insight

Fine-tuned Florence-2 improves detection on both TACO and Trash-ICRA19 compared with zero-shot Florence-2. It still trails YOLOv8 on in-domain TACO, but it outperforms the Stage 1 Grounding DINO cross-domain F1 baseline on Trash-ICRA19.

This is one of the strongest Stage 2 findings: a single unified model can preserve strong cross-domain detection while also handling classification and segmentation.

## Segmentation Results

| Dataset | Evaluated | mIoU | Pixel Accuracy |
|---|---:|---:|---:|
| TACO segmentation (in-domain)  | 150 | 0.2223 | 0.9180 |
| DWSD segmentation (cross-domain) | 144 | 0.2214 | 0.6791 |

### Segmentation Insight

Fine-tuning improves TACO segmentation from 0.1773 mIoU to 0.2223 mIoU and pixel accuracy from 0.7902 to 0.9180. The cross-domain result is more striking: on DWSD the fine-tuned model reaches 0.2214 mIoU — essentially identical to its in-domain mIoU (0.2223), with a difference of only 0.0009. Florence-2's polygon-token segmentation therefore transfers across waste domains almost without degradation. The cross-domain DWSD mIoU is 2.2× SAM's zero-shot baseline (0.1023) and 2.6× the best Stage 1 specialist Mask R-CNN (0.0842). Fine-tuned Florence-2 still trails the in-domain specialist DeepLabV3+ on TACO (0.4541 vs 0.2223), but DeepLabV3+ collapses to 0.0483 on DWSD — exactly the lab-to-field robustness gap a foundation model with LoRA fine-tuning is meant to close.

## Key Takeaways

1. Fine-tuning clearly improves the unified Florence-2 model across all measured tasks.
2. The strongest improvement is classification, especially because the LoRA adapter teaches the model the fixed waste categories.
3. Fine-tuned Florence-2 gives the best Stage 2 cross-domain classification result: 56.68% accuracy on RealWaste.
4. Fine-tuned Florence-2 also gives strong cross-domain detection: 50.49% F1 on Trash-ICRA19.
5. Specialist models still win peak in-domain performance, especially ViT-Base for TrashNet, YOLOv8 for TACO detection, and DeepLabV3+ for TACO segmentation.
6. The practical Stage 2 contribution is model unification: one Florence-2 model performs classification, detection, and segmentation instead of requiring separate specialist pipelines.

## Final Conclusion

The full-dataset fine-tuned Florence-2 run validates the Stage 2 direction. LoRA fine-tuning turns Florence-2 from a weak zero-shot closed-set waste classifier into a useful unified multi-task model. It does not replace every specialist model in peak in-domain accuracy, but it gives better cross-domain classification and detection while using a single deployable model for all three tasks.
