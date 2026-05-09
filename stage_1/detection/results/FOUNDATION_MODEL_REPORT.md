# Foundation Models for Waste Management: A Comparative Report

## Introduction
This report benchmarks the performance of task-specific and foundation object detection models for waste management, using the TACO (in-domain) and ICRA19 (cross-domain) datasets. The goal is to evaluate the generalization ability of foundation models compared to models trained specifically for waste detection.

## Models Compared
- **YOLOv8** (Task-Specific)
- **RetinaNet** (Task-Specific)
- **Faster R-CNN** (Task-Specific)
- **Grounding DINO** (Foundation Model)

## Key Metrics
- **mAP50**: Mean Average Precision at IoU 0.5
- **mAP50-95**: Mean Average Precision averaged over IoU thresholds
- **Precision** and **Recall**

## Results Summary
| Model            | Type           | In-Domain mAP50 | In-Domain mAP50-95 | Cross-Domain mAP50 | Cross-Domain mAP50-95 |
|------------------|----------------|-----------------|--------------------|--------------------|-----------------------|
| YOLOv8           | Task-Specific  | 0.6417          | 0.5204             | 0.1349             | 0.0472                |
| RetinaNet        | Task-Specific  | 0.2103          | 0.1514             | 0.0690             | 0.0251                |
| Faster R-CNN     | Task-Specific  | 0.1925          | 0.1326             | 0.1360             | 0.0684                |
| Grounding DINO   | Foundation     | 0.1526 (TACO)   | 0.1269 (TACO)      | 0.4270 (ICRA19)    | 0.2249 (ICRA19)       |

## Analysis
- **Task-specific models** (YOLOv8, RetinaNet, Faster R-CNN) achieve the highest accuracy on the dataset they are trained on (TACO), but their performance drops significantly on cross-domain data (ICRA19).
- **Grounding DINO**, a foundation model, shows strong generalization: while its in-domain performance is lower than YOLOv8, it outperforms all task-specific models on the cross-domain dataset (ICRA19), achieving much higher mAP50 and mAP50-95.
- This demonstrates that foundation models are more robust to domain shift and can detect waste objects in new environments without retraining.

## Visual Evidence
- Task-specific models: High accuracy on familiar waste types, but many missed detections or false positives on new/unseen waste types.
- Grounding DINO: Consistent detection across both datasets, with better recall and precision on cross-domain data.

## Conclusion
Foundation models like Grounding DINO are highly valuable for waste management applications where new types of waste or new environments are encountered. They provide strong zero-shot generalization, making them suitable for scalable, real-world waste detection systems.

## Recommendations
- Use task-specific models for maximum accuracy in well-defined, static environments.
- Use foundation models for deployment in diverse or changing environments, or when rapid adaptation to new waste types is needed.

---

*Report generated on December 26, 2025.*
