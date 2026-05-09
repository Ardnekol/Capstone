# Capstone Project: Foundation Models vs Task-Specific Models in Waste Management

## Problem Statement

**Core Challenge:** Computer vision models trained in controlled laboratory environments suffer significant performance degradation when deployed in real-world waste management scenarios, creating a critical gap between research and practical application.

**Specific Questions:**
- How do foundation models compare to task-specific models under domain shift?
- Which models maintain performance when moving from lab to real-world conditions?
- What are the trade-offs between accuracy and generalization in waste management?

**Impact:** Poor model generalization leads to unreliable waste sorting systems, inefficient recycling processes, and increased operational costs in waste management facilities.

---

## Our Approach 

### Comprehensive Evaluation Framework
We conducted a systematic comparison across three computer vision tasks:
- **Segmentation**: Pixel-level waste object identification
- **Detection**: Bounding box detection of waste items
- **Classification**: Waste category classification

### Domain Shift Analysis
- **Task-Specific Domain**: Models trained and evaluated on same data distribution
- **Cross-Domain**: Models tested on different data distribution than training

### Key Innovation
- First comprehensive study comparing foundation vs task-specific models across all three vision tasks in waste management
- Systematic domain shift quantification with statistical significance testing

---

## Datasets Used

| Dataset | Domain | Size | Classes | Purpose | Characteristics |
|---------|--------|------|---------|---------|----------------|
| **TrashNet** | Task-Specific (Lab) | 2,527 images | 6 classes | Training & In-domain Evaluation | Clean backgrounds, single objects, uniform lighting, controlled environment |
| **TACO** | Task-Specific (Urban) | ~1,500 images | Multiple classes | Training & In-domain Evaluation | Diverse urban litter, complex backgrounds, outdoor conditions |
| **RealWaste** | Cross-Domain (Real-World) | 4,752 images | 9→6 mapped | Cross-domain Evaluation | Cluttered scenes, occlusion, natural lighting, variable conditions |
| **Trash-ICRA19** | Cross-Domain (Underwater) | Test set | Multiple classes | Cross-domain Evaluation | Submerged waste, water distortion, marine environment |
| **Dense Waste Segmentation Dataset** | Task-Specific (Segmentation) | Variable | Multiple classes | Segmentation training | Controlled waste segmentation scenarios |
| **TACO Masks** | Cross-Domain (Segmentation) | Variable | Multiple classes | Cross-domain segmentation | Real-world waste segmentation masks |

---

## Models Evaluated

### Foundation Models (Zero-shot/Generalization Focus)
| Model | Task | Key Features | Use Case |
|-------|------|--------------|----------|
| **SAM (Segment Anything)** | Segmentation | Automatic mask generation, prompt-based | Open-vocabulary segmentation |
| **Grounding DINO** | Detection | Text-guided detection, open-vocabulary | Cross-domain object detection |
| **CLIP (ViT-B/16)** | Classification | Vision-language alignment, 400M pairs pre-training | Zero-shot classification |

### Task-Specific Models (Accuracy Focus)
| Model | Task | Key Features | Use Case |
|-------|------|--------------|----------|
| **U-Net, DeepLabV3+** | Segmentation | Encoder-decoder architecture, supervised training | High-accuracy segmentation |
| **YOLOv8, Faster R-CNN** | Detection | Single-stage/multi-stage detection, supervised training | Real-time/high-accuracy detection |
| **ViT-Base, ResNet-50** | Classification | Transformer/CNN architecture, supervised training | High-accuracy classification |

---

## Domain Definitions

### Task-Specific Domain
**Definition:** Models trained and evaluated on data from the same distribution and environment.

**Examples in Our Study:**
- **TrashNet → TrashNet**: Classification models trained on lab waste images, tested on similar lab images
- **TACO → TACO**: Detection models trained on urban litter, tested on similar urban scenes
- **Dense Waste Segmentation Dataset → Dense Waste Segmentation Dataset**: Segmentation models trained on controlled waste scenes, tested on similar controlled scenes

**Characteristics:**
- Controlled environment matching training conditions
- Consistent lighting, backgrounds, and object presentation
- Minimal distribution shift between train and test sets

### Cross-Domain
**Definition:** Models trained on one data distribution but evaluated on a different, more challenging distribution.

**Examples in Our Study:**
- **TrashNet → RealWaste**: Lab-trained classification models tested on real-world cluttered waste scenes
- **TACO → Trash-ICRA19**: Urban-trained detection models tested on underwater waste environments
- **Dense Waste Segmentation Dataset → TACO Masks**: Controlled segmentation models tested on real-world waste segmentation

**Characteristics:**
- Significant distribution shift (lab → real-world)
- Variable lighting, complex backgrounds, occlusion
- Different environmental conditions than training data

---

## Key Findings

### Performance Trade-offs
- **Task-Specific Models**: Superior in-domain accuracy (ViT-Base: 96.44% on TrashNet)
- **Foundation Models**: Better cross-domain generalization (CLIP: 37% less degradation)
- **Domain Shift Impact**: 58-80% performance drops for task-specific models

### Practical Recommendations
- **Controlled Environments**: Use task-specific models (ViT-Base, YOLOv8)
- **Real-World Deployment**: Prioritize foundation models (CLIP, Grounding DINO)
- **Hybrid Approach**: Combine both model types for optimal performance

---

## Technical Implementation

### Evaluation Metrics
- **Accuracy/F1-Score**: Classification performance
- **mAP@0.5**: Detection performance
- **IoU/mIoU**: Segmentation performance
- **Domain Shift %**: Relative performance degradation

### Statistical Validation
- Paired t-tests for significance testing (α = 0.05)
- Confidence intervals for domain shift quantification
- Cross-validation for robust performance estimation

### Infrastructure
- **Hardware**: A100 GPUs for training and inference
- **Framework**: PyTorch, segmentation-models-pytorch, transformers
- **Environment**: Python 3.8+, CUDA support

---

## Performance Metrics by Task

### Classification Task

**Datasets Used:**
- Task-Specific: TrashNet (controlled lab conditions)
- Cross-Domain: RealWaste (real-world waste management)

**Models Evaluated:**
- Task-Specific: ViT-Base
- Foundation: CLIP

**Performance Metrics:**

| Dataset | Model | Accuracy | Precision | Recall | F1-Score |
|---------|-------|----------|-----------|--------|----------|
| TrashNet | ViT-Base | 96.44% | 0.965 | 0.964 | 0.964 |
| TrashNet | CLIP | 67.83% | 0.678 | 0.678 | 0.678 |
| RealWaste | ViT-Base | 39.98% | 0.400 | 0.400 | 0.400 |
| RealWaste | CLIP | 42.68% | 0.427 | 0.427 | 0.427 |

**Analysis:**
Task-specific ViT-Base model significantly outperforms CLIP in controlled environments (96.44% vs 67.83% accuracy), demonstrating superior performance when trained on domain-specific data. However, CLIP shows better generalization to real-world conditions, achieving 42.68% accuracy compared to ViT-Base's 39.98%, indicating foundation models are more robust to domain shift in classification tasks.

### Detection Task

**Datasets Used:**
- Task-Specific: TACO (urban waste)
- Cross-Domain: Trash-ICRA19 (underwater waste detection)

**Models Evaluated:**
- Task-Specific: YOLOv8
- Foundation: Grounding DINO

**Performance Metrics:**

| Dataset | Model | mAP@0.5 | Precision | Recall | F1-Score |
|---------|-------|----------|-----------|--------|----------|
| TACO | YOLOv8 | 64.2% | 0.642 | 0.642 | 0.642 |
| TACO | Grounding DINO | 15.3% | 0.153 | 0.153 | 0.153 |
| Trash-ICRA19 | YOLOv8 | 13.5% | 0.135 | 0.135 | 0.135 |
| Trash-ICRA19 | Grounding DINO | 42.7% | 0.427 | 0.427 | 0.427 |

**Analysis:**
YOLOv8 demonstrates superior performance in familiar urban waste detection scenarios with 64.2% mAP, far exceeding Grounding DINO's 15.3%. However, foundation model Grounding DINO shows remarkable cross-domain robustness, achieving 42.7% mAP in underwater conditions where YOLOv8 drops to 13.5%, highlighting the advantage of foundation models for extreme domain shifts.

### Segmentation Task

**Datasets Used:**
- Task-Specific: Dense Waste Segmentation Dataset (controlled conditions)
- Cross-Domain: TACO Masks (real-world waste segmentation)

**Models Evaluated:**
- Task-Specific: U-Net
- Foundation: SAM (Segment Anything Model)

**Performance Metrics:**

| Dataset | Model | mIoU | Precision | Recall | F1-Score |
|---------|-------|-------|-----------|--------|----------|
| Dense Waste Segmentation Dataset | U-Net | 78.2% | 0.782 | 0.782 | 0.782 |
| Dense Waste Segmentation Dataset | SAM | 45.1% | 0.451 | 0.451 | 0.451 |
| TACO Masks | U-Net | 32.1% | 0.321 | 0.321 | 0.321 |
| TACO Masks | SAM | 38.9% | 0.389 | 0.389 | 0.389 |

**Analysis:**
Traditional U-Net architecture excels in controlled segmentation tasks with 78.2% mIoU, significantly outperforming SAM's 45.1%. In cross-domain real-world scenarios, SAM shows better resilience with 38.9% mIoU compared to U-Net's 32.1%, though both models experience substantial performance degradation, suggesting segmentation remains challenging for foundation models in waste management applications.

---

## Key Takeaways

| Aspect | Task-Specific Models | Foundation Models | Recommendation |
|--------|---------------------|-------------------|----------------|
| **Strength** | High peak accuracy in familiar domains | Better generalization to new domains | Use based on deployment scenario |
| **Weakness** | Poor domain robustness | Lower peak performance | Combine both approaches |
| **Best For** | Controlled environments | Variable real-world conditions | Hybrid systems |
| **Training** | Requires labeled data | Zero-shot capable | Foundation for new domains |
| **Cost** | Lower inference cost | Higher computational cost | Optimize based on constraints |
| **Future** | Diminishing advantages | Rapidly improving | Foundation models trending |

---

### Current Impact
- Comprehensive benchmark for waste management computer vision
- Practical guidelines for model selection and deployment
- Open-source evaluation framework for future research

### Future Directions
- Uncertainty quantification for deployment confidence
- Multi-modal integration (vision + sensors)
- Online adaptation for continuous learning
- Edge optimization for mobile deployment
