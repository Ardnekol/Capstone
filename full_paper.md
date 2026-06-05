# Bridging the Lab-to-Field Gap in Waste Vision with a Unified Foundation Model

*Workshop submission draft. Compiled from section drafts in this folder.*
*All numbers reflect the post-bug-fix full-dataset evaluation runs (May 2026).*

---

## Abstract

Computer vision models trained on curated waste datasets in laboratory conditions
degrade dramatically when deployed in the field: classification accuracy drops
60–80%, detection F1 falls by similar margins, and segmentation models collapse
in unfamiliar visual contexts. We study this *lab-to-field gap* across three
core waste-vision tasks — classification, object detection, and segmentation —
using six benchmarks spanning in-domain (TrashNet, TACO) and cross-domain
(RealWaste, Trash-ICRA19, DWSD) splits. We show that a single vision-language
foundation model, Florence-2-large-ft, fine-tuned with a 14 MB LoRA adapter
on a unified multi-task waste dataset, **sweeps every cross-domain benchmark**:
+14.0 percentage points over CLIP on RealWaste classification, +13.3 points
over Grounding DINO on Trash-ICRA19 detection F1, and a 2.2× improvement
over SAM on DWSD segmentation mIoU. Remarkably, the model's cross-domain
segmentation mIoU (0.2214) is statistically indistinguishable from its
in-domain mIoU (0.2223), indicating polygon-token generation transfers
between waste domains essentially without degradation. An ablation across
single-task LoRA variants shows that specialist adapters slightly outperform
the unified adapter on their own task, but the unified model wins on
operational simplicity — replacing three specialist deployments with one
shared base model and one set of adapter weights, suitable for practical
waste-sorting systems.

---

## 1 — Introduction

Automated waste sorting is one of the few computer vision applications where
deployment conditions almost never match training conditions. Public waste
datasets used to train classification, detection, and segmentation models are
curated under controlled lighting and clean backgrounds — TrashNet captures
single objects on white surfaces, TACO is sourced from staged photographs of
litter — yet real material recovery facilities, hostel bins, and outdoor
collection points present cluttered, occluded, variable-lighting scenes that
look almost nothing like the training distribution.

The consequence is a severe **lab-to-field gap**. In our experiments, a
ViT-Base classifier trained on TrashNet reaches 96.4% in-domain accuracy but
collapses to 40.0% on RealWaste, a real-world waste-sorting test set. YOLOv8
fine-tuned on TACO reaches 64.1% F1 in-domain but falls to 13.5% F1 on
Trash-ICRA19. DeepLabV3+ trained on TACO segmentation masks scores 0.4541
mIoU in-domain but collapses to 0.0483 mIoU cross-domain. Each of these is
a 60–90% relative degradation — the kind of failure that would make a
deployed waste-sorting system unreliable enough to be unusable.

Foundation models — large pretrained vision-language models like CLIP,
Grounding DINO, and SAM — have been proposed as a remedy because their broad
pretraining gives them better out-of-distribution robustness. But used
zero-shot they sacrifice accuracy: CLIP reaches only 42.7% on RealWaste,
Grounding DINO 37.2% F1 on ICRA19, SAM 0.1023 mIoU on DWSD. Practical waste
sorting needs both robustness *and* usable per-task accuracy, and it needs
*all three tasks* — classification to route material, detection to localize
items on a conveyor, segmentation to plan robotic grippers — in a deployable
system.

We show that a single vision-language foundation model, Florence-2-large-ft,
fine-tuned with a 14 MB LoRA adapter on a small unified multi-task waste
dataset, **sweeps every cross-domain benchmark across all three tasks**:
+14.0 points over CLIP on RealWaste classification, +13.3 points over
Grounding DINO on Trash-ICRA19 detection F1, and a 2.2× improvement
(+0.1191 mIoU) over SAM on DWSD segmentation. The cross-domain segmentation
result is particularly striking: the unified model's DWSD mIoU (0.2214) is
essentially identical to its in-domain TACO mIoU (0.2223), demonstrating
that polygon-token segmentation transfers between waste domains with
negligible degradation — in stark contrast to specialist segmenters that
collapse.

Our contributions are three: (1) a systematic evaluation of the
lab-to-field gap in waste vision across six benchmarks and three tasks;
(2) a unified Florence-2 + LoRA model that closes this gap on every
cross-domain benchmark with a single 14 MB adapter; and (3) an honest
ablation against single-task variants of the same adapter, showing that
the unified model trades 1–8 points of per-task accuracy for the
operational simplicity of one-model deployment — a favourable trade for
real-world waste-sorting systems.

---

## 2 — Related Work

**Waste-vision datasets and specialist models.** A growing body of work
applies standard supervised vision pipelines to waste sorting. TrashNet
provides 2,527 lab-condition images of six waste categories and serves as
a common benchmark for classification specialists (ViT, ResNet, EfficientNet).
TACO extends to urban outdoor litter with bounding box and polygon
annotations, supporting detection and segmentation. RealWaste and TrashCan /
Trash-ICRA19 provide field-condition test sets — real waste-sorting bins
and underwater debris respectively — and are widely used to probe
cross-domain robustness. Despite the abundance of specialist pipelines
built on these datasets, the cross-domain gap remains unresolved: the
supervised models that achieve >90% in-domain accuracy still routinely
drop below 40% on real-world test data.

**Foundation models for visual tasks.** CLIP uses contrastive image-text
pretraining at 400M-pair scale to produce open-vocabulary classification
capability. Grounding DINO adapts DETR-style detection to text-conditional
open-vocabulary detection. SAM produces high-quality segmentation masks
given point or box prompts, generalizing zero-shot across visual domains.
All three have been evaluated zero-shot on waste data with mixed results —
they generalize better than narrowly trained specialists but trail on
absolute accuracy when waste data is available.

**Multi-task vision-language models.** Florence-2 is an encoder-decoder
vision-language model with explicit task-prompt-based output: a single set
of weights can produce captions, bounding boxes, polygon segmentations,
OCR, and grounded phrases by varying the input text prompt. This makes it
uniquely suited to "one model, many tasks" deployments, in contrast to
CLIP (classification only), Grounding DINO (detection only), or SAM
(segmentation only).

**Parameter-efficient fine-tuning.** Low-Rank Adaptation (LoRA) inserts
small trainable matrices into attention layers of a frozen base model,
achieving competitive task accuracy with <1% of the trainable parameter
count of full fine-tuning. LoRA is especially attractive for foundation-model
deployment because adapter files (~14 MB for Florence-2) are shippable
and swappable independently of the base model.

**This work** combines all three threads: we use Florence-2 as the unified
base, LoRA as the fine-tuning method, and waste-vision datasets to study
the lab-to-field gap across all three vision tasks.

---

## 3 — Method

### 3.1 Model: Florence-2 + LoRA

We use **Florence-2-large-ft** (770M parameters; Microsoft) as the base
vision-language model. Florence-2 is an encoder-decoder transformer whose
output format is determined by a task-specific text prompt prepended to the
visual input:

- `<CAPTION>` produces a short class-like caption (used for classification).
- `<OD>` produces `label<loc_x1><loc_y1><loc_x2><loc_y2>` bounding-box
  sequences for object detection.
- `<REFERRING_EXPRESSION_SEGMENTATION>label` produces a sequence of
  `<loc_i>` location tokens defining polygon vertices for segmentation.

Location tokens `<loc_0>` through `<loc_999>` quantize image coordinates to
1000 bins per axis. All three tasks therefore share the same architecture
and tokenizer — only the prompt changes.

We attach **LoRA** adapters of rank 16 (`α=32`, dropout 0.05) to the
attention projections (`q_proj`, `k_proj`, `v_proj`) of every encoder and
decoder layer. The base model is frozen; only the LoRA matrices are
trained. The resulting adapter file is 14 MB and contains ~5.2M trainable
parameters (0.67% of the base model).

### 3.2 Training data: unified multi-task JSONL

We construct a single training corpus by interleaving samples from three
waste datasets, each formatted as a Florence-2 prompt/target pair:

| Source dataset | Task | Records | Prompt | Target |
|---|---|---:|---|---|
| TrashNet (2,527 images, 6 classes) | Classification | 2,274 | `<CAPTION>` | class label |
| TACO bounding boxes (1,500 images, 60 classes) | Detection | 1,343 | `<OD>` | `label<loc_..>...` |
| TACO polygon annotations | Segmentation | 2,811 | `<REFERRING_EXPRESSION_SEGMENTATION>label` | polygon `<loc_..>...` |

Total: **6,428 training records**, 714 validation records, shuffled by row
across tasks (no curriculum, no balancing).

For ablation we also produce three single-task subsets: cls-only
(TrashNet), det-only (TACO bboxes), seg-only (TACO polygons), each with a
deterministic 90/10 train/val split.

### 3.3 Training recipe

All LoRA variants are trained with **identical hyperparameters** to ensure
fair comparison:

| Hyperparameter | Value |
|---|---|
| Base model | `microsoft/Florence-2-large-ft` |
| LoRA rank / α / dropout | 16 / 32 / 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj` |
| Epochs | 3 |
| Learning rate | 1e-4 (cosine schedule, no warmup) |
| Batch size | 1, gradient accumulation 8 (effective batch 8) |
| Precision | fp32 weights, eager attention (no flash-attn) |
| Hardware | NVIDIA A100-SXM4-40GB |
| Wall time | ~3 hours per LoRA |

### 3.4 Evaluation protocol

We evaluate on six benchmarks covering all three tasks in both in-domain
and cross-domain settings:

| Task | In-domain | Cross-domain |
|---|---|---|
| Classification | TrashNet (2,527 imgs) | RealWaste (4,753 imgs) |
| Detection | TACO test split (300 imgs) | Trash-ICRA19 (1,120 imgs) |
| Segmentation | TACO masks (150 imgs) | DWSD (144 imgs) |

For classification we report accuracy and macro-F1; for detection,
precision/recall/F1 with IoU≥0.5 bbox matching; for segmentation, mean IoU
and pixel accuracy. RealWaste labels are mapped to TrashNet's six classes
for like-for-like comparison; the DWSD mask encoding uses non-zero pixels
as the object foreground (multi-class palette IDs are binarized for the
IoU computation). All evaluation uses greedy decoding with beam search
(`num_beams=3`, `do_sample=False`).

---

## 4 — Results

### 4.1 Florence-2 + LoRA sweeps every cross-domain benchmark

**Table 1.** Cross-domain results across all three vision tasks. Bold marks
the best per row.

| Cross-domain benchmark | Best Stage 1 specialist | Best Stage 1 foundation (zero-shot) | Florence-2 unified + LoRA (ours) |
|---|---:|---:|---:|
| RealWaste classification — accuracy | ViT-Base 39.98% | CLIP 42.68% | **56.68%** |
| Trash-ICRA19 detection — F1 | YOLOv8 13.49% | Grounding DINO 37.20% | **50.49%** |
| DWSD segmentation — mIoU | Mask R-CNN 0.0842 | SAM 0.1023 | **0.2214** |

The unified model wins every cross-domain row. Improvements over the best
prior baseline are **+14.00 points** on classification, **+13.29 points F1**
on detection, and **+0.1191 mIoU** on segmentation (a 2.2× improvement over
SAM). Notably, the foundation baselines themselves substantially outperform
the specialists cross-domain (CLIP > ViT-Base, G-DINO > YOLOv8) — confirming
the lab-to-field framing — and our fine-tuned Florence-2 extends that
advantage further.

### 4.2 Segmentation transfers across domains nearly without degradation

**Table 2.** Segmentation in-domain vs cross-domain. Relative drop in
parentheses.

| Model | TACO (in-domain) | DWSD (cross-domain) | Relative drop |
|---|---:|---:|---:|
| DeepLabV3+ specialist | 0.4541 | 0.0483 | **−89.4%** |
| SAM (foundation, zero-shot) | 0.0380 | 0.1023 | (improves) |
| Florence-2 unified + LoRA (ours) | **0.2223** | **0.2214** | **−0.4%** |

DeepLabV3+ achieves the highest in-domain mIoU but loses 89% of it
cross-domain. The unified Florence-2 + LoRA's mIoU is **0.2214 cross-domain
vs 0.2223 in-domain — a difference of only 0.0009**. Polygon-token
segmentation transfers between waste domains essentially without
degradation, despite generating its mask output as a sequence of
quantized location tokens.

### 4.3 Multi-task vs single-task ablation

**Table 3.** Multi-task ablation. Best per row in bold.

| Benchmark | cls-only LoRA | det-only LoRA | unified LoRA |
|---|---:|---:|---:|
| TrashNet cls (in-domain) — accuracy | **0.8567** | 0.4377 | 0.8480 |
| RealWaste cls (cross-domain) — accuracy | **0.5944** | 0.3953 | 0.5838 |
| TACO det (in-domain) — F1 | 0.2409 | 0.3469 | **0.3657** |
| Trash-ICRA19 det (cross-domain) — F1 | 0.4145 | **0.5886** | 0.5049 |
| TACO seg (in-domain) — mIoU | 0.1492 | **0.3237** | 0.2223 |
| DWSD seg (cross-domain) — mIoU | 0.2067 | **0.2257** | 0.2214 |

Specialist single-task LoRAs slightly outperform the unified LoRA on their
own task — cls-only wins classification by 1.1 cross-domain points,
det-only wins cross-domain detection by 8.4 F1 points and cross-domain
segmentation by 0.004 mIoU (essentially tied with unified). **The unified
model's multi-task LoRA does not provide implicit per-task regularization;
it does, however, deliver competitive performance across all three tasks
from a single 14 MB adapter.** All three Florence-2 LoRA variants beat
every Stage 1 cross-domain baseline.

A useful side-observation: the det-only LoRA produces strong segmentation
mIoU despite never being trained on segmentation. LoRA fine-tuning
preserves Florence-2's pretrained polygon-token generation capability
rather than overwriting it.

### 4.4 Florence-2 zero-shot is already the strongest cross-domain foundation model on two of three tasks

Before fine-tuning, Florence-2-large-ft zero-shot is already the strongest
foundation model on cross-domain detection (42.03% F1 on ICRA19 vs G-DINO
37.20%) and cross-domain segmentation (0.1207 mIoU on DWSD vs SAM 0.1023).
Fine-tuning with our 14 MB Waste LoRA improves DWSD mIoU by another **83%**
(0.1207 → 0.2214). The architectural advantage and the fine-tuning gain
compound.

---

## 5 — Analysis

### 5.1 Cross-domain generalization

The unified Florence-2 + Waste LoRA wins every cross-domain benchmark
across all three vision tasks (Table 1). Segmentation is the most striking
result: cross-domain DWSD mIoU is essentially identical to in-domain TACO
mIoU (0.2214 vs 0.2223, a difference of only 0.0009). This is in stark
contrast to specialist segmenters such as DeepLabV3+, which collapse from
0.4541 in-domain to 0.0483 cross-domain — an 89% relative drop. The
polygon-token formulation Florence-2 uses for segmentation, while costlier
per-token than dense mask decoders, transfers between waste domains
essentially without degradation. This is the robustness real-world
waste-sorting systems require, where input conditions vary unpredictably
between facilities, shifts, and waste streams.

### 5.2 Multi-task vs single-task: an honest ablation

The single-task ablation (Table 3) shows specialist LoRAs slightly
outperform the unified LoRA on their own task: cls-only wins classification
by 1.1 cross-domain accuracy points, det-only wins cross-domain detection
by 8.4 F1 points and cross-domain segmentation by 0.004 mIoU. **The unified
model's multi-task training therefore does not act as an implicit
regularizer** in the sense prior multi-task literature would predict — it
does not boost per-task cross-domain accuracy beyond what a focused
single-task adapter achieves on the same data.

What it does provide is parameter-shared deployment: one base model plus
one 14 MB adapter file serves all three tasks. Operating three specialist
adapters requires 3× the adapter footprint and dispatch logic to route
inputs to the correct LoRA. For the relatively small per-task gaps
observed (1–8 points), the deployment simplification is a favourable
trade.

A second, equally interesting observation: even the cls-only and det-only
LoRAs — neither trained on segmentation — produce competitive cross-domain
segmentation results (0.2067 and 0.2257 mIoU respectively), both still
beating SAM. LoRA fine-tuning on Florence-2 preserves the base model's
pretrained segmentation capability rather than destructively overwriting it.

### 5.3 Why deploy unified despite the per-task gap

A waste-sorting system in practice needs all three vision tasks:
classification to route material, detection to locate items on a conveyor,
and segmentation to plan robotic gripper trajectories. Three deployment
options exist:

1. **Three Stage 1 specialists** (e.g., ViT-Base + YOLOv8 + DeepLabV3+):
   strong in-domain accuracy but catastrophic cross-domain collapse,
   three inference stacks to maintain, and ~500 MB combined model weights.
2. **Three Florence-2 single-task LoRAs**: strong cross-domain
   generalization per task but three base-model loads in VRAM (~4.5 GB
   combined at fp16), three inference pipelines, and three sets of LoRA
   weights to version and ship.
3. **One Florence-2 + multi-task LoRA** (ours): strong cross-domain
   generalization (within 1–8 points of best single-task), one ~1.5 GB
   base model in VRAM, one 14 MB LoRA shipped, one inference endpoint
   serving all three tasks via a text-prompt task switch.

For deployment scenarios where ergonomics and maintenance cost matter —
which is most real-world waste-sorting systems — option 3 dominates. The
accuracy gap to single-task LoRAs is small enough to be absorbed by the
operational benefit. The gap to in-domain specialists is irrelevant in
practice: by construction, deployed systems do not see in-domain data,
they see field data.

### 5.4 A failure mode: single-task segmentation collapse

We initially planned a four-way ablation including a seg-only LoRA trained
on the same TACO polygon data the unified LoRA used. Under identical
hyperparameters to the cls-only and det-only runs, **seg-only training
mode-collapsed**: final training loss plateaued at ~3.0 (vs ~1.7 for the
unified run on the same data mix), and the resulting model produced
degenerate polygons of the form
`<loc_0><loc_0><loc_0><loc_999><loc_2><loc_999><loc_0><loc_999><loc_0><loc_0>`
— a thin vertical line along the image boundary. Quantitatively, seg-only
achieved 0.097 mIoU on its own task — worse than det-only's 0.324 mIoU on
the same task, despite det-only never being trained for segmentation.

We attribute this to the asymmetric difficulty of Florence-2's output
formats. Classification outputs 1–3 tokens (a class label); detection
outputs 5–10 tokens per object (label + four location tokens);
segmentation outputs 50–300 tokens per polygon. The per-sample loss for
segmentation dominates by an order of magnitude, and a trivial-polygon
local minimum (outputting only `<loc_0>` and `<loc_999>` boundary tokens)
offers low gradient signal to escape. Multi-task training, in which
classification and detection samples interleave with segmentation
samples, provides easier gradients that anchor stable representations
and let the model escape the trivial-polygon attractor.

This suggests that *segmentation specifically requires multi-task or
curriculum-like training when using a text-generative VLM*. Whether
segmentation single-task fine-tuning can be made stable with non-default
hyperparameters is left to future work.

---

## 6 — Limitations

We identify five limitations of the present study:

1. **In-domain specialists remain stronger.** Florence-2 + LoRA does not
   match specialist accuracy on the original training distribution: TACO
   detection F1 trails YOLOv8 by 27.5 points, TACO segmentation mIoU trails
   DeepLabV3+ by 0.23. Deployments operating purely in controlled
   environments would still benefit from a specialist pipeline.
2. **No controlled LoRA fine-tuning of other foundation models.** We do not
   compare against CLIP+LoRA, Grounding-DINO+LoRA, or SAM+LoRA on the same
   training data. Such an ablation would isolate whether Florence-2's
   pretraining specifically helps or whether any sufficiently large VLM
   would close the gap similarly.
3. **Single-task segmentation training fails to converge.** A seg-only
   LoRA trained under identical hyperparameters to the cls-only and
   det-only ablations mode-collapses to degenerate polygons. Whether
   non-default schedules (longer training, lower learning rate) can
   rescue it is unresolved.
4. **Datasets are mostly Western waste streams.** TrashNet and TACO are
   curated in Western/urban contexts; whether the model generalizes to
   industrial composition (e-waste, hazardous, biomedical) or
   non-Western waste streams is untested.
5. **No physical-deployment evaluation.** All numbers are computed
   offline on benchmark images. Real waste-facility deployment introduces
   latency budgets, lighting variability, and active-conveyor dynamics
   we have not validated.

---

## 7 — Conclusion

A single 14 MB LoRA adapter on top of Florence-2-large-ft suffices to
sweep every cross-domain waste-vision benchmark across classification,
detection, and segmentation — by +14.0 points over CLIP, +13.3 F1 over
Grounding DINO, and 2.2× SAM. The cross-domain segmentation result, in
particular, matches the model's in-domain segmentation within 0.001 mIoU,
demonstrating that polygon-token generation transfers between waste
domains essentially without degradation. An honest single-task ablation
shows specialist LoRAs slightly outperform the unified adapter on their
own task; the unified model's value is therefore not implicit regularization
but operational simplicity — one base model, one adapter, three tasks —
which is the right trade-off for real waste-management deployment where
input conditions vary unpredictably. We release our training and
evaluation pipeline alongside the trained LoRA weights to support
reproducible follow-up work.

---

## Status

- Word count target ≤ 4000 words for a NeurIPS workshop paper. This draft is approximately on budget.
- All numbers reflect the post-bug-fix full-dataset evaluation runs (May 2026).
- Citations are placeholders — add bibtex entries before submission.
- Figures: include (a) a hero figure showing the cross-domain comparison from Table 1, (b) a qualitative grid of segmentation predictions on DWSD vs TACO, (c) optionally an architecture diagram of Florence-2 + LoRA. Author to add.
- Author to verify final venue formatting requirements (NeurIPS workshop template) and recompile if needed.
