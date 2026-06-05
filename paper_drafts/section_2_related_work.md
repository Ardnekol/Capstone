# Section 2 — Related Work (draft v1, ~330 words)

**Waste-vision datasets and specialist models.** A growing body of work
applies standard supervised vision pipelines to waste sorting. TrashNet
[Yang & Thung, 2016] provides 2,527 lab-condition images of six waste
categories and serves as a common benchmark for classification specialists
(ViT, ResNet, EfficientNet). TACO [Proença & Simões, 2020] extends to urban
outdoor litter with bounding box and polygon annotations, supporting
detection and segmentation. RealWaste [Single et al., 2023] and TrashCan /
Trash-ICRA19 [Hong et al., 2020] provide field-condition test sets — real
waste-sorting bins and underwater debris respectively — and are widely used
to probe cross-domain robustness. Despite the abundance of specialist
pipelines built on these datasets, the cross-domain gap remains unresolved:
the supervised models that achieve >90% in-domain accuracy still routinely
drop below 40% on real-world test data.

**Foundation models for visual tasks.** CLIP [Radford et al., 2021] uses
contrastive image-text pretraining at 400M-pair scale to produce open-vocabulary
classification capability. Grounding DINO [Liu et al., 2023] adapts
DETR-style detection to text-conditional open-vocabulary detection. SAM
[Kirillov et al., 2023] produces high-quality segmentation masks given point
or box prompts, generalizing zero-shot across visual domains. All three
have been evaluated zero-shot on waste data with mixed results — they
generalize better than narrowly trained specialists but trail on absolute
accuracy when waste data is available.

**Multi-task vision-language models.** Florence-2 [Xiao et al., 2024] is an
encoder-decoder vision-language model with explicit task-prompt-based
output: a single set of weights can produce captions, bounding boxes,
polygon segmentations, OCR, and grounded phrases by varying the input text
prompt. This makes it uniquely suited to "one model, many tasks" deployments,
in contrast to CLIP (classification only), Grounding DINO (detection only),
or SAM (segmentation only).

**Parameter-efficient fine-tuning.** Low-Rank Adaptation (LoRA) [Hu et al.,
2022] inserts small trainable matrices into attention layers of a frozen
base model, achieving competitive task accuracy with <1% of the trainable
parameter count of full fine-tuning. LoRA is especially attractive for
foundation-model deployment because adapter files (~14 MB for Florence-2)
are shippable and swappable independently of the base model.

**This work** combines all three threads: we use Florence-2 as the unified
base, LoRA as the fine-tuning method, and waste-vision datasets to study
the lab-to-field gap across all three vision tasks.

---

## Notes for the author

- All citations are placeholders — fill in actual bibtex entries before submission.
- Compressed to ~330 words to fit workshop page budget. Can expand if the venue allows more.
- No multi-task-learning citations because we deliberately do NOT claim the multi-task regularization effect.
