# Section 3 — Method (draft v1, ~500 words)

## 3.1 Model: Florence-2 + LoRA

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

We attach **LoRA** [Hu et al., 2022] adapters of rank 16 (`α=32`, dropout
0.05) to the attention projections (`q_proj`, `k_proj`, `v_proj`) of every
encoder and decoder layer. The base model is frozen; only the LoRA
matrices are trained. The resulting adapter file is 14 MB and contains
~5.2M trainable parameters (0.67% of the base model).

## 3.2 Training data: unified multi-task JSONL

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

## 3.3 Training recipe

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

We use the HuggingFace `Trainer` with PEFT, no curriculum learning, and the
naturally-skewed task distribution (35% cls, 21% det, 44% seg) of the
unified JSONL.

## 3.4 Evaluation protocol

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
IoU computation).

All evaluation uses greedy decoding with beam search (`num_beams=3`,
`do_sample=False`) to keep the comparison deterministic. Full-dataset
evaluation (`max-images=0`) is reported throughout.

---

## Notes for the author

- §3.1 references the actual training-time hyperparameters from `train_unified.sh`.
- §3.4's "binarized" note on DWSD is important — it's the bug-fix we applied to the evaluator. Mentioning it preempts the obvious reviewer question.
- Adapter size is 14 MB (matches `adapter_model.safetensors`). The 5.2M trainable params is from `model.print_trainable_parameters()`.
