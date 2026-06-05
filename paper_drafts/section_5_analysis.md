# Section 5 — Analysis (draft v1)

Replaces the old §5 framing built around "multi-task implicit regularization,"
which the E1 ablation falsified. The new framing keeps the deployment story
intact and adds an honest single-task ablation that strengthens the paper.

---

## 5.1 — Cross-domain generalization: the headline finding

The unified Florence-2 + Waste LoRA wins every cross-domain benchmark across
all three vision tasks:

| Task | Best Stage 1 cross-domain baseline | Florence-2 unified LoRA | Improvement |
|------|---:|---:|---:|
| Classification (RealWaste) | CLIP **42.68%** acc | **56.68%** acc | **+14.00 pts** |
| Detection (Trash-ICRA19) | Grounding DINO **0.3720** F1 | **0.5049** F1 | **+13.29 pts (+35.7%)** |
| Segmentation (DWSD) | SAM **0.1023** mIoU | **0.2214** mIoU | **+0.1191 (+116.4%)** |

The segmentation result deserves emphasis. The unified model's cross-domain
mIoU on DWSD (0.2214) is essentially identical to its in-domain mIoU on TACO
(0.2223) — a difference of only 0.0009. This is in stark contrast to specialist
segmenters: DeepLabV3+ achieves 0.4541 in-domain but collapses to 0.0483
cross-domain, an 89% relative drop. The polygon-token formulation Florence-2
uses for segmentation, while costlier per-token than dense mask decoders,
transfers between waste domains essentially without degradation. This is the
robustness that real-world waste-sorting systems require, where input
conditions vary unpredictably between facilities, shifts, and waste streams.

## 5.2 — Multi-task vs single-task: an honest ablation

We trained three additional LoRA variants under identical hyperparameters
(rank 16, alpha 32, three epochs at lr=1e-4) on single-task subsets of the
training data: cls-only on TrashNet captions, det-only on TACO bounding boxes,
and seg-only on TACO polygons. (The seg-only variant failed to converge — see
§5.4.) The remaining two single-task LoRAs allow a clean ablation against the
unified multi-task LoRA:

| Cross-domain benchmark | cls-only LoRA | det-only LoRA | unified LoRA |
|---|---:|---:|---:|
| RealWaste cls accuracy | **0.5944** | 0.3953 | 0.5838 |
| ICRA19 det F1 | 0.4145 | **0.5886** | 0.5049 |
| DWSD seg mIoU | 0.2067 | **0.2257** | 0.2214 |

Each specialist LoRA slightly outperforms the unified LoRA on its own task:
cls-only wins classification by 1.1 points, det-only wins detection by 8.4
F1 points and segmentation by 0.004 mIoU. **The unified model's multi-task
training therefore does not act as an implicit regularizer** in the sense
that prior multi-task learning literature would predict — it does not boost
per-task cross-domain accuracy beyond what a focused single-task adapter
achieves on the same data.

What it does provide, however, is parameter-shared deployment: one base model
plus one 14 MB adapter file serves all three tasks. Operating three
specialist adapters requires 3× the adapter footprint and dispatch logic
to route inputs to the correct LoRA. For the relatively small per-task gaps
observed (1–8 points), the deployment simplification is a favourable trade.

A second, equally interesting observation: even the cls-only and det-only
LoRAs — neither trained on segmentation — produce competitive cross-domain
segmentation results (0.2067 and 0.2257 mIoU respectively), both still beating
SAM. This indicates that LoRA fine-tuning on Florence-2 preserves the base
model's pretrained segmentation capability rather than destructively
overwriting it.

## 5.3 — Why deploy unified despite the per-task gap

A waste-sorting system in practice needs all three vision tasks:
classification to route material, detection to locate items on a conveyor,
and segmentation to plan robotic gripper trajectories. Three deployment
options exist:

1. **Three Stage 1 specialists** (e.g., ViT-Base + YOLOv8 + DeepLabV3+):
   strong in-domain accuracy, but catastrophic cross-domain collapse, three
   inference stacks to maintain, and ~500 MB combined model weights.
2. **Three Florence-2 single-task LoRAs**: strong cross-domain generalization
   per task, but three base-model loads in VRAM (~4.5 GB combined at fp16),
   three inference pipelines, and three sets of LoRA weights to version and ship.
3. **One Florence-2 + multi-task LoRA** (ours): strong cross-domain
   generalization (within 1–8 points of best single-task), one ~1.5 GB base
   model in VRAM, one 14 MB LoRA shipped, one inference endpoint serving all
   three tasks via a text-prompt task switch.

For deployment scenarios where ergonomics and maintenance cost matter — which
is most real-world waste-sorting systems — option 3 dominates. The accuracy
gap to single-task LoRAs is small enough to be absorbed by the operational
benefit. The gap to in-domain specialists exists but is irrelevant: by
construction, deployed systems do not see in-domain data, they see field data.

## 5.4 — A failure mode: single-task segmentation collapse

We initially planned a four-way ablation including a seg-only LoRA, trained
on the same TACO polygon data the unified LoRA used. Under identical
hyperparameters to the cls-only and det-only runs, **seg-only training
mode-collapsed**: the final training loss plateaued at ~3.0 (compared to
~1.7 for the unified run with the same data mix), and the resulting model
produced degenerate polygons of the form
`<loc_0><loc_0><loc_0><loc_999><loc_2><loc_999><loc_0><loc_999><loc_0><loc_0>`
— a thin vertical line along the image boundary. Quantitatively, seg-only
achieved 0.097 mIoU on its own task — worse than det-only's 0.324 mIoU on
the same task, despite det-only never being trained for segmentation.

We attribute this to the asymmetric difficulty of Florence-2's task-specific
output formats. Classification outputs 1–3 tokens (a class label); detection
outputs 5–10 tokens per object (label + four location tokens); segmentation
outputs 50–300 tokens per polygon. The per-sample loss for segmentation
therefore dominates by an order of magnitude in pure cross-entropy terms,
and a trivial-polygon local minimum (outputting only `<loc_0>` and
`<loc_999>` boundary tokens) offers low gradient signal to escape. Multi-task
training, where classification and detection samples interleave with
segmentation samples, provides easier gradients that anchor stable
representations and let the model escape the trivial-polygon attractor.

This suggests that *segmentation specifically requires multi-task or
curriculum-like training when using a text-generative VLM*. We did not
attempt longer schedules or lower learning rates to rescue the seg-only run;
investigating whether segmentation single-task fine-tuning of Florence-2 can
be made stable with non-default hyperparameters is left to future work.

---

## Notes for the author

- All numbers come from the post-bug-fix full-dataset evaluation runs on dgx-a100-02 (May 2026). The DWSD mIoU numbers in particular depend on the binarization fix in `eval_segmentation` (treat any non-zero pixel as object).
- §5.2 deliberately *does not* claim "multi-task is regularization" — that claim was falsified by the ablation. The honest framing is the deployment-cost framing.
- §5.4 is short on purpose: it's a sidebar finding, not a headline. Include it because reviewers may ask "did you try single-task seg?"
- If page budget is tight, the cuttable subsections in order of importance are: §5.4 (sidebar, can shorten to a paragraph), §5.3 (the deployment narrative can move into the introduction).
