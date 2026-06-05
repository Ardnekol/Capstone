# Abstract (draft v1, ~210 words)

Computer vision models trained on curated waste datasets in laboratory conditions
degrade dramatically when deployed in the field: classification accuracy drops
60–80%, detection F1 falls by similar margins, and segmentation models collapse
in unfamiliar visual contexts. We study this *lab-to-field gap* across three
core waste-vision tasks — classification, object detection, and segmentation —
using six benchmarks spanning in-domain (TrashNet, TACO) and cross-domain
(RealWaste, Trash-ICRA19, DWSD) splits.

We show that a single vision-language foundation model, Florence-2-large-ft,
fine-tuned with a 14 MB LoRA adapter on a unified multi-task waste dataset,
**sweeps every cross-domain benchmark**: +14.0 percentage points over CLIP on
RealWaste classification, +13.3 points over Grounding DINO on Trash-ICRA19
detection F1, and a 2.2× improvement over SAM on DWSD segmentation mIoU.
Remarkably, the model's cross-domain segmentation mIoU (0.2214) is statistically
indistinguishable from its in-domain mIoU (0.2223), indicating polygon-token
generation transfers between waste domains essentially without degradation.

An ablation across single-task LoRA variants shows that specialist adapters
slightly outperform the unified adapter on their own task, but the unified
model wins on operational simplicity — replacing three specialist deployments
with one shared base model and one set of adapter weights, suitable for
practical waste-sorting systems.

---

## Alternative 150-word version (if page-budget tight)

Vision models trained on curated lab waste datasets lose 60–80% of their
performance when deployed in the field. We close this *lab-to-field gap* by
fine-tuning Florence-2-large-ft with a 14 MB LoRA adapter on a unified
multi-task waste dataset. The resulting model sweeps every cross-domain
benchmark across classification, detection, and segmentation: +14.0 points
over CLIP on RealWaste, +13.3 points over Grounding DINO on Trash-ICRA19,
and 2.2× SAM on DWSD. Cross-domain segmentation mIoU is nearly identical
to in-domain (0.2214 vs 0.2223), demonstrating that polygon-token generation
transfers between waste domains essentially without degradation. Ablations
show specialist single-task LoRAs slightly outperform the unified adapter
on their own tasks, but the unified variant enables one-model deployment
for all three vision tasks — a favourable trade-off for real-world waste
sorting where conditions vary unpredictably.

---

## Notes for the author

- All numbers are from full-dataset evaluation runs on the deploy GPU; no max-images cap.
- "Sweeps every cross-domain benchmark" is the load-bearing claim — it relies on the +0.12 mIoU DWSD finding from the post-bug-fix evaluator run.
- Keep CLIP, G-DINO, SAM mentions — they are the most recognisable foundation baselines and reviewers will look for them.
- Do NOT claim "multi-task implicit regularization" — that claim was falsified by the E1 ablation. The honest framing is the deployment-cost framing.
