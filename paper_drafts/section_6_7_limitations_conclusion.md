# Section 6 — Limitations (draft v1, ~210 words)

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
   non-default schedules (longer training, lower learning rate) can rescue
   it is unresolved.
4. **Datasets are mostly Western waste streams.** TrashNet and TACO are
   curated in Western/urban contexts; whether the model generalizes to
   industrial composition (e-waste, hazardous, biomedical) or
   non-Western waste streams is untested.
5. **No physical-deployment evaluation.** All numbers are computed
   offline on benchmark images. Real waste-facility deployment introduces
   latency budgets, lighting variability, and active-conveyor dynamics
   we have not validated.

---

# Section 7 — Conclusion (draft v1, ~180 words)

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

## Notes for the author

- §6 explicitly admits the negative result on seg-only and the missing CLIP+LoRA / SAM+LoRA ablations. Reviewers respect honest limitations.
- §7 deliberately echoes the abstract structure for symmetry — sweep, transfer-without-degradation, honest ablation, deployment claim.
- The "release pipeline + weights" claim is conditional on actually publishing them; remove if you decide otherwise before submission.
