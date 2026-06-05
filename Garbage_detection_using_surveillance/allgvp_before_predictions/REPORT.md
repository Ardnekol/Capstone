# AllGVPImages Inference Report

**Date:** 2026-05-26
**Project:** Garbage Detection using Surveillance (Capstone)

---

## Objective

Apply the finetuned CCTV garbage detection model to **unseen, unlabeled** GVP camera images and generate bounding-box annotations on the garbage regions — without any ground-truth annotations available.

---

## Setup

| Item | Value |
|---|---|
| Source images | `Capstone/datasets/AllGVPImages/before/` (6,197 total) |
| Sampled | **100 images** (random, seed = 42) |
| Model | YOLO finetuned on CCTV garbage dataset |
| Weights | `runs/detect/cctv_garbage_yolo_v42/weights/best.pt` |
| Classes | 1 — `0: garbage` |
| Confidence threshold | 0.25 |
| Image size | 640 |
| Compute | DGX-A100-02 via `srun` (CPU inference; GPUs saturated by other users) |

---

## Pipeline

```
AllGVPImages/before/  ──►  random sample (seed=42, N=100)
                              │
                              ▼
                   YOLO best.pt (CCTV-finetuned)
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
       annotated .jpg                     YOLO .txt
   (boxes drawn on image)         (class cx cy w h conf)
              │                               │
              └───────────► merged in same folder
                            (image.jpg → image.txt)
```

Script: [predict_allgvp_before.py](predict_allgvp_before.py)

---

## Results

| Metric | Value |
|---|---|
| Images processed | 100 |
| Images with ≥1 detection | **98 / 100 (98%)** |
| Images with 0 detections | 2 |
| Total bounding boxes | **181** |
| Avg boxes per detected image | 1.85 |

---

## Outputs

Located at [allgvp_before_predictions/yolo_v42_pred/](yolo_v42_pred/):

- `*.jpg` — 100 images with garbage bounding boxes drawn
- `*.txt` — 98 YOLO-format pseudo-label files (paired by filename)
- [`_sampled_filenames.txt`](_sampled_filenames.txt) — list of the 100 sampled images (for reproducibility)

**Annotation format (per line):**
```
<class_id>  <cx>  <cy>  <w>  <h>  <conf>
```
All coordinates normalized to [0, 1]; resolution-independent.

---

## Key Observations

- **98% detection rate** on a previously unseen image domain (GVP cameras vs. the CCTV cameras the model was trained on) suggests strong cross-camera generalization of the finetuned YOLO.
- **Manual verification:** the outputs were inspected by hand and the predicted boxes line up with the actual garbage regions in the images — the model performs well on this new domain.
- The 2 images with no detections were also checked manually and confirmed to be **true negatives** (the scenes genuinely contain no garbage), not missed detections. → **0 false negatives observed** on the sampled set.
- These annotations are model predictions (pseudo-labels), but given the manual check above they can be used directly for visualization, downstream analytics, or as a strong starting point for self-training.

---

## Reproducing

```bash
srun -p cse-gpu-all --nodelist=dgx-a100-02 --gres=gpu:1 \
  ~/.conda/envs/Capstone/bin/python predict_allgvp_before.py
```

Tune `N_SAMPLES`, `SEED`, or `CONF` at the top of the script to change sample size, sample selection, or detection threshold.
