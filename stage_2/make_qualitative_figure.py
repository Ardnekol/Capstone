#!/usr/bin/env python3
"""Qualitative cross-domain figure: detection (row 1) + segmentation (row 2).

Layout — 2 rows x 4 columns (8 panels):
    Input | Ground truth | Baseline | Ours (Florence-2)
  Row 1 (detection):    baseline = YOLOv8m       (boxes)
  Row 2 (segmentation): baseline = DeepLabV3+    (masks)  [or --seg-baseline maskrcnn]

Reuses the EXACT prediction logic from cross_domain_eval/ (Florence region-proposal
for detection, cascade for segmentation, YOLOv8m, DeepLabV3+/Mask R-CNN). Nothing
fabricated.

EXAMPLE SELECTION (honesty): we do NOT pick the single best image (that would be a
misleading outlier). For each task we scan several images, compute BOTH models'
scores, and pick a *representative* example — the smallest positive margin where the
baseline is non-trivial (baseline metric >= MIN_BASE). All candidates are printed.

Run on a machine with the Capstone env (GPU or CPU):
    python make_qualitative_figure.py --device cpu
    python make_qualitative_figure.py --det-index 3 --seg-index 7   # force exact rows
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
CDE = HERE / "cross_domain_eval"
sys.path.insert(0, str(CDE))

from eval_segmentation import list_pairs, list_pairs_dwsd, gt_binary       # noqa: E402
from eval_detection import load_zerowaste, load_warpd, score_image         # noqa: E402

GREEN, RED, BLUE = (0.1, 0.8, 0.2), (0.95, 0.2, 0.2), (0.2, 0.45, 1.0)
MIN_BASE = 0.10   # baseline must do at least this well for a "fair" representative example


def resize_mask(mask, size_wh):
    im = Image.fromarray((mask.astype(np.uint8) * 255)).resize(size_wh, Image.NEAREST)
    return (np.array(im) > 127).astype(np.uint8)


def biou(pred, gt):
    inter, union = np.sum(pred & gt), np.sum(pred | gt)
    return float(inter) / float(union) if union > 0 else (1.0 if pred.sum() == 0 else 0.0)


def overlay(img, mask, color):
    base = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    m = resize_mask(mask, img.size).astype(bool)
    out = base.copy()
    out[m] = 0.45 * base[m] + 0.55 * np.array(color, np.float32)
    return np.clip(out, 0, 1)


def f1_image(pred_boxes, gt_boxes):
    tp, fp, fn = score_image(pred_boxes, gt_boxes)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * p * r / (p + r) if (p + r) else 0.0


def show_boxes(ax, img, boxes, color):
    ax.imshow(np.array(img.convert("RGB")) / 255.0)
    for b in boxes:
        x1, y1, x2, y2 = b[:4]
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                               edgecolor=color, linewidth=1.4))


def pick_representative(scored):
    """scored: list of (idx, ours, base). Pick smallest positive margin with a
    non-trivial baseline; else the example whose 'ours' is nearest the median."""
    fair = [(i, o, b) for (i, o, b) in scored if b >= MIN_BASE and o > b]
    if fair:
        return min(fair, key=lambda c: c[1] - c[2])[0]
    med = sorted(o for _, o, _ in scored)[len(scored) // 2]
    return min(scored, key=lambda c: abs(c[1] - med))[0]


# ── baselines ───────────────────────────────────────────────────────────────
def load_deeplab(weights, device):
    import torch
    import segmentation_models_pytorch as smp
    m = smp.DeepLabV3Plus(encoder_name="resnet101", encoder_weights=None,
                          in_channels=3, classes=1).to(device).eval()
    st = torch.load(weights, map_location="cpu")
    m.load_state_dict(st.get("state_dict", st) if isinstance(st, dict) else st)
    return m


def deeplab_mask(model, img, device):
    import torch
    from torchvision import transforms
    tf = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    with torch.no_grad():
        prob = torch.sigmoid(model(tf(img).unsqueeze(0).to(device)))[0, 0].cpu().numpy()
    return (prob > 0.5).astype(np.uint8)


def load_maskrcnn(weights, device):
    import torch
    from torchvision.models.detection import maskrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    state = torch.load(weights, map_location="cpu")
    sd = state.get("state_dict", state) if isinstance(state, dict) else state
    ncls = sd["roi_heads.box_predictor.cls_score.weight"].shape[0]
    model = maskrcnn_resnet50_fpn(weights=None)
    inf = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(inf, ncls)
    infm = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(infm, 256, ncls)
    model.load_state_dict(sd)
    return model.to(device).eval()


def maskrcnn_mask(model, img, device, thr=0.5):
    import torch
    import torchvision.transforms.functional as TF
    with torch.no_grad():
        out = model([TF.to_tensor(img).to(device)])[0]
    pred = np.zeros((img.height, img.width), np.uint8)
    for m, s in zip(out["masks"], out["scores"]):
        if float(s) >= thr:
            pred = np.maximum(pred, (m[0].cpu().numpy() > 0.5).astype(np.uint8))
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det-dataset", choices=["zerowaste", "warpd"], default="zerowaste")
    ap.add_argument("--seg-dataset", choices=["zerowaste", "dwsd"], default="dwsd")
    ap.add_argument("--seg-baseline", choices=["deeplab", "maskrcnn"], default="deeplab")
    ap.add_argument("--det-index", type=int, default=-1)
    ap.add_argument("--seg-index", type=int, default=-1)
    ap.add_argument("--scan", type=int, default=10)
    ap.add_argument("--zerowaste-root", default=str(CDE / "../../datasets/zerowaste-f"))
    ap.add_argument("--warpd-root", default=str(CDE / "../../datasets/WARP/Warp-D"))
    ap.add_argument("--dwsd-root",
                    default=str(CDE / "../../datasets/segmentation/Dense Waste Segmentation Dataset"))
    ap.add_argument("--yolo",
                    default=str(CDE / "../../stage_1/detection/runs/detect/yolov8m_20251225_182218/weights/best.pt"))
    ap.add_argument("--deeplab", default=str(CDE / "../../stage_1/segmentation/results/deeplabv3plus_best.pth"))
    ap.add_argument("--maskrcnn", default=str(CDE / "../../stage_1/segmentation/results/maskrcnn_best.pth"))
    ap.add_argument("--lora", default=str(CDE / "../finetuned/florence2_unified_multitask_lora"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=str(HERE / "eval_results/qualitative"))
    args = ap.parse_args()

    import torch
    device = args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    from florence_common import load_florence2, run_boxes, run_seg_mask_cascade

    print("Loading Florence-2 + LoRA ...")
    flor, proc = load_florence2(args.lora, device)

    # ── DETECTION ──
    det_items = (load_zerowaste(Path(args.zerowaste_root).resolve()) if args.det_dataset == "zerowaste"
                 else load_warpd(Path(args.warpd_root).resolve()))
    print(f"Detection: {len(det_items)} images ({args.det_dataset}). Loading YOLOv8m ...")
    from ultralytics import YOLO
    yolo = YOLO(args.yolo)

    def yolo_boxes(path):
        r = yolo(str(path), verbose=False)[0]
        return r.boxes.xyxy.cpu().numpy().tolist() if r.boxes is not None else []

    if args.det_index >= 0:
        det_pick = args.det_index
    else:
        scored = []
        for i in range(min(args.scan, len(det_items))):
            p, gt = det_items[i]
            of = f1_image(run_boxes(flor, proc, Image.open(p).convert("RGB"), device, "region_proposal"), gt)
            ob = f1_image(yolo_boxes(p), gt)
            scored.append((i, of, ob))
            print(f"  det[{i}] {Path(p).name}: Ours F1={of:.3f} | YOLO F1={ob:.3f}")
        det_pick = pick_representative(scored)
    dpath, dgt = det_items[det_pick]
    dimg = Image.open(dpath).convert("RGB")
    df, dy = run_boxes(flor, proc, dimg, device, "region_proposal"), yolo_boxes(dpath)
    print(f"  -> detection row = index {det_pick}: {Path(dpath).name} "
          f"(Ours F1={f1_image(df, dgt):.3f}, YOLO F1={f1_image(dy, dgt):.3f})")

    # ── SEGMENTATION ──
    seg_pairs = (list_pairs_dwsd(Path(args.dwsd_root).resolve(), 0) if args.seg_dataset == "dwsd"
                 else list_pairs(Path(args.zerowaste_root).resolve(), 0))
    if args.seg_baseline == "deeplab":
        print(f"Segmentation: {len(seg_pairs)} pairs ({args.seg_dataset}). Loading DeepLabV3+ ...")
        seg_model = load_deeplab(args.deeplab, device)
        base_fn, base_name = (lambda im: deeplab_mask(seg_model, im, device)), "DeepLabV3+"
    else:
        print(f"Segmentation: {len(seg_pairs)} pairs ({args.seg_dataset}). Loading Mask R-CNN ...")
        seg_model = load_maskrcnn(args.maskrcnn, device)
        base_fn, base_name = (lambda im: maskrcnn_mask(seg_model, im, device)), "Mask R-CNN"

    def seg_iou(mask, im, gtm):
        return biou(resize_mask(mask, im.size), resize_mask(gtm, im.size))

    if args.seg_index >= 0:
        seg_pick = args.seg_index
    else:
        scored = []
        for i in range(min(args.scan, len(seg_pairs))):
            ip, mp = seg_pairs[i]
            im = Image.open(ip).convert("RGB"); gtm = gt_binary(mp)
            of = seg_iou(run_seg_mask_cascade(flor, proc, im, device, phrase="waste"), im, gtm)
            ob = seg_iou(base_fn(im), im, gtm)
            scored.append((i, of, ob))
            print(f"  seg[{i}] {Path(ip).name}: Ours mIoU={of:.3f} | {base_name} mIoU={ob:.3f}")
        seg_pick = pick_representative(scored)
    spath, smp_path = seg_pairs[seg_pick]
    simg = Image.open(spath).convert("RGB"); sgt = gt_binary(smp_path)
    sf, sb = run_seg_mask_cascade(flor, proc, simg, device, phrase="waste"), base_fn(simg)
    f_iou, b_iou = seg_iou(sf, simg, sgt), seg_iou(sb, simg, sgt)
    print(f"  -> segmentation row = index {seg_pick}: {Path(spath).name} "
          f"(Ours mIoU={f_iou:.3f}, {base_name} mIoU={b_iou:.3f})")

    # ── render 2x4 ──
    fig, ax = plt.subplots(2, 4, figsize=(8.2, 4.4))
    titles = ["Input", "Ground truth", "Baseline", "Ours (Florence-2)"]
    show_boxes(ax[0, 0], dimg, [], None)
    show_boxes(ax[0, 1], dimg, dgt, GREEN)
    show_boxes(ax[0, 2], dimg, dy, RED);  ax[0, 2].set_xlabel(f"YOLOv8m  F1 {f1_image(dy, dgt):.2f}", fontsize=8)
    show_boxes(ax[0, 3], dimg, df, BLUE); ax[0, 3].set_xlabel(f"F1 {f1_image(df, dgt):.2f}", fontsize=8)
    ax[1, 0].imshow(np.array(simg) / 255.0)
    ax[1, 1].imshow(overlay(simg, sgt, GREEN))
    ax[1, 2].imshow(overlay(simg, sb, RED));  ax[1, 2].set_xlabel(f"{base_name}  mIoU {b_iou:.2f}", fontsize=8)
    ax[1, 3].imshow(overlay(simg, sf, BLUE)); ax[1, 3].set_xlabel(f"mIoU {f_iou:.2f}", fontsize=8)
    for c in range(4):
        ax[0, c].set_title(titles[c], fontsize=9)
    ax[0, 0].set_ylabel("Detection", fontsize=9)
    ax[1, 0].set_ylabel("Segmentation", fontsize=9)
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    plt.tight_layout()

    out_dir = Path(args.out).resolve(); out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "qualitative_panel"
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", dpi=200)
    fig.savefig(f"{stem}.png", bbox_inches="tight", dpi=200)
    print(f"\nSaved:\n  {stem}.pdf\n  {stem}.png")


if __name__ == "__main__":
    main()
