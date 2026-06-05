#!/usr/bin/env python3
"""Cross-domain SEGMENTATION comparison on ZeroWaste-f.

Three regimes, ONE consistent metric (binary mIoU + pixel accuracy, any non-zero
GT pixel = foreground — identical to stage_2/evaluate_unified_model.py and
stage_1 eval_sam.py).

  1. Task-Specific : DeepLabV3+ (resnet101, binary) — Stage 1 checkpoint
  2. Foundation    : SAM ViT-H (automatic mask generator, best mask by pred IoU)
  3. Unified FT     : Florence-2 + LoRA (<REFERRING_EXPRESSION_SEGMENTATION>waste)

Dataset: ZeroWaste-f test split (datasets/zerowaste-f/splits_final_deblurred/test)
  images : data/*.PNG    masks : sem_seg/*.PNG  (>0 = waste foreground)

Each regime guarded by try/except on imports → missing deps skip cleanly.

Usage:
  python3 eval_segmentation.py --max-images 0 \
      --deeplab ../../stage_1/segmentation/results/deeplabv3plus_best.pth \
      --sam ../../stage_1/segmentation/sam_vit_h_4b8939.pth \
      --lora ../finetuned/florence2_unified_multitask_lora \
      --output-dir ../eval_results/segmentation_zerowaste
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
def list_pairs(root: Path, max_images: int) -> List[Tuple[Path, Path]]:
    test = root / "splits_final_deblurred" / "test"
    data, seg = test / "data", test / "sem_seg"
    pairs = []
    for p in sorted(data.iterdir()):
        if p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        m = seg / p.name
        if not m.exists():
            m = seg / (p.stem + ".PNG")
        if m.exists():
            pairs.append((p, m))
    return pairs[:max_images] if max_images > 0 else pairs


def list_pairs_dwsd(root: Path, max_images: int) -> List[Tuple[Path, Path]]:
    """DWSD test: Image/img_*.png ↔ Mask_renamed/img_*.png (same filename)."""
    img_dir = root / "DSWD" / "Test" / "Image"
    mask_dir = root / "DSWD" / "Test" / "Mask_renamed"
    pairs = []
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        m = mask_dir / p.name
        if m.exists():
            pairs.append((p, m))
    return pairs[:max_images] if max_images > 0 else pairs


def gt_binary(mask_path: Path) -> np.ndarray:
    return (np.array(Image.open(mask_path).convert("L")) > 0).astype(np.uint8)


def iou_pixacc(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    if pred.shape != gt.shape:
        pred = np.array(Image.fromarray(pred * 255).resize(
            (gt.shape[1], gt.shape[0]), Image.NEAREST)) > 127
        pred = pred.astype(np.uint8)
    inter = np.sum(pred & gt)
    union = np.sum(pred | gt)
    iou = float(inter) / float(union) if union > 0 else (1.0 if pred.sum() == 0 else 0.0)
    pix = float(np.sum(pred == gt)) / float(gt.size)
    return iou, pix


# ──────────────────────────────────────────────────────────────────────────────
# Regime predictors → yield per-image binary mask, then aggregate mIoU
# ──────────────────────────────────────────────────────────────────────────────
def eval_deeplab(weights: str, pairs, device: str) -> Dict:
    import torch
    import segmentation_models_pytorch as smp
    from torchvision import transforms
    model = smp.DeepLabV3Plus(encoder_name="resnet101", encoder_weights=None,
                              in_channels=3, classes=1).to(device).eval()
    state = torch.load(weights, map_location="cpu")
    model.load_state_dict(state.get("state_dict", state) if isinstance(state, dict) else state)
    tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ious, pixs = [], []
    with torch.no_grad():
        for i, (ip, mp) in enumerate(pairs):
            img = Image.open(ip).convert("RGB")
            x = tf(img).unsqueeze(0).to(device)
            logit = model(x)
            prob = torch.sigmoid(logit)[0, 0].cpu().numpy()
            pred = (prob > 0.5).astype(np.uint8)
            io, px = iou_pixacc(pred, gt_binary(mp))
            ious.append(io); pixs.append(px)
            if (i + 1) % 100 == 0:
                print(f"    [DeepLab] {i + 1}/{len(pairs)}")
    return {"mIoU": round(float(np.mean(ious)), 4), "pixel_acc": round(float(np.mean(pixs)), 4),
            "evaluated": len(ious), "per_image_iou": [round(float(x), 4) for x in ious]}


def eval_unet(weights: str, pairs, device: str) -> Dict:
    import torch
    import segmentation_models_pytorch as smp
    from torchvision import transforms
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                     in_channels=3, classes=1).to(device).eval()
    state = torch.load(weights, map_location="cpu")
    model.load_state_dict(state.get("state_dict", state) if isinstance(state, dict) else state)
    tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ious, pixs = [], []
    with torch.no_grad():
        for i, (ip, mp) in enumerate(pairs):
            img = Image.open(ip).convert("RGB")
            x = tf(img).unsqueeze(0).to(device)
            prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
            pred = (prob > 0.5).astype(np.uint8)
            io, px = iou_pixacc(pred, gt_binary(mp))
            ious.append(io); pixs.append(px)
            if (i + 1) % 100 == 0:
                print(f"    [UNet] {i + 1}/{len(pairs)}")
    return {"mIoU": round(float(np.mean(ious)), 4), "pixel_acc": round(float(np.mean(pixs)), 4),
            "evaluated": len(ious), "per_image_iou": [round(float(x), 4) for x in ious]}


def eval_maskrcnn(weights: str, pairs, device: str, score_thr: float = 0.5) -> Dict:
    import torch
    import torchvision.transforms.functional as TF
    from torchvision.models.detection import maskrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    state = torch.load(weights, map_location="cpu")
    sd = state.get("state_dict", state) if isinstance(state, dict) else state
    ncls = sd["roi_heads.box_predictor.cls_score.weight"].shape[0]  # auto-infer (=2)
    model = maskrcnn_resnet50_fpn(weights=None)
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, ncls)
    in_feat_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_feat_mask, 256, ncls)
    model.load_state_dict(sd)
    model = model.to(device).eval()
    ious, pixs = [], []
    with torch.no_grad():
        for i, (ip, mp) in enumerate(pairs):
            img = Image.open(ip).convert("RGB")
            out = model([TF.to_tensor(img).to(device)])[0]
            # Combine instance masks into one binary foreground (mirrors train_maskrcnn).
            pred = np.zeros((img.height, img.width), dtype=np.uint8)
            for m, s in zip(out["masks"], out["scores"]):
                if float(s) < score_thr:
                    continue
                pred = np.maximum(pred, (m[0].cpu().numpy() > 0.5).astype(np.uint8))
            io, px = iou_pixacc(pred, gt_binary(mp))
            ious.append(io); pixs.append(px)
            if (i + 1) % 50 == 0:
                print(f"    [MaskRCNN] {i + 1}/{len(pairs)}")
    return {"mIoU": round(float(np.mean(ious)), 4), "pixel_acc": round(float(np.mean(pixs)), 4),
            "evaluated": len(ious), "per_image_iou": [round(float(x), 4) for x in ious]}


def eval_sam(checkpoint: str, pairs, device: str) -> Dict:
    import torch
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    sam = sam_model_registry["vit_h"](checkpoint=checkpoint).to(device)
    # Shared-GPU friendly: fewer query points + smaller batch drastically cut peak
    # memory (we only need the single best mask as foreground, so dense sampling
    # is unnecessary). points_per_side 32->16, batch 64.
    gen = SamAutomaticMaskGenerator(sam, points_per_side=16, points_per_batch=64)
    ious, pixs = [], []
    for i, (ip, mp) in enumerate(pairs):
        img_np = np.array(Image.open(ip).convert("RGB"))
        try:
            masks = gen.generate(img_np)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            # retry once with the sparsest sampling
            gen = SamAutomaticMaskGenerator(sam, points_per_side=8, points_per_batch=32)
            masks = gen.generate(img_np)
        if not masks:
            pred = np.zeros(img_np.shape[:2], dtype=np.uint8)
        else:
            best = max(masks, key=lambda m: m["predicted_iou"])
            pred = best["segmentation"].astype(np.uint8)
        io, px = iou_pixacc(pred, gt_binary(mp))
        ious.append(io); pixs.append(px)
        if (i + 1) % 50 == 0:
            print(f"    [SAM] {i + 1}/{len(pairs)}")
    return {"mIoU": round(float(np.mean(ious)), 4), "pixel_acc": round(float(np.mean(pixs)), 4),
            "evaluated": len(ious), "per_image_iou": [round(float(x), 4) for x in ious]}


def eval_florence(lora: str, pairs, device: str, method: str = "referring") -> Dict:
    from florence_common import load_florence2, run_seg_mask, run_seg_mask_cascade
    seg_fn = run_seg_mask_cascade if method == "cascade" else run_seg_mask
    model, processor = load_florence2(lora, device)
    ious, pixs = [], []
    for i, (ip, mp) in enumerate(pairs):
        img = Image.open(ip).convert("RGB")
        pred = seg_fn(model, processor, img, device, phrase="waste")
        io, px = iou_pixacc(pred, gt_binary(mp))
        ious.append(io); pixs.append(px)
        if (i + 1) % 50 == 0:
            print(f"    [Florence/{method}] {i + 1}/{len(pairs)}")
    return {"mIoU": round(float(np.mean(ious)), 4), "pixel_acc": round(float(np.mean(pixs)), 4),
            "evaluated": len(ious), "per_image_iou": [round(float(x), 4) for x in ious]}


# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="3-regime cross-domain segmentation")
    ap.add_argument("--dataset", choices=["zerowaste", "dwsd"], default="zerowaste")
    ap.add_argument("--zerowaste-root", default="../../datasets/zerowaste-f")
    ap.add_argument("--dwsd-root", default="../../datasets/segmentation/Dense Waste Segmentation Dataset")
    ap.add_argument("--deeplab", default="../../stage_1/segmentation/results/deeplabv3plus_best.pth")
    ap.add_argument("--unet", default="../../stage_1/segmentation/results/unet_best.pth")
    ap.add_argument("--maskrcnn", default="../../stage_1/segmentation/results/maskrcnn_best.pth")
    ap.add_argument("--sam", default="../../stage_1/segmentation/sam_vit_h_4b8939.pth")
    ap.add_argument("--lora", default="../finetuned/florence2_unified_multitask_lora")
    ap.add_argument("--seg-method", default="referring", choices=["referring", "cascade"],
                    help="Florence segmentation: single referring call, or multi-instance cascade")
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--output-dir", default="../eval_results/segmentation_zerowaste")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip", default="")
    args = ap.parse_args()

    import torch
    device = args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    if args.dataset == "dwsd":
        pairs = list_pairs_dwsd(Path(args.dwsd_root).expanduser().resolve(), args.max_images)
        ds_label = "DWSD"
    else:
        pairs = list_pairs(Path(args.zerowaste_root).expanduser().resolve(), args.max_images)
        ds_label = "ZeroWaste-f"
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{ds_label} seg: {len(pairs)} image/mask pairs | device={device} | florence={args.seg_method}\n")

    results: Dict[str, Dict] = {}
    regimes = [
        ("deeplabv3plus", "task-specific", lambda: eval_deeplab(args.deeplab, pairs, device)),
        ("unet", "task-specific", lambda: eval_unet(args.unet, pairs, device)),
        ("maskrcnn", "task-specific", lambda: eval_maskrcnn(args.maskrcnn, pairs, device)),
        ("sam", "foundation", lambda: eval_sam(args.sam, pairs, device)),
        ("florence2_ft", "unified-ft", lambda: eval_florence(args.lora, pairs, device, args.seg_method)),
    ]
    for key, regime, fn in regimes:
        if key in skip:
            continue
        print(f"==== {regime}: {key} ====")
        try:
            results[key] = fn()
            results[key]["regime"] = regime
            r = results[key]
            print(f"  {key}: mIoU={r['mIoU']} pixAcc={r['pixel_acc']}\n")
        except Exception as e:
            print(f"  [SKIP] {key}: {type(e).__name__}: {str(e)[:160]}\n")

    results["_meta"] = {"florence_seg_method": args.seg_method, "n_pairs": len(pairs)}
    stem = f"segmentation_{args.dataset}"
    (out_dir / f"{stem}_results.json").write_text(json.dumps(results, indent=2))

    name_map = {"deeplabv3plus": "DeepLabV3+", "unet": "U-Net", "maskrcnn": "Mask R-CNN",
                "sam": "SAM ViT-H", "florence2_ft": "Florence-2 + LoRA"}
    lab = {"task-specific": "Task-Specific", "foundation": "Foundation", "unified-ft": "Unified FT"}
    lines = [f"# {ds_label} Cross-Domain Segmentation (binary mIoU) — Florence: {args.seg_method}\n",
             f"Test pairs: **{len(pairs)}**\n",
             "| Model | Regime | mIoU | Pixel Acc |", "|---|---|---:|---:|"]
    for k in ["deeplabv3plus", "unet", "maskrcnn", "sam", "florence2_ft"]:
        if k in results:
            r = results[k]
            lines.append(f"| {name_map[k]} | {lab[r['regime']]} | {r['mIoU']:.4f} | {r['pixel_acc']:.4f} |")
    (out_dir / f"{stem}_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {out_dir}/{stem}_results.json")


if __name__ == "__main__":
    main()
