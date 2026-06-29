#!/usr/bin/env python3
"""Deployment-cost comparison: one unified model vs a stack of specialists.

Measures, on the same GPU:
  - peak GPU memory (VRAM) to serve each model
  - inference latency per image (and throughput)
  - on-disk size (adapter / weights)

Unified = Florence-2 + LoRA serving all three tasks.
Specialist stack = ViT-Base (cls) + YOLOv8m (det) + DeepLabV3+ (seg).

Outputs a markdown table for the paper.
"""
from __future__ import annotations
import argparse, json, time, gc
from pathlib import Path
import numpy as np
from PIL import Image


def _sz_mb(p: Path) -> float:
    if p.is_file():
        return p.stat().st_size / 1048576
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1048576


def _peak_vram_mb() -> float:
    import torch
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1048576
    return 0.0


def _reset_vram():
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()


def time_fn(fn, n: int) -> float:
    """Return mean seconds/call over n calls (after 1 warmup)."""
    import torch
    fn()  # warmup
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.time() - t0) / n


def bench_florence(lora: str, img: Image.Image, device: str, n: int) -> dict:
    import torch
    from florence_common import load_florence2, run_boxes, run_seg_mask, _run
    _reset_vram()
    model, proc = load_florence2(lora, device)
    res = {}
    res["cls_s"] = time_fn(lambda: _run(model, proc, img, "<CAPTION>", device, 64), n)
    res["det_s"] = time_fn(lambda: run_boxes(model, proc, img, device, "region_proposal"), n)
    res["seg_s"] = time_fn(lambda: run_seg_mask(model, proc, img, device, "waste"), n)
    res["vram_mb"] = _peak_vram_mb()
    del model, proc; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res


def bench_specialists(ckpt_dir: Path, yolo: str, deeplab: str, img: Image.Image, device: str, n: int) -> dict:
    """Load ViT + YOLO + DeepLab together (the deployed stack) and time each."""
    import torch
    import torchvision.transforms as T
    _reset_vram()
    out = {}
    # ViT-Base classifier
    import timm
    vit = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=6)
    vit.load_state_dict(torch.load(ckpt_dir / "vit_base_best.pth", map_location="cpu"))
    vit = vit.to(device).eval()
    tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    x = tf(img).unsqueeze(0).to(device)
    out["cls_s"] = time_fn(lambda: vit(x), n)
    # YOLOv8
    from ultralytics import YOLO
    yo = YOLO(yolo)
    dev = 0 if device.startswith("cuda") else "cpu"
    out["det_s"] = time_fn(lambda: yo.predict(np.array(img), device=dev, verbose=False), n)
    # DeepLabV3+
    import segmentation_models_pytorch as smp
    dl = smp.DeepLabV3Plus(encoder_name="resnet101", encoder_weights=None, in_channels=3, classes=1)
    st = torch.load(deeplab, map_location="cpu")
    dl.load_state_dict(st.get("state_dict", st) if isinstance(st, dict) else st)
    dl = dl.to(device).eval()
    tf2 = T.Compose([T.Resize((256,256)), T.ToTensor(),
                     T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    x2 = tf2(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out["seg_s"] = time_fn(lambda: dl(x2), n)
    out["vram_mb"] = _peak_vram_mb()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", default="../finetuned/florence2_unified_multitask_lora")
    ap.add_argument("--ckpt-dir", default="../../stage_1/classification/results")
    ap.add_argument("--yolo", default="../../stage_1/detection/runs/detect/yolov8m_20251225_182218/weights/best.pt")
    ap.add_argument("--deeplab", default="../../stage_1/segmentation/results/deeplabv3plus_best.pth")
    ap.add_argument("--image", default="../../datasets/zerowaste-f/splits_final_deblurred/test/data")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output-dir", default="../eval_results/deployment")
    args = ap.parse_args()

    import torch
    device = args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    out_dir = Path(args.output_dir).expanduser().resolve(); out_dir.mkdir(parents=True, exist_ok=True)

    # pick one test image
    p = Path(args.image).expanduser()
    if p.is_dir():
        p = sorted(f for f in p.iterdir() if f.suffix.lower() in (".png",".jpg",".jpeg"))[0]
    img = Image.open(p).convert("RGB")
    print(f"Image: {p.name}  device={device}  n={args.n}\n")

    results = {}
    print("== Florence-2 unified ==")
    fl = bench_florence(args.lora, img, device, args.n)
    fl["adapter_mb"] = round(_sz_mb(Path(args.lora).expanduser() / "adapter_model.safetensors"), 1)
    results["unified"] = fl
    print(fl, "\n")

    print("== Specialist stack (ViT + YOLOv8 + DeepLabV3+) ==")
    try:
        sp = bench_specialists(Path(args.ckpt_dir).expanduser(), args.yolo, args.deeplab, img, device, args.n)
        sp["weights_mb"] = round(
            _sz_mb(Path(args.ckpt_dir).expanduser() / "vit_base_best.pth")
            + _sz_mb(Path(args.yolo).expanduser())
            + _sz_mb(Path(args.deeplab).expanduser()), 1)
        results["specialists"] = sp
        print(sp, "\n")
    except Exception as e:
        print(f"[specialists skipped] {type(e).__name__}: {str(e)[:160]}")
        results["specialists"] = {"error": str(e)[:200]}

    (out_dir / "deployment.json").write_text(json.dumps(results, indent=2))

    # markdown table
    u, s = results["unified"], results.get("specialists", {})
    lines = ["# Deployment cost: one unified model vs specialist stack\n",
             "| | Models | Peak VRAM (MB) | Cls (ms) | Det (ms) | Seg (ms) | On-disk |",
             "|---|---|---:|---:|---:|---:|---:|",
             f"| Unified Florence-2 | 1 | {u['vram_mb']:.0f} | {u['cls_s']*1000:.0f} | "
             f"{u['det_s']*1000:.0f} | {u['seg_s']*1000:.0f} | {u['adapter_mb']:.0f} MB adapter |"]
    if "vram_mb" in s:
        lines.append(f"| Specialist stack | 3 | {s['vram_mb']:.0f} | {s['cls_s']*1000:.0f} | "
                     f"{s['det_s']*1000:.0f} | {s['seg_s']*1000:.0f} | {s['weights_mb']:.0f} MB weights |")
    (out_dir / "deployment.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {out_dir}/deployment.md")


if __name__ == "__main__":
    main()
