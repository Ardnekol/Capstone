#!/usr/bin/env python3
"""Cross-domain CLASSIFICATION comparison on WaRP-C.

Compares three regimes on the WaRP-C (recycling-plant) classification test set,
a NEW cross-domain benchmark not seen by any model during training:

  1. Task-Specific : ViT-Base, ResNet-50, EfficientNet-B0 (Stage 1 checkpoints)
  2. Foundation    : CLIP ViT-B/16 (zero-shot)
  3. Unified FT     : Florence-2-large-ft + unified multitask LoRA (Stage 2)

All models predict into the TrashNet 6-class label space
(cardboard, glass, metal, paper, plastic, trash). WaRP-C's 5 classes are mapped
onto that space via WARPC_TO_TRASHNET below; unmapped classes are skipped.

Reuses the EXACT scoring logic from stage_1 (specialist/CLIP) and
stage_2/evaluate_unified_model.py (Florence <CAPTION>) so numbers are directly
comparable to the existing four-regime table.

Usage:
  python3 eval_warpc_classification.py \
      --warpc-root ../../datasets/WARP/Warp-C/test_crops \
      --lora ../finetuned/florence2_unified_multitask_lora \
      --ckpt-dir ../../stage_1/classification/results \
      --max-per-class 0 \
      --output-dir ../eval_results/warpc_classification
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Label space + WaRP-C → TrashNet mapping
# ──────────────────────────────────────────────────────────────────────────────
TRASHNET_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# WaRP-C folder names (lowercased) → TrashNet class. None = skip (ambiguous).
WARPC_TO_TRASHNET: Dict[str, Optional[str]] = {
    "bottle": "plastic",       # plastic bottles
    "cans": "metal",           # aluminium cans
    "cardboard": "cardboard",
    "detergent": "plastic",    # plastic detergent bottles
    "canister": "plastic",     # plastic canisters
}

CLIP_TEXT_PROMPTS = [
    "a photo of cardboard waste",
    "a photo of glass waste",
    "a photo of metal waste",
    "a photo of paper waste",
    "a photo of plastic waste",
    "a photo of miscellaneous trash",
]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def list_mapped_samples(root: Path, max_per_class: int) -> List[tuple]:
    """Return [(img_path, trashnet_label)] for WaRP-C, applying the mapping."""
    samples: List[tuple] = []
    for cls_dir in sorted(root.iterdir()):
        if not cls_dir.is_dir() or cls_dir.name.startswith("."):
            continue
        mapped = WARPC_TO_TRASHNET.get(cls_dir.name.lower().strip())
        if mapped is None:
            continue
        # WaRP-C test_crops nests images one level deeper:
        # test_crops/<coarse_class>/<fine_class>/*.jpg  → recurse.
        imgs = sorted(
            f for f in cls_dir.rglob("*")
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        )
        if max_per_class > 0:
            imgs = imgs[:max_per_class]
        samples.extend((p, mapped) for p in imgs)
    return samples


def compute_metrics(y_true: List[str], y_pred: List[str], dataset_name: str) -> Dict:
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / len(y_true) if y_true else 0.0
    all_labels = sorted(set(y_true + y_pred))
    per_class: Dict[str, Dict] = {}
    for label in all_labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[label] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}
    # Macro-F1 over the mapped target classes only (those present in y_true)
    target_labels = sorted(set(y_true))
    macro_f1 = float(np.mean([per_class[l]["f1"] for l in target_labels])) if target_labels else 0.0
    return {
        "dataset": dataset_name,
        "n": len(y_true),
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Regime 1: task-specific CNN/ViT classifiers
# ──────────────────────────────────────────────────────────────────────────────
def build_specialist(name: str, num_classes: int = 6):
    import torch.nn as nn
    if name == "vit_base":
        import timm
        return timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=num_classes)
    from torchvision import models
    if name == "resnet50":
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if name == "efficientnetb0":
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m
    raise ValueError(name)


def eval_specialist(name: str, ckpt: Path, samples: List[tuple], device: str) -> Dict:
    import torch
    from torchvision import transforms

    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    model = build_specialist(name)
    state = torch.load(ckpt, map_location="cpu")
    state = state.get("state_dict", state) if isinstance(state, dict) else state
    model.load_state_dict(state)
    model = model.to(device).eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for i, (path, gt) in enumerate(samples):
            try:
                img = Image.open(path).convert("RGB")
                x = tf(img).unsqueeze(0).to(device)
                logits = model(x)
                idx = int(logits.argmax(dim=-1).item())
                y_true.append(gt)
                y_pred.append(TRASHNET_CLASSES[idx])
            except Exception as e:
                print(f"    [{name}] error {Path(path).name}: {str(e)[:120]}")
            if (i + 1) % 200 == 0:
                print(f"    [{name}] {i + 1}/{len(samples)}")
    return compute_metrics(y_true, y_pred, "WaRP-C")


# ──────────────────────────────────────────────────────────────────────────────
# Regime 2: CLIP zero-shot
# ──────────────────────────────────────────────────────────────────────────────
def eval_clip(samples: List[tuple], device: str) -> Dict:
    import torch
    import open_clip
    from torchvision import transforms

    model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    model = model.to(device).eval()
    text_inputs = tokenizer(CLIP_TEXT_PROMPTS).to(device)

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711)),
    ])
    y_true, y_pred = [], []
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        for i, (path, gt) in enumerate(samples):
            try:
                img = Image.open(path).convert("RGB")
                x = tf(img).unsqueeze(0).to(device)
                feat = model.encode_image(x)
                feat /= feat.norm(dim=-1, keepdim=True)
                sim = (100.0 * feat @ text_features.T).softmax(dim=-1)
                idx = int(sim.argmax(dim=-1).item())
                y_true.append(gt)
                y_pred.append(TRASHNET_CLASSES[idx])
            except Exception as e:
                print(f"    [CLIP] error {Path(path).name}: {str(e)[:120]}")
            if (i + 1) % 200 == 0:
                print(f"    [CLIP] {i + 1}/{len(samples)}")
    return compute_metrics(y_true, y_pred, "WaRP-C")


# ──────────────────────────────────────────────────────────────────────────────
# Regime 3: Florence-2 + LoRA (<CAPTION>) — mirrors evaluate_unified_model.py
# ──────────────────────────────────────────────────────────────────────────────
def load_florence2(model_id: str, device: str):
    import torch
    import transformers.dynamic_module_utils as _dmu
    _orig = _dmu.check_imports

    def _patched(filename, *a, **k):
        try:
            return _orig(filename, *a, **k)
        except ImportError as e:
            if "flash_attn" in str(e):
                return []
            raise

    from transformers import AutoModelForCausalLM, AutoProcessor
    _dmu.check_imports = _patched
    try:
        adapter = Path(model_id).expanduser()
        is_local = adapter.is_dir() and (adapter / "adapter_config.json").exists()
        if is_local:
            cfg = json.loads((adapter / "adapter_config.json").read_text())
            base_id = cfg.get("base_model_name_or_path", "microsoft/Florence-2-large-ft")
            print(f"[Florence] LoRA {adapter} (base={base_id})")
            base = AutoModelForCausalLM.from_pretrained(base_id, trust_remote_code=True, attn_implementation="eager")
            from peft import PeftModel
            model = PeftModel.from_pretrained(base, str(adapter))
            processor = AutoProcessor.from_pretrained(str(adapter), trust_remote_code=True)
        else:
            print(f"[Florence] hub {model_id}")
            model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, attn_implementation="eager")
            processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    finally:
        _dmu.check_imports = _orig
    model = model.to(device, dtype=torch.float16 if device.startswith("cuda") else torch.float32).eval()
    return model, processor


def eval_florence(model_id: str, samples: List[tuple], device: str) -> Dict:
    import torch
    model, processor = load_florence2(model_id, device)
    known = set(TRASHNET_CLASSES)
    y_true, y_pred = [], []
    for i, (path, gt) in enumerate(samples):
        try:
            img = Image.open(path).convert("RGB")
            inputs = processor(text="<CAPTION>", images=img, return_tensors="pt")
            dtype = torch.float16 if device.startswith("cuda") else torch.float32
            inputs = {k: (v.to(device, dtype) if v.is_floating_point() else v.to(device))
                      for k, v in inputs.items()}
            with torch.inference_mode():
                gen = model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                                     max_new_tokens=64, num_beams=3, early_stopping=True)
            text = processor.batch_decode(gen, skip_special_tokens=False)[0]
            result = processor.post_process_generation(text, task="<CAPTION>",
                                                       image_size=(img.width, img.height))
            pred_text = ""
            for v in result.values():
                if isinstance(v, str):
                    pred_text = v.strip().lower()
                    break
            pred_label = "unknown"
            for k in known:
                if k in pred_text:
                    pred_label = k
                    break
            y_true.append(gt)
            y_pred.append(pred_label)
        except Exception as e:
            print(f"    [Florence] error {Path(path).name}: {str(e)[:120]}")
        if (i + 1) % 50 == 0:
            print(f"    [Florence] {i + 1}/{len(samples)}")
    return compute_metrics(y_true, y_pred, "WaRP-C")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="WaRP-C 3-regime classification comparison")
    ap.add_argument("--warpc-root", default="../../datasets/WARP/Warp-C/test_crops")
    ap.add_argument("--lora", default="../finetuned/florence2_unified_multitask_lora")
    ap.add_argument("--ckpt-dir", default="../../stage_1/classification/results")
    ap.add_argument("--max-per-class", type=int, default=0, help="0 = all images")
    ap.add_argument("--output-dir", default="../eval_results/warpc_classification")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip", default="", help="comma list: vit_base,resnet50,efficientnetb0,clip,florence")
    args = ap.parse_args()

    import torch
    device = args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    root = Path(args.warpc_root).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.ckpt_dir).expanduser().resolve()

    if not root.exists():
        raise SystemExit(f"WaRP-C root not found: {root}")

    samples = list_mapped_samples(root, args.max_per_class)
    print(f"WaRP-C test samples (mapped): {len(samples)} | device={device}")
    print(f"Mapping in use: {WARPC_TO_TRASHNET}\n")

    results: Dict[str, Dict] = {}

    specialists = {
        "vit_base": ckpt_dir / "vit_base_best.pth",
        "resnet50": ckpt_dir / "resnet50_best.pth",
        "efficientnetb0": ckpt_dir / "efficientnetb0_best.pth",
    }
    for name, ckpt in specialists.items():
        if name in skip:
            continue
        if not ckpt.exists():
            print(f"[skip] {name}: checkpoint missing {ckpt}")
            continue
        print(f"\n==== Task-Specific: {name} ====")
        try:
            results[name] = eval_specialist(name, ckpt, samples, device)
            results[name]["regime"] = "task-specific"
        except Exception as e:
            print(f"[FAIL] {name}: {e}")

    if "clip" not in skip:
        print("\n==== Foundation: CLIP ViT-B/16 (zero-shot) ====")
        try:
            results["clip"] = eval_clip(samples, device)
            results["clip"]["regime"] = "foundation"
        except Exception as e:
            print(f"[FAIL] clip: {e}")

    if "florence" not in skip:
        print("\n==== Unified FT: Florence-2 + LoRA ====")
        try:
            results["florence2_ft"] = eval_florence(args.lora, samples, device)
            results["florence2_ft"]["regime"] = "unified-ft"
        except Exception as e:
            print(f"[FAIL] florence: {e}")

    # Write JSON
    (out_dir / "warpc_classification_results.json").write_text(json.dumps(results, indent=2))

    # Write markdown comparison
    label = {"task-specific": "Task-Specific", "foundation": "Foundation", "unified-ft": "Unified FT"}
    name_map = {"vit_base": "ViT-Base", "resnet50": "ResNet-50",
                "efficientnetb0": "EfficientNet-B0", "clip": "CLIP ViT-B/16",
                "florence2_ft": "Florence-2 + LoRA"}
    lines = ["# WaRP-C Cross-Domain Classification (3-regime)\n",
             f"Test images (mapped to TrashNet space): **{len(samples)}**\n",
             f"Mapping: `{WARPC_TO_TRASHNET}`\n",
             "| Model | Regime | Accuracy | Macro-F1 |",
             "|---|---|---:|---:|"]
    order = ["vit_base", "resnet50", "efficientnetb0", "clip", "florence2_ft"]
    for k in order:
        if k in results:
            r = results[k]
            lines.append(f"| {name_map[k]} | {label.get(r.get('regime',''), '')} | "
                         f"{r['accuracy']*100:.2f}% | {r['macro_f1']:.4f} |")
    (out_dir / "warpc_classification_summary.md").write_text("\n".join(lines) + "\n")

    print("\n" + "\n".join(lines))
    print(f"\nWrote {out_dir/'warpc_classification_results.json'}")
    print(f"Wrote {out_dir/'warpc_classification_summary.md'}")


if __name__ == "__main__":
    main()
