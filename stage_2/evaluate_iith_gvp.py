#!/usr/bin/env python3
"""IITH-GVP field deployment evaluation.

Runs Florence-2 + LoRA on the IITH Garbage Vulnerable Points (GVP) surveillance
dataset and reports a quantitative comparison between "before" and "after"
cleanup images. The natural temporal signal in the dataset (each GVP location
has a snapshot just before cleanup and just after) gives an empirical
validation that does not require manual labels.

Hypothesis: the unified Florence-2 + LoRA should detect more garbage
(more bounding boxes, larger mask area) in "before" images than in
"after" images.

Outputs:
  <output-dir>/results.json    — per-image stats + aggregate metrics
  <output-dir>/summary.md      — human-readable comparison table

Usage:
  python3 evaluate_iith_gvp.py \\
      --model-id finetuned/florence2_unified_multitask_lora \\
      --gvp-root /u/student/2024/cs24mtech11024/Capstone/datasets/AllGVPImages \\
      --max-per-class 500 \\
      --output-dir eval_results/iith_gvp_unified
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

warnings.filterwarnings("ignore")


def _stage2_root() -> Path:
    return Path(__file__).resolve().parent


def _resolve_path_arg(path_str: Optional[str]) -> Optional[Path]:
    if path_str is None:
        return None
    p = Path(path_str).expanduser()
    if p.is_absolute():
        return p.resolve()
    candidates = [Path.cwd() / p, _stage2_root() / p, _stage2_root().parent / p]
    for c in candidates:
        if c.exists():
            return c.resolve()
    return candidates[0].resolve()


# ──────────────────────────────────────────────────────────────────────────────
# Florence-2 + LoRA loading (mirrors evaluate_unified_model.py)
# ──────────────────────────────────────────────────────────────────────────────

def load_florence2(model_id: str, device: str):
    import transformers.dynamic_module_utils as _dmu
    _orig = _dmu.check_imports

    def _patched(filename, *args, **kwargs):
        try:
            return _orig(filename, *args, **kwargs)
        except ImportError as e:
            if "flash_attn" in str(e):
                return []
            raise

    from transformers import AutoModelForCausalLM, AutoProcessor
    _dmu.check_imports = _patched
    try:
        adapter_dir = Path(model_id).expanduser()
        is_local = (
            adapter_dir.exists()
            and adapter_dir.is_dir()
            and (adapter_dir / "adapter_config.json").exists()
        )
        if is_local:
            cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
            base_id = cfg.get("base_model_name_or_path", "microsoft/Florence-2-large-ft")
            print(f"[Florence] Loading LoRA from {adapter_dir} (base={base_id})")
            base = AutoModelForCausalLM.from_pretrained(
                base_id, trust_remote_code=True, attn_implementation="eager"
            )
            from peft import PeftModel
            model = PeftModel.from_pretrained(base, str(adapter_dir))
            processor = AutoProcessor.from_pretrained(str(adapter_dir), trust_remote_code=True)
        else:
            print(f"[Florence] Loading hub model {model_id}")
            model = AutoModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=True, attn_implementation="eager"
            )
            processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    finally:
        _dmu.check_imports = _orig

    model = model.to(device, dtype=torch.float16 if device.startswith("cuda") else torch.float32).eval()
    return model, processor


def run_florence_task(model, processor, image: Image.Image, prompt: str, device: str) -> dict:
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    inputs = {
        k: (v.to(device, dtype) if v.is_floating_point() else v.to(device))
        for k, v in inputs.items()
    }
    with torch.no_grad():
        out = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=2048,
            num_beams=3,
            do_sample=False,
            early_stopping=False,
        )
    text = processor.batch_decode(out, skip_special_tokens=False)[0]
    return processor.post_process_generation(text, task=prompt, image_size=(image.width, image.height))


# ──────────────────────────────────────────────────────────────────────────────
# Per-image scoring
# ──────────────────────────────────────────────────────────────────────────────

def score_image(model, processor, img_path: Path, device: str) -> dict:
    """Run detection + segmentation prompts on one image, return summary stats."""
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    img_area = float(W * H)

    record: dict = {"path": str(img_path), "width": W, "height": H, "img_area": img_area}

    # 1) Object Detection: count bboxes, total bbox-area-fraction
    try:
        od = run_florence_task(model, processor, img, "<OD>", device)
        bboxes = od.get("<OD>", {}).get("bboxes", []) if isinstance(od.get("<OD>"), dict) else []
        labels = od.get("<OD>", {}).get("labels", []) if isinstance(od.get("<OD>"), dict) else []
        bbox_count = len(bboxes)
        total_bbox_area = 0.0
        for bb in bboxes:
            if len(bb) >= 4:
                x1, y1, x2, y2 = bb[:4]
                total_bbox_area += max(0.0, x2 - x1) * max(0.0, y2 - y1)
        bbox_area_frac = total_bbox_area / img_area if img_area > 0 else 0.0
        record["detection"] = {
            "count": bbox_count,
            "bbox_area_frac": round(float(bbox_area_frac), 6),
            "labels": [str(l) for l in labels],
        }
    except Exception as e:
        record["detection"] = {"error": str(e)[:200]}

    # 2) Generic segmentation ("waste" prompt). Measure foreground mask fraction.
    try:
        seg_prompt = "<REFERRING_EXPRESSION_SEGMENTATION>waste"
        seg = run_florence_task(model, processor, img, seg_prompt, device)
        polys = []
        seg_data = seg.get("<REFERRING_EXPRESSION_SEGMENTATION>")
        if isinstance(seg_data, dict):
            polys = seg_data.get("polygons", []) or []
        # Render polygons → binary mask, measure fraction of pixels
        if polys:
            from PIL import ImageDraw
            mask_img = Image.new("L", (W, H), 0)
            drw = ImageDraw.Draw(mask_img)
            for grp in polys:
                # grp is a list of polygons (each is a flat list of x1,y1,x2,y2,...)
                for poly in (grp if isinstance(grp[0], (list, tuple)) else [grp]):
                    if len(poly) >= 6:
                        pts = [(float(poly[i]), float(poly[i + 1])) for i in range(0, len(poly) - 1, 2)]
                        drw.polygon(pts, fill=255)
            mask_arr = np.array(mask_img)
            fg_frac = float((mask_arr > 0).sum()) / float(mask_arr.size)
        else:
            fg_frac = 0.0
        record["segmentation"] = {
            "polygon_groups": len(polys),
            "fg_frac": round(fg_frac, 6),
        }
    except Exception as e:
        record["segmentation"] = {"error": str(e)[:200]}

    return record


# ──────────────────────────────────────────────────────────────────────────────
# Sampling
# ──────────────────────────────────────────────────────────────────────────────

def list_gvp_images(root: Path, status: str) -> list[Path]:
    """List all images matching '*_<status>_*' across the GVP dataset tree."""
    pattern = re.compile(rf".*_{status}_\d+_\d+_\d+_\d+_\d+\.jpg$", re.IGNORECASE)
    out: list[Path] = []
    for p in root.rglob("*.jpg"):
        if pattern.fullmatch(p.name):
            out.append(p)
    return out


def sample_pairs(before: list[Path], after: list[Path], n_each: int, seed: int = 42) -> tuple[list[Path], list[Path]]:
    rng = random.Random(seed)
    rng.shuffle(before)
    rng.shuffle(after)
    return before[:n_each], after[:n_each]


# ──────────────────────────────────────────────────────────────────────────────
# Aggregation
# ──────────────────────────────────────────────────────────────────────────────

def aggregate(records: list[dict]) -> dict:
    n = len(records)
    if n == 0:
        return {"n": 0}
    det_counts = [r.get("detection", {}).get("count", 0) for r in records]
    bbox_fracs = [r.get("detection", {}).get("bbox_area_frac", 0.0) for r in records]
    seg_fracs = [r.get("segmentation", {}).get("fg_frac", 0.0) for r in records]
    return {
        "n": n,
        "detection_count_mean": float(np.mean(det_counts)),
        "detection_count_median": float(np.median(det_counts)),
        "bbox_area_frac_mean": float(np.mean(bbox_fracs)),
        "seg_fg_frac_mean": float(np.mean(seg_fracs)),
        "seg_fg_frac_median": float(np.median(seg_fracs)),
        "n_with_any_detection": int(sum(1 for c in det_counts if c > 0)),
        "n_with_any_segmentation": int(sum(1 for f in seg_fracs if f > 0.005)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Field deployment evaluation on IITH-GVP")
    ap.add_argument("--model-id", type=str, default="finetuned/florence2_unified_multitask_lora",
                    help="HF model id or local LoRA dir")
    ap.add_argument("--gvp-root", type=str,
                    default="/u/student/2024/cs24mtech11024/Capstone/datasets/AllGVPImages",
                    help="IITH GVP dataset root")
    ap.add_argument("--max-per-class", type=int, default=500,
                    help="Number of before / after images to sample (each); 0 = all")
    ap.add_argument("--output-dir", type=str, default="eval_results/iith_gvp_unified")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = _resolve_path_arg(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gvp_root = _resolve_path_arg(args.gvp_root)
    if not gvp_root or not gvp_root.exists():
        raise SystemExit(f"GVP root not found: {gvp_root}")

    print(f"Scanning {gvp_root} ...")
    before_all = list_gvp_images(gvp_root, "before")
    after_all = list_gvp_images(gvp_root, "after")
    print(f"  found {len(before_all)} before images")
    print(f"  found {len(after_all)} after images")

    if args.max_per_class > 0:
        before_sample, after_sample = sample_pairs(before_all, after_all, args.max_per_class, seed=args.seed)
    else:
        before_sample, after_sample = before_all, after_all
    print(f"Evaluating {len(before_sample)} before + {len(after_sample)} after images")

    device = args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    model, processor = load_florence2(args.model_id, device)

    records: dict[str, list[dict]] = {"before": [], "after": []}
    for cls, files in [("before", before_sample), ("after", after_sample)]:
        print(f"\n──── Scoring {cls} ({len(files)} images) ────")
        for i, p in enumerate(files):
            try:
                rec = score_image(model, processor, p, device)
            except Exception as e:
                rec = {"path": str(p), "error": str(e)[:200]}
            rec["status"] = cls
            records[cls].append(rec)
            if (i + 1) % 25 == 0:
                print(f"  {cls}: processed {i + 1}/{len(files)}")

    agg_before = aggregate(records["before"])
    agg_after = aggregate(records["after"])

    results = {
        "model_id": args.model_id,
        "gvp_root": str(gvp_root),
        "n_before_sampled": len(before_sample),
        "n_after_sampled": len(after_sample),
        "seed": args.seed,
        "before": agg_before,
        "after": agg_after,
        "per_image": {"before": records["before"], "after": records["after"]},
    }

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    # Human-readable summary
    lines: list[str] = []
    lines.append("# IITH-GVP Field Deployment Evaluation\n")
    lines.append(f"Model: `{args.model_id}`\n")
    lines.append(f"Dataset root: `{gvp_root}`")
    lines.append(f"Sample: {len(before_sample)} before / {len(after_sample)} after, seed={args.seed}\n")
    lines.append("## Aggregate metrics (before vs after cleanup)\n")
    lines.append("| Metric | Before | After | Δ (Before − After) |")
    lines.append("|---|---:|---:|---:|")

    def _row(name: str, b: float | int, a: float | int, fmt: str = "{:.4f}"):
        delta = (b - a) if isinstance(b, (int, float)) and isinstance(a, (int, float)) else None
        lines.append(f"| {name} | {fmt.format(b)} | {fmt.format(a)} | {fmt.format(delta) if delta is not None else '—'} |")

    _row("Mean detected objects per image", agg_before.get("detection_count_mean", 0), agg_after.get("detection_count_mean", 0))
    _row("Median detected objects per image", agg_before.get("detection_count_median", 0), agg_after.get("detection_count_median", 0))
    _row("Mean bbox area fraction", agg_before.get("bbox_area_frac_mean", 0), agg_after.get("bbox_area_frac_mean", 0))
    _row("Mean segmentation fg fraction", agg_before.get("seg_fg_frac_mean", 0), agg_after.get("seg_fg_frac_mean", 0))
    _row("Median segmentation fg fraction", agg_before.get("seg_fg_frac_median", 0), agg_after.get("seg_fg_frac_median", 0))
    _row("Images with any detection (count)", agg_before.get("n_with_any_detection", 0), agg_after.get("n_with_any_detection", 0), "{:d}")
    _row("Images with any segmentation (count)", agg_before.get("n_with_any_segmentation", 0), agg_after.get("n_with_any_segmentation", 0), "{:d}")

    lines.append("")
    lines.append("## Interpretation\n")
    lines.append("Florence-2 + LoRA was trained on TrashNet / TACO data and never on this surveillance footage.")
    lines.append("Without manual labels, the natural before/after temporal signal in the IITH-GVP dataset")
    lines.append("provides empirical validation: if the model correctly recognises waste in field conditions,")
    lines.append("it should report higher detection counts and larger segmentation foreground area on")
    lines.append("\"before-cleanup\" images than on \"after-cleanup\" images.")

    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")

    print(f"\nWrote {out_dir / 'results.json'}")
    print(f"Wrote {out_dir / 'summary.md'}")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
