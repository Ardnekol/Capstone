"""Shared Florence-2 + LoRA loading and task runners.

Mirrors stage_2/evaluate_unified_model.py exactly so cross-domain numbers are
directly comparable to the existing four-regime table.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


def load_florence2(model_id: str, device: str):
    """Load Florence-2 (LoRA dir or hub id) with the flash_attn import shim."""
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
            base = AutoModelForCausalLM.from_pretrained(
                base_id, trust_remote_code=True, attn_implementation="eager")
            from peft import PeftModel
            model = PeftModel.from_pretrained(base, str(adapter))
            processor = AutoProcessor.from_pretrained(str(adapter), trust_remote_code=True)
        else:
            print(f"[Florence] hub {model_id}")
            model = AutoModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=True, attn_implementation="eager")
            processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    finally:
        _dmu.check_imports = _orig
    model = model.to(device, dtype=torch.float16 if device.startswith("cuda") else torch.float32).eval()
    return model, processor


def _run(model, processor, image: Image.Image, prompt: str, device: str, max_new_tokens: int) -> Dict:
    import torch
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    inputs = {k: (v.to(device, dtype) if v.is_floating_point() else v.to(device))
              for k, v in inputs.items()}
    with torch.inference_mode():
        gen = model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                             max_new_tokens=max_new_tokens, num_beams=3, early_stopping=True)
    text = processor.batch_decode(gen, skip_special_tokens=False)[0]
    return processor.post_process_generation(
        text, task=prompt.split(">")[0] + ">", image_size=(image.width, image.height))


DET_PROMPTS = {
    "od": ("<OD>", 256),
    "region_proposal": ("<REGION_PROPOSAL>", 1024),
    "dense_region_caption": ("<DENSE_REGION_CAPTION>", 1024),
}


def run_boxes(model, processor, image: Image.Image, device: str,
              method: str = "od") -> List[List[float]]:
    """Florence detection → list of [x1,y1,x2,y2] boxes (pixel coords).

    method='od' uses the labelled detector; 'region_proposal' uses the
    class-agnostic region-proposal head (better recall on dense scenes, and the
    right match for a class-agnostic localization metric).
    """
    prompt, mnt = DET_PROMPTS.get(method, DET_PROMPTS["od"])
    result = _run(model, processor, image, prompt, device, mnt)
    for k in result:
        v = result[k]
        if isinstance(v, dict) and "bboxes" in v:
            return [list(b) for b in v["bboxes"]]
    return []


def run_od(model, processor, image: Image.Image, device: str) -> List[List[float]]:
    """Backward-compatible <OD> runner."""
    return run_boxes(model, processor, image, device, "od")


def _draw_polys(draw, polygons) -> None:
    """Rasterize Florence polygon output, tolerant of nesting depth."""
    for grp in polygons or []:
        if not grp:
            continue
        sub = grp if isinstance(grp[0], (list, tuple)) else [grp]
        for poly in sub:
            if len(poly) >= 6:
                pts = [(poly[i], poly[i + 1]) for i in range(0, len(poly) - 1, 2)]
                draw.polygon(pts, fill=255)


def run_seg_mask(model, processor, image: Image.Image, device: str,
                 phrase: str = "waste") -> np.ndarray:
    """Florence referring-segmentation → binary mask (H,W) uint8 {0,1}."""
    from PIL import ImageDraw
    W, H = image.width, image.height
    result = _run(model, processor, image, f"<REFERRING_EXPRESSION_SEGMENTATION>{phrase}", device, 512)
    mask = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask)
    for v in result.values():
        if isinstance(v, dict) and v.get("polygons"):
            _draw_polys(draw, v["polygons"])
    return (np.array(mask) > 127).astype(np.uint8)


def run_seg_mask_cascade(model, processor, image: Image.Image, device: str,
                         phrase: str = "waste") -> np.ndarray:
    """Multi-instance cascade → binary mask (H,W) uint8 {0,1}.

    Stage 1: <CAPTION_TO_PHRASE_GROUNDING>phrase finds every instance box.
    Stage 2: per box, <REGION_TO_SEGMENTATION><loc_*> produces a polygon.
    Union all polygons. Mirrors the Playground <MULTI_INSTANCE_SEGMENTATION>.
    """
    from PIL import ImageDraw
    W, H = image.width, image.height
    # Stage 1: phrase grounding → instance bboxes
    g = _run(model, processor, image, f"<CAPTION_TO_PHRASE_GROUNDING>{phrase}", device, 1024)
    gd = g.get("<CAPTION_TO_PHRASE_GROUNDING>", {})
    bboxes = gd.get("bboxes", []) if isinstance(gd, dict) else []

    mask = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask)
    # Stage 2: per-box region-to-segmentation
    for bbox in bboxes:
        if len(bbox) < 4:
            continue
        x1, y1, x2, y2 = bbox[:4]
        loc = (f"<loc_{int(x1 / W * 1000)}><loc_{int(y1 / H * 1000)}>"
               f"<loc_{int(x2 / W * 1000)}><loc_{int(y2 / H * 1000)}>")
        s = _run(model, processor, image, f"<REGION_TO_SEGMENTATION>{loc}", device, 512)
        sd = s.get("<REGION_TO_SEGMENTATION>", {})
        if isinstance(sd, dict):
            _draw_polys(draw, sd.get("polygons", []))
    return (np.array(mask) > 127).astype(np.uint8)
