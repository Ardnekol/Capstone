#!/usr/bin/env python3
"""Unified Florence-2 runner for waste/garbage datasets.

Goal: Use ONE Florence-2 model to do all 3 tasks on the same images:
  1) Object detection
  2) Segmentation (prompted)
  3) Classification (open-vocabulary detection over your class set)

This script is designed to plug into your Stage 1 dataset layout and naming.

Examples:
  # Run on RealWaste (folder-of-classes). Produces outputs per image.
    python Capstone/stage_2/run_florence_2.py --preset realwaste --max-images 50

    # Run on ONE image (unified demo: classification + detection + segmentation)
    python Capstone/stage_2/run_florence_2.py --image-path /path/to/image.jpg --model large-ft --num-beams 3 --seg-phrase trash

  # Run on TrashNet (folder-of-classes)
  python Capstone/stage_2/run_florence_2.py --preset trashnet --max-images 50

  # Run on detection dataset prepared in Stage 1 (YOLO format)
  python Capstone/stage_2/run_florence_2.py --preset taco_det --max-images 50

  # Run on segmentation dataset prepared in Stage 1 (images + binary masks)
  python Capstone/stage_2/run_florence_2.py --preset taco_seg --max-images 50

Notes:
    - All run artifacts are saved under Capstone/stage_2/ by default (results.json + viz/).
    - Use --output-root to choose a subfolder under Capstone/stage_2/ (default: outputs).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor


# ----------------------------
# Florence-2 tasks (per official processor)
# ----------------------------
TASK_OD = "<OD>"
TASK_OPEN_VOCAB = "<OPEN_VOCABULARY_DETECTION>"
TASK_SEG_REF = "<REFERRING_EXPRESSION_SEGMENTATION>"
TASK_CAPTION = "<CAPTION>"


GENERIC_WASTE_TERMS = (
    "trash,garbage,litter,waste,rubbish,plastic,plastic bottle,bottle,can,"
    "cup,food,wrapper,bag,plastic bag,paper,cardboard,metal,glass,"
    "cigarette,straw,cap,lid,container,box"
)


MODELS: Dict[str, str] = {
    "base": "microsoft/Florence-2-base",
    "large": "microsoft/Florence-2-large",
    "base-ft": "microsoft/Florence-2-base-ft",
    "large-ft": "microsoft/Florence-2-large-ft",
}


def configure_hf_cache(cache_dir: Path) -> None:
    """Redirect Hugging Face caches away from $HOME to avoid disk-quota errors.

    Florence-2 with `trust_remote_code=True` writes dynamic modules under
    `~/.cache/huggingface/modules/...` by default, which often fails on quota-limited home dirs.
    """

    cache_dir = cache_dir.expanduser().resolve()
    (cache_dir / "hub").mkdir(parents=True, exist_ok=True)
    (cache_dir / "modules").mkdir(parents=True, exist_ok=True)

    # Hub + transformers
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_dir / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir / "hub"))

    # Dynamic modules cache (trust_remote_code)
    os.environ.setdefault("HF_MODULES_CACHE", str(cache_dir / "modules"))


def default_cache_dir() -> Path:
    # Prefer node-local SLURM temporary space if available.
    for key in ("SLURM_TMPDIR", "TMPDIR", "TMP"):
        val = os.environ.get(key)
        if val:
            return Path(val) / f"hf_cache_uid{os.getuid()}"
    return Path("/tmp") / f"hf_cache_uid{os.getuid()}"


@dataclass(frozen=True)
class Preset:
    name: str
    images_dir: Optional[Path] = None
    class_subfolders: bool = False
    yolo_labels_dir: Optional[Path] = None
    masks_dir: Optional[Path] = None


def _repo_root() -> Path:
    # Capstone/stage_2/run_florence_2.py -> Capstone
    return Path(__file__).resolve().parents[1]


def _stage2_root() -> Path:
    return _repo_root() / "stage_2"


def _resolve_path_arg(path_str: Optional[str]) -> Optional[Path]:
    """Resolve a user-provided path argument robustly across different CWDs.

    Users often run this script from either the workspace root or from Capstone/stage_2.
    This helper tries a few common bases for relative paths:
      1) current working directory
      2) Capstone/ (repo root for this file)
      3) workspace root (parent of Capstone/)

    Returns an absolute Path if found, otherwise an absolute Path candidate.
    """

    if path_str is None:
        return None

    p = Path(path_str).expanduser()
    if p.is_absolute():
        return p.resolve()

    capstone = _repo_root().resolve()
    workspace = capstone.parent

    candidates: List[Path] = [
        (Path.cwd() / p),
        (capstone / p),
        (workspace / p),
    ]

    # Special-case: user passes paths starting with "Capstone/...".
    # If running from Capstone/stage_2, that relative path won't exist; interpret it from workspace root.
    if len(p.parts) > 0 and p.parts[0] == "Capstone":
        candidates.insert(0, workspace / p)

    for c in candidates:
        if c.exists():
            return c.resolve()

    # Not found: return the CWD-based candidate for a stable error message.
    return candidates[0].resolve()


def _resolve_existing_dir(path_str: Optional[str]) -> Optional[Path]:
    """Resolve a string to an existing local directory.

    Unlike _resolve_path_arg(), this returns None when the directory does not exist.
    This is important for arguments like --model-id where values may be HF repo ids.
    """

    if path_str is None:
        return None

    raw = str(path_str).strip()
    if not raw:
        return None

    p = Path(raw).expanduser()
    if p.is_absolute():
        return p.resolve() if p.exists() and p.is_dir() else None

    capstone = _repo_root().resolve()
    workspace = capstone.parent
    candidates = [Path.cwd() / p, capstone / p, workspace / p]

    # Special-case: user passes paths starting with "Capstone/...".
    if len(p.parts) > 0 and p.parts[0] == "Capstone":
        candidates.insert(0, workspace / p)

    for c in candidates:
        if c.exists() and c.is_dir():
            return c.resolve()
    return None


def _ensure_stage2_on_syspath() -> None:
    """Ensure Capstone/stage_2 is on sys.path.

    This allows local stub modules (e.g. flash_attn) to be discovered during
    transformers' dynamic import checks.
    """

    stage2 = str(_stage2_root().resolve())
    if stage2 not in sys.path:
        sys.path.insert(0, stage2)


def build_presets() -> Dict[str, Preset]:
    capstone = _repo_root()
    return {
        # Classification datasets (folder-of-classes)
        "trashnet": Preset(
            name="trashnet",
            images_dir=capstone / "datasets/classification/trashnet/dataset-preprocessed",
            class_subfolders=True,
        ),
        "realwaste": Preset(
            name="realwaste",
            images_dir=capstone / "datasets/classification/realwaste/dataset-preprocessed",
            class_subfolders=True,
        ),
        # Detection datasets prepared in Stage 1 (YOLO images/labels)
        "taco_det": Preset(
            name="taco_det",
            images_dir=capstone / "stage_1/detection/data/taco_yolo/val/images",
            yolo_labels_dir=capstone / "stage_1/detection/data/taco_yolo/val/labels",
        ),
        "icra19_det": Preset(
            name="icra19_det",
            images_dir=capstone / "stage_1/detection/data/icra19_yolo/test/images",
            yolo_labels_dir=capstone / "stage_1/detection/data/icra19_yolo/test/labels",
        ),
        # Segmentation datasets prepared in Stage 1 (images + binary masks)
        "taco_seg": Preset(
            name="taco_seg",
            images_dir=capstone / "stage_1/segmentation/data/taco/val/images",
            masks_dir=capstone / "stage_1/segmentation/data/taco/val/masks",
        ),
        "dwsd_seg": Preset(
            name="dwsd_seg",
            images_dir=capstone / "stage_1/segmentation/data/dwsd/test/images",
            masks_dir=capstone / "stage_1/segmentation/data/dwsd/test/masks",
        ),
    }


def list_images(images_dir: Path) -> List[Path]:
    exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
    if not images_dir.exists():
        return []
    paths: List[Path] = []
    for p in images_dir.rglob("*"):
        if p.is_file() and p.suffix in exts:
            paths.append(p)
    return sorted(paths)


def iter_classification_images(root: Path) -> Iterator[Tuple[Path, str]]:
    # root/<class_name>/*.{jpg,jpeg,png} (case-insensitive)
    # Use the same robust image listing as list_images() so datasets with .JPG work.
    for class_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        label = class_dir.name
        for img_path in list_images(class_dir):
            yield img_path, label


def sample_class_balanced_images(root: Path, max_images: int) -> List[Tuple[Path, str]]:
    """Return up to max_images, mixed across class subfolders.

    This avoids taking only the first class (e.g., 'cardboard') when the dataset
    is laid out as root/<class_name>/*.jpg and --max-images is small.
    """

    if max_images <= 0 or not root.exists():
        return []

    class_dirs = [p for p in sorted(root.iterdir()) if p.is_dir()]
    per_class: List[Tuple[str, List[Path]]] = [(d.name, list_images(d)) for d in class_dirs]

    out: List[Tuple[Path, str]] = []
    i = 0
    while len(out) < max_images:
        progressed = False
        for label, imgs in per_class:
            if i < len(imgs):
                out.append((imgs[i], label))
                progressed = True
                if len(out) >= max_images:
                    break
        if not progressed:
            break
        i += 1
    return out


def sample_n_per_class_mixed(root: Path, max_per_class: int, max_total: int) -> List[Tuple[Path, str]]:
    """Sample up to max_per_class images per class and interleave classes.

    - If max_total > 0, cap the total returned.
    - If a class has fewer than max_per_class images, it contributes all it has.
    """

    if max_per_class <= 0 or not root.exists():
        return []

    class_dirs = [p for p in sorted(root.iterdir()) if p.is_dir()]
    per_class: List[Tuple[str, List[Path]]] = []
    for d in class_dirs:
        imgs = list_images(d)[: int(max_per_class)]
        if imgs:
            per_class.append((d.name, imgs))

    out: List[Tuple[Path, str]] = []
    i = 0
    while True:
        progressed = False
        for label, imgs in per_class:
            if i < len(imgs):
                out.append((imgs[i], label))
                progressed = True
                if max_total and max_total > 0 and len(out) >= int(max_total):
                    return out
        if not progressed:
            break
        i += 1
    return out


def infer_class_names_from_subfolders(root: Path) -> List[str]:
    """Infer class names from immediate subfolders (for folder-of-classes datasets)."""
    if not root.exists():
        return []
    names: List[str] = []
    for p in sorted([p for p in root.iterdir() if p.is_dir()]):
        # keep original folder name, but strip extra whitespace
        nm = p.name.strip()
        if nm:
            names.append(nm)
    return names


def normalize_label_for_prompt(label: str) -> str:
    # A small normalization for better language-model prompting.
    return " ".join(label.strip().lower().replace("_", " ").split())


def _phrase_matches_label(phrase: str, label: str) -> bool:
    p = normalize_label_for_prompt(str(phrase))
    l = normalize_label_for_prompt(str(label))
    if not p or not l:
        return False
    return (p in l) or (l in p)


def filter_detections_by_label_phrase(
    bboxes: List[List[float]],
    labels: List[str],
    phrase: Optional[str],
) -> Tuple[List[List[float]], List[str]]:
    """Keep only detections whose label matches phrase (case-insensitive substring).

    Useful when open-vocab emits multiple overlapping labels (e.g., 'trash', 'bottle',
    'cardboard') and you want only one target class.
    """

    if phrase is None or not str(phrase).strip():
        return bboxes, labels

    out_b: List[List[float]] = []
    out_l: List[str] = []
    for i, bb in enumerate(bboxes):
        lab = labels[i] if i < len(labels) else ""
        if _phrase_matches_label(str(phrase), str(lab)):
            out_b.append(bb)
            out_l.append(str(lab) if str(lab).strip() else str(phrase))
    return out_b, out_l


GENERIC_LABELS = {
    "trash",
    "garbage",
    "waste",
    "litter",
    "rubbish",
    "object",
    "objects",
    "item",
    "items",
    "thing",
    "things",
}


def drop_generic_detections_if_specific_exist(
    bboxes: List[List[float]],
    labels: List[str],
    generic_labels: Optional[set] = None,
) -> Tuple[List[List[float]], List[str]]:
    """Drop boxes whose label is generic if any non-generic labels exist.

    This is a pragmatic cleanup for open-vocab outputs where Florence may emit both
    specific and generic labels (e.g., 'bottle' + 'trash'). We keep generic only
    when it's all we have.
    """

    if not bboxes or not labels:
        return bboxes, labels
    g = generic_labels if generic_labels is not None else GENERIC_LABELS

    norm = [normalize_label_for_prompt(lab) for lab in labels]
    has_specific = any((n and n not in g) for n in norm)
    if not has_specific:
        return bboxes, labels

    out_b: List[List[float]] = []
    out_l: List[str] = []
    for bb, lab, n in zip(bboxes, labels, norm):
        if n in g:
            continue
        out_b.append(bb)
        out_l.append(lab)

    # Safety fallback: if filtering removed everything, keep originals.
    return (out_b, out_l) if out_b else (bboxes, labels)


def override_all_labels(labels: List[str], new_label: str) -> List[str]:
    if not str(new_label).strip():
        return labels
    return [str(new_label)] * len(labels)


def _uniq_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for it in items:
        key = it.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def read_yolo_labels(label_path: Path, image_size: Tuple[int, int]) -> Dict[str, np.ndarray]:
    # YOLO: class_id x_center y_center w h (normalized)
    w_img, h_img = image_size
    boxes = []
    labels = []
    if not label_path.exists():
        return {"boxes": np.zeros((0, 4), dtype=np.float32), "labels": np.zeros((0,), dtype=np.int64)}
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        x_c = float(parts[1]) * w_img
        y_c = float(parts[2]) * h_img
        bw = float(parts[3]) * w_img
        bh = float(parts[4]) * h_img
        x1 = x_c - bw / 2
        y1 = y_c - bh / 2
        x2 = x_c + bw / 2
        y2 = y_c + bh / 2
        boxes.append([x1, y1, x2, y2])
        labels.append(cls)
    return {"boxes": np.array(boxes, dtype=np.float32), "labels": np.array(labels, dtype=np.int64)}


def safe_font(size: int = 18) -> ImageFont.ImageFont:
    for font_name in ("DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(font_name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _sanitize_bbox_xyxy(
    bbox: List[float], image_size: Tuple[int, int]
) -> Optional[Tuple[float, float, float, float]]:
    """Return a safe (x0,y0,x1,y1) bbox in image coordinates or None.

    Florence post-processing should yield xyxy boxes, but in practice some boxes
    may have swapped corners, NaNs, or fall outside image bounds.
    """

    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None

    try:
        x0, y0, x1, y1 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    except Exception:
        return None

    if not all(np.isfinite([x0, y0, x1, y1])):
        return None

    # Ensure correct ordering
    x_min, x_max = (x0, x1) if x0 <= x1 else (x1, x0)
    y_min, y_max = (y0, y1) if y0 <= y1 else (y1, y0)

    w, h = image_size
    # Clamp to bounds
    x_min = max(0.0, min(float(w - 1), x_min))
    x_max = max(0.0, min(float(w - 1), x_max))
    y_min = max(0.0, min(float(h - 1), y_min))
    y_max = max(0.0, min(float(h - 1), y_max))

    # Skip degenerate boxes
    if (x_max - x_min) < 1.0 or (y_max - y_min) < 1.0:
        return None

    return x_min, y_min, x_max, y_max


def draw_bboxes(image: Image.Image, bboxes: List[List[float]], labels: List[str]) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    font = safe_font(18)

    for bbox, label in zip(bboxes, labels):
        safe = _sanitize_bbox_xyxy(bbox, (out.width, out.height))
        if safe is None:
            continue
        x0, y0, x1, y1 = safe
        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=3)

        text = str(label)
        tx, ty = x0, max(0, y0 - 22)
        tb = draw.textbbox((tx, ty), text, font=font)
        draw.rectangle(tb, fill=(255, 0, 0))
        draw.text((tx, ty), text, fill=(255, 255, 255), font=font)

    return out


def _iter_polygons(polygons_obj) -> Iterator[List[Tuple[float, float]]]:
    """Yield polygon point lists from Florence polygon output.

    Florence-2 post_process_generation returns:
      { '<TASK>': { 'polygons': [ _polygons ], 'labels': [...] } }
    where each `_polygons` is typically a list of polygons and each polygon is a list of [x,y] points.
    """
    def _try_float(x) -> Optional[float]:
        try:
            v = float(x)
        except Exception:
            return None
        if not np.isfinite(v):
            return None
        return v

    if polygons_obj is None:
        return

    # Sometimes polygons are wrapped in a dict.
    if isinstance(polygons_obj, dict):
        for key in ("polygons", "polygon", "points"):
            if key in polygons_obj:
                yield from _iter_polygons(polygons_obj[key])
                return
        return

    if not isinstance(polygons_obj, (list, tuple)) or len(polygons_obj) == 0:
        return

    # Case 1: flat coordinate list: [x0, y0, x1, y1, ...]
    if all(_try_float(v) is not None for v in polygons_obj):
        coords = [_try_float(v) for v in polygons_obj]
        coords = [c for c in coords if c is not None]
        pts = [(coords[i], coords[i + 1]) for i in range(0, len(coords) - 1, 2)]
        if len(pts) >= 3:
            yield [(float(x), float(y)) for x, y in pts]
        return

    # Case 2: list of point pairs: [[x,y], [x,y], ...]
    if all(
        isinstance(v, (list, tuple))
        and len(v) == 2
        and _try_float(v[0]) is not None
        and _try_float(v[1]) is not None
        for v in polygons_obj
    ):
        pts2 = [(_try_float(x), _try_float(y)) for x, y in polygons_obj]
        pts2 = [(x, y) for x, y in pts2 if x is not None and y is not None]
        if len(pts2) >= 3:
            yield [(float(x), float(y)) for x, y in pts2]
        return

    # Case 3: nested list structure (list of polygons, list of lists, etc.)
    for item in polygons_obj:
        yield from _iter_polygons(item)


def polygons_to_mask(image_size: Tuple[int, int], polygons: List) -> np.ndarray:
    w, h = image_size
    mask_img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask_img)
    for poly in _iter_polygons(polygons):
        draw.polygon(poly, fill=255)
    return np.array(mask_img, dtype=np.uint8)


def _choose_best_bbox_for_phrase(
    bboxes: List[List[float]],
    labels: List[str],
    phrase: str,
    image_size: Tuple[int, int],
) -> Optional[Tuple[int, int, int, int]]:
    """Pick a bbox index that best matches the phrase; fallback to largest area."""

    if not bboxes:
        return None
    w, h = image_size

    phrase_norm = normalize_label_for_prompt(phrase)
    # sanitize bboxes and compute areas
    candidates: List[Tuple[int, Tuple[int, int, int, int], float, str]] = []
    for i, bb in enumerate(bboxes):
        safe = _sanitize_bbox_xyxy(bb, (w, h))
        if safe is None:
            continue
        x0, y0, x1, y1 = safe
        area = float((x1 - x0) * (y1 - y0))
        lab = labels[i] if i < len(labels) else ""
        candidates.append((i, (int(x0), int(y0), int(x1), int(y1)), area, normalize_label_for_prompt(str(lab))))

    if not candidates:
        return None

    # 1) try label contains phrase, 2) try phrase contains label, else largest
    best = None
    for _, xyxy, area, lab in candidates:
        if phrase_norm and lab and (phrase_norm in lab or lab in phrase_norm):
            if best is None or area > best[2]:
                best = (xyxy, lab, area)
    if best is not None:
        return best[0]

    candidates.sort(key=lambda t: t[2], reverse=True)
    return candidates[0][1]


def _crop_with_padding(
    image: Image.Image,
    bbox_xyxy: Tuple[int, int, int, int],
    pad_frac: float,
) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    w, h = image.size
    x0, y0, x1, y1 = bbox_xyxy
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    pad_x = int(round(bw * pad_frac))
    pad_y = int(round(bh * pad_frac))

    cx0 = max(0, x0 - pad_x)
    cy0 = max(0, y0 - pad_y)
    cx1 = min(w, x1 + pad_x)
    cy1 = min(h, y1 + pad_y)

    crop = image.crop((cx0, cy0, cx1, cy1))
    return crop, (cx0, cy0, cx1, cy1)


def overlay_mask(image: Image.Image, mask: np.ndarray, color=(255, 0, 0), alpha: int = 120) -> Image.Image:
    """Overlay a binary mask onto an image.

    This is implemented with NumPy to avoid slow Python per-pixel loops.
    """
    base = np.array(image.convert("RGBA"), dtype=np.uint8)
    if mask.dtype != np.uint8:
        mask_u8 = (mask > 0).astype(np.uint8) * 255
    else:
        mask_u8 = mask

    # Ensure shape matches image
    if mask_u8.shape[0] != base.shape[0] or mask_u8.shape[1] != base.shape[1]:
        mask_u8 = np.array(Image.fromarray(mask_u8).resize((base.shape[1], base.shape[0]), Image.NEAREST))

    m = mask_u8 > 0
    overlay = np.zeros_like(base)
    overlay[..., 0] = color[0]
    overlay[..., 1] = color[1]
    overlay[..., 2] = color[2]
    overlay[..., 3] = alpha

    # Alpha composite only where mask is True
    out = base.copy()
    a = overlay[..., 3:4].astype(np.float32) / 255.0
    out[m, :3] = (base[m, :3].astype(np.float32) * (1 - a[m]) + overlay[m, :3].astype(np.float32) * a[m]).astype(np.uint8)
    out[m, 3] = 255

    return Image.fromarray(out, mode="RGBA").convert("RGB")


def load_sam(
    checkpoint_path: Path,
    model_type: str,
    device: str,
):
    """Lazy-load SAM to avoid forcing dependency unless requested."""

    try:
        from segment_anything import sam_model_registry, SamPredictor  # type: ignore[import-not-found]
    except Exception as e:
        raise SystemExit(
            "segment_anything is not installed. Install it (and its deps) to use --seg-backend sam_box. "
            f"Original error: {e}"
        )

    if not checkpoint_path.exists():
        raise SystemExit(
            f"SAM checkpoint not found: {checkpoint_path}\n"
            "Download sam_vit_h_4b8939.pth from:\n"
            "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth\n"
            "and place it under Capstone/stage_2/ (recommended) or provide --sam-checkpoint."
        )

    sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
    sam.to(device=device if device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    predictor = SamPredictor(sam)
    return predictor


def sam_mask_from_box(predictor, image_rgb: np.ndarray, bbox_xyxy: Tuple[int, int, int, int]) -> np.ndarray:
    """Run SAM predictor with a single box prompt; return binary uint8 mask (0/255)."""

    x0, y0, x1, y1 = bbox_xyxy
    box = np.array([x0, y0, x1, y1], dtype=np.float32)
    predictor.set_image(image_rgb)
    masks, scores, _ = predictor.predict(
        box=box[None, :],
        multimask_output=True,
    )
    if masks is None or len(masks) == 0:
        return np.zeros((image_rgb.shape[0], image_rgb.shape[1]), dtype=np.uint8)
    # Pick best score
    best_idx = int(np.argmax(scores)) if scores is not None and len(scores) else 0
    m = masks[best_idx].astype(np.uint8) * 255
    return m


def load_florence2(model_id: str, device: str) -> Tuple[AutoModelForCausalLM, AutoProcessor]:
    # Support both:
    #  - Full model checkpoints (HF repo id or local dir with config.json)
    #  - PEFT/LoRA adapter-only checkpoints saved by our fine-tune script
    #    (local dir containing adapter_config.json + adapter_model.safetensors)
    adapter_dir = Path(model_id).expanduser()
    is_local_adapter = adapter_dir.exists() and adapter_dir.is_dir() and (adapter_dir / "adapter_config.json").exists()

    if is_local_adapter:
        adapter_cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
        base_id = adapter_cfg.get("base_model_name_or_path")
        if not base_id or not str(base_id).strip():
            raise SystemExit(f"LoRA adapter is missing base_model_name_or_path in adapter_config.json: {adapter_dir}")

        # Processor: prefer local copy (tokenizer configs) if present, else fall back to base.
        try:
            processor = AutoProcessor.from_pretrained(str(adapter_dir), trust_remote_code=True)
        except Exception:
            processor = AutoProcessor.from_pretrained(str(base_id), trust_remote_code=True)

        base_model = AutoModelForCausalLM.from_pretrained(
            str(base_id),
            trust_remote_code=True,
            torch_dtype=torch.float32,
            attn_implementation="eager",  # avoids SDPA support issues on some builds
        )

        try:
            from peft import PeftModel  # type: ignore
        except Exception as e:
            raise SystemExit(
                "This checkpoint is a LoRA adapter, but `peft` is not available. "
                "Install peft (and accelerate) or run with a full model id. "
                f"Original error: {e}"
            )

        model = PeftModel.from_pretrained(base_model, str(adapter_dir))
        # Merge LoRA weights into base for faster inference when available.
        if hasattr(model, "merge_and_unload"):
            try:
                model = model.merge_and_unload()
            except Exception:
                pass
    else:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            attn_implementation="eager",  # avoids SDPA support issues on some builds
        )

    # Some custom model code may prefer FlashAttention if available.
    # We explicitly disable it where possible to keep runs portable.
    if hasattr(model, "config") and model.config is not None:
        if hasattr(model.config, "attn_implementation"):
            try:
                model.config.attn_implementation = "eager"
            except Exception:
                pass
        for flag_name in ("use_flash_attn", "use_flash_attn_2", "use_flash_attention_2", "flash_attn"):
            if hasattr(model.config, flag_name):
                try:
                    setattr(model.config, flag_name, False)
                except Exception:
                    pass

    model.to(device)
    model.eval()
    return model, processor


@torch.no_grad()
def run_task(
    model: AutoModelForCausalLM,
    processor: AutoProcessor,
    image: Image.Image,
    task_token: str,
    task_input: Optional[str],
    device: str,
    max_new_tokens: int,
    num_beams: int = 1,
) -> Dict:
    prompt = task_token if not task_input else f"{task_token} {task_input}"
    inputs = processor(text=prompt, images=image, return_tensors="pt")

    input_ids = inputs["input_ids"].to(device)
    pixel_values = inputs["pixel_values"].to(device=device, dtype=torch.float32)

    # Prevent generation from exceeding the model's max sequence length.
    # Warning seen on clusters: (input_len + max_new_tokens) > model_max_length.
    max_new = max_new_tokens
    tokenizer = getattr(processor, "tokenizer", None)
    model_max_len = getattr(tokenizer, "model_max_length", None)
    if isinstance(model_max_len, int) and 0 < model_max_len < 1_000_000:
        input_len = int(input_ids.shape[-1])
        budget = max(1, model_max_len - input_len)
        max_new = min(max_new, budget)

    generated_ids = model.generate(
        input_ids=input_ids,
        pixel_values=pixel_values,
        max_new_tokens=max_new,
        do_sample=False,
        num_beams=max(1, int(num_beams)),
        early_stopping=False,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(
        generated_text,
        task=task_token,
        image_size=(image.width, image.height),
    )
    return {"prompt": prompt, "generated_text": generated_text, "parsed": parsed}


def pick_class_from_open_vocab(parsed_open_vocab: Dict, fallback: str = "trash") -> str:
    payload = parsed_open_vocab.get(TASK_OPEN_VOCAB, {})

    labels = payload.get("bboxes_labels") or payload.get("polygons_labels") or payload.get("labels")
    if not labels:
        return fallback
    counts = Counter([str(x).strip().lower() for x in labels if str(x).strip()])
    if not counts:
        return fallback
    return counts.most_common(1)[0][0]


def extract_open_vocab_bboxes(parsed_open_vocab: Dict) -> Tuple[List[List[float]], List[str]]:
    payload = parsed_open_vocab.get(TASK_OPEN_VOCAB, {})
    bboxes = payload.get("bboxes", []) or []
    labels = payload.get("bboxes_labels", []) or payload.get("labels", []) or []
    labels = [str(x) for x in labels]
    if len(labels) != len(bboxes):
        labels = labels[: len(bboxes)] + ["object"] * max(0, len(bboxes) - len(labels))
    # Some Florence open-vocab outputs return polygons only.
    # In that case, derive a bbox per instance from polygon points.
    if not bboxes:
        polygons = payload.get("polygons", []) or []
        poly_labels = payload.get("polygons_labels", []) or payload.get("labels", []) or []
        poly_labels = [str(x) for x in poly_labels]

        def _bbox_from_poly_group(poly_group) -> Optional[List[float]]:
            xs: List[float] = []
            ys: List[float] = []
            for poly in _iter_polygons(poly_group):
                for x, y in poly:
                    xs.append(float(x))
                    ys.append(float(y))
            if not xs or not ys:
                return None
            return [min(xs), min(ys), max(xs), max(ys)]

        bb_out: List[List[float]] = []
        lb_out: List[str] = []
        for i, poly_group in enumerate(polygons):
            bb = _bbox_from_poly_group(poly_group)
            if bb is None:
                continue
            bb_out.append(bb)
            lb_out.append(poly_labels[i] if i < len(poly_labels) else "object")
        if bb_out:
            return bb_out, lb_out
    return bboxes, labels


def _filter_bboxes(
    bboxes: List[List[float]],
    labels: List[str],
    image_size: Tuple[int, int],
    select: str,
    max_boxes: int,
    min_area_frac: float,
    phrase: Optional[str],
) -> Tuple[List[List[float]], List[str]]:
    """Filter/sort/select bboxes for visualization and downstream cropping."""

    w, h = image_size
    img_area = float(max(1, w * h))
    items: List[Tuple[float, float, float, float, float, str]] = []  # x0,y0,x1,y1,area,label
    for i, bb in enumerate(bboxes):
        safe = _sanitize_bbox_xyxy(bb, (w, h))
        if safe is None:
            continue
        x0, y0, x1, y1 = safe
        area = float((x1 - x0) * (y1 - y0))
        if area / img_area < float(min_area_frac):
            continue
        lab = labels[i] if i < len(labels) else "object"
        items.append((x0, y0, x1, y1, area, str(lab)))

    if not items:
        return [], []

    items.sort(key=lambda t: t[4], reverse=True)

    select = str(select).lower().strip()
    if select == "largest":
        x0, y0, x1, y1, _, lab = items[0]
        return [[x0, y0, x1, y1]], [lab]

    if select == "phrase":
        if phrase and str(phrase).strip():
            phrase_norm = normalize_label_for_prompt(str(phrase))
            best = None
            for x0, y0, x1, y1, area, lab in items:
                lab_norm = normalize_label_for_prompt(lab)
                if phrase_norm and lab_norm and (phrase_norm in lab_norm or lab_norm in phrase_norm):
                    if best is None or area > best[4]:
                        best = (x0, y0, x1, y1, area, lab)
            if best is not None:
                x0, y0, x1, y1, _, lab = best
                return [[x0, y0, x1, y1]], [lab]
        # fallback
        x0, y0, x1, y1, _, lab = items[0]
        return [[x0, y0, x1, y1]], [lab]

    # topk/all
    if max_boxes is not None and int(max_boxes) > 0:
        items = items[: int(max_boxes)]
    out_b = [[x0, y0, x1, y1] for x0, y0, x1, y1, _, _ in items]
    out_l = [lab for *_rest, lab in items]
    return out_b, out_l


def _bbox_iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, (ax1 - ax0)) * max(0.0, (ay1 - ay0))
    area_b = max(0.0, (bx1 - bx0)) * max(0.0, (by1 - by0))
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _dedupe_bboxes(
    bboxes: List[List[float]],
    labels: List[str],
    image_size: Tuple[int, int],
    iou_thr: float,
) -> Tuple[List[List[float]], List[str]]:
    """Remove near-duplicate boxes (no confidence scores available).

    We keep larger-area boxes first; suppress boxes with IoU >= iou_thr.
    """

    w, h = image_size
    items: List[Tuple[float, float, float, float, float, str]] = []
    for i, bb in enumerate(bboxes):
        safe = _sanitize_bbox_xyxy(bb, (w, h))
        if safe is None:
            continue
        x0, y0, x1, y1 = safe
        area = float((x1 - x0) * (y1 - y0))
        lab = labels[i] if i < len(labels) else "object"
        items.append((x0, y0, x1, y1, area, str(lab)))

    if not items:
        return [], []

    items.sort(key=lambda t: t[4], reverse=True)
    kept: List[Tuple[float, float, float, float, float, str]] = []
    for x0, y0, x1, y1, area, lab in items:
        cur = (x0, y0, x1, y1)
        if any(_bbox_iou_xyxy(cur, (kx0, ky0, kx1, ky1)) >= float(iou_thr) for kx0, ky0, kx1, ky1, *_ in kept):
            continue
        kept.append((x0, y0, x1, y1, area, lab))

    out_b = [[x0, y0, x1, y1] for x0, y0, x1, y1, *_ in kept]
    out_l = [lab for *_rest, lab in kept]
    return out_b, out_l


def _iter_tiles(
    image_w: int,
    image_h: int,
    tile_size: int,
    overlap: float,
    max_tiles: int,
) -> Iterator[Tuple[int, int, int, int]]:
    """Yield (x0,y0,x1,y1) tiles covering the image."""

    ts = int(tile_size)
    if ts <= 0:
        ts = min(image_w, image_h)
    ov = float(overlap)
    ov = max(0.0, min(0.9, ov))
    step = max(1, int(round(ts * (1.0 - ov))))

    tiles = 0
    y = 0
    while y < image_h and tiles < int(max_tiles):
        x = 0
        y1 = min(image_h, y + ts)
        y0 = max(0, y1 - ts)
        while x < image_w and tiles < int(max_tiles):
            x1 = min(image_w, x + ts)
            x0 = max(0, x1 - ts)
            yield (x0, y0, x1, y1)
            tiles += 1
            if x1 >= image_w:
                break
            x += step
        if y1 >= image_h:
            break
        y += step


def _shift_bboxes(
    bboxes_xyxy: List[List[float]],
    dx: int,
    dy: int,
) -> List[List[float]]:
    out: List[List[float]] = []
    for bb in bboxes_xyxy:
        if not isinstance(bb, (list, tuple)) or len(bb) != 4:
            continue
        try:
            x0, y0, x1, y1 = float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])
        except Exception:
            continue
        out.append([x0 + dx, y0 + dy, x1 + dx, y1 + dy])
    return out


def florence_open_vocab_sliced(
    model: AutoModelForCausalLM,
    processor: AutoProcessor,
    image: Image.Image,
    prompt: str,
    device: str,
    max_new_tokens: int,
    num_beams: int,
    tile_size: int,
    overlap: float,
    max_tiles: int,
    merge_iou: float,
) -> Tuple[List[List[float]], List[str], Dict]:
    """Run open-vocab detection on overlapping tiles and merge results.

    Returns merged (bboxes, labels, debug_info).
    """

    full_w, full_h = image.size
    all_bboxes: List[List[float]] = []
    all_labels: List[str] = []
    debug = {"tiles": 0, "tile_size": int(tile_size), "overlap": float(overlap), "max_tiles": int(max_tiles)}

    for x0, y0, x1, y1 in _iter_tiles(full_w, full_h, tile_size=tile_size, overlap=overlap, max_tiles=max_tiles):
        tile = image.crop((x0, y0, x1, y1))
        out = run_task(
            model,
            processor,
            tile,
            TASK_OPEN_VOCAB,
            prompt,
            device=device,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
        parsed = out.get("parsed", {})
        tb, tl = extract_open_vocab_bboxes(parsed)
        if tb:
            shifted = _shift_bboxes(tb, dx=int(x0), dy=int(y0))
            all_bboxes.extend(shifted)
            all_labels.extend(tl)
        debug["tiles"] = int(debug["tiles"]) + 1

    # Sanitize and merge duplicates across tiles.
    b_s, l_s = _dedupe_bboxes(all_bboxes, all_labels, image_size=(full_w, full_h), iou_thr=float(merge_iou))
    return b_s, l_s, debug


def pick_class_from_labels(labels: List[str], fallback: str = "trash") -> str:
    if not labels:
        return fallback
    counts = Counter([str(x).strip().lower() for x in labels if str(x).strip()])
    if not counts:
        return fallback
    return counts.most_common(1)[0][0]


def _extract_caption_text(parsed_caption: Dict) -> str:
    payload = parsed_caption.get(TASK_CAPTION)
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload.strip()
    # Some processors return dicts like {"caption": "..."}
    if isinstance(payload, dict):
        for k in ("caption", "text", "result"):
            v = payload.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return str(payload).strip()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_csv(path: Path, header: List[str], rows: List[List[str]]) -> None:
    lines = [",".join(header)]
    for r in rows:
        # naive CSV escaping (sufficient for paths/labels in this repo)
        lines.append(",".join(str(x).replace("\n", " ").replace(",", " ") for x in r))
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    _ensure_stage2_on_syspath()
    presets = build_presets()

    parser = argparse.ArgumentParser(description="Unified Florence-2 pipeline for garbage datasets")
    parser.add_argument(
        "--preset",
        choices=sorted(presets.keys()) + ["custom"],
        default="realwaste",
        help="Dataset preset (ignored if --image-path is provided)",
    )
    parser.add_argument(
        "--image-path",
        type=str,
        default=None,
        help="Run unified Florence-2 on a single image (overrides --preset/--images-dir).",
    )
    parser.add_argument("--images-dir", type=str, default=None, help="(custom) images directory")
    parser.add_argument("--class-subfolders", action="store_true", help="(custom) treat images-dir as folder-of-classes")
    parser.add_argument("--yolo-labels-dir", type=str, default=None, help="(custom) YOLO labels directory")
    parser.add_argument("--masks-dir", type=str, default=None, help="(custom) binary masks directory")

    parser.add_argument(
        "--model",
        choices=sorted(MODELS.keys()),
        default="base",
        help="Shortcut model key (ignored if --model-id is provided).",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help=(
            "Hugging Face model id or local path to a fine-tuned checkpoint. "
            "If set, overrides --model."
        ),
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max-images", type=int, default=25)
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=0,
        help=(
            "For folder-of-classes datasets (--class-subfolders / TrashNet/RealWaste presets), "
            "take at most this many images from each class folder (0 disables)."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--num-beams", type=int, default=1)

    parser.add_argument(
        "--hf-cache-dir",
        type=str,
        default=None,
        help="Optional Hugging Face cache dir (recommended: /tmp/$USER/hf) to avoid filling $HOME/.cache",
    )

    parser.add_argument("--dry-run", action="store_true", help="Only list/validate images and exit (no model load)")
    parser.add_argument("--print-presets", action="store_true", help="Print preset paths and exit")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for Hugging Face caches (recommended on GPU nodes to avoid $HOME quota).",
    )

    parser.add_argument(
        "--class-names",
        type=str,
        default="auto",
        help=(
            "Comma-separated waste class names for open-vocab. Use 'auto' to infer from "
            "folder-of-classes datasets (TrashNet/RealWaste presets)."
        ),
    )

    parser.add_argument(
        "--open-vocab-extra",
        type=str,
        default="trash,garbage,litter,waste,rubbish,plastic bottle,bottle,can,paper,cardboard",
        help=(
            "Extra comma-separated phrases appended to the open-vocab prompt. "
            "This often improves bbox quality on waste images (e.g., add 'plastic bottle')."
        ),
    )

    parser.add_argument(
        "--open-vocab-terms",
        type=str,
        default="",
        help=(
            "Comma-separated open-vocab terms to use as the prompt (overrides --class-names and --open-vocab-extra). "
            "Example: 'plastic bottle'."
        ),
    )

    parser.add_argument(
        "--open-vocab-retry-generic",
        action="store_true",
        help=(
            "If the first open-vocab pass yields no bboxes, retry once with a broader "
            "generic waste-object term list (still Florence-2 only)."
        ),
    )
    parser.add_argument(
        "--bbox-dedupe-iou",
        type=float,
        default=0.85,
        help="IoU threshold for suppressing near-duplicate detection boxes (0 disables dedupe).",
    )

    parser.add_argument(
        "--detection-mode",
        choices=["open_vocab", "od", "both"],
        default="open_vocab",
        help=(
            "Which detection output to visualize/save. 'open_vocab' usually works better for "
            "waste categories because it uses your class list."
        ),
    )

    parser.add_argument(
        "--slice-inference",
        action="store_true",
        help=(
            "SAHI-style slicing for better small-object detection: run open-vocab detection on overlapping tiles "
            "and merge boxes back into the full image. (Florence-2 only)"
        ),
    )
    parser.add_argument("--slice-size", type=int, default=768, help="Tile size (pixels) for --slice-inference.")
    parser.add_argument(
        "--slice-overlap",
        type=float,
        default=0.25,
        help="Overlap ratio between tiles for --slice-inference (0..0.9).",
    )
    parser.add_argument(
        "--slice-max-tiles",
        type=int,
        default=64,
        help="Safety cap on number of tiles per image for --slice-inference.",
    )
    parser.add_argument(
        "--slice-merge-iou",
        type=float,
        default=0.60,
        help="IoU threshold for merging duplicate boxes across tiles.",
    )

    parser.add_argument(
        "--det-select",
        choices=["all", "topk", "largest", "phrase"],
        default="largest",
        help="How to select bboxes for visualization/cropping from detection outputs.",
    )
    parser.add_argument("--det-max-boxes", type=int, default=5, help="Max boxes to keep when --det-select is all/topk")
    parser.add_argument(
        "--det-min-area-frac",
        type=float,
        default=0.01,
        help="Drop very tiny boxes: minimum area fraction of image for drawing/cropping.",
    )
    parser.add_argument(
        "--det-fallback-min-area-frac",
        type=float,
        default=0.0,
        help=(
            "If filtering removes all boxes, retry filtering with this min-area-frac "
            "(useful when objects are small)."
        ),
    )
    parser.add_argument(
        "--det-phrase",
        type=str,
        default=None,
        help="Phrase used when --det-select phrase (defaults to seg phrase if omitted).",
    )

    parser.add_argument(
        "--det-keep-label",
        type=str,
        default=None,
        help=(
            "If set, keep only detection boxes whose predicted label matches this phrase (case-insensitive substring). "
            "Example: --det-keep-label 'plastic bottle'."
        ),
    )

    parser.add_argument(
        "--det-drop-generic-labels",
        action="store_true",
        help=(
            "If set, drop generic labels like 'trash/garbage/waste/object' when any specific label exists. "
            "Useful in real-world mode to avoid duplicate labels for the same object."
        ),
    )

    parser.add_argument(
        "--det-single-class-from-classification",
        action="store_true",
        help=(
            "If set, keep only detections matching the per-image classification label, and rename detection labels to that class. "
            "Works best when --classification-mode caption is fine-tuned to output a single class name (e.g., TrashNet)."
        ),
    )

    parser.add_argument("--no-detection", action="store_true")
    parser.add_argument("--no-classification", action="store_true")
    parser.add_argument("--no-segmentation", action="store_true")

    parser.add_argument(
        "--classification-mode",
        choices=["open_vocab", "caption"],
        default="open_vocab",
        help=(
            "How to produce a classification-like output. "
            "open_vocab: picks most frequent label from open-vocab detection. "
            "caption: runs <CAPTION> and writes the caption text."
        ),
    )

    parser.add_argument(
        "--seg-phrase",
        type=str,
        default=None,
        help=(
            "Force a fixed referring-expression phrase for segmentation (e.g. 'trash' or 'waste'). "
            "If omitted, the script uses the predicted class label."
        ),
    )

    parser.add_argument(
        "--seg-try-phrases",
        type=str,
        default="",
        help=(
            "Comma-separated fallback phrases to try for Florence segmentation if the first phrase yields no mask. "
            "Example: 'trash,garbage,litter,waste'."
        ),
    )
    parser.add_argument(
        "--seg-merge-mode",
        choices=["first_nonempty", "union"],
        default="first_nonempty",
        help=(
            "How to combine multiple segmentation-phrase attempts: pick the first that yields a non-empty mask, "
            "or union all non-empty masks."
        ),
    )

    parser.add_argument(
        "--seg-from-detection",
        choices=["none", "open_vocab"],
        default="open_vocab",
        help=(
            "Improve segmentation by cropping around a detection box before running segmentation. "
            "Uses open-vocab detection bboxes when available."
        ),
    )
    parser.add_argument(
        "--seg-crop-pad",
        type=float,
        default=0.25,
        help="Padding fraction around the chosen bbox when --seg-from-detection is enabled.",
    )

    parser.add_argument(
        "--seg-backend",
        choices=["florence", "sam_box"],
        default="florence",
        help=(
            "Segmentation backend. florence uses Florence-2 referring-expression segmentation. "
            "sam_box uses SAM prompted by the selected detection box (usually much better masks)."
        ),
    )
    parser.add_argument(
        "--sam-checkpoint",
        type=str,
        default=str(_stage2_root() / "sam_vit_h_4b8939.pth"),
        help="Path to SAM checkpoint (e.g., sam_vit_h_4b8939.pth).",
    )
    parser.add_argument(
        "--sam-model-type",
        type=str,
        default="vit_h",
        help="SAM model type key in sam_model_registry (e.g., vit_h, vit_l, vit_b).",
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs",
        help=(
            "Output subdirectory under Capstone/stage_2/. "
            "Can be a relative path like 'outputs' or 'runs/florence2'. "
            "Absolute paths are rejected to keep everything in one place."
        ),
    )

    args = parser.parse_args()

    if args.print_presets:
        print("Available presets:")
        for k in sorted(presets.keys()):
            p = presets[k]
            print(f"- {k}: images_dir={p.images_dir} class_subfolders={p.class_subfolders} yolo_labels_dir={p.yolo_labels_dir} masks_dir={p.masks_dir}")
        return

    cache_dir = Path(args.cache_dir) if args.cache_dir else default_cache_dir()
    configure_hf_cache(cache_dir)

    # Resolve dataset or single image
    single_image_path: Optional[Path] = Path(args.image_path).expanduser().resolve() if args.image_path else None

    if single_image_path is not None:
        if not single_image_path.exists():
            raise SystemExit(f"--image-path not found: {single_image_path}")
        preset = Preset(name="single_image", images_dir=None, class_subfolders=False)
    else:
        if args.preset == "custom":
            if not args.images_dir:
                raise SystemExit("--images-dir is required for --preset custom")
            preset = Preset(
                name="custom",
                images_dir=_resolve_path_arg(args.images_dir),
                class_subfolders=bool(args.class_subfolders),
                yolo_labels_dir=_resolve_path_arg(args.yolo_labels_dir) if args.yolo_labels_dir else None,
                masks_dir=_resolve_path_arg(args.masks_dir) if args.masks_dir else None,
            )
        else:
            preset = presets[args.preset]

        if preset.images_dir is None or not preset.images_dir.exists():
            capstone = _repo_root().resolve()
            workspace = capstone.parent
            cwd = Path.cwd().resolve()
            raise SystemExit(
                "Images dir not found.\n"
                f"  Provided: {args.images_dir}\n"
                f"  CWD:      {cwd}\n"
                f"  Tried:    {cwd / Path(args.images_dir)}\n"
                f"            {capstone / Path(args.images_dir)}\n"
                f"            {workspace / Path(args.images_dir)}\n"
            )

    if args.hf_cache_dir:
        cache = Path(args.hf_cache_dir)
        cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(cache))
        os.environ.setdefault("HF_HUB_CACHE", str(cache / "hub"))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(cache / "transformers"))

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
        print("⚠️ CUDA not available; using CPU")

    model_id = args.model_id if args.model_id else MODELS[args.model]
    # If the user passed a local checkpoint dir, resolve it so HF doesn't treat it as a repo id.
    if args.model_id is not None:
        resolved_model_dir = _resolve_existing_dir(args.model_id)
        if resolved_model_dir is not None:
            model_id = str(resolved_model_dir)

    # Resolve class list
    class_names: List[str]
    if str(args.class_names).strip().lower() == "auto":
        if preset.class_subfolders and preset.images_dir:
            class_names = infer_class_names_from_subfolders(preset.images_dir)
        else:
            # Reasonable default prompt set for waste images.
            class_names = [
                "plastic",
                "paper",
                "cardboard",
                "glass",
                "metal",
                "food organics",
                "textile trash",
                "vegetation",
                "miscellaneous trash",
            ]
    else:
        class_names = [c.strip() for c in str(args.class_names).split(",") if c.strip()]
        if not class_names:
            class_names = ["glass", "paper", "cardboard", "plastic", "metal", "trash"]

    class_names_for_prompt = [normalize_label_for_prompt(c) for c in class_names]
    extra_terms = [normalize_label_for_prompt(x) for x in str(args.open_vocab_extra).split(",") if x.strip()]

    if str(args.open_vocab_terms).strip():
        user_terms = [normalize_label_for_prompt(x) for x in str(args.open_vocab_terms).split(",") if x.strip()]
        open_vocab_terms = _uniq_keep_order(user_terms)
    else:
        open_vocab_terms = _uniq_keep_order(class_names_for_prompt + extra_terms)
    open_vocab_prompt = ", ".join(open_vocab_terms)

    # Build image list
    if single_image_path is not None:
        image_items = [(single_image_path, None)]
    else:
        if preset.class_subfolders:
            if args.max_per_class and int(args.max_per_class) > 0:
                image_items = sample_n_per_class_mixed(
                    preset.images_dir,
                    max_per_class=int(args.max_per_class),
                    max_total=int(args.max_images) if (args.max_images and int(args.max_images) > 0) else 0,
                )
            elif args.max_images and args.max_images > 0:
                image_items = sample_class_balanced_images(preset.images_dir, args.max_images)
            else:
                image_items = list(iter_classification_images(preset.images_dir))
        else:
            image_items = [(p, None) for p in list_images(preset.images_dir)]

    if (not preset.class_subfolders) and args.max_images and args.max_images > 0:
        image_items = image_items[: args.max_images]

    if not image_items:
        raise SystemExit(f"No images found under: {preset.images_dir}")

    if args.dry_run:
        location = str(preset.images_dir) if preset.images_dir is not None else str(single_image_path)
        print(f"✅ Dry run OK. Found {len(image_items)} images under {location}")
        print("First 5:")
        for p, lbl in image_items[:5]:
            print(f"  - {p} (gt_label={lbl})")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    stage2_root = _stage2_root().resolve()
    output_root_arg = str(args.output_root).strip()
    if not output_root_arg:
        output_root_arg = "outputs"

    output_root_path = Path(output_root_arg)
    if output_root_path.is_absolute():
        raise SystemExit(
            "--output-root must be a relative subdirectory under Capstone/stage_2/ "
            "(absolute paths are disabled to keep outputs tracked in one directory)."
        )

    out_root = (stage2_root / output_root_path).resolve()
    try:
        out_root.relative_to(stage2_root)
    except ValueError:
        raise SystemExit("--output-root escapes Capstone/stage_2/; choose a simpler relative path.")

    out_dir = out_root / f"florence2_unified_{preset.name}_{timestamp}"
    ensure_dir(out_dir)

    # Task-specific folders (same image stems across folders)
    cls_dir = out_dir / "classification"
    det_dir = out_dir / "detection"
    seg_dir = out_dir / "segmentation"
    ensure_dir(cls_dir)
    ensure_dir(det_dir)
    ensure_dir(seg_dir)
    ensure_dir(det_dir / "viz")
    ensure_dir(seg_dir / "viz")
    ensure_dir(seg_dir / "masks")

    # Create output files early so they exist even if the run is interrupted.
    (cls_dir / "predictions.csv").write_text("stem,image_path,gt_label,pred_label\n")
    (det_dir / "predictions.jsonl").write_text("")
    (seg_dir / "predictions.jsonl").write_text("")

    print(f"Loading Florence-2: {model_id} on {device} ...")
    model, processor = load_florence2(model_id, device)

    sam_predictor = None
    if not args.no_segmentation and args.seg_backend == "sam_box":
        sam_predictor = load_sam(
            checkpoint_path=Path(args.sam_checkpoint).expanduser().resolve(),
            model_type=str(args.sam_model_type),
            device=device,
        )

    summary = {
        "preset": preset.name,
        "images_dir": str(preset.images_dir),
        "model": model_id,
        "device": device,
        "class_names": class_names,
        "num_images": len(image_items),
        "results": [],
    }

    cls_rows: List[List[str]] = []
    det_rows: List[Dict] = []
    seg_rows: List[Dict] = []

    cls_csv_path = cls_dir / "predictions.csv"
    det_jsonl_path = det_dir / "predictions.jsonl"
    seg_jsonl_path = seg_dir / "predictions.jsonl"

    for img_path, gt_label in tqdm(image_items, desc="Florence-2 unified", unit="img"):
        image = Image.open(img_path).convert("RGB")
        stem = Path(img_path).stem
        item = {
            "image_path": str(img_path),
            "gt_label": gt_label,
            "image_size": [image.width, image.height],
        }

        wrote_cls_row = False

        # A per-image class hint that can be used to restrict detection labels.
        # Set from classification output (open_vocab or caption).
        class_hint: Optional[str] = None

        open_vocab_out = None
        open_vocab_parsed = None
        ov_bboxes_sliced: Optional[List[List[float]]] = None
        ov_labels_sliced: Optional[List[str]] = None
        ov_slice_debug: Optional[Dict] = None
        open_vocab_top_label: Optional[str] = None

        # ---- Classification (open vocab) ----
        pred_class = None
        need_open_vocab = (
            (not args.no_classification and args.classification_mode == "open_vocab")
            or (args.detection_mode in ("open_vocab", "both"))
        )
        if need_open_vocab:
            # Pass 1: user class list + extra terms
            open_vocab_out = run_task(
                model,
                processor,
                image,
                TASK_OPEN_VOCAB,
                open_vocab_prompt,
                device=device,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
            )
            open_vocab_parsed = open_vocab_out["parsed"]

            # Optional pass 2: broaden prompt if no boxes found
            if args.open_vocab_retry_generic:
                b0, _l0 = extract_open_vocab_bboxes(open_vocab_parsed)
                if not b0:
                    retry_prompt = ", ".join(_uniq_keep_order(open_vocab_terms + [normalize_label_for_prompt(x) for x in GENERIC_WASTE_TERMS.split(",")]))
                    open_vocab_out = run_task(
                        model,
                        processor,
                        image,
                        TASK_OPEN_VOCAB,
                        retry_prompt,
                        device=device,
                        max_new_tokens=args.max_new_tokens,
                        num_beams=args.num_beams,
                    )
                    open_vocab_parsed = open_vocab_out["parsed"]

            item["open_vocab"] = open_vocab_out

            # Cache a simple "best" label from open-vocab outputs for downstream phrasing.
            # This is useful even when classification-mode=caption.
            try:
                open_vocab_top_label = pick_class_from_open_vocab(open_vocab_parsed, fallback="trash")
            except Exception:
                open_vocab_top_label = None

            # Optional SAHI-style slice inference: replace box/label source with merged tile detections.
            if args.slice_inference:
                b_s, l_s, dbg = florence_open_vocab_sliced(
                    model=model,
                    processor=processor,
                    image=image,
                    prompt=open_vocab_prompt,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams,
                    tile_size=args.slice_size,
                    overlap=args.slice_overlap,
                    max_tiles=args.slice_max_tiles,
                    merge_iou=args.slice_merge_iou,
                )
                ov_bboxes_sliced, ov_labels_sliced, ov_slice_debug = b_s, l_s, dbg
                item["open_vocab_slice"] = {"debug": dbg, "num_boxes": len(b_s)}

            if not args.no_classification and args.classification_mode == "open_vocab":
                if ov_labels_sliced is not None:
                    pred_class = pick_class_from_labels(ov_labels_sliced, fallback="trash")
                else:
                    pred_class = pick_class_from_open_vocab(open_vocab_parsed, fallback="trash")
                item["pred_class"] = pred_class
                class_hint = str(pred_class) if pred_class is not None else None

                cls_rows.append([
                    stem,
                    str(img_path),
                    str(gt_label) if gt_label is not None else "",
                    str(pred_class).replace(",", " "),
                ])
                with cls_csv_path.open("a") as f:
                    safe_pred = str(pred_class).replace(",", " ")
                    f.write(f"{stem},{str(img_path)},{str(gt_label) if gt_label is not None else ''},{safe_pred}\n")
                wrote_cls_row = True

        # ---- Classification (caption) ----
        if not args.no_classification and args.classification_mode == "caption":
            out_cap = run_task(
                model,
                processor,
                image,
                TASK_CAPTION,
                None,
                device=device,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
            )
            item["caption"] = out_cap
            caption_text = _extract_caption_text(out_cap.get("parsed", {}))
            item["caption_text"] = caption_text
            if caption_text and str(caption_text).strip():
                class_hint = str(caption_text).strip()
            cls_rows.append([
                stem,
                str(img_path),
                str(gt_label) if gt_label is not None else "",
                caption_text,
            ])
            with cls_csv_path.open("a") as f:
                f.write(f"{stem},{str(img_path)},{str(gt_label) if gt_label is not None else ''},{caption_text.replace(',', ' ')}\n")
            wrote_cls_row = True

        # If classification is disabled (or produced no row), still write a placeholder row
        # so the CSV has the same number of rows as images processed.
        if not wrote_cls_row:
            cls_rows.append([
                stem,
                str(img_path),
                str(gt_label) if gt_label is not None else "",
                "",
            ])
            with cls_csv_path.open("a") as f:
                f.write(f"{stem},{str(img_path)},{str(gt_label) if gt_label is not None else ''},\n")

        # ---- Object Detection ----
        if not args.no_detection:
            if args.detection_mode in ("open_vocab", "both") and open_vocab_parsed is not None:
                if ov_bboxes_sliced is not None and ov_labels_sliced is not None:
                    bboxes_raw, labels_raw = ov_bboxes_sliced, ov_labels_sliced
                else:
                    bboxes_raw, labels_raw = extract_open_vocab_bboxes(open_vocab_parsed)

                # Optional strict label filtering (e.g. keep only 'plastic bottle')
                if args.det_keep_label and str(args.det_keep_label).strip():
                    bboxes_raw, labels_raw = filter_detections_by_label_phrase(
                        bboxes=bboxes_raw,
                        labels=labels_raw,
                        phrase=str(args.det_keep_label),
                    )

                if args.det_single_class_from_classification and class_hint and str(class_hint).strip():
                    # Keep only boxes that match the per-image predicted class and force the output label to that class.
                    bboxes_raw, labels_raw = filter_detections_by_label_phrase(
                        bboxes=bboxes_raw,
                        labels=labels_raw,
                        phrase=str(class_hint),
                    )
                    labels_raw = override_all_labels(labels_raw, str(class_hint))

                if args.det_drop_generic_labels:
                    bboxes_raw, labels_raw = drop_generic_detections_if_specific_exist(
                        bboxes=bboxes_raw,
                        labels=labels_raw,
                    )
                det_phrase = None
                if args.det_phrase and str(args.det_phrase).strip():
                    det_phrase = normalize_label_for_prompt(str(args.det_phrase))
                elif args.seg_phrase and str(args.seg_phrase).strip():
                    det_phrase = normalize_label_for_prompt(str(args.seg_phrase))
                elif pred_class:
                    det_phrase = normalize_label_for_prompt(str(pred_class))
                bboxes, labels = _filter_bboxes(
                    bboxes=bboxes_raw,
                    labels=labels_raw,
                    image_size=(image.width, image.height),
                    select=args.det_select,
                    max_boxes=args.det_max_boxes,
                    min_area_frac=args.det_min_area_frac,
                    phrase=det_phrase,
                )
                # If filtering removed everything, retry with a smaller min-area threshold.
                if not bboxes and float(args.det_fallback_min_area_frac) < float(args.det_min_area_frac):
                    bboxes, labels = _filter_bboxes(
                        bboxes=bboxes_raw,
                        labels=labels_raw,
                        image_size=(image.width, image.height),
                        select=args.det_select,
                        max_boxes=args.det_max_boxes,
                        min_area_frac=float(args.det_fallback_min_area_frac),
                        phrase=det_phrase,
                    )

                if bboxes and float(args.bbox_dedupe_iou) > 0:
                    bboxes, labels = _dedupe_bboxes(
                        bboxes=bboxes,
                        labels=labels,
                        image_size=(image.width, image.height),
                        iou_thr=float(args.bbox_dedupe_iou),
                    )
                # Always save a viz image so detection/viz has the same number of images as inputs.
                viz_od = draw_bboxes(image, bboxes, labels) if bboxes else image.convert("RGB")
                viz_od.save(det_dir / "viz" / f"{stem}_ovd.jpg")

                det_rows.append(
                    {
                        "stem": stem,
                        "image_path": str(img_path),
                        "mode": "open_vocab",
                        "bboxes": bboxes,
                        "labels": labels,
                    }
                )
                with det_jsonl_path.open("a") as f:
                    f.write(json.dumps(det_rows[-1]) + "\n")

            if args.detection_mode in ("od", "both"):
                out_od = run_task(
                    model,
                    processor,
                    image,
                    TASK_OD,
                    None,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams,
                )
                item["od"] = out_od
                od_payload = out_od["parsed"].get(TASK_OD, {})
                bboxes = od_payload.get("bboxes", [])
                labels = od_payload.get("labels", ["object"] * len(bboxes))
                # Always save a viz image so detection/viz has the same number of images as inputs.
                viz_od = draw_bboxes(image, bboxes, labels) if bboxes else image.convert("RGB")
                viz_od.save(det_dir / "viz" / f"{stem}_od.jpg")

                det_rows.append(
                    {
                        "stem": stem,
                        "image_path": str(img_path),
                        "mode": "od",
                        "bboxes": bboxes,
                        "labels": labels,
                    }
                )
                with det_jsonl_path.open("a") as f:
                    f.write(json.dumps(det_rows[-1]) + "\n")

        # ---- Segmentation (prompted) ----
        if not args.no_segmentation:
            if args.seg_phrase and str(args.seg_phrase).strip():
                primary_phrase = normalize_label_for_prompt(str(args.seg_phrase))
            else:
                # Prefer an actual detected label over a generic word like "trash".
                if pred_class:
                    primary_phrase = normalize_label_for_prompt(pred_class)
                elif class_hint and str(class_hint).strip():
                    primary_phrase = normalize_label_for_prompt(str(class_hint))
                elif open_vocab_top_label:
                    primary_phrase = normalize_label_for_prompt(open_vocab_top_label)
                else:
                    primary_phrase = "trash"

            extra_phrases = [normalize_label_for_prompt(p) for p in str(args.seg_try_phrases).split(",") if p.strip()]
            phrase_candidates = _uniq_keep_order([primary_phrase] + extra_phrases)
            if not phrase_candidates:
                phrase_candidates = ["trash"]

            polygons: List = []
            merged = np.zeros((image.height, image.width), dtype=np.uint8)
            used_phrase = phrase_candidates[0]

            if args.seg_backend == "sam_box":
                if open_vocab_parsed is None:
                    raise SystemExit("--seg-backend sam_box requires open-vocab detection to be enabled")
                bboxes_raw, labels_raw = extract_open_vocab_bboxes(open_vocab_parsed)
                bboxes_f, labels_f = _filter_bboxes(
                    bboxes=bboxes_raw,
                    labels=labels_raw,
                    image_size=(image.width, image.height),
                    select=args.det_select,
                    max_boxes=args.det_max_boxes,
                    min_area_frac=args.det_min_area_frac,
                    phrase=primary_phrase,
                )
                if not bboxes_f and float(args.det_fallback_min_area_frac) < float(args.det_min_area_frac):
                    bboxes_f, labels_f = _filter_bboxes(
                        bboxes=bboxes_raw,
                        labels=labels_raw,
                        image_size=(image.width, image.height),
                        select=args.det_select,
                        max_boxes=args.det_max_boxes,
                        min_area_frac=float(args.det_fallback_min_area_frac),
                        phrase=primary_phrase,
                    )
                if bboxes_f and float(args.bbox_dedupe_iou) > 0:
                    bboxes_f, labels_f = _dedupe_bboxes(
                        bboxes=bboxes_f,
                        labels=labels_f,
                        image_size=(image.width, image.height),
                        iou_thr=float(args.bbox_dedupe_iou),
                    )
                best_xyxy = _choose_best_bbox_for_phrase(
                    bboxes=bboxes_f,
                    labels=labels_f,
                    phrase=primary_phrase,
                    image_size=(image.width, image.height),
                )
                if best_xyxy is not None and sam_predictor is not None:
                    img_np = np.array(image.convert("RGB"))
                    merged = sam_mask_from_box(sam_predictor, img_np, best_xyxy)
            else:
                seg_image = image
                crop_rect = None
                have_detection_crop = False

                # If enabled, pick a crop from open-vocab detection once (we reuse it across phrase attempts).
                if (
                    args.seg_from_detection == "open_vocab"
                    and open_vocab_parsed is not None
                    and args.detection_mode in ("open_vocab", "both")
                ):
                    if ov_bboxes_sliced is not None and ov_labels_sliced is not None:
                        bboxes_raw, labels_raw = ov_bboxes_sliced, ov_labels_sliced
                    else:
                        bboxes_raw, labels_raw = extract_open_vocab_bboxes(open_vocab_parsed)
                    # Use primary phrase for bbox selection.
                    bboxes_f, labels_f = _filter_bboxes(
                        bboxes=bboxes_raw,
                        labels=labels_raw,
                        image_size=(image.width, image.height),
                        select=args.det_select,
                        max_boxes=args.det_max_boxes,
                        min_area_frac=args.det_min_area_frac,
                        phrase=primary_phrase,
                    )
                    if not bboxes_f and float(args.det_fallback_min_area_frac) < float(args.det_min_area_frac):
                        bboxes_f, labels_f = _filter_bboxes(
                            bboxes=bboxes_raw,
                            labels=labels_raw,
                            image_size=(image.width, image.height),
                            select=args.det_select,
                            max_boxes=args.det_max_boxes,
                            min_area_frac=float(args.det_fallback_min_area_frac),
                            phrase=primary_phrase,
                        )
                    if bboxes_f and float(args.bbox_dedupe_iou) > 0:
                        bboxes_f, labels_f = _dedupe_bboxes(
                            bboxes=bboxes_f,
                            labels=labels_f,
                            image_size=(image.width, image.height),
                            iou_thr=float(args.bbox_dedupe_iou),
                        )
                    best_xyxy = _choose_best_bbox_for_phrase(
                        bboxes=bboxes_f,
                        labels=labels_f,
                        phrase=primary_phrase,
                        image_size=(image.width, image.height),
                    )
                    if best_xyxy is not None:
                        seg_image, crop_rect = _crop_with_padding(image, best_xyxy, pad_frac=float(args.seg_crop_pad))
                        have_detection_crop = True

                # If user requested segmentation-from-detection but we couldn't find a detection bbox,
                # do not run referring-expression segmentation on the full image (it often segments background).
                if args.seg_from_detection == "open_vocab" and open_vocab_parsed is not None and not have_detection_crop:
                    item["seg_attempts"] = [{"phrase": primary_phrase, "has_polygons": False, "skipped": "no_detection_bbox"}]
                    # merged stays empty; we'll still write empty viz/mask later.
                    polygons = []
                else:

                    # Try multiple phrases to improve chances of getting polygons.
                    seg_attempts: List[Dict] = []
                    for phrase in phrase_candidates:
                        used_phrase = phrase
                        out_seg = run_task(
                            model,
                            processor,
                            seg_image,
                            TASK_SEG_REF,
                            phrase,
                            device=device,
                            max_new_tokens=args.max_new_tokens,
                            num_beams=args.num_beams,
                        )
                        seg_payload = out_seg["parsed"].get(TASK_SEG_REF, {})
                        polys = seg_payload.get("polygons", [])
                        seg_attempts.append({"phrase": phrase, "has_polygons": bool(polys)})

                        if polys:
                            # Convert polygons into a mask, mapping back from crop if needed.
                            if crop_rect is None:
                                tmp = np.zeros((image.height, image.width), dtype=np.uint8)
                                for poly_group in polys:
                                    tmp = np.maximum(tmp, polygons_to_mask((image.width, image.height), poly_group))
                            else:
                                cx0, cy0, cx1, cy1 = crop_rect
                                crop_w = max(1, cx1 - cx0)
                                crop_h = max(1, cy1 - cy0)
                                crop_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
                                for poly_group in polys:
                                    crop_mask = np.maximum(crop_mask, polygons_to_mask((crop_w, crop_h), poly_group))
                                tmp = np.zeros((image.height, image.width), dtype=np.uint8)
                                tmp[cy0:cy1, cx0:cx1] = crop_mask

                            if args.seg_merge_mode == "union":
                                merged = np.maximum(merged, tmp)
                                polygons = polys
                                item["seg"] = out_seg
                            else:
                                merged = tmp
                                polygons = polys
                                item["seg"] = out_seg
                                break

                    item["seg_attempts"] = seg_attempts

            # Always save viz + mask so segmentation outputs align 1:1 with input images.
            if np.any(merged > 0):
                viz_seg = overlay_mask(image, merged)
                mask_u8 = (merged > 0).astype(np.uint8) * 255
            else:
                viz_seg = image.convert("RGB")
                mask_u8 = np.zeros((image.height, image.width), dtype=np.uint8)

            viz_seg.save(seg_dir / "viz" / f"{stem}_seg.jpg")
            Image.fromarray(mask_u8).save(seg_dir / "masks" / f"{stem}.png")

            seg_rows.append(
                {
                    "stem": stem,
                    "image_path": str(img_path),
                    "phrase": used_phrase,
                    "seg_backend": args.seg_backend,
                    "has_polygons": bool(polygons),
                }
            )
            with seg_jsonl_path.open("a") as f:
                f.write(json.dumps(seg_rows[-1]) + "\n")

        summary["results"].append(item)

    (out_dir / "results.json").write_text(json.dumps(summary, indent=2))

    # Files are written incrementally; keep the aggregated summary only.

    print(f"✅ Done. Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
