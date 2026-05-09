#!/usr/bin/env python3
"""Prepare a JSONL dataset for fine-tuning Florence-2 on TrashNet as "caption-as-class".

Why this exists:
- TrashNet is a folder-of-classes classification dataset (no bounding boxes).
- We can still LoRA fine-tune Florence-2 to output the correct class name by
  training the <CAPTION> task to generate the class label.

Output JSONL format (one record per image):
  {"image_path": "/abs/path.jpg", "prefix": "<CAPTION>", "suffix": "plastic"}

By default, it splits into train/val JSONL under Capstone/stage_2/finetune_data.

Note: This fine-tunes classification-like behavior; it does NOT create detection
bounding boxes (TrashNet has no boxes).
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _stage2_root() -> Path:
    return _repo_root() / "stage_2"


def _resolve_path_arg(path_str: str) -> Path:
    p = Path(path_str).expanduser()
    if p.is_absolute():
        return p.resolve()

    capstone = _repo_root().resolve()
    workspace = capstone.parent
    candidates = [Path.cwd() / p, capstone / p, workspace / p]
    if len(p.parts) > 0 and p.parts[0] == "Capstone":
        candidates.insert(0, workspace / p)

    for c in candidates:
        if c.exists():
            return c.resolve()

    return candidates[0].resolve()


def list_images(images_dir: Path) -> List[Path]:
    exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
    if not images_dir.exists():
        return []
    paths: List[Path] = []
    for p in images_dir.rglob("*"):
        if p.is_file() and p.suffix in exts:
            paths.append(p)
    return sorted(paths)


def iter_trashnet_items(root: Path) -> List[Tuple[Path, str]]:
    items: List[Tuple[Path, str]] = []
    for class_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        label = class_dir.name.strip()
        if not label:
            continue
        for img_path in list_images(class_dir):
            items.append((img_path, label))
    return items


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare TrashNet caption-as-label JSONL for Florence-2 fine-tuning")
    ap.add_argument(
        "--images-root",
        type=str,
        default=str(_repo_root() / "datasets/classification/trashnet/dataset-preprocessed"),
        help="TrashNet root directory containing class subfolders",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=str(_stage2_root() / "finetune_data"),
        help="Output directory for JSONL files (must be under Capstone/stage_2)",
    )
    ap.add_argument("--train-name", type=str, default="trashnet_caption_train.jsonl")
    ap.add_argument("--val-name", type=str, default="trashnet_caption_val.jsonl")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--label-template",
        type=str,
        default="{label}",
        help="Suffix template. Use '{label}' placeholder. Examples: '{label}' or 'a photo of {label}'.",
    )

    args = ap.parse_args()

    images_root = _resolve_path_arg(args.images_root)
    out_dir = _resolve_path_arg(args.out_dir)

    stage2 = _stage2_root().resolve()
    try:
        out_dir.resolve().relative_to(stage2)
    except Exception:
        raise SystemExit(f"--out-dir must be inside {stage2} (got {out_dir})")

    items = iter_trashnet_items(images_root)
    if not items:
        raise SystemExit(f"No images found under: {images_root}")

    random.seed(int(args.seed))
    random.shuffle(items)

    val_ratio = float(args.val_ratio)
    val_ratio = max(0.0, min(0.9, val_ratio))
    n_val = int(round(len(items) * val_ratio))
    val_items = items[:n_val]
    train_items = items[n_val:]

    ensure_dir(out_dir)

    def _write(path: Path, split_items: List[Tuple[Path, str]]) -> None:
        lines: List[str] = []
        for img_path, label in split_items:
            suffix = str(args.label_template).format(label=label)
            rec: Dict[str, str] = {
                "image_path": str(Path(img_path).resolve()),
                "prefix": "<CAPTION>",
                "suffix": str(suffix).strip(),
            }
            lines.append(json.dumps(rec, ensure_ascii=False))
        path.write_text("\n".join(lines) + "\n")

    train_path = (out_dir / str(args.train_name)).resolve()
    val_path = (out_dir / str(args.val_name)).resolve()

    _write(train_path, train_items)
    _write(val_path, val_items)

    print("✅ Wrote TrashNet caption-as-label JSONL")
    print(f"- Train: {train_path} ({len(train_items)} images)")
    print(f"- Val:   {val_path} ({len(val_items)} images)")


if __name__ == "__main__":
    main()
