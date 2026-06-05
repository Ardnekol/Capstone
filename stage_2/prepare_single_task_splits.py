#!/usr/bin/env python3
"""Create deterministic 90/10 train/val splits for the per-task JSONLs.

The unified pipeline already has a val split for TrashNet captions, but the
TACO OD and TACO segmentation JSONLs only have train files. E1 (single-task
ablation) needs eval signals for each single-task run, so we produce:

  finetune_data/taco_od_train_split.jsonl
  finetune_data/taco_od_val_split.jsonl
  finetune_data/taco_seg_train_split.jsonl
  finetune_data/taco_seg_val_split.jsonl

These mirror the 90/10 split used for the unified run. Seed is fixed so
re-running produces the same split.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

STAGE2 = Path(__file__).resolve().parent
DATA = STAGE2 / "finetune_data"

SPLITS = [
    ("taco_od_train.jsonl", "taco_od_train_split.jsonl", "taco_od_val_split.jsonl"),
    ("taco_seg_train.jsonl", "taco_seg_train_split.jsonl", "taco_seg_val_split.jsonl"),
]

VAL_FRACTION = 0.10
SEED = 1337


def split_one(src: Path, train_out: Path, val_out: Path) -> None:
    if not src.exists():
        raise SystemExit(f"Source JSONL missing: {src}")
    lines = [ln for ln in src.read_text().splitlines() if ln.strip()]
    rng = random.Random(SEED)
    rng.shuffle(lines)
    n_val = max(1, int(round(len(lines) * VAL_FRACTION)))
    val_lines = lines[:n_val]
    train_lines = lines[n_val:]
    train_out.write_text("\n".join(train_lines) + "\n")
    val_out.write_text("\n".join(val_lines) + "\n")
    print(f"  {src.name}: {len(lines)} total -> {len(train_lines)} train, {len(val_lines)} val")


def main() -> None:
    print(f"Creating 90/10 splits (seed={SEED}) in {DATA}")
    for src_name, train_name, val_name in SPLITS:
        split_one(DATA / src_name, DATA / train_name, DATA / val_name)
    print("Done.")


if __name__ == "__main__":
    main()
