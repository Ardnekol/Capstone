#!/usr/bin/env python3
"""Combine multiple Florence-2 task-specific JSONLs into one unified multi-task JSONL.

Merges OD, caption (classification), and segmentation JSONLs.
Supports optional task balancing via oversampling/undersampling.

Output: a single shuffled JSONL with mixed task prefixes.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _stage2_root() -> Path:
    return _repo_root() / "stage_2"


def _resolve_path_arg(path_str: Optional[str]) -> Optional[Path]:
    if path_str is None:
        return None
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


def _load_jsonl(path: Path) -> List[Dict]:
    records = []
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        records.append(json.loads(ln))
    return records


def _task_label(prefix: str) -> str:
    """Return a short name for the task based on prefix."""
    if "<OD>" in prefix:
        return "od"
    elif "<CAPTION>" in prefix:
        return "caption"
    elif "<REFERRING_EXPRESSION_SEGMENTATION>" in prefix:
        return "seg"
    return "unknown"


def main() -> None:
    ft_dir = _stage2_root() / "finetune_data"

    ap = argparse.ArgumentParser(
        description="Combine task-specific JSONLs into unified multi-task JSONL"
    )
    ap.add_argument(
        "--od-jsonl", type=str,
        default=str(ft_dir / "taco_od_train.jsonl"),
        help="Object detection JSONL",
    )
    ap.add_argument(
        "--caption-jsonl", type=str,
        default=str(ft_dir / "trashnet_caption_train.jsonl"),
        help="Caption/classification JSONL",
    )
    ap.add_argument(
        "--seg-jsonl", type=str,
        default=str(ft_dir / "taco_seg_train.jsonl"),
        help="Segmentation JSONL",
    )
    ap.add_argument(
        "--caption-val-jsonl", type=str,
        default=str(ft_dir / "trashnet_caption_val.jsonl"),
        help="Caption validation JSONL (used for unified val split)",
    )
    ap.add_argument(
        "--out-train", type=str,
        default=str(ft_dir / "unified_multitask_train.jsonl"),
        help="Output unified training JSONL",
    )
    ap.add_argument(
        "--out-val", type=str,
        default=str(ft_dir / "unified_multitask_val.jsonl"),
        help="Output unified validation JSONL",
    )
    ap.add_argument(
        "--val-ratio", type=float, default=0.1,
        help="Fraction of OD and seg data to hold out for validation",
    )
    ap.add_argument(
        "--balance", choices=["none", "oversample", "undersample"], default="none",
        help="Task balancing strategy: none (keep as-is), "
             "oversample (repeat minority tasks to match majority), "
             "undersample (cap majority tasks to match minority)",
    )
    ap.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")

    args = ap.parse_args()
    rng = random.Random(args.seed)

    # Load all JSONLs
    od_path = _resolve_path_arg(args.od_jsonl)
    caption_path = _resolve_path_arg(args.caption_jsonl)
    seg_path = _resolve_path_arg(args.seg_jsonl)
    caption_val_path = _resolve_path_arg(args.caption_val_jsonl)

    task_records: Dict[str, List[Dict]] = {}
    task_val_records: Dict[str, List[Dict]] = {}

    for name, path in [("od", od_path), ("caption", caption_path), ("seg", seg_path)]:
        if path is None or not path.exists():
            print(f"  [SKIP] {name}: {path} not found")
            continue
        records = _load_jsonl(path)
        print(f"  [LOAD] {name}: {len(records)} records from {path}")

        # Split into train/val
        rng.shuffle(records)
        val_count = int(len(records) * args.val_ratio)
        task_val_records[name] = records[:val_count]
        task_records[name] = records[val_count:]

    # For caption, use the dedicated val file if available
    if caption_val_path and caption_val_path.exists():
        caption_val = _load_jsonl(caption_val_path)
        print(f"  [LOAD] caption val: {len(caption_val)} records from {caption_val_path}")
        task_val_records["caption"] = caption_val
        # Use full caption train (no split needed since we have dedicated val)
        if caption_path and caption_path.exists():
            task_records["caption"] = _load_jsonl(caption_path)

    if not task_records:
        raise SystemExit("No task records loaded. Check your JSONL paths.")

    # Print task counts
    print("\nTask counts (train):")
    for name, records in sorted(task_records.items()):
        print(f"  {name}: {len(records)}")

    # Apply balancing
    train_records: List[Dict] = []
    if args.balance == "none":
        for records in task_records.values():
            train_records.extend(records)
    elif args.balance == "oversample":
        max_count = max(len(r) for r in task_records.values())
        for name, records in task_records.items():
            if len(records) >= max_count:
                train_records.extend(records)
            else:
                # Repeat records to reach max_count
                repeats = max_count // len(records)
                remainder = max_count % len(records)
                oversampled = records * repeats + rng.sample(records, remainder)
                train_records.extend(oversampled)
                print(f"  [OVERSAMPLE] {name}: {len(records)} -> {len(oversampled)}")
    elif args.balance == "undersample":
        min_count = min(len(r) for r in task_records.values())
        for name, records in task_records.items():
            if len(records) <= min_count:
                train_records.extend(records)
            else:
                sampled = rng.sample(records, min_count)
                train_records.extend(sampled)
                print(f"  [UNDERSAMPLE] {name}: {len(records)} -> {len(sampled)}")

    # Combine val records
    val_records: List[Dict] = []
    for records in task_val_records.values():
        val_records.extend(records)

    # Shuffle
    rng.shuffle(train_records)
    rng.shuffle(val_records)

    # Write
    out_train = _resolve_path_arg(args.out_train)
    out_val = _resolve_path_arg(args.out_val)
    assert out_train is not None and out_val is not None

    out_train.parent.mkdir(parents=True, exist_ok=True)
    out_val.parent.mkdir(parents=True, exist_ok=True)

    out_train.write_text(
        "\n".join(json.dumps(r) for r in train_records) + ("\n" if train_records else "")
    )
    out_val.write_text(
        "\n".join(json.dumps(r) for r in val_records) + ("\n" if val_records else "")
    )

    # Summary
    print(f"\nUnified train: {len(train_records)} records -> {out_train}")
    print(f"Unified val:   {len(val_records)} records -> {out_val}")

    # Task breakdown in final train
    from collections import Counter
    task_counts = Counter(_task_label(r.get("prefix", "")) for r in train_records)
    print("\nFinal train task breakdown:")
    for task, count in sorted(task_counts.items()):
        print(f"  {task}: {count} ({100*count/len(train_records):.1f}%)")


if __name__ == "__main__":
    main()
