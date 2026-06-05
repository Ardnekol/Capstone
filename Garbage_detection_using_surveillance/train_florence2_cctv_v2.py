#!/usr/bin/env python3
"""Florence-2 LoRA fine-tuning v2 for CCTV garbage detection.

Improvements over the Stage-2 baseline trainer:
  1. Accepts v2 JSONL with `kind: "neg"` records (empty suffix, optional crop_xyxy)
     so the model learns 'no garbage' on background crops.
  2. Color/brightness augmentation on positive images (CCTV-relevant; no geometric
     transform so loc tokens stay valid).
  3. Optional partial unfreeze of the DaViT vision encoder's last N blocks —
     LoRA on the language head alone cannot fix bad visual features for OOD CCTV.
  4. Expanded LoRA target modules (FFN projections in addition to attention).
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoProcessor


# ---------- HF cache plumbing (mirrors Stage 2) ----------

def configure_hf_cache(cache_dir: Path) -> None:
    cache_dir = cache_dir.expanduser().resolve()
    (cache_dir / "hub").mkdir(parents=True, exist_ok=True)
    (cache_dir / "modules").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_dir / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir / "hub"))
    os.environ.setdefault("HF_MODULES_CACHE", str(cache_dir / "modules"))


def default_cache_dir() -> Path:
    for key in ("SLURM_TMPDIR", "TMPDIR", "TMP"):
        val = os.environ.get(key)
        if val:
            return Path(val) / f"hf_cache_uid{os.getuid()}"
    return Path("/tmp") / f"hf_cache_uid{os.getuid()}"


# ---------- Dataset with negative + augmentation support ----------

class CctvV2Dataset(Dataset):
    def __init__(self, jsonl_path: Path, augment: bool = True, seed: int = 42):
        self.augment = augment
        self.rng = random.Random(seed)
        self.records: List[Dict[str, Any]] = []
        for ln in jsonl_path.read_text().splitlines():
            ln = ln.strip()
            if ln:
                self.records.append(json.loads(ln))
        if not self.records:
            raise ValueError(f"No records in {jsonl_path}")
        n_pos = sum(1 for r in self.records if r.get("kind", "pos") == "pos")
        n_neg = len(self.records) - n_pos
        print(f"[Dataset] {jsonl_path.name}: pos={n_pos} neg={n_neg} total={len(self.records)}")

    def __len__(self) -> int:
        return len(self.records)

    def _augment_image(self, img: Image.Image) -> Image.Image:
        # Color/brightness jitter — no geometric ops so loc tokens stay valid
        if self.rng.random() < 0.7:
            img = ImageEnhance.Brightness(img).enhance(self.rng.uniform(0.7, 1.3))
        if self.rng.random() < 0.5:
            img = ImageEnhance.Contrast(img).enhance(self.rng.uniform(0.8, 1.2))
        if self.rng.random() < 0.5:
            img = ImageEnhance.Color(img).enhance(self.rng.uniform(0.7, 1.3))
        return img

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.records[idx]
        kind = r.get("kind", "pos")
        img_path = Path(r["image_path"]).expanduser()
        img = Image.open(img_path).convert("RGB")

        if kind == "neg" and "crop_xyxy" in r:
            img = img.crop(tuple(r["crop_xyxy"]))

        if self.augment and kind == "pos":
            img = self._augment_image(img)

        prefix = str(r.get("prefix", "<OD>")).strip() or "<OD>"
        suffix = str(r.get("suffix", ""))   # may be "" for negatives
        return {"image": img, "prefix": prefix, "suffix": suffix, "kind": kind}


# ---------- Collator (mirrors Stage 2 with empty-suffix tolerance) ----------

@dataclass
class V2Collator:
    processor: Any
    max_length: int

    def __post_init__(self) -> None:
        try:
            sig = inspect.signature(self.processor.__call__)
            self._supports_suffix = "suffix" in sig.parameters
        except Exception:
            self._supports_suffix = False
        self._tok = getattr(self.processor, "tokenizer", None)
        self._img_proc = getattr(self.processor, "image_processor", None)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        images   = [b["image"]  for b in batch]
        prefixes = [b["prefix"] for b in batch]
        # For neg records we still need a non-empty target for the EOS-only path
        # to provide *some* loss signal. Use the processor's suffix path if supported.
        suffixes = [b["suffix"] if b["suffix"] else "" for b in batch]

        if self._supports_suffix and all(s for s in suffixes):
            enc = self.processor(
                text=prefixes, images=images, suffix=suffixes,
                return_tensors="pt", padding=True, truncation=True,
                max_length=self.max_length,
            )
            if "labels" not in enc:
                raise ValueError("Processor did not return labels with suffix path.")
            return {
                "input_ids":      enc["input_ids"],
                "attention_mask": enc.get("attention_mask"),
                "pixel_values":   enc["pixel_values"],
                "labels":         enc["labels"],
            }

        # Manual path (handles empty suffixes; required for neg records).
        if self._tok is None or self._img_proc is None:
            raise ValueError("Processor missing tokenizer/image_processor")

        pixel_values = self._img_proc(images, return_tensors="pt")["pixel_values"]
        prefix_enc = self._tok(prefixes, padding=False, truncation=True,
                               max_length=self.max_length)
        prefix_ids_list: List[List[int]] = [list(x) for x in prefix_enc["input_ids"]]

        eos = self._tok.eos_token_id
        suffix_ids_list: List[List[int]] = []
        for s in suffixes:
            if s:
                ids = list(self._tok(s, add_special_tokens=False)["input_ids"])
                ids = ids[: self.max_length - 1]
                if eos is not None:
                    ids = ids + [eos]
            else:
                # Negative record: target is just EOS — tells the decoder "stop, no boxes"
                ids = [eos] if eos is not None else [0]
            suffix_ids_list.append(ids)

        pad_id = self._tok.pad_token_id if self._tok.pad_token_id is not None else 0

        def _pad(seq, length, value):
            return seq + [value] * (length - len(seq)) if len(seq) < length else seq[:length]

        max_pre = max(len(x) for x in prefix_ids_list)
        max_suf = max(len(x) for x in suffix_ids_list)
        input_ids      = torch.tensor([_pad(x, max_pre, pad_id) for x in prefix_ids_list], dtype=torch.long)
        attention_mask = (input_ids != pad_id).long()
        labels         = torch.tensor([_pad(x, max_suf, -100)   for x in suffix_ids_list], dtype=torch.long)

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "pixel_values":   pixel_values,
            "labels":         labels,
        }


# ---------- LoRA + vision-block unfreezing ----------

EXPANDED_LORA_TARGETS = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "out_proj", "fc1", "fc2",
    "gate_proj", "up_proj", "down_proj",
)


def _discover_lora_targets(model: torch.nn.Module) -> List[str]:
    found = set()
    for name, _ in model.named_modules():
        last = name.split(".")[-1]
        if last in EXPANDED_LORA_TARGETS:
            found.add(last)
    if not found:
        found = {"q_proj", "k_proj", "v_proj", "o_proj"}
    return sorted(found)


_VISION_RE = re.compile(r"(vision|visual|davit|image_encoder|patch_embed)", re.IGNORECASE)
_BLOCK_IDX_RE = re.compile(r"(?:blocks?|layers?)\.(\d+)\.")


def unfreeze_vision_last_blocks(model: torch.nn.Module, n_blocks: int) -> Tuple[int, int]:
    """Set requires_grad=True on the last `n_blocks` vision-encoder blocks.

    Heuristic: among params whose names match _VISION_RE, parse out the
    'blocks.<idx>' / 'layers.<idx>' integer, and unfreeze params whose idx is
    within the top-N highest indices observed.

    Returns (num_params_unfrozen, num_modules_touched).
    """
    if n_blocks <= 0:
        return (0, 0)

    vision_params: List[Tuple[str, int, torch.nn.Parameter]] = []
    no_idx_params: List[Tuple[str, torch.nn.Parameter]] = []
    for name, p in model.named_parameters():
        if not _VISION_RE.search(name):
            continue
        m = _BLOCK_IDX_RE.search(name)
        if m:
            vision_params.append((name, int(m.group(1)), p))
        else:
            no_idx_params.append((name, p))

    if not vision_params:
        print("[unfreeze] WARNING: no vision params with block index found; nothing unfrozen.")
        return (0, 0)

    max_idx = max(idx for _, idx, _ in vision_params)
    threshold = max_idx - (n_blocks - 1)
    n_params = 0
    for name, idx, p in vision_params:
        if idx >= threshold:
            p.requires_grad_(True)
            n_params += p.numel()
    # Also unfreeze post-block norms / heads that lack an index
    for name, p in no_idx_params:
        if any(k in name for k in ("norm", "head", "proj")):
            p.requires_grad_(True)
            n_params += p.numel()
    print(f"[unfreeze] Unfroze vision blocks idx>={threshold} (max={max_idx}); "
          f"{n_params:,} extra trainable params")
    return (n_params, len(vision_params))


# ---------- Main ----------

def main() -> None:
    ap = argparse.ArgumentParser(description="Florence-2 CCTV LoRA fine-tune v2")
    ap.add_argument("--model-id",   default="microsoft/Florence-2-large")
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--eval-jsonl",  default=None)
    ap.add_argument("--output-dir",  required=True)
    ap.add_argument("--cache-dir",   default=None)

    ap.add_argument("--device",      default="cuda:0")
    ap.add_argument("--max-length",  type=int, default=1024)
    ap.add_argument("--per-device-train-batch-size", type=int, default=2)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=5e-5)
    ap.add_argument("--num-train-epochs", type=float, default=40.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--lr-scheduler-type", default="cosine")

    ap.add_argument("--logging-steps", type=int, default=20)
    ap.add_argument("--save-steps",    type=int, default=500)
    ap.add_argument("--save-total-limit", type=int, default=2)

    ap.add_argument("--lora-r",       type=int, default=32)
    ap.add_argument("--lora-alpha",   type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--unfreeze-vision-blocks", type=int, default=1,
                    help="Unfreeze the last N DaViT blocks in addition to LoRA")

    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--resume-from-checkpoint", default=None)

    args = ap.parse_args()

    # Cache
    cache_dir = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else default_cache_dir()
    configure_hf_cache(cache_dir)

    # Import Trainer lazily — same defensive message as Stage 2
    try:
        from transformers import Trainer, TrainingArguments
    except Exception as e:
        raise SystemExit(
            "Failed to import transformers Trainer. Likely a transformers/peft/torch version mismatch.\n"
            "Recommended pins on torch 2.0.1:\n"
            "  pip install --user -U 'transformers==4.40.2' 'tokenizers==0.19.1' "
            "'huggingface_hub==0.23.4' 'peft==0.10.0' accelerate\n"
            f"Original error: {type(e).__name__}: {e}"
        )

    train_path = Path(args.train_jsonl).expanduser().resolve()
    if not train_path.exists():
        raise SystemExit(f"train JSONL not found: {train_path}")
    eval_path = Path(args.eval_jsonl).expanduser().resolve() if args.eval_jsonl else None
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    print(f"Loading processor/model: {args.model_id}  device={device}")
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, trust_remote_code=True,
        torch_dtype=torch.float32, attn_implementation="eager",
    )
    if hasattr(model, "config") and hasattr(model.config, "attn_implementation"):
        try:
            model.config.attn_implementation = "eager"
        except Exception:
            pass

    # LoRA
    from peft import LoraConfig, TaskType, get_peft_model  # noqa: WPS433
    targets = _discover_lora_targets(model)
    print(f"[LoRA] r={args.lora_r} alpha={args.lora_alpha} targets={targets}")
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        target_modules=targets,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Partial vision unfreeze (on top of LoRA)
    unfreeze_vision_last_blocks(model, int(args.unfreeze_vision_blocks))

    # Datasets + collator
    train_ds = CctvV2Dataset(train_path, augment=not args.no_augment)
    eval_ds  = CctvV2Dataset(eval_path,  augment=False) if eval_path else None
    collator = V2Collator(processor=processor, max_length=int(args.max_length))

    model.train()

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=int(args.per_device_train_batch_size),
        gradient_accumulation_steps=int(args.gradient_accumulation_steps),
        learning_rate=float(args.learning_rate),
        num_train_epochs=float(args.num_train_epochs),
        max_steps=int(args.max_steps),
        warmup_ratio=float(args.warmup_ratio),
        weight_decay=float(args.weight_decay),
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=int(args.logging_steps),
        save_steps=int(args.save_steps),
        save_total_limit=int(args.save_total_limit),
        remove_unused_columns=False,
        report_to=[],
        bf16=bool(args.bf16),
        fp16=bool(args.fp16),
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=collator,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(out_dir))
    try:
        processor.save_pretrained(str(out_dir))
    except Exception:
        pass
    print(f"Done. Adapter saved to: {out_dir}")


if __name__ == "__main__":
    main()
