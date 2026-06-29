#!/usr/bin/env python3
"""Fine-tune Florence-2 for object detection (OD) on a JSONL dataset (e.g., TACO).

This script keeps the "one model" constraint: it fine-tunes Florence-2 itself.

Expected JSONL lines:
  {"image_path": "/abs/path.jpg", "prefix": "<OD>", "suffix": "label<loc_..><loc_..><loc_..><loc_..>..."}

By default it attempts a LoRA fine-tune via `peft`. If `peft` isn't installed,
use `--no-lora` for a full fine-tune (much heavier), or install peft.

Notes:
- Uses `trust_remote_code=True` to load Florence-2.
- Sets Hugging Face cache under /tmp by default to avoid $HOME quota issues.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoProcessor


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _stage2_root() -> Path:
    return _repo_root() / "stage_2"


def _ensure_stage2_on_syspath() -> None:
    stage2 = str(_stage2_root().resolve())
    if stage2 not in sys.path:
        sys.path.insert(0, stage2)


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


class JsonlODDataset(Dataset):
    def __init__(self, jsonl_path: Path):
        self.jsonl_path = jsonl_path
        self.records: List[Dict[str, Any]] = []
        for ln in jsonl_path.read_text().splitlines():
            ln = ln.strip()
            if not ln:
                continue
            self.records.append(json.loads(ln))
        if not self.records:
            raise ValueError(f"No records in {jsonl_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.records[idx]
        image_path = Path(r["image_path"]).expanduser()
        prefix = str(r.get("prefix", "")).strip()
        suffix = str(r.get("suffix", "")).strip()
        if not prefix:
            raise ValueError("Missing prefix")
        if not suffix:
            raise ValueError("Missing suffix")
        img = Image.open(image_path).convert("RGB")
        return {"image": img, "prefix": prefix, "suffix": suffix}


@dataclass
class FlorenceCollator:
    processor: Any
    max_length: int

    def __post_init__(self) -> None:
        try:
            sig = inspect.signature(self.processor.__call__)
            self._supports_suffix = "suffix" in sig.parameters
        except Exception:
            self._supports_suffix = False

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        images = [b["image"] for b in batch]
        prefixes = [b["prefix"] for b in batch]
        suffixes = [b["suffix"] for b in batch]

        if self._supports_suffix:
            enc = self.processor(
                text=prefixes,
                images=images,
                suffix=suffixes,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            # Florence processor returns labels when suffix is provided.
            if "labels" not in enc:
                raise ValueError("Processor did not return labels; try running with --no-suffix-path")
            return {
                "input_ids": enc["input_ids"],
                "attention_mask": enc.get("attention_mask"),
                "pixel_values": enc["pixel_values"],
                "labels": enc["labels"],
            }

        # Fallback path (older processor): build labels manually.
        # Florence-2 is encoder-decoder: input_ids → encoder (merged with image),
        # labels → decoder (shifted to create decoder_input_ids).
        # So input_ids = prefix ONLY, labels = suffix ONLY.
        tok = getattr(self.processor, "tokenizer", None)
        img_proc = getattr(self.processor, "image_processor", None)
        if tok is None or img_proc is None:
            raise ValueError("Processor missing tokenizer/image_processor")

        pixel_values = img_proc(images, return_tensors="pt")["pixel_values"]

        # Tokenize prefixes (encoder input) and suffixes (decoder target) separately.
        prefix_enc = tok(prefixes, return_tensors=None, padding=False, truncation=True, max_length=self.max_length)
        prefix_ids_list: List[List[int]] = [list(x) for x in prefix_enc["input_ids"]]

        suffix_ids_list: List[List[int]] = [
            list(tok(s, add_special_tokens=False)["input_ids"])[: self.max_length - 1]
            + ([tok.eos_token_id] if tok.eos_token_id is not None else [])
            for s in suffixes
        ]

        # Pad prefix (input_ids) batch
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
        max_prefix_len = max(len(x) for x in prefix_ids_list)

        def _pad(seq: List[int], target_len: int, pad_value: int) -> List[int]:
            return seq + [pad_value] * (target_len - len(seq)) if len(seq) < target_len else seq[:target_len]

        input_ids = torch.tensor([_pad(x, max_prefix_len, pad_id) for x in prefix_ids_list], dtype=torch.long)
        attention_mask = (input_ids != pad_id).long()

        # Pad suffix (labels) batch; use -100 as pad (ignored in loss)
        max_suffix_len = max(len(x) for x in suffix_ids_list)
        labels = torch.tensor([_pad(x, max_suffix_len, -100) for x in suffix_ids_list], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "labels": labels,
        }


def _guess_lora_target_modules(model: torch.nn.Module) -> List[str]:
    # Try to match common projection layer names.
    common = ("q_proj", "k_proj", "v_proj", "o_proj")
    found = set()
    for name, _ in model.named_modules():
        last = name.split(".")[-1]
        if last in common:
            found.add(last)
    return sorted(found) if found else ["q_proj", "k_proj", "v_proj", "o_proj"]


def main() -> None:
    _ensure_stage2_on_syspath()

    ap = argparse.ArgumentParser(description="Fine-tune Florence-2 OD on a JSONL dataset")
    ap.add_argument("--model-id", type=str, default="microsoft/Florence-2-base", help="Base model id or local path")
    ap.add_argument(
        "--train-jsonl",
        type=str,
        default=str(_stage2_root() / "finetune_data" / "taco_od_train.jsonl"),
        help="Training JSONL produced by prepare_taco_florence2_od_jsonl.py",
    )
    ap.add_argument("--eval-jsonl", type=str, default=None, help="Optional eval JSONL")
    ap.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for checkpoints (default: stage_2/finetuned/florence2_od_<timestamp>)",
    )
    ap.add_argument("--cache-dir", type=str, default=None, help="HF cache dir (recommended on clusters)")

    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--max-length", type=int, default=1024)

    ap.add_argument("--per-device-train-batch-size", type=int, default=1)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--num-train-epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1, help="If >0, overrides num_train_epochs")

    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--save-total-limit", type=int, default=2)
    ap.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Checkpoint path to resume from, or 'latest' to pick the newest checkpoint in output-dir",
    )

    ap.add_argument("--no-lora", action="store_true", help="Disable LoRA and full fine-tune the model")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)

    ap.add_argument("--bf16", action="store_true", help="Use bf16 mixed precision (Ampere+)")
    ap.add_argument("--fp16", action="store_true", help="Use fp16 mixed precision")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed (default 42 = original paper run). Set per run for multi-seed results.")

    args = ap.parse_args()

    # Import Trainer lazily so we can show a helpful error when transformers/peft versions are incompatible.
    try:
        from transformers import Trainer, TrainingArguments, set_seed  # noqa: WPS433
        # Seed everything (Python/NumPy/Torch) BEFORE model + LoRA init so the
        # adapter's random initialization is reproducible and varies per seed.
        set_seed(int(args.seed))
        print(f"[seed] global seed set to {args.seed}")
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "Failed to import transformers Trainer. This is almost always a dependency mismatch.\n\n"
            "Common causes on DGX clusters:\n"
            "  1) You installed `transformers>=4.55` or `transformers>=5` but the node has torch 2.0.x. Those transformers builds may require torch>=2.1/2.4 and disable torch.\n"
            "  2) `peft` is newer/older than your `transformers` (API mismatch).\n\n"
            "Fix (recommended pins for Florence-2 + LoRA on torch 2.0.1):\n"
            "  python -m pip uninstall -y transformers tokenizers huggingface_hub peft\n"
            "  python -m pip install --user -U 'transformers==4.40.2' 'tokenizers==0.19.1' 'huggingface_hub==0.23.4' 'peft==0.10.0' accelerate\n"
            "Then rerun this script.\n\n"
            f"Original error: {type(e).__name__}: {e}"
        )

    cache_dir = _resolve_path_arg(args.cache_dir) if args.cache_dir else default_cache_dir()
    configure_hf_cache(cache_dir)

    train_jsonl = _resolve_path_arg(args.train_jsonl)
    if train_jsonl is None or not train_jsonl.exists():
        raise SystemExit(f"train JSONL not found: {train_jsonl}")

    eval_jsonl = _resolve_path_arg(args.eval_jsonl) if args.eval_jsonl else None
    out_dir = _resolve_path_arg(args.output_dir) if args.output_dir else (_stage2_root() / "finetuned" / f"florence2_od_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    assert out_dir is not None
    out_dir.mkdir(parents=True, exist_ok=True)

    resume_from_checkpoint: Optional[str] = None
    if args.resume_from_checkpoint:
        if str(args.resume_from_checkpoint).lower() == "latest":
            checkpoints = sorted(
                (p for p in out_dir.glob("checkpoint-*") if p.is_dir()),
                key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
            )
            if not checkpoints:
                raise SystemExit(f"No checkpoints found in output directory: {out_dir}")
            resume_from_checkpoint = str(checkpoints[-1].resolve())
        else:
            candidate = _resolve_path_arg(args.resume_from_checkpoint)
            if candidate is None or not candidate.exists():
                candidate = (out_dir / str(args.resume_from_checkpoint)).resolve()
            if candidate is None or not candidate.exists():
                raise SystemExit(f"Checkpoint not found: {args.resume_from_checkpoint}")
            resume_from_checkpoint = str(candidate)

    device = args.device
    if not (device.startswith("cuda") and torch.cuda.is_available()):
        device = "cpu"

    print(f"Loading processor/model: {args.model_id} (device={device})")
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )

    if hasattr(model, "config") and model.config is not None:
        if hasattr(model.config, "attn_implementation"):
            try:
                model.config.attn_implementation = "eager"
            except Exception:
                pass

    if not args.no_lora:
        try:
            from peft import LoraConfig, TaskType, get_peft_model  # type: ignore

            targets = _guess_lora_target_modules(model)
            print(f"Enabling LoRA on target modules: {targets}")
            lora_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=int(args.lora_r),
                lora_alpha=int(args.lora_alpha),
                lora_dropout=float(args.lora_dropout),
                target_modules=targets,
            )
            model = get_peft_model(model, lora_cfg)
            model.print_trainable_parameters()
        except Exception as e:
            raise SystemExit(
                "LoRA requested but peft is not available or failed to initialize. "
                "Install peft (`pip install peft`) or re-run with --no-lora.\n"
                f"Original error: {type(e).__name__}: {e}"
            )

    train_ds = JsonlODDataset(train_jsonl)
    eval_ds = JsonlODDataset(eval_jsonl) if eval_jsonl else None

    collator = FlorenceCollator(processor=processor, max_length=int(args.max_length))

    # Trainer will move tensors to the proper device; model.to() is not necessary here.
    # But we set model to train mode.
    model.train()

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        seed=int(args.seed),
        data_seed=int(args.seed),
        per_device_train_batch_size=int(args.per_device_train_batch_size),
        gradient_accumulation_steps=int(args.gradient_accumulation_steps),
        learning_rate=float(args.learning_rate),
        num_train_epochs=float(args.num_train_epochs),
        max_steps=int(args.max_steps),
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
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    if resume_from_checkpoint:
        print(f"Resuming from checkpoint: {resume_from_checkpoint}")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(str(out_dir))
    try:
        processor.save_pretrained(str(out_dir))
    except Exception:
        pass

    print(f"Done. Saved fine-tuned model to: {out_dir}")


if __name__ == "__main__":
    main()
