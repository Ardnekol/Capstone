#!/usr/bin/env python3
"""Report which evaluation regimes are runnable in THIS interpreter.

Run this first (on the GPU node) to know what the overnight job will actually
produce. Each regime needs a specific set of packages; missing ones mean that
regime is skipped, not that the whole run fails.
"""
from __future__ import annotations
import importlib

REGIMES = {
    "Florence-2 + LoRA (all tasks)": ["torch", "transformers", "peft", "PIL", "numpy"],
    "Classification: ViT/ResNet/EffNet": ["torch", "timm", "torchvision"],
    "Classification: CLIP": ["open_clip"],
    "Detection: YOLOv8": ["ultralytics"],
    "Detection: Grounding DINO": ["transformers"],
    "Segmentation: DeepLabV3+": ["segmentation_models_pytorch"],
    "Segmentation: SAM": ["segment_anything"],
}


def have(mod: str) -> bool:
    try:
        importlib.import_module(mod)
        return True
    except Exception:
        return False


def main() -> None:
    import sys
    print(f"Python: {sys.version.split()[0]}  ({sys.executable})")
    try:
        import torch
        print(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
    except Exception:
        print("torch: MISSING")
    print()
    cache = {}
    for regime, mods in REGIMES.items():
        missing = [m for m in mods if not cache.setdefault(m, have(m))]
        status = "RUNNABLE " if not missing else "SKIP     "
        extra = "" if not missing else f"  (missing: {', '.join(missing)})"
        print(f"  [{status}] {regime}{extra}")
    print("\nTip: install missing user-site packages with e.g.")
    print("     pip install --user open_clip_torch ultralytics segmentation_models_pytorch segment_anything")


if __name__ == "__main__":
    main()
