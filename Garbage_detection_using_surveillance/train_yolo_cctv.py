#!/usr/bin/env python3
"""YOLOv8 training for CCTV garbage detection — improved config."""

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def train(
    model_name: str = "yolov8m",
    epochs: int = 150,
    imgsz: int = 1024,
    batch_size: int = 4,
    device: int = 0,
    patience: int = 40,
):
    data_yaml = Path(__file__).parent / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_yaml}")

    if not torch.cuda.is_available():
        device_str = "cpu"
        print("Warning: CUDA not available, training on CPU.")
    else:
        gpu_count = torch.cuda.device_count()
        device = min(device, gpu_count - 1)
        device_str = device
        print(f"GPU: {torch.cuda.get_device_name(device)} "
              f"({torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB)")

    print(f"\nModel={model_name}  epochs={epochs}  imgsz={imgsz}  batch={batch_size}\n")

    # accept bare size letter ("m") or full name ("yolov8m")
    pt = model_name if model_name.startswith("yolo") else f"yolov8{model_name}"
    model = YOLO(f"{pt}.pt")

    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device_str,
        patience=patience,

        # Checkpointing
        save=True,
        save_period=10,

        # Learning rate — lowered to prevent early peaking with SGD
        lr0=0.0005,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,

        # Augmentation — tuned for fixed-angle CCTV footage
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,        # brightness variation for low-light
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        flipud=0.0,        # CCTV is fixed orientation — never flip upside-down
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,    # reduced from 0.3 — less noise with small batch
        multi_scale=False, # disabled — adds variance that hurts small-batch training

        # Regularisation — none needed, YOLO head handles it
        dropout=0.0,

        # Loss weights
        box=7.5,
        cls=0.5,
        dfl=1.5,

        # SGD proven better than AdamW on this dataset
        optimizer="SGD",
        # nbs=64 → ultralytics auto-computes accumulate=64/4=16 steps
        # effective batch = 64, stable gradients despite small physical batch
        nbs=64,
        close_mosaic=15,
        warmup_epochs=3,
        seed=42,

        project="runs/detect",
        name="cctv_garbage_yolo_v4",
        exist_ok=False,
        verbose=True,
    )

    best = Path("runs/detect/cctv_garbage_yolo_v4/weights/best.pt")
    print(f"\nBest model: {best}")
    print(f"Evaluate: python3 evaluate_yolo_cctv.py --model {best}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",   default="yolov8m", help="yolov8n/s/m/l/x")
    p.add_argument("--epochs",  type=int, default=150)
    p.add_argument("--imgsz",   type=int, default=1024)
    p.add_argument("--batch",   type=int, default=4)
    p.add_argument("--device",  type=int, default=0)
    p.add_argument("--patience",type=int, default=40)
    args = p.parse_args()
    train(args.model, args.epochs, args.imgsz, args.batch, args.device, args.patience)


if __name__ == "__main__":
    main()
