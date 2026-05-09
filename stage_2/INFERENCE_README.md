# Unified Florence-2 Inference System

## Overview

The unified Florence-2 model takes **one image** as input and produces **three task outputs**:

1. **Classification** (text): Waste class label (cardboard, glass, metal, paper, plastic, trash)
2. **Object Detection** (image): Input image with bounding boxes
3. **Segmentation** (image): Input image with polygon masks

All three outputs come from the **same model** via three different task prompts.

---

## Quick Start

### 1. Demo on TACO Dataset

```bash
cd ~/Capstone/stage_2
bash demo_inference.sh
```

This runs inference on a sample TACO image and saves outputs to `./inference_outputs/demo/`.

### 2. Custom Image

```bash
export PATH="$HOME/.conda/envs/Capstone/bin:$PATH"
python3 inference_unified.py \
    --image /path/to/your/image.jpg \
    --model-id finetuned/florence2_unified_multitask_lora \
    --output-dir ./my_results
```

---

## Output Files

For each inference run, you get three outputs:

1. **image_with_detection.jpg**
   - Original image with bounding boxes
   - Color-coded by waste class:
     - Cardboard: Orange
     - Glass: Cyan
     - Metal: Gray
     - Paper: Yellow
     - Plastic: Red
     - Trash: Purple

2. **image_with_segmentation.jpg**
   - Original image with semi-transparent polygon overlays
   - Class labels displayed at polygon centroids

3. **results.json**
   - Structured output with:
     - Classification label (text)
     - Detections: list of {label, bbox, confidence}
     - Segmentations: list of {label, polygon coordinates}

---

## How It Works

```
One Input Image
       ↓
[Florence-2 Model]
  Base: microsoft/Florence-2-large-ft (0.77B params)
  Fine-tuned: LoRA on multi-task dataset
       ↓
Three Task Prompts (three separate passes):
  1. "<CAPTION>" → Classification
  2. "<OD>" → Object Detection  
  3. "<REFERRING_EXPRESSION_SEGMENTATION>waste" → Segmentation
       ↓
Location Token Parsing:
  - Detections: label<loc_x1><loc_y1><loc_x2><loc_y2>
  - Segmentation: label<loc_x1><loc_y1>...<loc_xn><loc_yn>
  - Quantization: 1000 bins per axis (0-999)
       ↓
Three Output Images + JSON
```

---

## Model Information

- **Base Model**: `microsoft/Florence-2-large-ft`
- **Parameters**: 0.77B (base) + 5.2M (LoRA)
- **Training**: 3 epochs on unified multi-task dataset
- **Fine-tuned Model Path**: `finetuned/florence2_unified_multitask_lora/`

---

## Requirements

- Python 3.8+
- Conda environment: `Capstone`
- Packages: torch, transformers, peft, opencv-python, pillow
- GPU recommended (can run on CPU, slower)

Install via:
```bash
conda activate Capstone
```

---

## Example Output

```
============================================================
UNIFIED FLORENCE-2 INFERENCE
============================================================
Image: /path/to/waste.jpg
Model: finetuned/florence2_unified_multitask_lora
Device: cuda

Image size: 640x480

[Loading model...]
✓ Model loaded

[1/3] Classification...
  ➜ Label: plastic

[2/3] Object Detection...
  ➜ Detected 2 objects

[3/3] Segmentation...
  ➜ Segmented 1 regions

[4/4] Drawing visualizations...
  ✓ Detection: ./inference_outputs/demo/image_with_detection.jpg
  ✓ Segmentation: ./inference_outputs/demo/image_with_segmentation.jpg
  ✓ Results: ./inference_outputs/demo/results.json

============================================================
UNIFIED INFERENCE COMPLETE
============================================================
```

---

## Troubleshooting

**Q: "Model not found" error**
```bash
# First fine-tune the model:
cd ~/Capstone/stage_2
bash train_unified.sh --skip-prep
```

**Q: "Image not found" error**
```bash
# Use absolute path:
python3 inference_unified.py --image /full/path/to/image.jpg
```

**Q: Out of GPU memory**
```bash
# Run on CPU:
python3 inference_unified.py --image image.jpg --device cpu
```

**Q: Very slow inference**
- Check GPU usage: `nvidia-smi`
- Ensure CUDA device is correct: `echo $CUDA_VISIBLE_DEVICES`

---

## Files

| File | Purpose |
|------|---------|
| `inference_unified.py` | Main inference pipeline (9.8K) |
| `demo_inference.sh` | Quick demo script (1.1K) |
| `INFERENCE_README.md` | This documentation |

---

## Next Steps

- Batch process multiple images
- Integrate with downstream applications
- Retrain on tri-task proxy dataset for improved performance
- Adjust visualization colors/transparency in `inference_unified.py`

