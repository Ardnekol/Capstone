#!/usr/bin/env python3
"""
Unified Florence-2 Inference Pipeline
======================================
Takes one image as input and produces three task outputs:
1. Classification (text): Waste class label
2. Object Detection (image): Bounding boxes
3. Segmentation (image): Polygon masks
"""

import argparse, json, re, os
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import cv2
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from peft import PeftModel
from PIL import Image

WASTE_CLASSES = {"cardboard", "glass", "metal", "paper", "plastic", "trash"}
CLASS_COLORS = {
    "cardboard": (0, 165, 255), "glass": (0, 255, 255), "metal": (128, 128, 128),
    "paper": (0, 255, 255), "plastic": (0, 0, 255), "trash": (128, 0, 128),
}

def _dequantize_location_token(token_idx: int, image_size: int) -> float:
    return (token_idx + 0.5) * (image_size / 1000.0)

def _parse_loc_tokens(text: str) -> List[int]:
    tokens = []
    matches = re.findall(r'<loc_(\d+)>', text)
    for match in matches:
        tokens.append(int(match))
    return tokens

def _extract_label_from_output(text: str) -> str:
    text_lower = text.lower()
    for cls in WASTE_CLASSES:
        if cls in text_lower:
            return cls
    return "trash"

def _extract_detections(output_text: str, image_height: int, image_width: int) -> List[Dict]:
    detections = []
    pattern = r'([a-z\s]+?)(<loc_\d+><loc_\d+><loc_\d+><loc_\d+>)'
    matches = re.finditer(pattern, output_text.lower())
    
    for match in matches:
        label_text = match.group(1).strip()
        loc_tokens_str = match.group(2)
        tokens = _parse_loc_tokens(loc_tokens_str)
        if len(tokens) < 4:
            continue
        
        x1 = int(_dequantize_location_token(tokens[0], image_width))
        y1 = int(_dequantize_location_token(tokens[1], image_height))
        x2 = int(_dequantize_location_token(tokens[2], image_width))
        y2 = int(_dequantize_location_token(tokens[3], image_height))
        
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        label = _extract_label_from_output(label_text)
        detections.append({"label": label, "bbox": [x1, y1, x2, y2], "confidence": 0.95})
    
    return detections

def _extract_segmentation(output_text: str, image_height: int, image_width: int) -> List[Dict]:
    segmentations = []
    pattern = r'([a-z\s]+?)(<loc_\d+>(?:<loc_\d+>)+)'
    matches = re.finditer(pattern, output_text.lower())
    
    for match in matches:
        label_text = match.group(1).strip()
        loc_tokens_str = match.group(2)
        tokens = _parse_loc_tokens(loc_tokens_str)
        
        if len(tokens) < 2 or len(tokens) % 2 != 0:
            continue
        
        polygon = []
        for i in range(0, len(tokens), 2):
            x = int(_dequantize_location_token(tokens[i], image_width))
            y = int(_dequantize_location_token(tokens[i+1], image_height))
            polygon.append([x, y])
        
        if len(polygon) >= 3:
            label = _extract_label_from_output(label_text)
            segmentations.append({"label": label, "polygon": polygon})
    
    return segmentations

def _extract_segmentation_from_postprocess(result: Dict) -> List[Dict]:
    segmentations = []

    for value in result.values():
        if not isinstance(value, dict):
            continue
        polygons = value.get("polygons")
        labels = value.get("labels")
        if not polygons:
            continue

        for idx, poly_group in enumerate(polygons):
            if not poly_group:
                continue

            label = "trash"
            if isinstance(labels, list) and idx < len(labels):
                label = _extract_label_from_output(str(labels[idx]))

            for poly in poly_group:
                if isinstance(poly, list) and len(poly) >= 6:
                    polygon = [[int(poly[i]), int(poly[i + 1])] for i in range(0, len(poly) - 1, 2)]
                    if len(polygon) >= 3:
                        segmentations.append({"label": label, "polygon": polygon})

    return segmentations

def draw_detections(image_path: str, detections: List[Dict]) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    
    for det in detections:
        label = det["label"]
        x1, y1, x2, y2 = det["bbox"]
        color = CLASS_COLORS.get(label, (255, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale, thickness = 0.5, 1
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        cv2.rectangle(img, (x1, y1 - text_size[1] - 5), (x1 + text_size[0] + 5, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 5), font, font_scale, (255, 255, 255), thickness)
    
    return img

def draw_segmentation(image_path: str, segmentations: List[Dict]) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    
    overlay = img.copy()
    for seg in segmentations:
        label = seg["label"]
        polygon = seg["polygon"]
        if len(polygon) < 3:
            continue
        
        color = CLASS_COLORS.get(label, (255, 255, 255))
        pts = np.array(polygon, dtype=np.int32)
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=2)
    
    alpha = 0.3
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale, thickness = 0.5, 1
    for seg in segmentations:
        label = seg["label"]
        polygon = seg["polygon"]
        if len(polygon) >= 3:
            pts = np.array(polygon, dtype=np.float32)
            cx, cy = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))
            color = CLASS_COLORS.get(label, (255, 255, 255))
            cv2.putText(img, label, (cx, cy), font, font_scale, color, thickness)
    
    return img

def run_inference(image_path: str, model_id: str, output_dir: str = "./inference_outputs", device: str = "cuda") -> Dict:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("UNIFIED FLORENCE-2 INFERENCE")
    print("="*60)
    print(f"Image: {image_path}\nModel: {model_id}\nDevice: {device}")
    print("="*60 + "\n")
    
    img = Image.open(image_path)
    image_height, image_width = img.size[1], img.size[0]
    print(f"Image size: {image_width}x{image_height}\n")
    
    print("[Loading model...]")
    processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large-ft", trust_remote_code=True)
    
    # Load base model in float32 first (to avoid exp_vml_cpu error)
    base_model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-large-ft",
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    
    # Load LoRA adapter
    if os.path.isdir(model_id):
        model = PeftModel.from_pretrained(base_model, model_id)
    else:
        model = base_model
    
    # Move to device
    model = model.to(device)
    
    # Convert to float16 on GPU for inference
    if device == "cuda":
        model = model.half()
    
    print("✓ Model loaded\n")
    
    results = {}
    
    print("[1/3] Classification...")
    inputs = processor(text="<CAPTION>", images=img, return_tensors="pt").to(device)
    # Convert inputs to model's dtype
    if device == "cuda":
        inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
    with torch.no_grad():
        generated_ids = model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=100, do_sample=False)
    classification_output = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    classification_label = _extract_label_from_output(classification_output)
    results["classification"] = {"raw_output": classification_output.strip(), "label": classification_label}
    print(f"  ➜ Label: {classification_label}\n")
    
    print("[2/3] Object Detection...")
    inputs = processor(text="<OD>", images=img, return_tensors="pt").to(device)
    # Convert inputs to model's dtype
    if device == "cuda":
        inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
    with torch.no_grad():
        generated_ids = model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=300, do_sample=False)
    detection_output = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    detections = _extract_detections(detection_output, image_height, image_width)
    results["detections"] = detections
    print(f"  ➜ Detected {len(detections)} objects\n")
    
    print("[3/3] Segmentation...")
    task_prompt = f"<REFERRING_EXPRESSION_SEGMENTATION>{classification_label}"
    inputs = processor(text=task_prompt, images=img, return_tensors="pt").to(device)
    # Convert inputs to model's dtype
    if device == "cuda":
        inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=256,
            do_sample=False,
            num_beams=3,
            early_stopping=True,
        )
    segmentation_output = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    postprocessed = processor.post_process_generation(
        segmentation_output,
        task="<REFERRING_EXPRESSION_SEGMENTATION>",
        image_size=(image_width, image_height),
    )
    segmentations = _extract_segmentation_from_postprocess(postprocessed)
    if not segmentations:
        segmentations = _extract_segmentation(segmentation_output, image_height, image_width)
    results["segmentations"] = segmentations
    print(f"  ➜ Segmented {len(segmentations)} regions\n")
    
    print("[4/4] Drawing visualizations...")
    if detections:
        det_img = draw_detections(image_path, detections)
        det_output = os.path.join(output_dir, "image_with_detection.jpg")
        cv2.imwrite(det_output, det_img)
        print(f"  ✓ Detection: {det_output}")
    
    if segmentations:
        seg_img = draw_segmentation(image_path, segmentations)
        seg_output = os.path.join(output_dir, "image_with_segmentation.jpg")
        cv2.imwrite(seg_output, seg_img)
        print(f"  ✓ Segmentation: {seg_output}")
    
    results_json_path = os.path.join(output_dir, "results.json")
    with open(results_json_path, "w") as f:
        json_results = {
            "classification": results["classification"],
            "detections": [{"label": d["label"], "bbox": [int(x) for x in d["bbox"]], "confidence": float(d["confidence"])} for d in detections],
            "segmentations": [{"label": s["label"], "polygon": [[int(x), int(y)] for x, y in s["polygon"]]} for s in segmentations]
        }
        json.dump(json_results, f, indent=2)
    print(f"  ✓ Results: {results_json_path}")
    
    print("\n" + "="*60)
    print("UNIFIED INFERENCE COMPLETE")
    print("="*60 + "\n")
    return results

def main():
    parser = argparse.ArgumentParser(description="Unified Florence-2 Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model-id", type=str, default="finetuned/florence2_unified_multitask_lora", help="Model path")
    parser.add_argument("--output-dir", type=str, default="./inference_outputs", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    
    args = parser.parse_args()
    
    if not os.path.isfile(args.image):
        print(f"ERROR: Image not found: {args.image}")
        exit(1)
    
    if not os.path.isdir(args.model_id):
        print(f"ERROR: Model not found: {args.model_id}")
        exit(1)
    
    run_inference(image_path=args.image, model_id=args.model_id, output_dir=args.output_dir, device=args.device)

if __name__ == "__main__":
    main()
