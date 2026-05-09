#!/usr/bin/env python3
"""
CLIP (ViT-B/16) Evaluation Script for Waste Classification

Evaluate CLIP zero-shot performance on TrashNet and RealWaste datasets.

Usage:
    python eval_clip.py
"""

import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix
import open_clip
from PIL import Image
import json
import numpy as np

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CONFIG = {
    'model_name': 'ViT-B-16',
    'pretrained': 'openai',
    'classes': ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash'],
    'batch_size': 32,
    'img_size': 224
}

# ============================================================================
# Dataset Class
# ============================================================================

class WasteDataset(Dataset):
    def __init__(self, root_dir, transform=None, classes=None):
        self.root_dir = root_dir
        self.transform = transform
        if classes is None:
            # Auto-detect classes from directory
            self.classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
            self.classes.sort()  # Ensure consistent ordering
        else:
            self.classes = classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = []

        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            if os.path.exists(cls_dir):
                for img_name in os.listdir(cls_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.samples.append((os.path.join(cls_dir, img_name), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# ============================================================================
# CLIP Evaluation Function
# ============================================================================

def evaluate_clip(model, preprocess, tokenizer, text_inputs, test_loader, device, classes):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            # Encode images
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # Encode text
            text_features = model.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # Compute similarity
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            preds = similarity.argmax(dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Classification report
    report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    return report, cm

# ============================================================================
# Main Function
# ============================================================================

def main():
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load CLIP model
    model, _, preprocess = open_clip.create_model_and_transforms(DEFAULT_CONFIG['model_name'], pretrained=DEFAULT_CONFIG['pretrained'])
    tokenizer = open_clip.get_tokenizer(DEFAULT_CONFIG['model_name'])
    model = model.to(device)

    # Text prompts for each class
    text_prompts = [
        "a photo of cardboard waste",
        "a photo of glass waste",
        "a photo of metal waste",
        "a photo of paper waste",
        "a photo of plastic waste",
        "a photo of miscellaneous trash"
    ]

    text_inputs = tokenizer(text_prompts).to(device)

    # Data transform
    data_transform = transforms.Compose([
        transforms.Resize((DEFAULT_CONFIG['img_size'], DEFAULT_CONFIG['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])

    # Evaluate on TrashNet
    print("Evaluating CLIP on TrashNet...")
    trashnet_dataset = WasteDataset('/u/student/2024/cs24mtech11024/Capstone/stage_1/datasets/classification/trashnet/dataset-preprocessed/', data_transform, classes=DEFAULT_CONFIG['classes'])
    trashnet_loader = DataLoader(trashnet_dataset, batch_size=DEFAULT_CONFIG['batch_size'], shuffle=False, num_workers=4)
    trashnet_report, trashnet_cm = evaluate_clip(model, preprocess, tokenizer, text_inputs, trashnet_loader, device, DEFAULT_CONFIG['classes'])

    # Evaluate on RealWaste
    print("Evaluating CLIP on RealWaste...")
    realwaste_dataset = WasteDataset('/u/student/2024/cs24mtech11024/Capstone/stage_1/datasets/classification/realwaste/dataset-preprocessed/', data_transform)  # Auto-detect classes
    realwaste_loader = DataLoader(realwaste_dataset, batch_size=DEFAULT_CONFIG['batch_size'], shuffle=False, num_workers=4)

    # For RealWaste evaluation, we need to handle the different class structure
    # Map RealWaste classes to TrashNet classes where possible
    realwaste_class_mapping = {
        'Cardboard': 'cardboard',
        'Glass': 'glass',
        'Metal': 'metal',
        'Paper': 'paper',
        'Plastic': 'plastic',
        'Miscellaneous Trash': 'trash',
        'Food Organics': 'trash',  # Map to trash
        'Textile Trash': 'trash',  # Map to trash
        'Vegetation': 'trash'      # Map to trash
    }

    # Filter RealWaste samples to only include classes that can be mapped
    filtered_samples = []
    for img_path, class_idx in realwaste_dataset.samples:
        realwaste_class = realwaste_dataset.classes[class_idx]
        if realwaste_class in realwaste_class_mapping:
            trashnet_class = realwaste_class_mapping[realwaste_class]
            trashnet_idx = DEFAULT_CONFIG['classes'].index(trashnet_class)
            filtered_samples.append((img_path, trashnet_idx))

    if filtered_samples:
        # Create a filtered dataset with mapped classes
        realwaste_dataset.samples = filtered_samples
        realwaste_dataset.classes = DEFAULT_CONFIG['classes']  # Use TrashNet classes for evaluation
        realwaste_dataset.class_to_idx = {cls: idx for idx, cls in enumerate(DEFAULT_CONFIG['classes'])}

        realwaste_report, realwaste_cm = evaluate_clip(model, preprocess, tokenizer, text_inputs, realwaste_loader, device, DEFAULT_CONFIG['classes'])
    else:
        print("Warning: No mappable classes found in RealWaste dataset")
        realwaste_report = {'accuracy': 0, 'macro avg': {'precision': 0, 'recall': 0, 'f1-score': 0}}
        realwaste_cm = None

    # Save results
    results = {
        'model': 'CLIP (ViT-B/16)',
        'trashnet_accuracy': trashnet_report['accuracy'],
        'realwaste_accuracy': realwaste_report['accuracy'],
        'trashnet_report': trashnet_report,
        'realwaste_report': realwaste_report,
        'text_prompts': text_prompts
    }

    os.makedirs('results', exist_ok=True)
    with open('results/clip_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("CLIP evaluation complete! Results saved to results/clip_results.json")

if __name__ == '__main__':
    main()