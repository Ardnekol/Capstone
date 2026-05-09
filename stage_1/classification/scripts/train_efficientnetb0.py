#!/usr/bin/env python3
"""
EfficientNet-B0 Training Script for Waste Classification

Train EfficientNet-B0 on TrashNet dataset and evaluate on both TrashNet and RealWaste.

Usage:
    python train_efficientnetb0.py --epochs 50 --batch 32
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from PIL import Image
import json
from datetime import datetime
import numpy as np

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CONFIG = {
    'model_name': 'efficientnetb0',
    'epochs': 50,
    'batch_size': 32,
    'learning_rate': 0.001,
    'img_size': 224,
    'num_classes': 6,
    'classes': ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
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
# Model Setup
# ============================================================================

def create_model(num_classes):
    model = models.efficientnet_b0(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

# ============================================================================
# Training Function
# ============================================================================

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # Validation phase
        model.eval()
        running_loss = 0.0
        running_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = running_corrects.double() / len(val_loader.dataset)

        print(f'Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        scheduler.step()

        # Save best model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), f'results/efficientnetb0_best.pth')

    return model

# ============================================================================
# Evaluation Function
# ============================================================================

def evaluate_model(model, test_loader, device, classes):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Classification report
    report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    return report, cm

# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train EfficientNet-B0 for waste classification')
    parser.add_argument('--epochs', type=int, default=DEFAULT_CONFIG['epochs'], help='Number of epochs')
    parser.add_argument('--batch', type=int, default=DEFAULT_CONFIG['batch_size'], help='Batch size')
    parser.add_argument('--lr', type=float, default=DEFAULT_CONFIG['learning_rate'], help='Learning rate')
    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Data transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Create datasets
    train_dataset = WasteDataset('/u/student/2024/cs24mtech11024/Capstone/stage_1/datasets/classification/trashnet/dataset-preprocessed/', data_transforms['train'], classes=DEFAULT_CONFIG['classes'])
    val_dataset = WasteDataset('/u/student/2024/cs24mtech11024/Capstone/stage_1/datasets/classification/trashnet/dataset-preprocessed/', data_transforms['val'], classes=DEFAULT_CONFIG['classes'])

    # Split train into train/val (80/20)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False, num_workers=4)

    # Model
    model = create_model(DEFAULT_CONFIG['num_classes'])
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Train
    print("Starting training...")
    trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, args.epochs, device)

    # Evaluate on TrashNet
    print("Evaluating on TrashNet...")
    trashnet_report, trashnet_cm = evaluate_model(trained_model, val_loader, device, DEFAULT_CONFIG['classes'])

    # Evaluate on RealWaste
    print("Evaluating on RealWaste...")
    realwaste_dataset = WasteDataset('/u/student/2024/cs24mtech11024/Capstone/stage_1/datasets/classification/realwaste/dataset-preprocessed/', data_transforms['val'])  # Auto-detect classes
    realwaste_loader = DataLoader(realwaste_dataset, batch_size=args.batch, shuffle=False, num_workers=4)

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

        realwaste_report, realwaste_cm = evaluate_model(trained_model, realwaste_loader, device, DEFAULT_CONFIG['classes'])
    else:
        print("Warning: No mappable classes found in RealWaste dataset")
        realwaste_report = {'accuracy': 0, 'macro avg': {'precision': 0, 'recall': 0, 'f1-score': 0}}
        realwaste_cm = None

    # Save results
    results = {
        'model': 'EfficientNet-B0',
        'trashnet_accuracy': trashnet_report['accuracy'],
        'realwaste_accuracy': realwaste_report['accuracy'],
        'trashnet_report': trashnet_report,
        'realwaste_report': realwaste_report,
        'training_config': {
            'epochs': args.epochs,
            'batch_size': args.batch,
            'learning_rate': args.lr
        }
    }

    os.makedirs('results', exist_ok=True)
    with open('results/efficientnetb0_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("Training complete! Results saved to results/efficientnetb0_results.json")

if __name__ == '__main__':
    main()