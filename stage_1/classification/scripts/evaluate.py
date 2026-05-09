#!/usr/bin/env python3
"""
Generate Classification Results Report

Compile results from all models and create comparison tables and plots.

Usage:
    python evaluate.py
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

def load_results():
    """Load results from all model evaluation files."""
    results = {}

    result_files = {
        'ResNet-50': 'results/resnet50_results.json',
        'EfficientNet-B0': 'results/efficientnetb0_results.json',
        'ViT-Base': 'results/vit_base_results.json',
        'CLIP (ViT-B/16)': 'results/clip_results.json'
    }

    for model_name, file_path in result_files.items():
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                results[model_name] = json.load(f)
        else:
            print(f"Warning: {file_path} not found")

    return results

def create_accuracy_table(results):
    """Create accuracy comparison table."""
    data = []
    for model_name, result in results.items():
        data.append({
            'Model': model_name,
            'TrashNet Accuracy': f"{result.get('trashnet_accuracy', 0):.4f}",
            'RealWaste Accuracy': f"{result.get('realwaste_accuracy', 0):.4f}"
        })

    df = pd.DataFrame(data)
    return df

def create_detailed_metrics_table(results):
    """Create detailed metrics table for each model and dataset."""
    rows = []

    for model_name, result in results.items():
        for dataset in ['trashnet', 'realwaste']:
            report = result.get(f'{dataset}_report', {})
            if 'macro avg' in report:
                macro = report['macro avg']
                rows.append({
                    'Model': model_name,
                    'Dataset': dataset.title(),
                    'Precision': f"{macro.get('precision', 0):.4f}",
                    'Recall': f"{macro.get('recall', 0):.4f}",
                    'F1-Score': f"{macro.get('f1-score', 0):.4f}",
                    'Accuracy': f"{result.get(f'{dataset}_accuracy', 0):.4f}"
                })

    df = pd.DataFrame(rows)
    return df

def plot_accuracy_comparison(results):
    """Plot accuracy comparison across models and datasets."""
    models = list(results.keys())
    trashnet_acc = [results[m].get('trashnet_accuracy', 0) for m in models]
    realwaste_acc = [results[m].get('realwaste_accuracy', 0) for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, trashnet_acc, width, label='TrashNet', alpha=0.8)
    ax.bar(x + width/2, realwaste_acc, width, label='RealWaste', alpha=0.8)

    ax.set_xlabel('Models')
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Accuracy Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_markdown_report(results):
    """Generate comprehensive markdown report."""
    report = "# Waste Classification Model Comparison Report\n\n"

    report += "## Overview\n\n"
    report += "This report compares the performance of different classification models on waste classification tasks.\n\n"
    report += "- **Task-Specific Models**: ResNet-50, EfficientNet-B0\n"
    report += "- **Foundation Model**: CLIP (ViT-B/16) for zero-shot classification\n"
    report += "- **Training Dataset**: TrashNet\n"
    report += "- **Evaluation Datasets**: TrashNet (in-domain), RealWaste (cross-domain)\n\n"

    # Accuracy Table
    acc_table = create_accuracy_table(results)
    report += "## Accuracy Comparison\n\n"
    report += acc_table.to_markdown(index=False) + "\n\n"

    # Detailed Metrics
    metrics_table = create_detailed_metrics_table(results)
    report += "## Detailed Metrics\n\n"
    report += metrics_table.to_markdown(index=False) + "\n\n"

    # Analysis
    report += "## Analysis\n\n"

    # Find best models
    best_trashnet = max(results.items(), key=lambda x: x[1].get('trashnet_accuracy', 0))
    best_realwaste = max(results.items(), key=lambda x: x[1].get('realwaste_accuracy', 0))

    report += f"- **Best on TrashNet**: {best_trashnet[0]} ({best_trashnet[1].get('trashnet_accuracy', 0):.4f})\n"
    report += f"- **Best on RealWaste**: {best_realwaste[0]} ({best_realwaste[1].get('realwaste_accuracy', 0):.4f})\n\n"

    # Cross-domain performance
    report += "### Cross-Domain Performance\n\n"
    for model_name, result in results.items():
        trashnet_acc = result.get('trashnet_accuracy', 0)
        realwaste_acc = result.get('realwaste_accuracy', 0)
        drop = trashnet_acc - realwaste_acc
        report += f"- **{model_name}**: {drop:+.4f} accuracy change from TrashNet to RealWaste\n"

    report += "\n## Conclusion\n\n"
    report += "This analysis provides insights into model performance across different waste classification scenarios.\n"
    report += "Task-specific models generally perform better on in-domain data, while foundation models offer\n"
    report += "competitive zero-shot performance without fine-tuning.\n"

    return report

def main():
    print("Loading results...")
    results = load_results()

    if not results:
        print("No results found. Please run model training/evaluation first.")
        return

    print(f"Found results for {len(results)} models")

    # Create plots
    print("Creating plots...")
    plot_accuracy_comparison(results)

    # Generate report
    print("Generating report...")
    report = generate_markdown_report(results)

    # Save report
    os.makedirs('results', exist_ok=True)
    with open('results/COMPARISON_REPORT.md', 'w') as f:
        f.write(report)

    print("Report saved to results/COMPARISON_REPORT.md")
    print("Plot saved to results/accuracy_comparison.png")

if __name__ == '__main__':
    main()