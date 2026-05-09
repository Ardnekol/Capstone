#!/usr/bin/env python3
"""
Generate Segmentation Results Report

Compile results from all segmentation models and create comparison tables and plots.

Usage:
    python evaluate_segmentation.py
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
        'U-Net': 'results/unet_results.json',
        'DeepLabV3+': 'results/deeplabv3plus_results.json',
        'Mask R-CNN': 'results/maskrcnn_results.json',
        'SAM (ViT-H)': 'results/sam_results.json'
    }

    for model_name, file_path in result_files.items():
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                results[model_name] = json.load(f)
        else:
            print(f"Warning: {file_path} not found")

    return results

def create_iou_table(results):
    """Create IoU comparison table."""
    data = []
    for model_name, result in results.items():
        data.append({
            'Model': model_name,
            'TACO IoU': f"{result.get('taco_iou', 0):.4f}",
            'DWSD IoU': f"{result.get('dwsd_iou', 0):.4f}"
        })

    df = pd.DataFrame(data)
    return df

def create_detailed_metrics_table(results):
    """Create detailed metrics table."""
    data = []
    for model_name, result in results.items():
        # TACO metrics
        taco_metrics = result.get('taco_metrics', {})
        data.append({
            'Model': model_name,
            'Dataset': 'TACO',
            'IoU': f"{taco_metrics.get('iou', 0):.4f}",
            'Precision': f"{taco_metrics.get('precision', 0):.4f}",
            'Recall': f"{taco_metrics.get('recall', 0):.4f}",
            'F1-Score': f"{taco_metrics.get('f1', 0):.4f}"
        })

        # DWSD metrics
        dwsd_metrics = result.get('dwsd_metrics', {})
        data.append({
            'Model': model_name,
            'Dataset': 'DWSD',
            'IoU': f"{dwsd_metrics.get('iou', 0):.4f}",
            'Precision': f"{dwsd_metrics.get('precision', 0):.4f}",
            'Recall': f"{dwsd_metrics.get('recall', 0):.4f}",
            'F1-Score': f"{dwsd_metrics.get('f1', 0):.4f}"
        })

    df = pd.DataFrame(data)
    return df

def create_comparison_plot(results):
    """Create comparison bar plot."""
    models = list(results.keys())
    taco_ious = [results[model].get('taco_iou', 0) for model in models]
    dwsd_ious = [results[model].get('dwsd_iou', 0) for model in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, taco_ious, width, label='TACO (In-Domain)', alpha=0.8)
    bars2 = ax.bar(x + width/2, dwsd_ious, width, label='DWSD (Cross-Domain)', alpha=0.8)

    ax.set_xlabel('Models')
    ax.set_ylabel('IoU Score')
    ax.set_title('Segmentation Model Comparison: IoU Scores Across Datasets')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    add_value_labels(bars1)
    add_value_labels(bars2)

    plt.tight_layout()
    plt.savefig('results/segmentation_iou_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_report(results):
    """Generate comprehensive markdown report."""
    report = "# Segmentation Model Comparison Report\n\n"
    report += "## Overview\n\n"
    report += "This report compares the performance of different segmentation models on waste segmentation tasks.\n\n"
    report += "- **Task-Specific Models**: U-Net, DeepLabV3+, Mask R-CNN\n"
    report += "- **Foundation Model**: SAM (ViT-H) for zero-shot segmentation\n"
    report += "- **Training Dataset**: TACO (urban waste)\n"
    report += "- **Evaluation Datasets**: TACO (in-domain), DWSD (cross-domain - campus waste)\n\n"

    # IoU Comparison
    report += "## IoU Comparison\n\n"
    iou_df = create_iou_table(results)
    report += iou_df.to_markdown(index=False) + "\n\n"

    # Detailed Metrics
    report += "## Detailed Metrics\n\n"
    metrics_df = create_detailed_metrics_table(results)
    report += metrics_df.to_markdown(index=False) + "\n\n"

    # Analysis
    report += "## Analysis\n\n"

    # Find best models
    best_taco = max(results.items(), key=lambda x: x[1].get('taco_iou', 0))
    best_dwsd = max(results.items(), key=lambda x: x[1].get('dwsd_iou', 0))

    report += f"- **Best on TACO**: {best_taco[0]} ({best_taco[1].get('taco_iou', 0):.4f})\n"
    report += f"- **Best on DWSD**: {best_dwsd[0]} ({best_dwsd[1].get('dwsd_iou', 0):.4f})\n\n"

    # Domain shift analysis
    report += "### Cross-Domain Performance\n\n"
    report += "#### TACO → DWSD (Urban → Campus)\n"
    for model_name, result in results.items():
        taco_iou = result.get('taco_iou', 0)
        dwsd_iou = result.get('dwsd_iou', 0)
        drop = taco_iou - dwsd_iou
        relative_drop = (drop / taco_iou * 100) if taco_iou > 0 else 0
        report += f"- **{model_name}**: {drop:.4f} IoU drop ({relative_drop:.1f}% relative)\n"

    report += "\n## Conclusion\n\n"
    report += "This analysis provides insights into model performance across different waste segmentation scenarios.\n"
    report += "Task-specific models generally perform better on in-domain data, while foundation models offer\n"
    report += "competitive zero-shot performance without fine-tuning.\n\n"
    report += "The DWSD dataset enables meaningful urban-urban domain shift analysis, providing a complementary\n"
    report += "evaluation scenario to the extreme urban-beach shift observed with BePLi.\n\n"

    # Save report
    with open('results/SEGMENTATION_REPORT.md', 'w') as f:
        f.write(report)

    print("Report saved to results/SEGMENTATION_REPORT.md")

def main():
    print("Loading results...")
    results = load_results()
    print(f"Found results for {len(results)} models")

    if not results:
        print("No results found. Please run the training/evaluation scripts first.")
        return

    print("Creating plots...")
    create_comparison_plot(results)
    create_performance_profile_plot(results)
    create_domain_shift_summary_plot(results)

    print("Generating report...")
    generate_report(results)

    print("✅ Report generation complete!")
    print("📁 Report: results/SEGMENTATION_REPORT.md")
    print("📊 Plots: results/segmentation_iou_comparison.png")
    print("📈 Additional plots: results/performance_profile.png, results/domain_shift_summary.png")

def create_performance_profile_plot(results):
    """Create a performance profile radar chart"""
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))

    models = list(results.keys())
    metrics = ['iou', 'precision', 'recall', 'f1']

    # Calculate average performance across datasets
    for i, model in enumerate(models):
        taco_metrics = results[model]['taco_metrics']
        dwsd_metrics = results[model]['dwsd_metrics']

        # Average metrics across datasets
        avg_metrics = {}
        for metric in metrics:
            avg_metrics[metric] = (taco_metrics[metric] + dwsd_metrics[metric]) / 2

        # Create radar plot
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        values = [avg_metrics[metric] for metric in metrics]
        values += values[:1]  # Close the loop
        angles += angles[:1]

        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])

    ax.set_xticks(np.linspace(0, 2*np.pi, len(metrics), endpoint=False))
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_title('Average Performance Profile Across Datasets', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/performance_profile.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_domain_shift_summary_plot(results):
    """Create a domain shift summary bar chart"""
    fig, ax = plt.subplots(figsize=(12, 6))

    models = list(results.keys())
    taco_ious = [results[m]['taco_iou'] for m in models]
    dwsd_ious = [results[m]['dwsd_iou'] for m in models]
    drops = [taco - dwsd for taco, dwsd in zip(taco_ious, dwsd_ious)]

    x = np.arange(len(models))
    width = 0.25

    # Create grouped bars
    bars1 = ax.bar(x - width, taco_ious, width, label='TACO (In-Domain)', alpha=0.8, color='#2E86AB')
    bars2 = ax.bar(x, dwsd_ious, width, label='DWSD (Cross-Domain)', alpha=0.8, color='#A23B72')
    bars3 = ax.bar(x + width, drops, width, label='Performance Drop', alpha=0.8,
                   color=['red' if d > 0 else 'green' for d in drops])

    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('IoU Score / Drop', fontsize=12)
    ax.set_title('Domain Shift Summary: Performance Drop Analysis', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.,
                   height + (0.01 if height >= 0 else -0.03),
                   f'{height:.3f}', ha='center',
                   va='bottom' if height >= 0 else 'top', fontsize=8)

    # Add reference line at y=0
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('results/domain_shift_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main()