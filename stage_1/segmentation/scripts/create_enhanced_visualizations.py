#!/usr/bin/env python3
"""
Enhanced Visualization Script for Segmentation Analysis
Generates comprehensive plots for the detailed segmentation report
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.patches as mpatches

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_results():
    """Load all model results from JSON files"""
    results_dir = Path('results')
    results = {}

    result_files = {
        'U-Net': 'unet_results.json',
        'DeepLabV3+': 'deeplabv3plus_results.json',
        'Mask R-CNN': 'maskrcnn_results.json',
        'SAM (ViT-H)': 'sam_results.json'
    }

    for model_name, filename in result_files.items():
        with open(results_dir / filename, 'r') as f:
            results[model_name] = json.load(f)

    return results

def create_domain_shift_plot(results):
    """Create domain shift visualization"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    models = list(results.keys())
    taco_ious = [results[m]['taco_iou'] for m in models]
    dwsd_ious = [results[m]['dwsd_iou'] for m in models]

    # Domain shift bars
    x = np.arange(len(models))
    width = 0.35

    bars1 = ax1.bar(x - width/2, taco_ious, width, label='TACO (In-Domain)', alpha=0.8, color='#2E86AB')
    bars2 = ax1.bar(x + width/2, dwsd_ious, width, label='DWSD (Cross-Domain)', alpha=0.8, color='#A23B72')

    ax1.set_xlabel('Models', fontsize=12)
    ax1.set_ylabel('IoU Score', fontsize=12)
    ax1.set_title('Domain Shift Analysis: TACO → DWSD', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    # Performance drop plot
    drops = [taco_ious[i] - dwsd_ious[i] for i in range(len(models))]
    colors = ['red' if drop > 0 else 'green' for drop in drops]

    bars = ax2.bar(models, drops, color=colors, alpha=0.7)
    ax2.set_ylabel('IoU Drop (TACO - DWSD)', fontsize=12)
    ax2.set_title('Performance Degradation Analysis', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)

    # Add value labels
    for bar, drop in zip(bars, drops):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.,
                height + (0.01 if height >= 0 else -0.03),
                f'{drop:.3f}', ha='center',
                va='bottom' if height >= 0 else 'top', fontsize=9)

    # Radar plot for balanced comparison
    angles = np.linspace(0, 2*np.pi, len(models), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    taco_ious_closed = taco_ious + [taco_ious[0]]
    dwsd_ious_closed = dwsd_ious + [dwsd_ious[0]]

    ax3.plot(angles, taco_ious_closed, 'o-', linewidth=2, label='TACO', color='#2E86AB')
    ax3.fill(angles, taco_ious_closed, alpha=0.25, color='#2E86AB')
    ax3.plot(angles, dwsd_ious_closed, 'o-', linewidth=2, label='DWSD', color='#A23B72')
    ax3.fill(angles, dwsd_ious_closed, alpha=0.25, color='#A23B72')

    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(models)
    ax3.set_title('Model Performance Comparison (Radar)', fontsize=14, fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    # Precision vs Recall scatter plot
    taco_precision = [results[m]['taco_metrics']['precision'] for m in models]
    taco_recall = [results[m]['taco_metrics']['recall'] for m in models]
    dwsd_precision = [results[m]['dwsd_metrics']['precision'] for m in models]
    dwsd_recall = [results[m]['dwsd_metrics']['recall'] for m in models]

    ax4.scatter(taco_precision, taco_recall, s=100, label='TACO', color='#2E86AB', marker='o')
    ax4.scatter(dwsd_precision, dwsd_recall, s=100, label='DWSD', color='#A23B72', marker='s')

    for i, model in enumerate(models):
        ax4.annotate(model, (taco_precision[i], taco_recall[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax4.annotate(model, (dwsd_precision[i], dwsd_recall[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)

    ax4.set_xlabel('Precision', fontsize=12)
    ax4.set_ylabel('Recall', fontsize=12)
    ax4.set_title('Precision vs Recall Analysis', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig('results/enhanced_domain_shift_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_model_comparison_heatmap(results):
    """Create a heatmap comparing all metrics across models and datasets"""
    models = list(results.keys())
    metrics = ['iou', 'precision', 'recall', 'f1']
    datasets = ['taco', 'dwsd']

    # Create data matrix
    data = []
    for model in models:
        for dataset in datasets:
            row = [model, dataset.upper()]
            for metric in metrics:
                value = results[model][f'{dataset}_metrics'][metric]
                row.append(value)
            data.append(row)

    df = pd.DataFrame(data, columns=['Model', 'Dataset', 'IoU', 'Precision', 'Recall', 'F1-Score'])

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))

    # Pivot data for heatmap
    heatmap_data = df.pivot(index=['Model', 'Dataset'], columns=None, values=['IoU', 'Precision', 'Recall', 'F1-Score'])
    heatmap_data = heatmap_data.droplevel(0, axis=1)  # Remove multiindex column

    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax,
                cbar_kws={'label': 'Score'}, linewidths=0.5)

    ax.set_title('Comprehensive Model Performance Heatmap', fontsize=16, fontweight='bold')
    ax.set_ylabel('Model & Dataset', fontsize=12)

    plt.tight_layout()
    plt.savefig('results/model_performance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_deployment_recommendation_plot(results):
    """Create a deployment recommendation visualization"""
    fig, ax = plt.subplots(figsize=(14, 8))

    models = list(results.keys())

    # Define criteria scores (0-10 scale)
    criteria = {
        'Accuracy': [7, 9, 6, 2],  # DeepLabV3+ highest
        'Speed': [8, 6, 4, 5],     # U-Net fastest
        'Memory': [8, 6, 3, 2],    # U-Net most efficient
        'Domain Adaptability': [4, 3, 5, 9],  # SAM best
        'Ease of Deployment': [9, 7, 5, 6]   # U-Net easiest
    }

    # Normalize scores
    normalized_criteria = {}
    for criterion, scores in criteria.items():
        max_score = max(scores)
        normalized_criteria[criterion] = [score/max_score * 10 for score in scores]

    # Create radar chart for each model
    angles = np.linspace(0, 2*np.pi, len(criteria), endpoint=False).tolist()
    angles += angles[:1]

    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

    for i, (model, color) in enumerate(zip(models, colors)):
        values = [normalized_criteria[criterion][i] for criterion in criteria.keys()]
        values += values[:1]  # Close the loop

        ax.plot(angles, values, 'o-', linewidth=2, label=model, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(list(criteria.keys()))
    ax.set_ylim(0, 10)
    ax.set_title('Deployment Suitability Analysis', fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    ax.grid(True, alpha=0.3)

    # Add recommendation text
    recommendations = {
        'U-Net': 'Best for resource-constrained deployment',
        'DeepLabV3+': 'Optimal for high-accuracy applications',
        'Mask R-CNN': 'Ideal for detailed instance analysis',
        'SAM (ViT-H)': 'Superior for cross-domain scenarios'
    }

    y_pos = 0.02
    for model, rec in recommendations.items():
        ax.text(1.2, y_pos, f'{model}: {rec}', transform=ax.transAxes,
               fontsize=9, verticalalignment='bottom')
        y_pos += 0.08

    plt.tight_layout()
    plt.savefig('results/deployment_recommendations.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Generate all enhanced visualizations"""
    print("Loading results...")
    results = load_results()

    print("Creating domain shift analysis plot...")
    create_domain_shift_plot(results)

    print("Creating performance heatmap...")
    create_model_comparison_heatmap(results)

    print("Creating deployment recommendations...")
    create_deployment_recommendation_plot(results)

    print("✅ Enhanced visualizations generated!")
    print("📁 Saved to results/ directory:")
    print("  - enhanced_domain_shift_analysis.png")
    print("  - model_performance_heatmap.png")
    print("  - deployment_recommendations.png")

if __name__ == '__main__':
    main()