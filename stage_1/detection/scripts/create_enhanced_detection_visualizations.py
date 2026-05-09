#!/usr/bin/env python3
"""
Enhanced Visualization Script for Detection Analysis
Generates comprehensive plots for the detailed detection report
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_results():
    """Load all detection model results"""
    results_file = Path('results/all_results.json')
    with open(results_file, 'r') as f:
        results = json.load(f)

    # Also load CSV for additional data
    csv_file = Path('results/model_comparison.csv')
    csv_data = pd.read_csv(csv_file)

    return results, csv_data

def create_domain_shift_visualization(results, csv_data):
    """Create comprehensive domain shift visualization"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Extract data
    models = []
    in_domain_map50 = []
    cross_domain_map50 = []
    retention_rates = []

    for _, row in csv_data.iterrows():
        if pd.notna(row['In-Domain mAP50']) and pd.notna(row['Cross-Domain mAP50']):
            models.append(row['Model'])
            in_domain_map50.append(row['In-Domain mAP50'])
            cross_domain_map50.append(row['Cross-Domain mAP50'])
            retention_rates.append(row['Retention %'] / 100)  # Convert to decimal

    # Domain shift comparison
    x = np.arange(len(models))
    width = 0.35

    bars1 = ax1.bar(x - width/2, in_domain_map50, width, label='TACO (In-Domain)', alpha=0.8, color='#2E86AB')
    bars2 = ax1.bar(x + width/2, cross_domain_map50, width, label='ICRA19 (Cross-Domain)', alpha=0.8, color='#A23B72')

    ax1.set_xlabel('Models', fontsize=12)
    ax1.set_ylabel('mAP@0.5', fontsize=12)
    ax1.set_title('Domain Shift Analysis: TACO → ICRA19', fontsize=14, fontweight='bold')
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

    # Retention rate analysis
    colors = ['red' if r < 1 else 'green' for r in retention_rates]
    bars = ax2.bar(models, retention_rates, color=colors, alpha=0.7)
    ax2.set_ylabel('Retention Rate (Cross/In-Domain)', fontsize=12)
    ax2.set_title('Model Robustness: Domain Retention Analysis', fontsize=14, fontweight='bold')
    ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='No Change')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Add value labels
    for bar, rate in zip(bars, retention_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.,
                height + (0.02 if height >= 1 else -0.05),
                f'{rate:.2f}', ha='center',
                va='bottom' if height >= 1 else 'top', fontsize=9)

    # Performance profile radar chart
    angles = np.linspace(0, 2*np.pi, 3, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    # Normalize metrics for radar (0-1 scale)
    max_map50 = max(max(in_domain_map50), max(cross_domain_map50))

    for i, model in enumerate(models):
        taco_val = in_domain_map50[i] / max_map50
        icra_val = cross_domain_map50[i] / max_map50
        retention_val = retention_rates[i]

        values = [taco_val, icra_val, retention_val]
        values += values[:1]

        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        ax3.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i % len(colors)])
        ax3.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])

    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(['TACO mAP', 'ICRA19 mAP', 'Retention'])
    ax3.set_ylim(0, 1.2)
    ax3.set_title('Model Performance Profiles', fontsize=14, fontweight='bold')
    ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax3.grid(True, alpha=0.3)

    # Grounding DINO detailed analysis
    if 'grounding_dino' in results:
        gd_results = results['grounding_dino']
        datasets = ['taco', 'icra19']
        metrics = ['precision', 'recall']

        data = []
        for dataset in datasets:
            for metric in metrics:
                data.append({
                    'Dataset': dataset.upper(),
                    'Metric': metric.capitalize(),
                    'Value': gd_results['datasets'][dataset][metric]
                })

        df = pd.DataFrame(data)

        sns.barplot(data=df, x='Dataset', y='Value', hue='Metric', ax=ax4, palette=['#2E86AB', '#A23B72'])
        ax4.set_title('Grounding DINO: Precision vs Recall Analysis', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Score', fontsize=12)
        ax4.grid(True, alpha=0.3)

        # Add value labels
        for container in ax4.containers:
            ax4.bar_label(container, fmt='%.3f', fontsize=9)

    plt.tight_layout()
    plt.savefig('results/enhanced_detection_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_model_comparison_heatmap(csv_data):
    """Create a heatmap comparing all metrics across models"""
    # Filter out rows with NaN values for key metrics
    valid_data = csv_data.dropna(subset=['In-Domain mAP50', 'In-Domain mAP50-95'])

    fig, ax = plt.subplots(figsize=(12, 6))

    # Prepare data for heatmap
    models = valid_data['Model'].values
    metrics_data = valid_data[['In-Domain mAP50', 'In-Domain mAP50-95']].values

    # Create heatmap
    sns.heatmap(metrics_data, annot=True, fmt='.3f', cmap='YlOrRd',
                xticklabels=['mAP@0.5', 'mAP@0.5:0.95'],
                yticklabels=models, ax=ax)

    ax.set_title('In-Domain Performance Heatmap', fontsize=16, fontweight='bold')
    ax.set_ylabel('Models', fontsize=12)

    plt.tight_layout()
    plt.savefig('results/detection_performance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_deployment_recommendation_plot(csv_data):
    """Create deployment recommendation visualization"""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Define criteria scores (0-10 scale) based on results
    criteria = {
        'Accuracy': [10, 3, 4, 6],  # YOLOv8 highest, Grounding DINO good cross-domain
        'Speed': [9, 5, 7, 6],      # YOLOv8 fastest
        'Domain Robustness': [2, 7, 4, 10],  # Grounding DINO best
        'Ease of Deployment': [8, 6, 7, 5],   # YOLOv8 easiest
        'Training Requirements': [7, 4, 5, 10]  # Grounding DINO no training
    }

    models = ['YOLOv8', 'Faster R-CNN', 'RetinaNet', 'Grounding DINO']

    # Create radar chart for each model
    angles = np.linspace(0, 2*np.pi, len(criteria), endpoint=False).tolist()
    angles += angles[:1]

    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

    for i, model in enumerate(models):
        values = [criteria[criterion][i] for criterion in criteria.keys()]
        values += values[:1]  # Close the loop

        ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(list(criteria.keys()))
    ax.set_ylim(0, 10)
    ax.set_title('Detection Model Deployment Suitability', fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    ax.grid(True, alpha=0.3)

    # Add recommendation text
    recommendations = {
        'YOLOv8': 'Best for real-time urban waste detection',
        'Faster R-CNN': 'Strong baseline for research applications',
        'RetinaNet': 'Good for imbalanced datasets',
        'Grounding DINO': 'Superior for cross-domain scenarios'
    }

    y_pos = 0.02
    for model, rec in recommendations.items():
        ax.text(1.2, y_pos, f'{model}: {rec}', transform=ax.transAxes,
               fontsize=9, verticalalignment='bottom')
        y_pos += 0.08

    plt.tight_layout()
    plt.savefig('results/detection_deployment_guide.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Generate all enhanced detection visualizations"""
    print("Loading detection results...")
    results, csv_data = load_results()

    print("Creating domain shift analysis plot...")
    create_domain_shift_visualization(results, csv_data)

    print("Creating performance heatmap...")
    create_model_comparison_heatmap(csv_data)

    print("Creating deployment recommendations...")
    create_deployment_recommendation_plot(csv_data)

    print("✅ Enhanced detection visualizations generated!")
    print("📁 Saved to results/ directory:")
    print("  - enhanced_detection_analysis.png")
    print("  - detection_performance_heatmap.png")
    print("  - detection_deployment_guide.png")

if __name__ == '__main__':
    main()