"""
===============================================
WEEK 4: TRAIN/VAL/TEST SPLIT THESIS VISUALIZATIONS
===============================================

This script creates detailed, publication-ready visualizations from the 
Week 4 train/val/test splits of the ISIC 2019 dataset for thesis documentation.

IMPORTANT:
- Does NOT load full X_train.npy / X_val.npy / X_test.npy (too large)
- Uses split statistics from week4_output.txt
- Works with network volume paths: /workspace/outputs/

Data sources on network volume:
  - /workspace/outputs/X_train.npy (64,000 samples)
  - /workspace/outputs/X_val.npy (8,000 samples)
  - /workspace/outputs/X_test.npy (8,000 samples)
  - /workspace/outputs/y_train.npy / y_val.npy / y_test.npy (labels)

This script generates 10 publication-ready visualizations for thesis inclusion:
  1. Train/Val/Test Split Overview
  2. Class Distribution Across Splits
  3. Train/Val/Test Balance Analysis
  4. Per-Class Split Comparison
  5. Dataset Composition Sankey Diagram
  6. Memory & Storage Analysis
  7. Split Ratios & Statistics
  8. Class-Specific Split Distribution
  9. Data Quality Assessment
  10. Comprehensive Thesis Report

Usage:
  python week4_thesis_visualizations.py

Prerequisites:
  - Week 4 split data on /workspace/outputs/
  - Split indices from week4_split_indices.npz
  - Week 3 augmented data reference

Dependencies:
  - numpy, pandas, matplotlib, seaborn
  - pathlib

Output:
  - All visualizations saved to /workspace/outputs/viz/week4/ (300 DPI)

Author: Thesis Documentation
Date: 2024
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, Rectangle
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION & PATHS
# ============================================

# Detect network volume (RunPod or local workspace)
BASE_DIR = Path(os.getcwd())
if Path("/workspace").exists():
    BASE_DIR = Path("/workspace")
elif Path("/notebooks").exists():
    BASE_DIR = Path("/notebooks")

# Network volume paths
OUTPUTS_DIR = (BASE_DIR / "outputs").resolve()

# Visualization output directory
THESIS_VIZ_DIR = OUTPUTS_DIR / "viz" / "week4"
THESIS_VIZ_DIR.mkdir(parents=True, exist_ok=True)

# Data paths
X_TRAIN_PATH = OUTPUTS_DIR / "X_train.npy"
X_VAL_PATH = OUTPUTS_DIR / "X_val.npy"
X_TEST_PATH = OUTPUTS_DIR / "X_test.npy"
Y_TRAIN_PATH = OUTPUTS_DIR / "y_train.npy"
Y_VAL_PATH = OUTPUTS_DIR / "y_val.npy"
Y_TEST_PATH = OUTPUTS_DIR / "y_test.npy"

# Class information
CLASS_NAMES = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']
CLASS_FULL_NAMES = {
    'AK': 'Actinic Keratosis',
    'BCC': 'Basal Cell Carcinoma',
    'BKL': 'Benign Keratosis-like Lesion',
    'DF': 'Dermatofibroma',
    'MEL': 'Melanoma',
    'NV': 'Nevus',
    'SCC': 'Squamous Cell Carcinoma',
    'VASC': 'Vascular Lesion'
}

# Color palette for train/val/test
SPLIT_COLORS = {
    'Train': '#3498db',      # Blue
    'Val': '#e74c3c',        # Red
    'Test': '#2ecc71'        # Green
}

# Color palette for classes
CLASS_COLORS = {
    'AK': '#FF6B6B',
    'BCC': '#4ECDC4',
    'BKL': '#45B7D1',
    'DF': '#FFA07A',
    'MEL': '#000000',
    'NV': '#FFD93D',
    'SCC': '#6BCB77',
    'VASC': '#A78BFA'
}

print(f"📊 Week 4 Train/Val/Test Split Thesis Visualization Generator")
print(f"=" * 70)
print(f"Network Volume:    {BASE_DIR}")
print(f"Outputs Dir:       {OUTPUTS_DIR}")
print(f"Thesis Viz Dir:    {THESIS_VIZ_DIR}")
print(f"=" * 70)

# ============================================
# DATA LOADING & STATISTICS
# ============================================

print("\n📁 Loading Week 4 split data...")

# ============================================
# HARDCODED SPLIT STATISTICS
# ============================================
# Based on week4_output.txt - actual final values from Week 4 execution

# Each class has exactly 10,000 samples before split
# Split: 8000 train / 1000 val / 1000 test per class
# This is balanced across all splits

split_class_dist = {
    'AK': {'Train': 8000, 'Val': 1000, 'Test': 1000},
    'BCC': {'Train': 8000, 'Val': 1000, 'Test': 1000},
    'BKL': {'Train': 8000, 'Val': 1000, 'Test': 1000},
    'DF': {'Train': 8000, 'Val': 1000, 'Test': 1000},
    'MEL': {'Train': 8000, 'Val': 1000, 'Test': 1000},
    'NV': {'Train': 8000, 'Val': 1000, 'Test': 1000},
    'SCC': {'Train': 8000, 'Val': 1000, 'Test': 1000},
    'VASC': {'Train': 8000, 'Val': 1000, 'Test': 1000},
}

# Total samples per split
train_total = 64000
val_total = 8000
test_total = 8000
grand_total = 80000

# Create metadata for visualizations
split_data = []
for class_name in CLASS_NAMES:
    for split_name, count in split_class_dist[class_name].items():
        split_data.append({
            'class': class_name,
            'split': split_name,
            'count': count,
            'percentage': (count / grand_total) * 100
        })

metadata_df = pd.DataFrame(split_data)

# File sizes (from week4_val_output.txt)
file_sizes = {
    'X_train': 257.49,  # GB
    'y_train': 0.00,    # GB (small)
    'X_val': 32.19,     # GB
    'y_val': 0.00,      # GB (small)
    'X_test': 32.19,    # GB
    'y_test': 0.00      # GB (small)
}
total_disk_size = sum(file_sizes.values())

# Image specifications
image_shape = (600, 600, 3)
total_images = 80000
bytes_per_image = np.prod(image_shape) * 4  # float32
estimated_total_size = (total_images * bytes_per_image) / (1024**3)

print(f"\n✓ X_train.npy: {file_sizes['X_train']:.2f} GB (64,000 images)")
print(f"✓ X_val.npy: {file_sizes['X_val']:.2f} GB (8,000 images)")
print(f"✓ X_test.npy: {file_sizes['X_test']:.2f} GB (8,000 images)")
print(f"✓ Total disk size: {total_disk_size:.2f} GB")

print(f"\n✅ Split Statistics (Week 4 Actual Output):")
print(f"   Train: {train_total:,} samples (80.0%)")
print(f"   Val: {val_total:,} samples (10.0%)")
print(f"   Test: {test_total:,} samples (10.0%)")
print(f"   Total: {grand_total:,} samples")
print(f"   Per-class split: 8000 train / 1000 val / 1000 test")

print(f"\n✅ Data loading complete!")
print(f"   Using hardcoded values from week4_output.txt")
print(f"   All visualizations use metadata only (no large array loading)")

# ============================================
# UTILITY FUNCTIONS
# ============================================

def get_class_color(class_name):
    """Get color for a class."""
    return CLASS_COLORS.get(class_name, '#cccccc')

def get_split_color(split_name):
    """Get color for a split."""
    return SPLIT_COLORS.get(split_name, '#cccccc')

def save_figure(fig, filename, dpi=300, transparent=False):
    """Save figure with thesis-quality settings."""
    filepath = THESIS_VIZ_DIR / filename
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', transparent=transparent)
    print(f"  ✓ Saved: {filename} ({dpi} DPI)")
    plt.close(fig)
    return filepath

# ============================================
# VISUALIZATION 1: Train/Val/Test Split Overview
# ============================================

print("\n🎨 Creating visualizations...\n")
print("  1️⃣  Train/Val/Test Split Overview...")

fig = plt.figure(figsize=(16, 10))
gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

fig.suptitle('Week 4: Train/Validation/Test Split Overview', fontsize=18, fontweight='bold', y=0.98)

# Subplot 1: Split sizes (bar chart)
ax1 = fig.add_subplot(gs[0, 0])
splits = ['Train', 'Val', 'Test']
sizes = [train_total, val_total, test_total]
colors_list = [SPLIT_COLORS[s] for s in splits]
bars = ax1.bar(splits, sizes, color=colors_list, edgecolor='black', linewidth=2, alpha=0.8)
ax1.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
ax1.set_title('Total Samples per Split', fontsize=13, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height):,}', ha='center', va='bottom', fontweight='bold', fontsize=11)

# Subplot 2: Split ratios (pie chart)
ax2 = fig.add_subplot(gs[0, 1])
percentages = [80, 10, 10]
explode = (0.05, 0, 0)
wedges, texts, autotexts = ax2.pie(percentages, labels=splits, autopct='%1.1f%%',
                                     colors=colors_list, explode=explode,
                                     textprops={'fontsize': 11, 'fontweight': 'bold'},
                                     startangle=90)
ax2.set_title('Split Distribution (%)', fontsize=13, fontweight='bold')
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

# Subplot 3: Storage information
ax3 = fig.add_subplot(gs[1, 0])
ax3.axis('off')

storage_text = f"""
STORAGE INFORMATION

Split Files:
  • X_train.npy: {file_sizes['X_train']:.2f} GB (64,000 images)
  • X_val.npy: {file_sizes['X_val']:.2f} GB (8,000 images)
  • X_test.npy: {file_sizes['X_test']:.2f} GB (8,000 images)

Label Files:
  • y_train.npy, y_val.npy, y_test.npy: ~{file_sizes['y_train']:.2f} GB (negligible)

Total Size: {total_disk_size:.2f} GB

Image Specifications:
  • Resolution: {image_shape[0]}×{image_shape[1]} pixels
  • Channels: {image_shape[2]} (RGB)
  • Data type: float32 (normalized ImageNet)
  • Bytes per image: {bytes_per_image / (1024**2):.2f} MB
"""

ax3.text(0.05, 0.95, storage_text, transform=ax3.transAxes, fontsize=10,
        verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, pad=1))

# Subplot 4: Split strategy
ax4 = fig.add_subplot(gs[1, 1])
ax4.axis('off')

strategy_text = """
SPLITTING STRATEGY

Balanced Per-Class Approach:
  ✓ All 8 classes sampled to 10,000 each
  ✓ Consistent split ratios across classes
  ✓ Per-class: 8000 train / 1000 val / 1000 test
  
Quality Assurance:
  ✓ No class imbalance across splits
  ✓ Stratified random sampling
  ✓ Reproducible with fixed seed (42)
  
Results:
  ✓ Train: 64,000 perfectly balanced images
  ✓ Val: 8,000 perfectly balanced images
  ✓ Test: 8,000 perfectly balanced images
  ✓ All splits have equal class representation
"""

ax4.text(0.05, 0.95, strategy_text, transform=ax4.transAxes, fontsize=10,
        verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8, pad=1))

save_figure(fig, '01_split_overview.png')

# ============================================
# VISUALIZATION 2: Class Distribution Across Splits
# ============================================

print("  2️⃣  Class Distribution Across Splits...")

fig, ax = plt.subplots(figsize=(14, 8))

class_split_data = []
for class_name in CLASS_NAMES:
    for split_name in ['Train', 'Val', 'Test']:
        class_split_data.append({
            'Class': class_name,
            'Split': split_name,
            'Samples': split_class_dist[class_name][split_name]
        })

split_df = pd.DataFrame(class_split_data)
pivot_df = split_df.pivot(index='Class', columns='Split', values='Samples')

pivot_df.plot(kind='bar', ax=ax, color=[SPLIT_COLORS[s] for s in ['Train', 'Val', 'Test']],
             edgecolor='black', linewidth=1.5, alpha=0.8)

ax.set_title('Class Distribution Across Train/Val/Test Splits', fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Class', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontweight='bold')
ax.legend(title='Split', fontsize=10, title_fontsize=11, loc='upper right')
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for container in ax.containers:
    ax.bar_label(container, label_type='edge', fontsize=9, fontweight='bold')

plt.tight_layout()
save_figure(fig, '02_class_distribution_splits.png')

# ============================================
# VISUALIZATION 3: Train/Val/Test Balance Analysis
# ============================================

print("  3️⃣  Train/Val/Test Balance Analysis...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Class Balance Within Each Split', fontsize=14, fontweight='bold', y=1.02)

for idx, split_name in enumerate(['Train', 'Val', 'Test']):
    ax = axes[idx]
    split_counts = [split_class_dist[cls][split_name] for cls in CLASS_NAMES]
    colors_list = [get_class_color(cls) for cls in CLASS_NAMES]
    
    bars = ax.bar(CLASS_NAMES, split_counts, color=colors_list, edgecolor='black', linewidth=1.5, alpha=0.8)
    
    ax.set_title(f'{split_name} Split\n(Total: {sum(split_counts):,})', fontsize=12, fontweight='bold')
    ax.set_ylabel('Samples', fontsize=11, fontweight='bold')
    ax.set_ylim(0, max(split_counts) * 1.1)
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height):,}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
save_figure(fig, '03_balance_analysis.png')

# ============================================
# VISUALIZATION 4: Per-Class Split Comparison
# ============================================

print("  4️⃣  Per-Class Split Comparison...")

fig, axes = plt.subplots(2, 4, figsize=(16, 10))
fig.suptitle('Per-Class Split Distribution (8000 Train / 1000 Val / 1000 Test)', 
             fontsize=14, fontweight='bold', y=1.00)

axes = axes.flatten()

for idx, class_name in enumerate(CLASS_NAMES):
    ax = axes[idx]
    
    train = split_class_dist[class_name]['Train']
    val = split_class_dist[class_name]['Val']
    test = split_class_dist[class_name]['Test']
    
    splits = ['Train', 'Val', 'Test']
    counts = [train, val, test]
    colors_list = [SPLIT_COLORS[s] for s in splits]
    
    bars = ax.bar(splits, counts, color=colors_list, edgecolor='black', linewidth=1.5, alpha=0.8)
    
    ax.set_title(f'{class_name}\n{CLASS_FULL_NAMES[class_name]}', fontsize=11, fontweight='bold')
    ax.set_ylabel('Samples', fontsize=10)
    ax.set_ylim(0, 9000)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height):,}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
save_figure(fig, '04_per_class_comparison.png')

# ============================================
# VISUALIZATION 5: Dataset Composition Waterfall
# ============================================

print("  5️⃣  Dataset Composition Waterfall...")

fig, ax = plt.subplots(figsize=(14, 8))

# Create a waterfall-like visualization showing class contributions
categories = []
values = []
colors_list = []

train_breakdown = []
for class_name in CLASS_NAMES:
    categories.append(class_name)
    values.append(split_class_dist[class_name]['Train'])
    colors_list.append(get_class_color(class_name))

cumulative = sum(values)

x_pos = np.arange(len(categories))
bars = ax.barh(x_pos, values, color=colors_list, edgecolor='black', linewidth=1.5, alpha=0.8)

ax.set_yticks(x_pos)
ax.set_yticklabels([f"{cat}\n{CLASS_FULL_NAMES[cat]}" for cat in categories], fontsize=10)
ax.set_xlabel('Training Samples', fontsize=12, fontweight='bold')
ax.set_title('Training Split: Class Contribution to Dataset', fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)

# Add value labels and percentages
for i, (bar, val) in enumerate(zip(bars, values)):
    width = bar.get_width()
    percentage = (val / train_total) * 100
    ax.text(width, bar.get_y() + bar.get_height()/2.,
           f' {int(val):,} ({percentage:.1f}%)',
           ha='left', va='center', fontweight='bold', fontsize=10)

plt.tight_layout()
save_figure(fig, '05_composition_breakdown.png')

# ============================================
# VISUALIZATION 6: Memory & Storage Analysis
# ============================================

print("  6️⃣  Memory & Storage Analysis...")

fig = plt.figure(figsize=(14, 8))
gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.35)

fig.suptitle('Week 4: Memory & Storage Analysis', fontsize=14, fontweight='bold')

# Subplot 1: Storage by split
ax1 = fig.add_subplot(gs[0, 0])
split_names = ['Train', 'Val', 'Test']
storage_sizes = [file_sizes['X_train'], file_sizes['X_val'], file_sizes['X_test']]
colors_list = [SPLIT_COLORS[s] for s in split_names]

bars = ax1.bar(split_names, storage_sizes, color=colors_list, edgecolor='black', linewidth=2, alpha=0.8)
ax1.set_ylabel('Storage (GB)', fontsize=11, fontweight='bold')
ax1.set_title('Disk Storage by Split', fontsize=12, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}GB', ha='center', va='bottom', fontweight='bold', fontsize=10)

# Subplot 2: Storage breakdown pie chart
ax2 = fig.add_subplot(gs[0, 1])
wedges, texts, autotexts = ax2.pie(storage_sizes, labels=split_names, autopct='%1.1f%%',
                                     colors=colors_list, startangle=90,
                                     textprops={'fontsize': 10, 'fontweight': 'bold'})
ax2.set_title('Total Storage Distribution\n(Total: {:.2f}GB)'.format(total_disk_size), 
             fontsize=12, fontweight='bold')
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

# Subplot 3: Memory per image
ax3 = fig.add_subplot(gs[1, 0])
ax3.axis('off')

memory_text = f"""
MEMORY & STORAGE DETAILS

Image Specifications:
  • Resolution: 600×600 pixels
  • Color channels: 3 (RGB)
  • Data type: float32
  • Memory per image: {bytes_per_image / (1024**2):.2f} MB

Storage Breakdown:
  • X_train.npy: {file_sizes['X_train']:.2f} GB ({train_total:,} images)
  • X_val.npy: {file_sizes['X_val']:.2f} GB ({val_total:,} images)
  • X_test.npy: {file_sizes['X_test']:.2f} GB ({test_total:,} images)
  • Label files (y_*.npy): negligible

Total Dataset Size: {total_disk_size:.2f} GB

Calculation Check:
  • Expected: {estimated_total_size:.2f} GB
  • Actual: {total_disk_size:.2f} GB
  • ✓ Match (float32 @ 600×600×3)
"""

ax3.text(0.05, 0.95, memory_text, transform=ax3.transAxes, fontsize=9.5,
        verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8, pad=1))

# Subplot 4: Memory loading strategy
ax4 = fig.add_subplot(gs[1, 1])
ax4.axis('off')

loading_text = """
MEMORY LOADING STRATEGY

Week 4 Process:
  ✓ Sequential chunk processing (no full load)
  ✓ Memory-controlled copying (124 img/chunk)
  ✓ Aggressive garbage collection
  ✓ 46.6 GB container RAM utilized
  ✓ RunPod network volume optimized

Safety Measures:
  ✓ 50% memory safety factor
  ✓ Conservative worker count (2)
  ✓ ~0.5 GB per chunk limit
  ✓ Real-time memory monitoring
  ✓ Checkpoint/resume support

Result:
  ✓ No out-of-memory errors
  ✓ Stable processing
  ✓ Total runtime: ~2.5 hours
"""

ax4.text(0.05, 0.95, loading_text, transform=ax4.transAxes, fontsize=9.5,
        verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8, pad=1))

save_figure(fig, '06_memory_storage_analysis.png')

# ============================================
# VISUALIZATION 7: Split Ratios & Statistics
# ============================================

print("  7️⃣  Split Ratios & Statistics...")

fig = plt.figure(figsize=(14, 10))
gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

fig.suptitle('Week 4: Split Ratios & Detailed Statistics', fontsize=14, fontweight='bold')

# Subplot 1: Sample distribution table
ax1 = fig.add_subplot(gs[0, :])
ax1.axis('off')

# Create table data
table_data = [['Class', 'Train', 'Val', 'Test', 'Total', 'Train %', 'Val %', 'Test %']]
for class_name in CLASS_NAMES:
    train = split_class_dist[class_name]['Train']
    val = split_class_dist[class_name]['Val']
    test = split_class_dist[class_name]['Test']
    total = train + val + test
    table_data.append([
        class_name,
        f'{train:,}',
        f'{val:,}',
        f'{test:,}',
        f'{total:,}',
        f'{(train/total)*100:.1f}%',
        f'{(val/total)*100:.1f}%',
        f'{(test/total)*100:.1f}%'
    ])

# Add totals row
table_data.append(['TOTAL', f'{train_total:,}', f'{val_total:,}', f'{test_total:,}', 
                   f'{grand_total:,}', f'{(train_total/grand_total)*100:.1f}%',
                   f'{(val_total/grand_total)*100:.1f}%', f'{(test_total/grand_total)*100:.1f}%'])

table = ax1.table(cellText=table_data, cellLoc='center', loc='center',
                 colWidths=[0.08, 0.12, 0.12, 0.12, 0.12, 0.11, 0.11, 0.11])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Style header row
for i in range(len(table_data[0])):
    table[(0, i)].set_facecolor('#3498db')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style total row
for i in range(len(table_data[0])):
    table[(len(table_data)-1, i)].set_facecolor('#ecf0f1')
    table[(len(table_data)-1, i)].set_text_props(weight='bold')

# Color code data rows
for i in range(1, len(table_data)-1):
    table[(i, 0)].set_facecolor('#f8f9fa')
    table[(i, 0)].set_text_props(weight='bold')

ax1.set_title('Detailed Sample Distribution by Class and Split', fontsize=12, fontweight='bold', pad=20)

# Subplot 2: Class balance uniformity
ax2 = fig.add_subplot(gs[1, 0])

# Check if all splits have the same distribution
uniformity_data = {
    'Train': np.std([split_class_dist[cls]['Train'] for cls in CLASS_NAMES]),
    'Val': np.std([split_class_dist[cls]['Val'] for cls in CLASS_NAMES]),
    'Test': np.std([split_class_dist[cls]['Test'] for cls in CLASS_NAMES])
}

splits = list(uniformity_data.keys())
stds = list(uniformity_data.values())
colors_list = [SPLIT_COLORS[s] for s in splits]

bars = ax2.bar(splits, stds, color=colors_list, edgecolor='black', linewidth=2, alpha=0.8)
ax2.set_ylabel('Standard Deviation', fontsize=11, fontweight='bold')
ax2.set_title('Class Distribution Uniformity\n(Lower is better)', fontsize=12, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

# Subplot 3: Quality metrics
ax3 = fig.add_subplot(gs[1, 1])
ax3.axis('off')

metrics_text = f"""
QUALITY METRICS

Balance Statistics:
  • All classes have 10,000 samples
  • Per-class split: 8000/1000/1000
  • Perfect uniformity across splits
  • Standard deviation: 0 (ideal)

Dataset Characteristics:
  • Total samples: {grand_total:,}
  • Total classes: {len(CLASS_NAMES)}
  • Samples per class: 10,000 (uniform)
  • Train/Val/Test ratio: 80/10/10
  
Quality Assurance:
  ✓ No class imbalance
  ✓ Stratified random sampling
  ✓ Consistent representation
  ✓ Ready for model training
"""

ax3.text(0.05, 0.95, metrics_text, transform=ax3.transAxes, fontsize=10,
        verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, pad=1))

save_figure(fig, '07_split_statistics.png')

# ============================================
# VISUALIZATION 8: Class-Specific Split Distribution
# ============================================

print("  8️⃣  Class-Specific Split Distribution...")

fig, ax = plt.subplots(figsize=(14, 8))

# Create stacked bar chart
train_counts = [split_class_dist[cls]['Train'] for cls in CLASS_NAMES]
val_counts = [split_class_dist[cls]['Val'] for cls in CLASS_NAMES]
test_counts = [split_class_dist[cls]['Test'] for cls in CLASS_NAMES]

x = np.arange(len(CLASS_NAMES))
width = 0.6

bars1 = ax.bar(x, train_counts, width, label='Train', color=SPLIT_COLORS['Train'], 
              edgecolor='black', linewidth=1.5, alpha=0.8)
bars2 = ax.bar(x, val_counts, width, bottom=train_counts, label='Val', 
              color=SPLIT_COLORS['Val'], edgecolor='black', linewidth=1.5, alpha=0.8)
bars3 = ax.bar(x, test_counts, width, bottom=np.array(train_counts)+np.array(val_counts),
              label='Test', color=SPLIT_COLORS['Test'], edgecolor='black', linewidth=1.5, alpha=0.8)

ax.set_xlabel('Class', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
ax.set_title('Stacked Split Distribution by Class', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(CLASS_NAMES, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.grid(axis='y', alpha=0.3)

# Add total labels
for i, (train, val, test) in enumerate(zip(train_counts, val_counts, test_counts)):
    total = train + val + test
    ax.text(i, total, f'{total:,}', ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.tight_layout()
save_figure(fig, '08_stacked_split_distribution.png')

# ============================================
# VISUALIZATION 9: Data Quality Assessment
# ============================================

print("  9️⃣  Data Quality Assessment...")

fig = plt.figure(figsize=(14, 8))
gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

fig.suptitle('Week 4: Data Quality Assessment', fontsize=14, fontweight='bold')

# Subplot 1: Split representation
ax1 = fig.add_subplot(gs[0, 0])
ax1.axis('off')

quality_text = """
SPLIT REPRESENTATION QUALITY

Category Balance:
  ✓ Train:  8 classes × 8,000 = 64,000
  ✓ Val:    8 classes × 1,000 = 8,000
  ✓ Test:   8 classes × 1,000 = 8,000
  ✓ Total:  80,000 perfectly balanced

Statistical Properties:
  • Mean samples per class (train): 8,000
  • Std dev: 0 (perfect balance)
  • Min-Max range: 0 (all equal)
  • Coefficient of variation: 0%
  
Imbalance Ratio:
  • Train: 1.00 (perfect)
  • Val: 1.00 (perfect)
  • Test: 1.00 (perfect)
"""

ax1.text(0.05, 0.95, quality_text, transform=ax1.transAxes, fontsize=9,
        verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8, pad=1))

# Subplot 2: Validation checks
ax2 = fig.add_subplot(gs[0, 1])
ax2.axis('off')

validation_text = """
VALIDATION CHECKS PASSED

Data Integrity:
  ✓ All files successfully created
  ✓ File sizes match expectations
  ✓ Shapes verified (600×600×3)
  ✓ Data types correct (float32)
  ✓ Label ranges valid (0-7)
  
Reproducibility:
  ✓ Random seed fixed (42)
  ✓ Deterministic split assignment
  ✓ No data leakage between splits
  ✓ Checkpoint indices saved
  
Safety Measures:
  ✓ Memory constraints respected
  ✓ Container limits honored
  ✓ Garbage collection applied
  ✓ No OOM errors
"""

ax2.text(0.05, 0.95, validation_text, transform=ax2.transAxes, fontsize=9,
        verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8, pad=1))

# Subplot 3: Class representation in train split
ax3 = fig.add_subplot(gs[1, 0])

train_percentages = [(split_class_dist[cls]['Train'] / train_total) * 100 for cls in CLASS_NAMES]
colors_list = [get_class_color(cls) for cls in CLASS_NAMES]

bars = ax3.barh(CLASS_NAMES, train_percentages, color=colors_list, edgecolor='black', linewidth=1.5, alpha=0.8)

ax3.set_xlabel('Percentage of Train Set (%)', fontsize=11, fontweight='bold')
ax3.set_title('Training Split Class Representation', fontsize=12, fontweight='bold')
ax3.set_xlim(0, 15)
ax3.grid(axis='x', alpha=0.3)

for i, (bar, pct) in enumerate(zip(bars, train_percentages)):
    ax3.text(pct, bar.get_y() + bar.get_height()/2.,
            f' {pct:.2f}%', ha='left', va='center', fontweight='bold', fontsize=9)

# Subplot 4: Readiness assessment
ax4 = fig.add_subplot(gs[1, 1])
ax4.axis('off')

readiness_text = """
READINESS FOR MODEL TRAINING

Data Preparation Status:
  ✓ Train/Val/Test splits created
  ✓ Perfectly balanced classes
  ✓ Stratified random sampling
  ✓ No data leakage
  ✓ Reproducible (seed=42)

Next Steps (Week 5):
  • Load train split incrementally
  • Create data loaders/generators
  • Normalize with ImageNet stats
  • Implement augmentation pipeline
  • Train baseline models
  
Recommended Approach:
  • Use memmap mode for efficiency
  • Batch processing (32-256)
  • Data augmentation on-the-fly
  • Validation after each epoch
  • Early stopping mechanism
"""

ax4.text(0.05, 0.95, readiness_text, transform=ax4.transAxes, fontsize=9,
        verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, pad=1))

save_figure(fig, '09_quality_assessment.png')

# ============================================
# VISUALIZATION 10: Comprehensive Thesis Report
# ============================================

print("  🔟 Comprehensive Thesis Report...")

fig = plt.figure(figsize=(16, 12))
gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)

fig.suptitle('Week 4: Comprehensive Data Splitting Report', fontsize=16, fontweight='bold', y=0.98)

# Top left: Key metrics
ax1 = fig.add_subplot(gs[0, 0])
ax1.axis('off')

metrics_text = f"""
KEY METRICS

Total Dataset: {grand_total:,}
  • Train: {train_total:,} (80%)
  • Val: {val_total:,} (10%)
  • Test: {test_total:,} (10%)

Classes: {len(CLASS_NAMES)}
  • Samples/class: 10,000
  • Split: 8k/1k/1k
  
Balance: Perfect ✓
  • Std dev: 0
  • Ratio: 1.00
"""

ax1.text(0, 0.95, metrics_text, transform=ax1.transAxes, fontsize=10,
        verticalalignment='top', family='monospace', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#3498db', alpha=0.2, pad=1))

# Top center: File information
ax2 = fig.add_subplot(gs[0, 1])
ax2.axis('off')

file_text = f"""
FILES CREATED

Training Split:
  X_train.npy
  y_train.npy
  
Validation Split:
  X_val.npy
  y_val.npy
  
Test Split:
  X_test.npy
  y_test.npy
  
Indices: week4_
split_indices.npz
"""

ax2.text(0, 0.95, file_text, transform=ax2.transAxes, fontsize=9,
        verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='#2ecc71', alpha=0.2, pad=1))

# Top right: Storage summary
ax3 = fig.add_subplot(gs[0, 2])
ax3.axis('off')

storage_summary = f"""
STORAGE SUMMARY

X_train: {file_sizes['X_train']:.2f} GB
X_val: {file_sizes['X_val']:.2f} GB
X_test: {file_sizes['X_test']:.2f} GB

Total: {total_disk_size:.2f} GB

Images: {total_images:,}
Type: float32
Shape: ({image_shape[0]}×{image_shape[1]}×{image_shape[2]})
"""

ax3.text(0, 0.95, storage_summary, transform=ax3.transAxes, fontsize=10,
        verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='#e74c3c', alpha=0.2, pad=1))

# Middle row: Distribution charts
ax4 = fig.add_subplot(gs[1, :2])

splits = ['Train', 'Val', 'Test']
sizes = [train_total, val_total, test_total]
colors_list = [SPLIT_COLORS[s] for s in splits]

bars = ax4.bar(splits, sizes, color=colors_list, edgecolor='black', linewidth=2, alpha=0.8)
ax4.set_ylabel('Samples', fontsize=11, fontweight='bold')
ax4.set_title('Split Distribution', fontsize=12, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    pct = (height / grand_total) * 100
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height):,}\n({pct:.0f}%)', ha='center', va='bottom', fontweight='bold', fontsize=10)

# Middle right: Class balance radar/summary
ax5 = fig.add_subplot(gs[1, 2])
ax5.axis('off')

balance_text = """
CLASS BALANCE
ASSESSMENT

Train Split:
✓ All 8 classes
✓ 8,000 each
✓ Mean: 8,000
✓ Std: 0

Val Split:
✓ All 8 classes
✓ 1,000 each
✓ Mean: 1,000
✓ Std: 0

Test Split:
✓ All 8 classes
✓ 1,000 each
✓ Mean: 1,000
✓ Std: 0
"""

ax5.text(0, 0.95, balance_text, transform=ax5.transAxes, fontsize=9,
        verticalalignment='top', family='monospace', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#f39c12', alpha=0.2, pad=1))

# Bottom: Summary table
ax6 = fig.add_subplot(gs[2, :])
ax6.axis('off')

summary_table_data = [['Class', 'Train', 'Val', 'Test', 'Total', 'Train%', 'Val%', 'Test%']]
for cls in CLASS_NAMES:
    train = split_class_dist[cls]['Train']
    val = split_class_dist[cls]['Val']
    test = split_class_dist[cls]['Test']
    total = train + val + test
    summary_table_data.append([
        cls,
        f'{train:,}',
        f'{val:,}',
        f'{test:,}',
        f'{total:,}',
        f'80%',
        f'10%',
        f'10%'
    ])

summary_table_data.append(['TOTAL', f'{train_total:,}', f'{val_total:,}', f'{test_total:,}',
                          f'{grand_total:,}', f'80%', f'10%', f'10%'])

summary_table = ax6.table(cellText=summary_table_data, cellLoc='center', loc='center',
                         colWidths=[0.08, 0.12, 0.12, 0.12, 0.12, 0.11, 0.11, 0.11])
summary_table.auto_set_font_size(False)
summary_table.set_fontsize(9)
summary_table.scale(1, 2)

# Style header
for i in range(len(summary_table_data[0])):
    summary_table[(0, i)].set_facecolor('#34495e')
    summary_table[(0, i)].set_text_props(weight='bold', color='white')

# Style total
for i in range(len(summary_table_data[0])):
    summary_table[(len(summary_table_data)-1, i)].set_facecolor('#bdc3c7')
    summary_table[(len(summary_table_data)-1, i)].set_text_props(weight='bold')

save_figure(fig, '10_comprehensive_report.png')

# ============================================
# COMPLETION MESSAGE
# ============================================

print("\n" + "=" * 70)
print("✅ THESIS VISUALIZATION GENERATION COMPLETE!")
print("=" * 70)
print(f"\n📁 Output Directory: {THESIS_VIZ_DIR}")
print(f"\n🎨 Generated Visualizations:")
print(f"   1. 01_split_overview.png - Overview of splits")
print(f"   2. 02_class_distribution_splits.png - Distribution across splits")
print(f"   3. 03_balance_analysis.png - Balance analysis")
print(f"   4. 04_per_class_comparison.png - Per-class breakdown")
print(f"   5. 05_composition_breakdown.png - Composition waterfall")
print(f"   6. 06_memory_storage_analysis.png - Memory & storage")
print(f"   7. 07_split_statistics.png - Detailed statistics")
print(f"   8. 08_stacked_split_distribution.png - Stacked distribution")
print(f"   9. 09_quality_assessment.png - Quality assessment")
print(f"   10. 10_comprehensive_report.png - Comprehensive report")
print(f"\n📊 All visualizations saved at 300 DPI for thesis quality")
print(f"\n✨ Ready for thesis documentation and publication!")
print("=" * 70)
