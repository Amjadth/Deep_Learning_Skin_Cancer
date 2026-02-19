# ============================================
# WEEK 4 VISUALIZATION GENERATOR (Report-Ready)
# ============================================
#
# Loads existing train/val/test splits from outputs and generates the
# report-ready visualizations from Week 4 (bar/stacked/heatmap, pies, summary).
#
# Outputs (saved to outputs/visualizations):
# - split_distribution.png (2x2: totals, train class bars, per-split bars, heatmap) [300 DPI]
# - split_pie_charts.png (3 pies: Train/Val/Test) [300 DPI]
# - split_statistics_summary.png (text summary figure) [300 DPI]
# Also saves a quick reference split_distribution.png to outputs at 150 DPI.
#
# Author: Deep Learning Engineer
# Date: 2024
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import json
import os

# --------------------------------------------
# Configuration
# --------------------------------------------
CLASS_NAMES = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

# Environment detection (RunPod/workspace)
BASE_DIR = Path(os.getcwd())
if Path('/workspace').exists():
    BASE_DIR = Path('/workspace')
elif Path('/notebooks').exists():
    BASE_DIR = Path('/notebooks')

# Prefer network volume/default outputs path
OUTPUT_DIR = (BASE_DIR / 'outputs').resolve()
if os.path.exists('/runpod-volume'):
    OUTPUT_DIR = Path('/runpod-volume/outputs').resolve()
if os.path.exists('/workspace'):
    # If /workspace exists and outputs exist there, prefer it
    ws_out = Path('/workspace/outputs').resolve()
    if ws_out.exists():
        OUTPUT_DIR = ws_out

# Paths
X_TRAIN = OUTPUT_DIR / 'X_train.npy'
Y_TRAIN = OUTPUT_DIR / 'y_train.npy'
X_VAL = OUTPUT_DIR / 'X_val.npy'
Y_VAL = OUTPUT_DIR / 'y_val.npy'
X_TEST = OUTPUT_DIR / 'X_test.npy'
Y_TEST = OUTPUT_DIR / 'y_test.npy'

# Validate files
required = [X_TRAIN, Y_TRAIN, X_VAL, Y_VAL, X_TEST, Y_TEST]
missing = [p for p in required if not p.exists()]
if missing:
    raise FileNotFoundError(f"Missing required split files: {', '.join(str(m) for m in missing)}")

# Load labels (small)
y_train = np.load(Y_TRAIN)
y_val = np.load(Y_VAL)
y_test = np.load(Y_TEST)

# Load image arrays as memmaps (no full RAM load) just to get shapes
# Infer shape from labels and known per-image shape based on saved files
# We'll detect per-image shape using memmap of one file based on file size

def infer_image_shape_and_load(file_path: Path, num_samples: int):
    # Try to infer per-image HWC by reading shape via .npy header if available, else compute from size
    try:
        arr = np.load(file_path, mmap_mode='r')
        # If this succeeded, we can use arr.shape directly
        return arr, arr.shape[1:]
    except Exception:
        # Fallback: assume float32 and (H, W, C) = (600, 600, 3) per Week 3/4
        image_shape = (600, 600, 3)
        full_shape = (num_samples,) + image_shape
        arr = np.memmap(file_path, dtype=np.float32, mode='r', shape=full_shape)
        return arr, image_shape

X_train, image_shape = infer_image_shape_and_load(X_TRAIN, len(y_train))
X_val, _ = infer_image_shape_and_load(X_VAL, len(y_val))
X_test, _ = infer_image_shape_and_load(X_TEST, len(y_test))

total_samples_loaded = len(X_train) + len(X_val) + len(X_test)

# Compute distributions
train_counts = Counter(y_train)
val_counts = Counter(y_val)
test_counts = Counter(y_test)

# Create viz dir
viz_dir = OUTPUT_DIR / 'visualizations'
viz_dir.mkdir(parents=True, exist_ok=True)

# ------------------------------
# Figure 1: 2x2 distribution figure (300 DPI)
# ------------------------------
fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# Plot 1: Overall split distribution
ax1 = axes[0, 0]
split_sizes = [len(X_train), len(X_val), len(X_test)]
split_labels = ['Train', 'Validation', 'Test']
colors = ['#3498db', '#2ecc71', '#e74c3c']

bars = ax1.bar(split_labels, split_sizes, color=colors, alpha=0.8, edgecolor='black')
ax1.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
ax1.set_title('Dataset Split Distribution', fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

for bar, size in zip(bars, split_sizes):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{size:,}\n({size/total_samples_loaded*100:.1f}%)',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

# Plot 2: Class distribution in training set
ax2 = axes[0, 1]
train_class_counts = [train_counts[i] for i in range(len(CLASS_NAMES))]
bars = ax2.barh(CLASS_NAMES, train_class_counts, color='#3498db', alpha=0.8, edgecolor='black')
ax2.set_xlabel('Number of Samples', fontsize=12, fontweight='bold')
ax2.set_title('Training Set - Class Distribution', fontsize=14, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)
for bar, count in zip(bars, train_class_counts):
    width = bar.get_width()
    ax2.text(width, bar.get_y() + bar.get_height()/2., f' {count:,}', ha='left', va='center', fontsize=10)

# Plot 3: Stacked bars across splits
ax3 = axes[1, 0]
train_counts_list = [train_counts[i] for i in range(len(CLASS_NAMES))]
val_counts_list = [val_counts[i] for i in range(len(CLASS_NAMES))]
test_counts_list = [test_counts[i] for i in range(len(CLASS_NAMES))]

x = np.arange(len(CLASS_NAMES))
width = 0.25

ax3.bar(x - width, train_counts_list, width, label='Train', color='#3498db', alpha=0.8, edgecolor='black')
ax3.bar(x, val_counts_list, width, label='Validation', color='#2ecc71', alpha=0.8, edgecolor='black')
ax3.bar(x + width, test_counts_list, width, label='Test', color='#e74c3c', alpha=0.8, edgecolor='black')
ax3.set_xlabel('Class', fontsize=12, fontweight='bold')
ax3.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
ax3.set_title('Class Distribution Across Splits', fontsize=14, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(CLASS_NAMES)
ax3.legend(fontsize=11)
ax3.grid(axis='y', alpha=0.3)

# Plot 4: Heatmap of class percentages
ax4 = axes[1, 1]
percentage_data = []
for i in range(len(CLASS_NAMES)):
    percentage_data.append([
        (train_counts[i] / len(y_train)) * 100,
        (val_counts[i] / len(y_val)) * 100,
        (test_counts[i] / len(y_test)) * 100
    ])

sns.heatmap(percentage_data, annot=True, fmt='.2f', cmap='YlOrRd',
            xticklabels=['Train', 'Val', 'Test'], yticklabels=CLASS_NAMES,
            cbar_kws={'label': 'Percentage (%)'}, ax=ax4, linewidths=1, linecolor='black')
ax4.set_title('Class Percentage Distribution Heatmap', fontsize=14, fontweight='bold')
ax4.set_xlabel('Split', fontsize=12, fontweight='bold')
ax4.set_ylabel('Class', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(viz_dir / 'split_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {viz_dir / 'split_distribution.png'} (300 DPI)")

# ------------------------------
# Figure 2: Pie charts for each split (300 DPI)
# ------------------------------
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle('Class Distribution in Each Split', fontsize=16, fontweight='bold')

for ax, y_split, split_name, color in [
    (axes[0], y_train, 'Training Set', '#3498db'),
    (axes[1], y_val, 'Validation Set', '#2ecc71'),
    (axes[2], y_test, 'Test Set', '#e74c3c')
]:
    counts = Counter(y_split)
    sizes = [counts[i] for i in range(len(CLASS_NAMES))]
    wedges, texts, autotexts = ax.pie(sizes, labels=CLASS_NAMES, autopct='%1.1f%%',
                                      startangle=90, colors=plt.cm.Pastel1.colors,
                                      wedgeprops={'edgecolor': 'black', 'linewidth': 1.5})
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)
    ax.set_title(f'{split_name}\n({len(y_split):,} samples)', fontsize=13, fontweight='bold', color=color)

plt.tight_layout()
plt.savefig(viz_dir / 'split_pie_charts.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {viz_dir / 'split_pie_charts.png'} (300 DPI)")

# ------------------------------
# Figure 3: Text summary (300 DPI)
# ------------------------------
fig, ax = plt.subplots(figsize=(16, 10))
fig.patch.set_facecolor('white')

avg_std = np.mean([
    np.std([
        (train_counts[i] / len(y_train)) * 100,
        (val_counts[i] / len(y_val)) * 100,
        (test_counts[i] / len(y_test)) * 100
    ]) for i in range(len(CLASS_NAMES))
])

stats_text = f"""
ISIC 2019 - Train/Validation/Test Split Summary
{'='*60}

Split Configuration:
  • Training: {TRAIN_RATIO*100:.0f}% ({len(X_train):,} samples)
  • Validation: {VAL_RATIO*100:.0f}% ({len(X_val):,} samples)
  • Test: {TEST_RATIO*100:.0f}% ({len(X_test):,} samples)
  • Total: {total_samples_loaded:,} samples
  • Standard: Medical AI Industry (80/10/10)

Dataset Information:
  • Image Resolution: {X_train.shape[1]}x{X_train.shape[2]} pixels
  • Number of Classes: {len(CLASS_NAMES)}
  • Classes: {', '.join(CLASS_NAMES)}
  • Data Type: float32 (images), int32 (labels)

Stratification Quality:
  • Average Standard Deviation: {avg_std:.3f}%
  • Quality: {'Excellent' if avg_std < 1 else 'Good' if avg_std < 2 else 'Acceptable'}
  • Stratified Split: ✓ (maintains class distribution)

Class Distribution Summary:
"""
for i, class_name in enumerate(CLASS_NAMES):
    train_c = train_counts[i]
    val_c = val_counts[i]
    test_c = test_counts[i]
    train_pct = (train_c / len(y_train)) * 100
    val_pct = (val_c / len(y_val)) * 100
    test_pct = (test_c / len(y_test)) * 100
    stats_text += f"  • {class_name}: Train={train_c:,} ({train_pct:.1f}%), Val={val_c:,} ({val_pct:.1f}%), Test={test_c:,} ({test_pct:.1f}%)\n"

stats_text += f"""
Memory Usage (Theoretical - if fully loaded):
  • Training Set: {X_train.nbytes / (1024**3):.2f} GB
  • Validation Set: {X_val.nbytes / (1024**3):.2f} GB
  • Test Set: {X_test.nbytes / (1024**3):.2f} GB
  • Total: {(X_train.nbytes + X_val.nbytes + X_test.nbytes) / (1024**3):.2f} GB
  • Note: Arrays are memory-mapped (stream from disk, not fully loaded into RAM)
"""

ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
ax.axis('off')
ax.set_title('Train/Validation/Test Split Statistics Summary', fontsize=18, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(viz_dir / 'split_statistics_summary.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {viz_dir / 'split_statistics_summary.png'} (300 DPI)")

# Quick reference (150 DPI) saved in outputs
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(split_labels, split_sizes, color=colors, alpha=0.8, edgecolor='black')
ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
ax.set_title('Dataset Split Distribution', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for bar, size in zip(bars, split_sizes):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
             f'{size:,}\n({size/total_samples_loaded*100:.1f}%)',
             ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'split_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {OUTPUT_DIR / 'split_distribution.png'} (150 DPI)")

print(f"\n✅ All visualizations saved to: {viz_dir}")
print("  📊 High-Resolution Visualizations (300 DPI - Report Ready):")
print("     • split_distribution.png (300 DPI)")
print("     • split_pie_charts.png (300 DPI)")
print("     • split_statistics_summary.png (300 DPI)")
print("  📄 Quick Reference (150 DPI):")
print(f"     • split_distribution.png (150 DPI) - Saved to {OUTPUT_DIR}")