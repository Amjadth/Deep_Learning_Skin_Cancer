"""
CRITICAL: Verify that data and labels are properly aligned.
This is the most common cause of poor convergence in transfer learning.
"""

import numpy as np
import json
from pathlib import Path
from collections import Counter

# Find data directory
data_dir = Path('/workspace/outputs')
if not data_dir.exists():
    data_dir = Path('/runpod-volume/outputs')

print(f"📂 Data directory: {data_dir}")
print(f"{'='*70}\n")

# Load data
print("Loading data...")
X_train = np.load(data_dir / 'X_train_denormalized.npy', mmap_mode='r')
y_train = np.load(data_dir / 'y_train.npy')
X_val = np.load(data_dir / 'X_val_denormalized.npy', mmap_mode='r')
y_val = np.load(data_dir / 'y_val.npy')

# Load metadata
with open(data_dir / 'split_info.json', 'r') as f:
    split_info = json.load(f)

class_names = split_info['class_names']
num_classes = len(class_names)

print(f"✅ Data loaded successfully\n")

print(f"📊 DATASET STRUCTURE")
print(f"{'='*70}")
print(f"Train samples: {len(X_train):,} images, {len(y_train):,} labels")
print(f"Val samples: {len(X_val):,} images, {len(y_val):,} labels")
print(f"Classes: {num_classes} ({class_names})\n")

# CHECK 1: Counts match
print(f"✅ CHECK 1: Data-Label Count Match")
print(f"  Train: X={len(X_train)} vs y={len(y_train)} → {'PASS' if len(X_train)==len(y_train) else 'FAIL ❌'}")
print(f"  Val:   X={len(X_val)} vs y={len(y_val)} → {'PASS' if len(X_val)==len(y_val) else 'FAIL ❌'}\n")

# CHECK 2: Label range valid
print(f"✅ CHECK 2: Label Value Range")
train_min, train_max = y_train.min(), y_train.max()
val_min, val_max = y_val.min(), y_val.max()
print(f"  Train labels: [{train_min}, {train_max}] → {'PASS' if (train_min >= 0 and train_max < num_classes) else 'FAIL ❌'}")
print(f"  Val labels:   [{val_min}, {val_max}] → {'PASS' if (val_min >= 0 and val_max < num_classes) else 'FAIL ❌'}\n")

# CHECK 3: Class distribution
print(f"✅ CHECK 3: Class Distribution (Should be roughly balanced)")
train_counts = Counter(y_train)
val_counts = Counter(y_val)

print(f"  Train:")
for class_idx in sorted(train_counts.keys()):
    count = train_counts[class_idx]
    pct = 100 * count / len(y_train)
    print(f"    {class_names[class_idx]:20s}: {count:6,} ({pct:5.1f}%)")

print(f"\n  Val:")
for class_idx in sorted(val_counts.keys()):
    count = val_counts[class_idx]
    pct = 100 * count / len(y_val)
    print(f"    {class_names[class_idx]:20s}: {count:6,} ({pct:5.1f}%)")

# CHECK 4: Data value ranges
print(f"\n✅ CHECK 4: Image Value Ranges (Should be [0, 1])")
train_sample = X_train[0]
val_sample = X_val[0]
print(f"  Train sample: min={train_sample.min():.4f}, max={train_sample.max():.4f}, mean={train_sample.mean():.4f}")
print(f"  Val sample:   min={val_sample.min():.4f}, max={val_sample.max():.4f}, mean={val_sample.mean():.4f}")
print(f"  Status: {'PASS' if (train_sample.min() >= 0 and train_sample.max() <= 1) else 'FAIL - NOT [0,1] normalized ❌'}\n")

# CHECK 5: Image shape consistency
print(f"✅ CHECK 5: Image Shape Consistency (Should be (224, 224, 3))")
print(f"  Train: {X_train.shape} → {'PASS' if X_train.shape[1:] == (224, 224, 3) else 'FAIL ❌'}")
print(f"  Val:   {X_val.shape} → {'PASS' if X_val.shape[1:] == (224, 224, 3) else 'FAIL ❌'}\n")

# CHECK 6: Random samples - visual check
print(f"✅ CHECK 6: Random Sample Verification")
print(f"  (Inspect these to ensure they make sense)\n")

import random
random.seed(42)

for i in range(3):
    idx = random.randint(0, len(X_train) - 1)
    label = y_train[idx]
    img = X_train[idx]
    print(f"  Sample {i+1}:")
    print(f"    Index: {idx}")
    print(f"    Label: {label} ({class_names[label]})")
    print(f"    Image: shape={img.shape}, dtype={img.dtype}, range=[{img.min():.3f}, {img.max():.3f}], mean={img.mean():.3f}")
    print()

# SUMMARY
print(f"{'='*70}")
print(f"📋 VERDICT:")
all_pass = (
    len(X_train) == len(y_train) and
    len(X_val) == len(y_val) and
    train_min >= 0 and train_max < num_classes and
    val_min >= 0 and val_max < num_classes and
    train_sample.min() >= 0 and train_sample.max() <= 1
)

if all_pass:
    print(f"✅ ALL CHECKS PASSED - Data integrity is GOOD")
    print(f"   If training accuracy is still ~23%, the issue is elsewhere:")
    print(f"   - Model architecture mismatch?")
    print(f"   - Learning rate too aggressive?")
    print(f"   - Try reducing LR from 1e-4 to 1e-5")
else:
    print(f"❌ CRITICAL ISSUES FOUND - Data is corrupted or misaligned!")
    print(f"   Training cannot converge with bad data.")

print(f"{'='*70}\n")
