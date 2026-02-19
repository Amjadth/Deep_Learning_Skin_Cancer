# ============================================
# QUICK WEEK 4 SPLIT VALIDATION TEST
# ============================================
#
# Quick validation script for basic integrity checks.
# Runs faster than the full test suite.
#
# Tests:
# 1. File existence
# 2. File integrity (can be opened)
# 3. Shape consistency
# 4. Data types
# 5. Label ranges
# 6. Sample count verification
#
# Author: Deep Learning Engineer
# Date: 2024
# ============================================

import numpy as np
from pathlib import Path
import os
import sys

# Configuration
CLASS_NAMES = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']
NUM_CLASSES = len(CLASS_NAMES)
EXPECTED_IMAGE_SHAPE = (600, 600, 3)
EXPECTED_IMAGE_DTYPE = np.float32
EXPECTED_LABEL_DTYPE = np.int32

# Environment detection
BASE_DIR = Path(os.getcwd())
if Path('/workspace').exists():
    BASE_DIR = Path('/workspace')
elif Path('/notebooks').exists():
    BASE_DIR = Path('/notebooks')

NETWORK_VOLUME = None
if os.path.exists("/runpod-volume"):
    NETWORK_VOLUME = Path("/runpod-volume")
elif os.path.exists("/workspace"):
    NETWORK_VOLUME = Path("/workspace")

STORAGE_BASE = NETWORK_VOLUME if NETWORK_VOLUME else BASE_DIR
OUTPUT_DIR = (STORAGE_BASE / 'outputs').resolve()

print("=" * 70)
print("QUICK WEEK 4 SPLIT VALIDATION")
print("=" * 70)
print(f"📁 Output Directory: {OUTPUT_DIR}")
print("=" * 70)

# Test results
errors = []
warnings = []

def error(msg):
    errors.append(msg)
    print(f"❌ ERROR: {msg}")

def warning(msg):
    warnings.append(msg)
    print(f"⚠️  WARNING: {msg}")

def success(msg):
    print(f"✅ {msg}")

# Test 1: File existence
print("\n📂 Test 1: File Existence")
print("-" * 70)

required_files = {
    'X_train': OUTPUT_DIR / 'X_train.npy',
    'y_train': OUTPUT_DIR / 'y_train.npy',
    'X_val': OUTPUT_DIR / 'X_val.npy',
    'y_val': OUTPUT_DIR / 'y_val.npy',
    'X_test': OUTPUT_DIR / 'X_test.npy',
    'y_test': OUTPUT_DIR / 'y_test.npy',
}

all_exist = True
for name, path in required_files.items():
    if path.exists():
        size_gb = path.stat().st_size / (1024**3)
        success(f"{name}: {path.name} ({size_gb:.2f} GB)")
    else:
        error(f"{name}: Missing - {path}")
        all_exist = False

if not all_exist:
    print("\n❌ CRITICAL: Missing files. Cannot proceed.")
    sys.exit(1)

# Test 2: File integrity
print("\n🔍 Test 2: File Integrity")
print("-" * 70)

try:
    # First, try to load labels to get the sample counts
    # Labels are small and saved with np.save(), so they should load easily
    try:
        y_train = np.load(required_files['y_train'], allow_pickle=False)
        y_val = np.load(required_files['y_val'], allow_pickle=False)
        y_test = np.load(required_files['y_test'], allow_pickle=False)
        success("Labels loaded (y_train, y_val, y_test)")
    except:
        # Fallback: try with pickle allowed
        y_train = np.load(required_files['y_train'], allow_pickle=True)
        y_val = np.load(required_files['y_val'], allow_pickle=True)
        y_test = np.load(required_files['y_test'], allow_pickle=True)
        success("Labels loaded with pickle allowed")
    
    # Now load image files
    # These are memmap files created by week4.py
    # We know the image dimensions (600, 600, 3) and dtype (float32)
    # We can infer the number of samples from file size or use the label counts
    
    def load_image_memmap(file_path, num_samples, image_shape, dtype):
        """Load image memmap file with known shape."""
        full_shape = (num_samples,) + image_shape
        try:
            # Try loading as memmap with known shape
            data = np.memmap(file_path, dtype=dtype, mode='r', shape=full_shape)
            # Verify it works by checking shape
            assert data.shape == full_shape, f"Shape mismatch: expected {full_shape}, got {data.shape}"
            return data, 'memmap_known_shape'
        except Exception as e:
            # Fallback: Try to infer shape from file size
            file_size = file_path.stat().st_size
            bytes_per_element = np.dtype(dtype).itemsize
            total_elements = file_size // bytes_per_element
            elements_per_image = np.prod(image_shape)
            inferred_samples = total_elements // elements_per_image
            
            if inferred_samples * elements_per_image == total_elements:
                inferred_shape = (inferred_samples,) + image_shape
                data = np.memmap(file_path, dtype=dtype, mode='r', shape=inferred_shape)
                return data, f'memmap_inferred_{inferred_samples}_samples'
            else:
                raise Exception(f"Cannot infer shape from file size. File: {file_size} bytes, "
                              f"Expected elements: {num_samples * elements_per_image}, "
                              f"Got: {total_elements}. Error: {str(e)}")
    
    # Load image files using label counts to determine shape
    X_train, strategy_x_train = load_image_memmap(
        required_files['X_train'], len(y_train), EXPECTED_IMAGE_SHAPE, EXPECTED_IMAGE_DTYPE)
    success(f"X_train loaded ({strategy_x_train}): {X_train.shape}")
    
    X_val, strategy_x_val = load_image_memmap(
        required_files['X_val'], len(y_val), EXPECTED_IMAGE_SHAPE, EXPECTED_IMAGE_DTYPE)
    success(f"X_val loaded ({strategy_x_val}): {X_val.shape}")
    
    X_test, strategy_x_test = load_image_memmap(
        required_files['X_test'], len(y_test), EXPECTED_IMAGE_SHAPE, EXPECTED_IMAGE_DTYPE)
    success(f"X_test loaded ({strategy_x_test}): {X_test.shape}")
    
    success("All files can be opened")
except Exception as e:
    error(f"Cannot open files: {str(e)}")
    import traceback
    print(f"\nDetailed error:")
    traceback.print_exc()
    sys.exit(1)

# Test 3: Shape consistency
print("\n📐 Test 3: Shape Consistency")
print("-" * 70)

# Check sample counts
if len(X_train) == len(y_train):
    success(f"Train: {len(X_train):,} images, {len(y_train):,} labels")
else:
    error(f"Train mismatch: {len(X_train):,} images vs {len(y_train):,} labels")

if len(X_val) == len(y_val):
    success(f"Val: {len(X_val):,} images, {len(y_val):,} labels")
else:
    error(f"Val mismatch: {len(X_val):,} images vs {len(y_val):,} labels")

if len(X_test) == len(y_test):
    success(f"Test: {len(X_test):,} images, {len(y_test):,} labels")
else:
    error(f"Test mismatch: {len(X_test):,} images vs {len(y_test):,} labels")

# Check image shapes
train_shape = X_train.shape[1:]
val_shape = X_val.shape[1:]
test_shape = X_test.shape[1:]

if train_shape == EXPECTED_IMAGE_SHAPE:
    success(f"Train image shape: {train_shape}")
else:
    error(f"Train image shape: {train_shape}, Expected: {EXPECTED_IMAGE_SHAPE}")

if val_shape == EXPECTED_IMAGE_SHAPE:
    success(f"Val image shape: {val_shape}")
else:
    error(f"Val image shape: {val_shape}, Expected: {EXPECTED_IMAGE_SHAPE}")

if test_shape == EXPECTED_IMAGE_SHAPE:
    success(f"Test image shape: {test_shape}")
else:
    error(f"Test image shape: {test_shape}, Expected: {EXPECTED_IMAGE_SHAPE}")

# Test 4: Data types
print("\n🔢 Test 4: Data Types")
print("-" * 70)

if X_train.dtype == EXPECTED_IMAGE_DTYPE:
    success(f"X_train dtype: {X_train.dtype}")
else:
    error(f"X_train dtype: {X_train.dtype}, Expected: {EXPECTED_IMAGE_DTYPE}")

if y_train.dtype == EXPECTED_LABEL_DTYPE:
    success(f"y_train dtype: {y_train.dtype}")
else:
    error(f"y_train dtype: {y_train.dtype}, Expected: {EXPECTED_LABEL_DTYPE}")

# Test 5: Label ranges
print("\n🏷️  Test 5: Label Ranges")
print("-" * 70)

train_min, train_max = np.min(y_train), np.max(y_train)
val_min, val_max = np.min(y_val), np.max(y_val)
test_min, test_max = np.min(y_test), np.max(y_test)

if train_min == 0 and train_max == NUM_CLASSES - 1:
    success(f"Train labels: [{train_min}, {train_max}]")
else:
    error(f"Train labels: [{train_min}, {train_max}], Expected: [0, {NUM_CLASSES-1}]")

if val_min == 0 and val_max == NUM_CLASSES - 1:
    success(f"Val labels: [{val_min}, {val_max}]")
else:
    error(f"Val labels: [{val_min}, {val_max}], Expected: [0, {NUM_CLASSES-1}]")

if test_min == 0 and test_max == NUM_CLASSES - 1:
    success(f"Test labels: [{test_min}, {test_max}]")
else:
    error(f"Test labels: [{test_min}, {test_max}], Expected: [0, {NUM_CLASSES-1}]")

# Test 6: Sample counts
print("\n📊 Test 6: Sample Counts")
print("-" * 70)

total_samples = len(y_train) + len(y_val) + len(y_test)
success(f"Total samples: {total_samples:,}")

# Check split ratios (approximately 80/10/10)
train_ratio = len(y_train) / total_samples
val_ratio = len(y_val) / total_samples
test_ratio = len(y_test) / total_samples

print(f"   Train: {len(y_train):,} ({train_ratio*100:.1f}%)")
print(f"   Val: {len(y_val):,} ({val_ratio*100:.1f}%)")
print(f"   Test: {len(y_test):,} ({test_ratio*100:.1f}%)")

if 0.75 <= train_ratio <= 0.85:
    success("Train ratio is reasonable (~80%)")
else:
    warning(f"Train ratio: {train_ratio*100:.1f}% (expected ~80%)")

if 0.05 <= val_ratio <= 0.15:
    success("Val ratio is reasonable (~10%)")
else:
    warning(f"Val ratio: {val_ratio*100:.1f}% (expected ~10%)")

if 0.05 <= test_ratio <= 0.15:
    success("Test ratio is reasonable (~10%)")
else:
    warning(f"Test ratio: {test_ratio*100:.1f}% (expected ~10%)")

# Test 7: Class distribution (quick check)
print("\n📈 Test 7: Class Distribution (Quick Check)")
print("-" * 70)

from collections import Counter
train_counts = Counter(y_train)
val_counts = Counter(y_val)
test_counts = Counter(y_test)

print(f"{'Class':<8} {'Train':<10} {'Val':<10} {'Test':<10}")
print("-" * 40)
for i, class_name in enumerate(CLASS_NAMES):
    print(f"{class_name:<8} {train_counts[i]:<10} {val_counts[i]:<10} {test_counts[i]:<10}")

# Check if all classes are present
all_classes_present = True
for i in range(NUM_CLASSES):
    if train_counts[i] == 0 or val_counts[i] == 0 or test_counts[i] == 0:
        error(f"Class {CLASS_NAMES[i]} missing in one or more splits")
        all_classes_present = False

if all_classes_present:
    success("All classes present in all splits")

# Test 8: Quick corruption check (sample)
print("\n🔬 Test 8: Quick Corruption Check (Sample)")
print("-" * 70)

np.random.seed(42)
sample_size = 100

for split_name, X_split in [('Train', X_train), ('Val', X_val), ('Test', X_test)]:
    try:
        n_samples = min(sample_size, X_split.shape[0])
        sample_indices = np.random.choice(X_split.shape[0], size=n_samples, replace=False)
        sample = X_split[sample_indices]
        
        nan_count = np.isnan(sample).sum()
        inf_count = np.isinf(sample).sum()
        
        if nan_count == 0 and inf_count == 0:
            success(f"{split_name}: No corruption in {n_samples} samples")
        else:
            error(f"{split_name}: NaN: {nan_count}, Inf: {inf_count} in {n_samples} samples")
    except Exception as e:
        error(f"{split_name}: Error checking corruption: {str(e)}")

# Summary
print("\n" + "=" * 70)
print("QUICK VALIDATION SUMMARY")
print("=" * 70)

if len(errors) == 0:
    print("✅ ALL TESTS PASSED!")
    if len(warnings) > 0:
        print(f"⚠️  {len(warnings)} warning(s) - review above")
    print("\n💡 Your dataset splits appear to be valid.")
    print("   For comprehensive validation, run: python test_week4_splits.py")
    sys.exit(0)
else:
    print(f"❌ VALIDATION FAILED: {len(errors)} error(s)")
    print("\nErrors:")
    for err in errors:
        print(f"   - {err}")
    if len(warnings) > 0:
        print(f"\n⚠️  {len(warnings)} warning(s):")
        for warn in warnings:
            print(f"   - {warn}")
    sys.exit(1)

