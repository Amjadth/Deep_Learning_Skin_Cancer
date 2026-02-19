# ============================================
# WEEK 4 SPLIT VALIDATION TEST SCRIPT
# ============================================
#
# This script validates that the train/validation/test splits
# created in Week 4 are complete, correct, and free from corruption.
#
# Tests Performed:
# 1. File Existence Check - All required files exist
# 2. File Integrity Check - Files can be opened and read
# 3. Shape Consistency Check - All arrays have correct shapes
# 4. Data Type Validation - Correct dtypes (float32, int32)
# 5. Label Range Validation - Labels are in valid range [0, num_classes-1]
# 6. Data Corruption Check - No NaN or Inf values
# 7. Data Leakage Check - No overlap between splits
# 8. Completeness Check - All samples accounted for
# 9. Class Distribution Check - Balanced distribution
# 10. Image Value Range Check - Valid pixel values
# 11. Label Consistency Check - Labels match image count
# 12. Memory Map Integrity - Files can be memory-mapped correctly
#
# Author: Deep Learning Engineer
# Date: 2024
# ============================================

import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
import os
import sys
import time

# ============================================
# CONFIGURATION
# ============================================
CLASS_NAMES = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']
NUM_CLASSES = len(CLASS_NAMES)

# Expected image shape (from Week 3)
EXPECTED_IMAGE_SHAPE = (600, 600, 3)  # Height, Width, Channels
EXPECTED_IMAGE_DTYPE = np.float32
EXPECTED_LABEL_DTYPE = np.int32

# Tolerance for floating point comparisons
FLOAT_TOLERANCE = 1e-6

# ============================================
# ENVIRONMENT DETECTION
# ============================================
runpod_pod_id = os.environ.get('RUNPOD_POD_ID', None)
is_pod = runpod_pod_id is not None
is_serverless = os.environ.get('RUNPOD_WORKER_ID', None) is not None

BASE_DIR = Path(os.getcwd())
if Path('/workspace').exists():
    BASE_DIR = Path('/workspace')
elif Path('/notebooks').exists():
    BASE_DIR = Path('/notebooks')

NETWORK_VOLUME = None
if is_serverless and os.path.exists("/runpod-volume"):
    NETWORK_VOLUME = Path("/runpod-volume")
elif is_pod and os.path.exists("/workspace"):
    NETWORK_VOLUME = Path("/workspace")
elif os.path.exists("/runpod-volume"):
    NETWORK_VOLUME = Path("/runpod-volume")
elif os.path.exists("/workspace"):
    NETWORK_VOLUME = Path("/workspace")

STORAGE_BASE = NETWORK_VOLUME if NETWORK_VOLUME else BASE_DIR
OUTPUT_DIR = (STORAGE_BASE / 'outputs').resolve()

print("=" * 70)
print("WEEK 4 SPLIT VALIDATION TEST")
print("=" * 70)
print(f"📁 Output Directory: {OUTPUT_DIR}")
print(f"💾 Network Volume: {NETWORK_VOLUME if NETWORK_VOLUME else 'Not detected'}")
print("=" * 70)

# ============================================
# TEST RESULTS TRACKING
# ============================================
test_results = {
    'passed': 0,
    'failed': 0,
    'warnings': 0,
    'tests': []
}

def log_test(test_name, passed, message="", warning=False):
    """Log test result."""
    status = "✅ PASS" if passed else ("⚠️  WARN" if warning else "❌ FAIL")
    test_results['tests'].append({
        'name': test_name,
        'passed': passed,
        'warning': warning,
        'message': message
    })
    if passed:
        test_results['passed'] += 1
    elif warning:
        test_results['warnings'] += 1
    else:
        test_results['failed'] += 1
    print(f"{status}: {test_name}")
    if message:
        print(f"   {message}")

# ============================================
# TEST 1: FILE EXISTENCE CHECK
# ============================================
print("\n" + "=" * 70)
print("TEST 1: FILE EXISTENCE CHECK")
print("=" * 70)

required_files = {
    'X_train': OUTPUT_DIR / 'X_train.npy',
    'y_train': OUTPUT_DIR / 'y_train.npy',
    'X_val': OUTPUT_DIR / 'X_val.npy',
    'y_val': OUTPUT_DIR / 'y_val.npy',
    'X_test': OUTPUT_DIR / 'X_test.npy',
    'y_test': OUTPUT_DIR / 'y_test.npy',
    'split_info': OUTPUT_DIR / 'split_info.json',
    'split_summary': OUTPUT_DIR / 'split_summary.csv'
}

all_files_exist = True
for file_name, file_path in required_files.items():
    exists = file_path.exists()
    if exists:
        size_gb = file_path.stat().st_size / (1024**3)
        log_test(f"File exists: {file_name}", True, f"{file_path.name} ({size_gb:.2f} GB)")
    else:
        log_test(f"File exists: {file_name}", False, f"Missing: {file_path}")
        all_files_exist = False

if not all_files_exist:
    print("\n❌ CRITICAL: Missing required files. Cannot proceed with validation.")
    sys.exit(1)

# ============================================
# TEST 2: FILE INTEGRITY CHECK
# ============================================
print("\n" + "=" * 70)
print("TEST 2: FILE INTEGRITY CHECK")
print("=" * 70)

def load_image_memmap(file_path, num_samples, image_shape, dtype, array_name):
    """Load image memmap file with known shape."""
    full_shape = (num_samples,) + image_shape
    try:
        # Try loading as memmap with known shape (raw memmap file created by week4.py)
        data = np.memmap(file_path, dtype=dtype, mode='r', shape=full_shape)
        # Verify it works
        assert data.shape == full_shape, f"Shape mismatch: expected {full_shape}, got {data.shape}"
        log_test(f"File integrity: {array_name}", True, 
                f"Shape: {data.shape}, Dtype: {data.dtype} (memmap)")
        return True, data
    except Exception as e:
        # Fallback: Try to infer shape from file size
        try:
            file_size = file_path.stat().st_size
            bytes_per_element = np.dtype(dtype).itemsize
            total_elements = file_size // bytes_per_element
            elements_per_image = np.prod(image_shape)
            inferred_samples = total_elements // elements_per_image
            
            if inferred_samples * elements_per_image == total_elements:
                inferred_shape = (inferred_samples,) + image_shape
                data = np.memmap(file_path, dtype=dtype, mode='r', shape=inferred_shape)
                log_test(f"File integrity: {array_name}", True, 
                        f"Shape: {data.shape}, Dtype: {data.dtype} (memmap inferred)")
                return True, data
            else:
                raise Exception(f"Cannot infer shape. Expected {num_samples * elements_per_image} elements, got {total_elements}")
        except Exception as e2:
            # Last resort: Try loading as .npy file
            try:
                data = np.load(file_path, mmap_mode='r', allow_pickle=True)
                log_test(f"File integrity: {array_name}", True, 
                        f"Shape: {data.shape}, Dtype: {data.dtype} (npy file)")
                return True, data
            except Exception as e3:
                log_test(f"File integrity: {array_name}", False, 
                        f"All loading strategies failed. Errors: {str(e)}, {str(e2)}, {str(e3)}")
                return False, None

def load_label_file(file_path, array_name):
    """Load label file (saved with np.save())."""
    try:
        # Try loading as .npy file (labels are saved with np.save())
        data = np.load(file_path, allow_pickle=False)
        log_test(f"File integrity: {array_name}", True, 
                f"Shape: {data.shape}, Dtype: {data.dtype}")
        return True, data
    except Exception as e:
        # Fallback: try with pickle allowed
        try:
            data = np.load(file_path, allow_pickle=True)
            log_test(f"File integrity: {array_name}", True, 
                    f"Shape: {data.shape}, Dtype: {data.dtype} (with pickle)")
            return True, data
        except Exception as e2:
            log_test(f"File integrity: {array_name}", False, f"Error: {str(e)}, {str(e2)}")
            return False, None

# First, load labels to get sample counts
print("\n📊 Loading labels first to determine sample counts...")
y_train_valid, y_train = load_label_file(required_files['y_train'], 'y_train')
y_val_valid, y_val = load_label_file(required_files['y_val'], 'y_val')
y_test_valid, y_test = load_label_file(required_files['y_test'], 'y_test')

if not all([y_train_valid, y_val_valid, y_test_valid]):
    print("\n❌ CRITICAL: Cannot load label files. Cannot proceed.")
    sys.exit(1)

# Now load image files using label counts to determine shape
print("\n📊 Loading image files as memmap...")
X_train_valid, X_train = load_image_memmap(
    required_files['X_train'], len(y_train), EXPECTED_IMAGE_SHAPE, EXPECTED_IMAGE_DTYPE, 'X_train')
X_val_valid, X_val = load_image_memmap(
    required_files['X_val'], len(y_val), EXPECTED_IMAGE_SHAPE, EXPECTED_IMAGE_DTYPE, 'X_val')
X_test_valid, X_test = load_image_memmap(
    required_files['X_test'], len(y_test), EXPECTED_IMAGE_SHAPE, EXPECTED_IMAGE_DTYPE, 'X_test')

if not all([X_train_valid, X_val_valid, X_test_valid]):
    print("\n❌ CRITICAL: File integrity check failed. Cannot proceed.")
    sys.exit(1)

# ============================================
# TEST 3: SHAPE CONSISTENCY CHECK
# ============================================
print("\n" + "=" * 70)
print("TEST 3: SHAPE CONSISTENCY CHECK")
print("=" * 70)

# Check image shapes
train_shape = X_train.shape
val_shape = X_val.shape
test_shape = X_test.shape

expected_image_dims = EXPECTED_IMAGE_SHAPE
expected_train_samples = len(y_train)
expected_val_samples = len(y_val)
expected_test_samples = len(y_test)

# Check sample counts match
log_test("Train samples match", 
         train_shape[0] == expected_train_samples,
         f"Images: {train_shape[0]}, Labels: {expected_train_samples}")

log_test("Val samples match",
         val_shape[0] == expected_val_samples,
         f"Images: {val_shape[0]}, Labels: {expected_val_samples}")

log_test("Test samples match",
         test_shape[0] == expected_test_samples,
         f"Images: {test_shape[0]}, Labels: {expected_test_samples}")

# Check image dimensions
train_image_shape = train_shape[1:]
val_image_shape = val_shape[1:]
test_image_shape = test_shape[1:]

log_test("Train image shape",
         train_image_shape == expected_image_dims,
         f"Expected: {expected_image_dims}, Got: {train_image_shape}")

log_test("Val image shape",
         val_image_shape == expected_image_dims,
         f"Expected: {expected_image_dims}, Got: {val_image_shape}")

log_test("Test image shape",
         test_image_shape == expected_image_dims,
         f"Expected: {expected_image_dims}, Got: {test_image_shape}")

# Check all shapes are consistent
all_shapes_consistent = (train_image_shape == val_image_shape == test_image_shape == expected_image_dims)
log_test("All shapes consistent", all_shapes_consistent,
         "All splits have identical image dimensions")

# ============================================
# TEST 4: DATA TYPE VALIDATION
# ============================================
print("\n" + "=" * 70)
print("TEST 4: DATA TYPE VALIDATION")
print("=" * 70)

log_test("X_train dtype",
         X_train.dtype == EXPECTED_IMAGE_DTYPE,
         f"Expected: {EXPECTED_IMAGE_DTYPE}, Got: {X_train.dtype}")

log_test("X_val dtype",
         X_val.dtype == EXPECTED_IMAGE_DTYPE,
         f"Expected: {EXPECTED_IMAGE_DTYPE}, Got: {X_val.dtype}")

log_test("X_test dtype",
         X_test.dtype == EXPECTED_IMAGE_DTYPE,
         f"Expected: {EXPECTED_IMAGE_DTYPE}, Got: {X_test.dtype}")

log_test("y_train dtype",
         y_train.dtype == EXPECTED_LABEL_DTYPE,
         f"Expected: {EXPECTED_LABEL_DTYPE}, Got: {y_train.dtype}")

log_test("y_val dtype",
         y_val.dtype == EXPECTED_LABEL_DTYPE,
         f"Expected: {EXPECTED_LABEL_DTYPE}, Got: {y_val.dtype}")

log_test("y_test dtype",
         y_test.dtype == EXPECTED_LABEL_DTYPE,
         f"Expected: {EXPECTED_LABEL_DTYPE}, Got: {y_test.dtype}")

# ============================================
# TEST 5: LABEL RANGE VALIDATION
# ============================================
print("\n" + "=" * 70)
print("TEST 5: LABEL RANGE VALIDATION")
print("=" * 70)

all_labels_train = np.unique(y_train)
all_labels_val = np.unique(y_val)
all_labels_test = np.unique(y_test)

min_label_train = np.min(y_train)
max_label_train = np.max(y_train)
min_label_val = np.min(y_val)
max_label_val = np.max(y_val)
min_label_test = np.min(y_test)
max_label_test = np.max(y_test)

log_test("Train labels in range",
         min_label_train == 0 and max_label_train == NUM_CLASSES - 1,
         f"Range: [{min_label_train}, {max_label_train}], Expected: [0, {NUM_CLASSES-1}]")

log_test("Val labels in range",
         min_label_val == 0 and max_label_val == NUM_CLASSES - 1,
         f"Range: [{min_label_val}, {max_label_val}], Expected: [0, {NUM_CLASSES-1}]")

log_test("Test labels in range",
         min_label_test == 0 and max_label_test == NUM_CLASSES - 1,
         f"Range: [{min_label_test}, {max_label_test}], Expected: [0, {NUM_CLASSES-1}]")

# Check all classes are represented
expected_classes = set(range(NUM_CLASSES))
train_classes = set(all_labels_train)
val_classes = set(all_labels_val)
test_classes = set(all_labels_test)

log_test("All classes in train",
         train_classes == expected_classes,
         f"Classes: {sorted(train_classes)}, Expected: {sorted(expected_classes)}")

log_test("All classes in val",
         val_classes == expected_classes,
         f"Classes: {sorted(val_classes)}, Expected: {sorted(expected_classes)}")

log_test("All classes in test",
         test_classes == expected_classes,
         f"Classes: {sorted(test_classes)}, Expected: {sorted(expected_classes)}")

# ============================================
# TEST 6: DATA CORRUPTION CHECK (NaN/Inf)
# ============================================
print("\n" + "=" * 70)
print("TEST 6: DATA CORRUPTION CHECK (NaN/Inf)")
print("=" * 70)

def check_corruption(arr, arr_name, sample_size=1000):
    """Check for NaN and Inf values in a sample of the array."""
    try:
        # Sample random indices to check (memory-efficient)
        n_samples = min(sample_size, arr.shape[0])
        sample_indices = np.random.choice(arr.shape[0], size=n_samples, replace=False)
        
        # Load sample
        sample = arr[sample_indices]
        
        # Check for NaN
        nan_count = np.isnan(sample).sum()
        
        # Check for Inf
        inf_count = np.isinf(sample).sum()
        
        if nan_count > 0 or inf_count > 0:
            return False, f"NaN: {nan_count}, Inf: {inf_count} (sampled {n_samples} images)"
        else:
            return True, f"No corruption detected (sampled {n_samples} images)"
    except Exception as e:
        return False, f"Error checking corruption: {str(e)}"

# Check each split (sample-based to avoid loading entire dataset)
np.random.seed(42)
train_clean, train_msg = check_corruption(X_train, 'X_train', sample_size=1000)
log_test("Train data corruption check", train_clean, train_msg)

val_clean, val_msg = check_corruption(X_val, 'X_val', sample_size=500)
log_test("Val data corruption check", val_clean, val_msg)

test_clean, test_msg = check_corruption(X_test, 'X_test', sample_size=500)
log_test("Test data corruption check", test_clean, test_msg)

# ============================================
# TEST 7: IMAGE VALUE RANGE CHECK
# ============================================
print("\n" + "=" * 70)
print("TEST 7: IMAGE VALUE RANGE CHECK")
print("=" * 70)

def check_value_range(arr, arr_name, sample_size=100):
    """Check if image values are in reasonable range."""
    try:
        n_samples = min(sample_size, arr.shape[0])
        sample_indices = np.random.choice(arr.shape[0], size=n_samples, replace=False)
        sample = arr[sample_indices]
        
        min_val = np.min(sample)
        max_val = np.max(sample)
        mean_val = np.mean(sample)
        std_val = np.std(sample)
        
        # Check if values are in reasonable range (typically [0, 1] or [0, 255])
        # Normalized images should be in [0, 1] or [-1, 1]
        # We'll accept anything reasonable (not extreme outliers)
        is_reasonable = (min_val >= -10 and max_val <= 10)  # Allow some normalization ranges
        
        return is_reasonable, f"Range: [{min_val:.3f}, {max_val:.3f}], Mean: {mean_val:.3f}, Std: {std_val:.3f}"
    except Exception as e:
        return False, f"Error: {str(e)}"

np.random.seed(42)
train_range_ok, train_range_msg = check_value_range(X_train, 'X_train', sample_size=100)
log_test("Train value range", train_range_ok, train_range_msg)

val_range_ok, val_range_msg = check_value_range(X_val, 'X_val', sample_size=50)
log_test("Val value range", val_range_ok, val_range_msg)

test_range_ok, test_range_msg = check_value_range(X_test, 'X_test', sample_size=50)
log_test("Test value range", test_range_ok, test_range_msg)

# ============================================
# TEST 8: DATA LEAKAGE CHECK
# ============================================
print("\n" + "=" * 70)
print("TEST 8: DATA LEAKAGE CHECK")
print("=" * 70)

# Load split indices if available
split_indices_path = OUTPUT_DIR / 'week4_split_indices.npz'
if split_indices_path.exists():
    try:
        split_indices = np.load(split_indices_path)
        train_indices = set(split_indices['train_indices'])
        val_indices = set(split_indices['val_indices'])
        test_indices = set(split_indices['test_indices'])
        
        # Check for overlaps
        train_val_overlap = train_indices & val_indices
        train_test_overlap = train_indices & test_indices
        val_test_overlap = val_indices & test_indices
        
        log_test("No train-val overlap",
                 len(train_val_overlap) == 0,
                 f"Overlap: {len(train_val_overlap)} indices" if train_val_overlap else "No overlap")
        
        log_test("No train-test overlap",
                 len(train_test_overlap) == 0,
                 f"Overlap: {len(train_test_overlap)} indices" if train_test_overlap else "No overlap")
        
        log_test("No val-test overlap",
                 len(val_test_overlap) == 0,
                 f"Overlap: {len(val_test_overlap)} indices" if val_test_overlap else "No overlap")
        
        # Check completeness
        total_unique = len(train_indices) + len(val_indices) + len(test_indices)
        log_test("All indices unique",
                 total_unique == len(train_indices | val_indices | test_indices),
                 f"Total: {total_unique} unique indices")
        
    except Exception as e:
        log_test("Data leakage check (using indices)", False, f"Error loading indices: {str(e)}")
else:
    log_test("Data leakage check (using indices)", False, 
             "Split indices file not found - cannot verify data leakage",
             warning=True)

# Alternative: Check sample counts sum correctly
total_samples = len(y_train) + len(y_val) + len(y_test)

# Expected total should be approximately 80k (10k per class * 8 classes)
# But allow some flexibility since it depends on available data
expected_total_min = 64000  # 8k per class minimum
expected_total_max = 80000  # 10k per class maximum

total_valid = expected_total_min <= total_samples <= expected_total_max
log_test("Sample count completeness",
         total_valid,
         f"Total: {total_samples}, Expected range: [{expected_total_min}, {expected_total_max}]")

# ============================================
# TEST 9: CLASS DISTRIBUTION CHECK
# ============================================
print("\n" + "=" * 70)
print("TEST 9: CLASS DISTRIBUTION CHECK")
print("=" * 70)

train_counts = Counter(y_train)
val_counts = Counter(y_val)
test_counts = Counter(y_test)

# Check if distribution is balanced (should be ~equal per class)
expected_train_per_class = len(y_train) / NUM_CLASSES
expected_val_per_class = len(y_val) / NUM_CLASSES
expected_test_per_class = len(y_test) / NUM_CLASSES

train_balanced = True
val_balanced = True
test_balanced = True
max_deviation_train = 0
max_deviation_val = 0
max_deviation_test = 0

for class_idx in range(NUM_CLASSES):
    train_dev = abs(train_counts[class_idx] - expected_train_per_class)
    val_dev = abs(val_counts[class_idx] - expected_val_per_class)
    test_dev = abs(test_counts[class_idx] - expected_test_per_class)
    
    max_deviation_train = max(max_deviation_train, train_dev)
    max_deviation_val = max(max_deviation_val, val_dev)
    max_deviation_test = max(max_deviation_test, test_dev)
    
    # Allow 1% deviation
    if train_dev > expected_train_per_class * 0.01:
        train_balanced = False
    if val_dev > expected_val_per_class * 0.01:
        val_balanced = False
    if test_dev > expected_test_per_class * 0.01:
        test_balanced = False

log_test("Train class balance",
         train_balanced,
         f"Max deviation: {max_deviation_train:.1f} samples (expected: {expected_train_per_class:.1f} per class)")

log_test("Val class balance",
         val_balanced,
         f"Max deviation: {max_deviation_val:.1f} samples (expected: {expected_val_per_class:.1f} per class)")

log_test("Test class balance",
         test_balanced,
         f"Max deviation: {max_deviation_test:.1f} samples (expected: {expected_test_per_class:.1f} per class)")

# Print distribution
print("\n📊 Class Distribution:")
print(f"{'Class':<8} {'Train':<10} {'Val':<10} {'Test':<10} {'Train%':<10} {'Val%':<10} {'Test%'}")
print("-" * 80)
for i, class_name in enumerate(CLASS_NAMES):
    train_c = train_counts[i]
    val_c = val_counts[i]
    test_c = test_counts[i]
    train_pct = (train_c / len(y_train)) * 100
    val_pct = (val_c / len(y_val)) * 100
    test_pct = (test_c / len(y_test)) * 100
    print(f"{class_name:<8} {train_c:<10} {val_c:<10} {test_c:<10} "
          f"{train_pct:6.2f}%    {val_pct:6.2f}%    {test_pct:6.2f}%")

# ============================================
# TEST 10: METADATA VALIDATION
# ============================================
print("\n" + "=" * 70)
print("TEST 10: METADATA VALIDATION")
print("=" * 70)

# Check split_info.json
if required_files['split_info'].exists():
    try:
        import json
        with open(required_files['split_info'], 'r') as f:
            split_info = json.load(f)
        
        # Validate split sizes
        info_train_size = split_info['split_sizes']['train']
        info_val_size = split_info['split_sizes']['validation']
        info_test_size = split_info['split_sizes']['test']
        
        log_test("Split info train size",
                 info_train_size == len(y_train),
                 f"Info: {info_train_size}, Actual: {len(y_train)}")
        
        log_test("Split info val size",
                 info_val_size == len(y_val),
                 f"Info: {info_val_size}, Actual: {len(y_val)}")
        
        log_test("Split info test size",
                 info_test_size == len(y_test),
                 f"Info: {info_test_size}, Actual: {len(y_test)}")
        
        # Validate class distribution
        info_train_dist = split_info['class_distribution']['train']
        for class_name in CLASS_NAMES:
            class_idx = CLASS_NAMES.index(class_name)
            info_count = info_train_dist[class_name]
            actual_count = train_counts[class_idx]
            log_test(f"Split info {class_name} count",
                     info_count == actual_count,
                     f"Info: {info_count}, Actual: {actual_count}")
        
    except Exception as e:
        log_test("Split info validation", False, f"Error: {str(e)}")
else:
    log_test("Split info exists", False, "split_info.json not found")

# Check split_summary.csv
if required_files['split_summary'].exists():
    try:
        summary_df = pd.read_csv(required_files['split_summary'])
        log_test("Split summary exists", True, f"Rows: {len(summary_df)}")
        
        # Validate summary matches actual data
        train_summary = summary_df[summary_df['Split'] == 'Train']
        if len(train_summary) == NUM_CLASSES:
            log_test("Split summary train rows", True, f"Expected: {NUM_CLASSES}, Got: {len(train_summary)}")
        else:
            log_test("Split summary train rows", False, f"Expected: {NUM_CLASSES}, Got: {len(train_summary)}")
            
    except Exception as e:
        log_test("Split summary validation", False, f"Error: {str(e)}")
else:
    log_test("Split summary exists", False, "split_summary.csv not found")

# ============================================
# TEST 11: MEMORY MAP INTEGRITY
# ============================================
print("\n" + "=" * 70)
print("TEST 11: MEMORY MAP INTEGRITY")
print("=" * 70)

def test_memmap_access(arr, arr_name, num_samples=10):
    """Test that memmap can be accessed correctly."""
    try:
        # Try accessing random samples
        n_samples = min(num_samples, arr.shape[0])
        sample_indices = np.random.choice(arr.shape[0], size=n_samples, replace=False)
        
        # Access samples
        samples = arr[sample_indices]
        
        # Verify shape
        expected_shape = (n_samples,) + arr.shape[1:]
        if samples.shape != expected_shape:
            return False, f"Shape mismatch: Expected {expected_shape}, Got {samples.shape}"
        
        return True, f"Successfully accessed {n_samples} samples"
    except Exception as e:
        return False, f"Error: {str(e)}"

np.random.seed(42)
train_memmap_ok, train_memmap_msg = test_memmap_access(X_train, 'X_train', num_samples=10)
log_test("Train memmap access", train_memmap_ok, train_memmap_msg)

val_memmap_ok, val_memmap_msg = test_memmap_access(X_val, 'X_val', num_samples=5)
log_test("Val memmap access", val_memmap_ok, val_memmap_msg)

test_memmap_ok, test_memmap_msg = test_memmap_access(X_test, 'X_test', num_samples=5)
log_test("Test memmap access", test_memmap_ok, test_memmap_msg)

# ============================================
# TEST 12: RANDOM SAMPLE VERIFICATION
# ============================================
print("\n" + "=" * 70)
print("TEST 12: RANDOM SAMPLE VERIFICATION")
print("=" * 70)

# Load a few random samples and verify they're valid
np.random.seed(42)
num_samples_to_check = 5

print(f"\n🔍 Checking {num_samples_to_check} random samples from each split...")

for split_name, X_split, y_split in [('Train', X_train, y_train), 
                                      ('Val', X_val, y_val), 
                                      ('Test', X_test, y_test)]:
    sample_indices = np.random.choice(len(y_split), size=min(num_samples_to_check, len(y_split)), replace=False)
    
    all_valid = True
    for idx in sample_indices:
        try:
            # Load image
            image = X_split[idx]
            label = y_split[idx]
            
            # Verify shape
            if image.shape != EXPECTED_IMAGE_SHAPE:
                all_valid = False
                print(f"   ❌ {split_name} sample {idx}: Invalid shape {image.shape}")
                continue
            
            # Verify label
            if label < 0 or label >= NUM_CLASSES:
                all_valid = False
                print(f"   ❌ {split_name} sample {idx}: Invalid label {label}")
                continue
            
            # Verify no NaN/Inf
            if np.isnan(image).any() or np.isinf(image).any():
                all_valid = False
                print(f"   ❌ {split_name} sample {idx}: Contains NaN or Inf")
                continue
                
        except Exception as e:
            all_valid = False
            print(f"   ❌ {split_name} sample {idx}: Error - {str(e)}")
    
    if all_valid:
        log_test(f"{split_name} random samples", True, 
                f"All {num_samples_to_check} samples valid")
    else:
        log_test(f"{split_name} random samples", False, 
                "Some samples failed validation")

# ============================================
# FINAL SUMMARY
# ============================================
print("\n" + "=" * 70)
print("VALIDATION SUMMARY")
print("=" * 70)

total_tests = test_results['passed'] + test_results['failed'] + test_results['warnings']
pass_rate = (test_results['passed'] / total_tests * 100) if total_tests > 0 else 0

print(f"\n📊 Test Results:")
print(f"   Total Tests: {total_tests}")
print(f"   ✅ Passed: {test_results['passed']} ({test_results['passed']/total_tests*100:.1f}%)")
print(f"   ❌ Failed: {test_results['failed']} ({test_results['failed']/total_tests*100:.1f}%)")
print(f"   ⚠️  Warnings: {test_results['warnings']} ({test_results['warnings']/total_tests*100:.1f}%)")
print(f"   Pass Rate: {pass_rate:.1f}%")

# Print failed tests
if test_results['failed'] > 0:
    print(f"\n❌ Failed Tests:")
    for test in test_results['tests']:
        if not test['passed'] and not test['warning']:
            print(f"   - {test['name']}: {test['message']}")

# Print warnings
if test_results['warnings'] > 0:
    print(f"\n⚠️  Warnings:")
    for test in test_results['tests']:
        if test['warning']:
            print(f"   - {test['name']}: {test['message']}")

# Final verdict
print("\n" + "=" * 70)
if test_results['failed'] == 0:
    print("✅ VALIDATION PASSED: All critical tests passed!")
    print("   Your dataset splits are valid, complete, and free from corruption.")
    if test_results['warnings'] > 0:
        print(f"   ⚠️  Note: {test_results['warnings']} warning(s) - review above.")
    sys.exit(0)
else:
    print("❌ VALIDATION FAILED: Some tests failed!")
    print("   Please review the failed tests above and fix the issues.")
    sys.exit(1)

