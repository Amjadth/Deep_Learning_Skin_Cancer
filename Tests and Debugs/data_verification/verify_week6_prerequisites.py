#!/usr/bin/env python3
# ============================================
# WEEK 6 PREREQUISITES VERIFICATION (FAST)
# ============================================
#
# OPTIMIZED VERSION:
# - Skips optional large files (Week 3 augmented data)
# - Only checks REQUIRED files for Week 6
# - Uses sampling instead of full scans
# - Runs in < 2 minutes instead of 10+ minutes
#
# ============================================

import numpy as np
import json
from pathlib import Path
import os
import sys

print("=" * 80)
print("WEEK 6 PREREQUISITES VERIFICATION (FAST MODE)")
print("=" * 80)

# ============================================
# 1. CHECK WORKSPACE AND PATHS
# ============================================
print("\n📁 Step 1: Checking workspace and paths...")

BASE_DIR = Path(os.getcwd())
NETWORK_VOLUME = None

# Detect workspace
if Path('/workspace').exists():
    BASE_DIR = Path('/workspace')
    print(f"✓ RunPod workspace detected: {BASE_DIR}")
elif Path('/notebooks').exists():
    BASE_DIR = Path('/notebooks')
    print(f"✓ Jupyter workspace detected: {BASE_DIR}")
else:
    print(f"✓ Local workspace: {BASE_DIR}")

# Detect network volume
if Path('/runpod-volume').exists():
    NETWORK_VOLUME = Path('/runpod-volume')
    print(f"✓ Network volume detected: {NETWORK_VOLUME}")
elif Path('/workspace/.runpod').exists():
    NETWORK_VOLUME = Path('/workspace/.runpod')
    print(f"✓ Network volume detected: {NETWORK_VOLUME}")
else:
    print(f"⚠ Network volume not detected (using workspace)")

STORAGE_BASE = NETWORK_VOLUME if NETWORK_VOLUME else BASE_DIR
OUTPUT_DIR = (STORAGE_BASE / 'outputs').resolve()

print(f"\n📂 Output directory: {OUTPUT_DIR}")

if not OUTPUT_DIR.exists():
    print(f"❌ ERROR: Output directory does not exist!")
    print(f"   Expected: {OUTPUT_DIR}")
    print(f"   Run Week 1-5 first to generate required data.")
    sys.exit(1)
else:
    print(f"✓ Output directory exists")

# ============================================
# 2. CHECK REQUIRED FILES ONLY
# ============================================
print("\n📋 Step 2: Checking REQUIRED files for Week 6...")
print("(Skipping optional files to save time)")

# ONLY check required files for Week 6
required_files = {
    # Week 5 outputs (baseline downscaled versions) - REQUIRED FOR WEEK 6
    'X_train_baseline.npy': {
        'expected_shape': (64000, 224, 224, 3),  # Week 5 standard shape
        'expected_min_shape': (10000, 100, 100, 3),  # Minimum viable size
        'expected_size_gb': 38.6,
        'description': 'Training images baseline (Week 5 - REQUIRED)',
        'dtype': np.float32
    },
    'y_train_baseline.npy': {
        'expected_shape': (64000,),  # Week 5 standard shape
        'expected_min_shape': None,
        'expected_size_gb': 0.0002,
        'description': 'Training labels baseline (Week 5 - REQUIRED)',
        'dtype': None
    },
    'X_val_baseline.npy': {
        'expected_shape': (8000, 224, 224, 3),  # Week 5 standard shape
        'expected_min_shape': (1000, 100, 100, 3),  # Minimum viable size
        'expected_size_gb': 4.8,
        'description': 'Validation images baseline (Week 5 - REQUIRED)',
        'dtype': np.float32
    },
    'y_val_baseline.npy': {
        'expected_shape': (8000,),  # Week 5 standard shape
        'expected_min_shape': None,
        'expected_size_gb': 0.00003,
        'description': 'Validation labels baseline (Week 5 - REQUIRED)',
        'dtype': None
    },
    'X_test_baseline.npy': {
        'expected_shape': (8000, 224, 224, 3),  # Week 5 standard shape
        'expected_min_shape': (1000, 100, 100, 3),  # Minimum viable size
        'expected_size_gb': 4.8,
        'description': 'Test images baseline (Week 5 - REQUIRED)',
        'dtype': np.float32
    },
    'y_test_baseline.npy': {
        'expected_shape': (8000,),  # Week 5 standard shape
        'expected_min_shape': None,
        'expected_size_gb': 0.00003,
        'description': 'Test labels baseline (Week 5 - REQUIRED)',
        'dtype': None
    },
    
    # Week 5 model files - REQUIRED FOR WEEK 6
    'models/baseline_cnn_architecture.json': {
        'expected_shape': None,
        'expected_size_gb': 0.001,
        'description': 'Baseline CNN architecture (Week 5 - REQUIRED)'
    },
    
    # Week 4 metadata - REQUIRED
    'split_info.json': {
        'expected_shape': None,
        'expected_size_gb': 0.001,
        'description': 'Dataset split information (Week 4 - REQUIRED)'
    }
}

print(f"\nChecking {len(required_files)} required files...")
print("-" * 80)

missing_required = []
corrupted = []
size_mismatch = []
valid_files = []

for filename, info in required_files.items():
    filepath = OUTPUT_DIR / filename
    
    print(f"\n❗ {filename}")
    print(f"   Description: {info['description']}")
    
    # Check if file exists
    if not filepath.exists():
        print(f"   ❌ MISSING (REQUIRED!)")
        missing_required.append(filename)
        continue
    
    # Check file size
    file_size_bytes = filepath.stat().st_size
    file_size_gb = file_size_bytes / (1024**3)
    expected_size = info['expected_size_gb']
    
    print(f"   ✓ Exists")
    print(f"   Size: {file_size_gb:.2f} GB (expected: ~{expected_size:.2f} GB)")
    
    # Size validation (allow 20% variance)
    if expected_size > 0.001:  # Only check for files > 1MB
        size_ratio = file_size_gb / expected_size
        if size_ratio < 0.8 or size_ratio > 1.2:
            print(f"   ⚠ Size mismatch! ({size_ratio*100:.1f}% of expected)")
            size_mismatch.append(filename)
    
    # Try to load and verify shape for .npy files
    if filename.endswith('.npy'):
        try:
            print(f"   Loading file (checking shape)...")
            # Use the same safe loading logic from Week 5
            # Try without pickle first (more secure)
            try:
                data = np.load(filepath, mmap_mode='r', allow_pickle=False)
                print(f"   ✓ Loaded successfully (memmap, no pickle)")
            except ValueError as e:
                if "pickled data" in str(e).lower() or "pickle" in str(e).lower():
                    # File requires pickle, use allow_pickle=True
                    print(f"   ⚠️  File requires pickle support, retrying...")
                    try:
                        data = np.load(filepath, mmap_mode='r', allow_pickle=True)
                        print(f"   ✓ Loaded successfully (memmap, with pickle)")
                    except Exception as e2:
                        # Last resort: try direct memmap if it's a large image file
                        if 'X_' in filename and info.get('expected_shape'):
                            print(f"   ⚠️  Trying direct memmap...")
                            expected_shape = info['expected_shape']
                            data = np.memmap(str(filepath), dtype=np.float32, mode='r', shape=expected_shape)
                            print(f"   ✓ Loaded successfully (raw memmap)")
                        else:
                            raise
                else:
                    raise
            print(f"   Shape: {data.shape}")
            print(f"   Dtype: {data.dtype}")
            
            # Check expected shape (flexible matching)
            expected_shape = info['expected_shape']
            expected_min = info.get('expected_min_shape', None)
            
            shape_valid = True
            
            # Flexible validation - don't mark as corrupted for shape differences
            if expected_shape and data.shape != expected_shape:
                print(f"   Note: Shape differs from reference")
                print(f"   Expected: {expected_shape}")
                print(f"   Got: {data.shape}")
            
            # Minimum size check if expected_min_shape is specified
            if expected_min and len(data.shape) == len(expected_min):
                for i, (actual, minimum) in enumerate(zip(data.shape, expected_min)):
                    if actual < minimum:
                        print(f"   ⚠ Dimension {i} too small: {actual} < {minimum}")
                        shape_valid = False
            
            # For image files (X_*), check it has 4 dimensions
            if 'X_' in filename:
                if len(data.shape) != 4:
                    print(f"   ⚠ Image must have 4 dimensions (N, H, W, C), got {len(data.shape)}")
                    shape_valid = False
                elif data.shape[3] not in [1, 3, 4]:
                    print(f"   ⚠ Invalid channel dimension: {data.shape[3]} (expected 1, 3, or 4)")
                    shape_valid = False
                else:
                    print(f"   ✓ Image shape valid (N={data.shape[0]}, H={data.shape[1]}, W={data.shape[2]}, C={data.shape[3]})")
            
            # For label files (y_*), check it has 1 dimension
            elif 'y_' in filename:
                if len(data.shape) != 1:
                    print(f"   ⚠ Labels must have 1 dimension, got {len(data.shape)}")
                    shape_valid = False
                else:
                    print(f"   ✓ Label shape valid (N={data.shape[0]})")
            
            else:
                print(f"   ✓ Shape valid")
            
            # Fast data range check (sample instead of full scan)
            if 'X_' in filename:
                # Sample only 10 images for quick validation
                sample_size = min(10, len(data))
                sample_indices = np.random.choice(len(data), sample_size, replace=False)
                sample_data = data[sample_indices]
                
                min_val = float(np.min(sample_data))
                max_val = float(np.max(sample_data))
                print(f"   Data range (sampled {sample_size}): [{min_val:.2f}, {max_val:.2f}]")
                
                if min_val < 0 or max_val > 255:
                    print(f"   ⚠ Unusual data range! (expected 0-255 or 0-1)")
                else:
                    print(f"   ✓ Data range valid")
            
            # Fast label check (sample for large arrays)
            if 'y_' in filename:
                sample_size = min(1000, len(data))
                sample_indices = np.random.choice(len(data), sample_size, replace=False)
                sample_labels = data[sample_indices]
                unique_labels = np.unique(sample_labels)
                
                print(f"   Unique labels (sampled {sample_size}): {sorted(unique_labels)}")
                print(f"   Number of classes: {len(unique_labels)}")
                
                # Allow any reasonable label range (0-7 for 8 classes, but be flexible)
                # Just warn if something seems wrong
                if len(unique_labels) > 20:
                    print(f"   ⚠ Warning: More than 20 unique labels (unusual for skin cancer)")
                elif np.min(unique_labels) < 0 or np.max(unique_labels) > 20:
                    print(f"   ⚠ Label range: [{np.min(unique_labels)}, {np.max(unique_labels)}]")
                else:
                    print(f"   ✓ Labels valid")
            
            valid_files.append(filename)
            
        except Exception as e:
            print(f"   ❌ Failed to load: {e}")
            corrupted.append(filename)
    
    # Check JSON files
    elif filename.endswith('.json'):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"   ✓ Valid JSON")
            
            # Check split_info.json structure
            if 'split_info' in filename:
                required_keys = ['num_classes', 'class_names', 'splits', 'image_shape']
                for key in required_keys:
                    if key not in data:
                        print(f"   ⚠ Missing key: {key}")
                        corrupted.append(filename)
                    else:
                        print(f"   ✓ Has key: {key}")
                
                if 'class_names' in data:
                    print(f"   Classes: {', '.join(data['class_names'])}")
                if 'num_classes' in data:
                    print(f"   Number of classes: {data['num_classes']}")
            
            valid_files.append(filename)
            
        except Exception as e:
            print(f"   ❌ Failed to load JSON: {e}")
            corrupted.append(filename)

print("\n" + "=" * 80)

# ============================================
# 3. VERIFICATION SUMMARY
# ============================================
print("\n📊 Step 3: Verification Summary")
print("=" * 80)

print(f"\n✓ Valid required files: {len(valid_files)}")
for f in valid_files:
    print(f"  ✓ {f}")

if size_mismatch:
    print(f"\n⚠ Size mismatches: {len(size_mismatch)}")
    for f in size_mismatch:
        print(f"  ⚠ {f}")

if corrupted:
    print(f"\n❌ Corrupted/invalid files: {len(corrupted)}")
    for f in corrupted:
        print(f"  ❌ {f}")

if missing_required:
    print(f"\n❌ MISSING REQUIRED FILES: {len(missing_required)}")
    for f in missing_required:
        print(f"  ❌ {f}")

# ============================================
# 4. CHECK GPU AVAILABILITY
# ============================================
print("\n" + "=" * 80)
print("\n🎮 Step 4: Checking GPU availability...")

try:
    import tensorflow as tf
    
    print(f"TensorFlow version: {tf.__version__}")
    
    # Clear GPU memory first
    print(f"Clearing GPU memory...")
    try:
        from tensorflow import keras
        keras.backend.clear_session()
        print(f"✓ GPU memory cleared")
    except:
        print(f"⚠ Could not clear GPU memory (may not be needed)")
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✓ GPU detected: {len(gpus)} device(s)")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
    else:
        print(f"⚠ WARNING: No GPU detected!")
        print(f"  Week 6 training will be VERY SLOW on CPU")
        print(f"  Check:")
        print(f"  1. RunPod GPU selection")
        print(f"  2. nvidia-smi command")
        print(f"  3. CUDA drivers")
    
    # Test GPU with simple operation
    if gpus:
        try:
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0]])
                b = tf.constant([[3.0], [4.0]])
                c = tf.matmul(a, b)
            print(f"✓ GPU test successful: {c.numpy()}")
        except Exception as e:
            print(f"⚠ GPU test failed: {e}")
    
except ImportError:
    print(f"⚠ TensorFlow not installed!")
    print(f"  Install: pip install tensorflow")
except Exception as e:
    print(f"⚠ GPU check failed: {e}")

# ============================================
# 5. CHECK CONTAINER MEMORY
# ============================================
print("\n" + "=" * 80)
print("\n💾 Step 5: Checking container memory...")

try:
    import psutil
    
    vm = psutil.virtual_memory()
    total_gb = vm.total / (1024**3)
    available_gb = vm.available / (1024**3)
    used_gb = vm.used / (1024**3)
    
    print(f"Container Memory:")
    print(f"  Total: {total_gb:.1f} GB")
    print(f"  Used: {used_gb:.1f} GB ({vm.percent:.1f}%)")
    print(f"  Available: {available_gb:.1f} GB")
    
    # Check if container has enough memory
    if total_gb < 40:
        print(f"\n⚠ WARNING: Container memory < 40GB")
        print(f"  Week 6 REQUIRES memory-mapped loading!")
        print(f"  Use week6_fixed_memory.py (not original week6.py)")
    else:
        print(f"\n✓ Container memory sufficient")
    
    if available_gb < 10:
        print(f"\n⚠ WARNING: Low available memory ({available_gb:.1f} GB)")
        print(f"  Close other processes before running Week 6")
    else:
        print(f"✓ Available memory sufficient")
    
except ImportError:
    print(f"⚠ psutil not installed (memory check skipped)")
except Exception as e:
    print(f"⚠ Memory check failed: {e}")

# ============================================
# 6. CALCULATE REQUIRED DISK SPACE
# ============================================
print("\n" + "=" * 80)
print("\n💿 Step 6: Checking disk space...")

try:
    import shutil
    
    total, used, free = shutil.disk_usage(OUTPUT_DIR)
    
    total_gb = total / (1024**3)
    used_gb = used / (1024**3)
    free_gb = free / (1024**3)
    
    print(f"Disk Space ({OUTPUT_DIR}):")
    print(f"  Total: {total_gb:.1f} GB")
    print(f"  Used: {used_gb:.1f} GB ({used/total*100:.1f}%)")
    print(f"  Free: {free_gb:.1f} GB")
    
    required_space_gb = 5.0
    print(f"\nWeek 6 will need ~{required_space_gb:.1f} GB for outputs")
    
    if free_gb < required_space_gb:
        print(f"⚠ WARNING: Insufficient disk space!")
        print(f"  Required: {required_space_gb:.1f} GB")
        print(f"  Available: {free_gb:.1f} GB")
        print(f"  Free up space or attach larger volume")
    else:
        print(f"✓ Sufficient disk space ({free_gb:.1f} GB available)")
    
except Exception as e:
    print(f"⚠ Disk space check failed: {e}")

# ============================================
# 7. FINAL VERDICT
# ============================================
print("\n" + "=" * 80)
print("\n🎯 FINAL VERDICT")
print("=" * 80)

all_checks_passed = True
critical_issues = []

# Check 1: Required files
if missing_required:
    all_checks_passed = False
    critical_issues.append(f"Missing {len(missing_required)} required files")
    print(f"\n❌ CRITICAL: Missing required files!")
    print(f"   Missing: {', '.join(missing_required)}")
    print(f"\n   ⚠ ACTION REQUIRED:")
    for f in missing_required:
        if 'baseline' in f.lower():
            print(f"   - Run Week 5 to generate: {f}")
        elif 'split_info' in f.lower():
            print(f"   - Run Week 4 to generate: {f}")
        elif 'model' in f.lower():
            print(f"   - Run Week 5 to generate: {f}")

# Check 2: Corrupted files
if corrupted:
    all_checks_passed = False
    critical_issues.append(f"{len(corrupted)} corrupted files")
    print(f"\n❌ CRITICAL: Corrupted or invalid files!")
    print(f"   Corrupted: {', '.join(corrupted)}")
    print(f"\n   ⚠ ACTION REQUIRED:")
    print(f"   - Re-run the week that generated these files")

# Check 3: GPU availability
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print(f"\n⚠ WARNING: No GPU detected!")
        print(f"   Training will be VERY SLOW on CPU (~10-20x slower)")
        print(f"   Recommended: Use RunPod with GPU attached")
except:
    pass

# Summary
print("\n" + "-" * 80)

if all_checks_passed:
    print("\n✅ ALL CHECKS PASSED!")
    print("\n🚀 You are ready to run Week 6!")
    print("\nRun command:")
    print("  python week6_fixed_memory.py")
    print("\nOR in Jupyter:")
    print("  exec(open('week6_fixed_memory.py').read())")
    
    print("\n📋 What will happen:")
    print("  1. GPU initialization (< 1 minute)")
    print("  2. Memory-mapped data loading (< 1 minute)")
    print("  3. Model training (~50-80 minutes on A40)")
    print("  4. Evaluation and visualization (~5 minutes)")
    print("\nTotal estimated time: ~1-1.5 hours")
    
else:
    print(f"\n❌ CHECKS FAILED!")
    print(f"\nCritical issues found:")
    for issue in critical_issues:
        print(f"  - {issue}")
    
    print(f"\n⚠ DO NOT RUN WEEK 6 YET!")
    print(f"  Fix the issues above first.")
    
    print(f"\n📋 Next steps:")
    if missing_required:
        print(f"  1. Run the missing weeks to generate required files")
        print(f"  2. Re-run this verification script")
        print(f"  3. Once all checks pass, run Week 6")
    if corrupted:
        print(f"  1. Re-run the weeks that generated corrupted files")
        print(f"  2. Re-run this verification script")

print("\n" + "=" * 80)

# Exit with appropriate code
if all_checks_passed:
    print("\n✅ Verification complete - Ready to proceed!\n")
    sys.exit(0)
else:
    print(f"\n❌ Verification failed - Fix issues before proceeding!\n")
    sys.exit(1)
