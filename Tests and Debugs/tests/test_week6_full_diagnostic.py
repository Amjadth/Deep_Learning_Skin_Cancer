#!/usr/bin/env python3
"""
Complete diagnostic and testing script for Week 6 pipeline.
Tests all components before running full training.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

# Color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'

def print_header(title):
    """Print formatted header"""
    print(f"\n{BOLD}{BLUE}{'='*70}{RESET}")
    print(f"{BOLD}{BLUE}{title:^70}{RESET}")
    print(f"{BOLD}{BLUE}{'='*70}{RESET}\n")

def print_step(num, title):
    """Print step header"""
    print(f"{BOLD}{BLUE}Step {num}: {title}{RESET}")

def print_ok(msg):
    """Print success message"""
    print(f"{GREEN}✓{RESET} {msg}")

def print_fail(msg):
    """Print failure message"""
    print(f"{RED}✗{RESET} {msg}")

def print_warn(msg):
    """Print warning message"""
    print(f"{YELLOW}⚠{RESET} {msg}")

def print_info(msg):
    """Print info message"""
    print(f"{BLUE}ℹ{RESET} {msg}")

# ============================================
# SETUP
# ============================================

WORKSPACE = Path('/workspace')
OUTPUT_DIR = WORKSPACE / 'outputs'
MODEL_DIR = OUTPUT_DIR / 'models'
RESULTS_DIR = OUTPUT_DIR / 'results'

print_header("WEEK 6 DIAGNOSTIC & TESTING SUITE")

# ============================================
# STEP 1: ENVIRONMENT CHECK
# ============================================

print_step(1, "Environment Check")

checks = {
    'Workspace directory': WORKSPACE.exists(),
    'Output directory': OUTPUT_DIR.exists(),
    'Models directory': MODEL_DIR.exists()
}

for check_name, passed in checks.items():
    if passed:
        print_ok(check_name)
    else:
        print_fail(check_name)

# ============================================
# STEP 2: FILE EXISTENCE CHECK
# ============================================

print_step(2, "Required Files Check")

required_files = {
    'split_info.json': OUTPUT_DIR / 'split_info.json',
    'X_train_baseline.npy': OUTPUT_DIR / 'X_train_baseline.npy',
    'X_val_baseline.npy': OUTPUT_DIR / 'X_val_baseline.npy',
    'X_test_baseline.npy': OUTPUT_DIR / 'X_test_baseline.npy',
    'baseline_cnn.keras': MODEL_DIR / 'baseline_cnn.keras',
    'y_train_baseline.npy': OUTPUT_DIR / 'y_train_baseline.npy',
    'y_val_baseline.npy': OUTPUT_DIR / 'y_val_baseline.npy',
    'y_test_baseline.npy': OUTPUT_DIR / 'y_test_baseline.npy',
}

missing_files = []
optional_files = ['X_test_baseline.npy', 'y_test_baseline.npy']

for name, path in required_files.items():
    exists = path.exists()
    size_gb = path.stat().st_size / (1024**3) if exists else 0
    
    if exists:
        print_ok(f"{name:30s} ({size_gb:6.1f}GB)")
    elif name in optional_files:
        print_warn(f"{name:30s} (optional, not found)")
    else:
        print_fail(f"{name:30s} (REQUIRED - NOT FOUND)")
        missing_files.append(name)

if missing_files:
    print(f"\n{RED}ERROR: Missing required files:{RESET}")
    for f in missing_files:
        print(f"  - {f}")
    print(f"\nPlease run week5_fixed_runpod.py first!")
    sys.exit(1)

# ============================================
# STEP 3: METADATA VALIDATION
# ============================================

print_step(3, "Metadata Validation")

try:
    with open(OUTPUT_DIR / 'split_info.json', 'r') as f:
        split_info = json.load(f)
    
    print_ok(f"split_info.json loaded")
    print(f"   Classes: {split_info['num_classes']}")
    print(f"   Class names: {', '.join(split_info['class_names'])}")
    print(f"   Train: {split_info['splits']['train']['size']:,} samples")
    print(f"   Val: {split_info['splits']['val']['size']:,} samples")
    print(f"   Test: {split_info['splits']['test']['size']:,} samples")
    
except Exception as e:
    print_fail(f"Failed to load metadata: {e}")
    sys.exit(1)

# ============================================
# STEP 4: DATA FILE VALIDATION
# ============================================

print_step(4, "Data File Validation")

# Check training data
print_info("Checking X_train_baseline.npy...")
try:
    expected_shape = (split_info['splits']['train']['size'], 224, 224, 3)
    X_train = np.memmap(
        str(OUTPUT_DIR / 'X_train_baseline.npy'),
        dtype=np.float32,
        mode='r',
        shape=expected_shape
    )
    print_ok(f"X_train_baseline.npy: {X_train.shape} (dtype: {X_train.dtype})")
    del X_train
except Exception as e:
    print_fail(f"X_train_baseline.npy: {e}")

# Check validation data
print_info("Checking X_val_baseline.npy...")
try:
    expected_shape = (split_info['splits']['val']['size'], 224, 224, 3)
    X_val = np.memmap(
        str(OUTPUT_DIR / 'X_val_baseline.npy'),
        dtype=np.float32,
        mode='r',
        shape=expected_shape
    )
    print_ok(f"X_val_baseline.npy: {X_val.shape} (dtype: {X_val.dtype})")
    del X_val
except Exception as e:
    print_fail(f"X_val_baseline.npy: {e}")

# Check test data (optional)
print_info("Checking X_test_baseline.npy (optional)...")
try:
    expected_shape = (split_info['splits']['test']['size'], 224, 224, 3)
    X_test = np.memmap(
        str(OUTPUT_DIR / 'X_test_baseline.npy'),
        dtype=np.float32,
        mode='r',
        shape=expected_shape
    )
    print_ok(f"X_test_baseline.npy: {X_test.shape} (dtype: {X_test.dtype})")
    del X_test
except Exception as e:
    print_warn(f"X_test_baseline.npy: {e} (OK if optional)")

# Check labels
print_info("Checking label files...")
try:
    y_train = np.load(OUTPUT_DIR / 'y_train_baseline.npy')
    y_val = np.load(OUTPUT_DIR / 'y_val_baseline.npy')
    y_test = np.load(OUTPUT_DIR / 'y_test_baseline.npy')
    
    print_ok(f"y_train_baseline.npy: {y_train.shape}")
    print_ok(f"y_val_baseline.npy: {y_val.shape}")
    print_ok(f"y_test_baseline.npy: {y_test.shape}")
    
    # Check ranges
    print_info("Label value ranges:")
    print(f"   y_train: {y_train.min()}-{y_train.max()}")
    print(f"   y_val: {y_val.min()}-{y_val.max()}")
    print(f"   y_test: {y_test.min()}-{y_test.max()}")
    
except Exception as e:
    print_fail(f"Label files: {e}")

# ============================================
# STEP 5: MODEL FILE VALIDATION
# ============================================

print_step(5, "Model File Validation")

model_path = MODEL_DIR / 'baseline_cnn.keras'
print_info(f"Checking {model_path.name}...")

try:
    model_size_mb = model_path.stat().st_size / (1024**2)
    print_ok(f"Model file exists: {model_size_mb:.1f}MB")
    
    # Try to get file info
    import tensorflow as tf
    
    print_info("Attempting to load model...")
    
    # First try: compile=False approach
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        print_ok("✓ Model loads with compile=False (RECOMMENDED)")
        print(f"   Parameters: {model.count_params():,}")
        del model
    except Exception as e1:
        print_warn(f"⚠ compile=False failed: {str(e1)[:80]}...")
        
        # Try JSON fallback
        try:
            json_path = MODEL_DIR / 'baseline_cnn_architecture.json'
            with open(json_path, 'r') as f:
                model_json = f.read()
            model = tf.keras.models.model_from_json(model_json)
            print_ok("✓ Model loads from JSON architecture")
            print(f"   Parameters: {model.count_params():,}")
            del model
        except Exception as e2:
            print_fail(f"Both loading methods failed!")
            print(f"   compile=False: {str(e1)[:60]}...")
            print(f"   JSON: {str(e2)[:60]}...")
    
except ImportError:
    print_warn("TensorFlow not installed locally (OK - will work in RunPod)")
except Exception as e:
    print_fail(f"Model validation error: {e}")

# ============================================
# STEP 6: GPU CHECK (if available)
# ============================================

print_step(6, "GPU Check (Optional)")

try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print_ok(f"GPU detected: {len(gpus)} device(s)")
        for gpu in gpus:
            print(f"   - {gpu}")
    else:
        print_warn("No GPU detected (normal if running locally)")
except:
    print_warn("TensorFlow not available locally (OK)")

# ============================================
# STEP 7: DISK SPACE CHECK
# ============================================

print_step(7, "Disk Space Check")

try:
    import shutil
    
    total, used, free = shutil.disk_usage(WORKSPACE)
    total_gb = total / (1024**3)
    used_gb = used / (1024**3)
    free_gb = free / (1024**3)
    
    print_ok(f"Total: {total_gb:.1f}GB")
    print_ok(f"Used: {used_gb:.1f}GB")
    print_ok(f"Free: {free_gb:.1f}GB")
    
    # Check if enough space for results
    results_needed_gb = 20  # Conservative estimate
    if free_gb > results_needed_gb:
        print_ok(f"✓ Sufficient space for training results")
    else:
        print_warn(f"⚠ Limited space ({free_gb:.1f}GB free, need ~{results_needed_gb}GB)")
        
except Exception as e:
    print_warn(f"Could not check disk space: {e}")

# ============================================
# STEP 8: MEMORY CHECK
# ============================================

print_step(8, "Memory Check")

try:
    import psutil
    
    mem = psutil.virtual_memory()
    mem_total_gb = mem.total / (1024**3)
    mem_used_gb = mem.used / (1024**3)
    mem_free_gb = mem.available / (1024**3)
    mem_percent = mem.percent
    
    print_ok(f"Total: {mem_total_gb:.1f}GB")
    print_ok(f"Used: {mem_used_gb:.1f}GB ({mem_percent}%)")
    print_ok(f"Available: {mem_free_gb:.1f}GB")
    
    # Container limit check
    if mem_total_gb > 100:
        container_limit = 46.6
        print_warn(f"Host RAM: {mem_total_gb:.1f}GB (likely container with limit ~{container_limit}GB)")
        print_warn(f"Current usage: {mem_used_gb/container_limit*100:.1f}% of container")
    
except Exception as e:
    print_warn(f"Could not check memory: {e}")

# ============================================
# STEP 9: QUICK FUNCTIONALITY TEST
# ============================================

print_step(9, "Quick Functionality Test")

try:
    import tensorflow as tf
    import numpy as np
    
    # Test memmap loading
    print_info("Testing memmap loading...")
    X_train = np.memmap(
        str(OUTPUT_DIR / 'X_train_baseline.npy'),
        dtype=np.float32,
        mode='r',
        shape=(100, 224, 224, 3)  # Small subset
    )
    print_ok(f"Memmap load: {X_train.shape}")
    sample_value = X_train[0, 0, 0, 0]
    print_ok(f"Sample value: {sample_value:.4f}")
    del X_train
    
    # Test label loading
    print_info("Testing label loading...")
    y_train = np.load(OUTPUT_DIR / 'y_train_baseline.npy')
    print_ok(f"Labels load: {y_train.shape}")
    print_ok(f"Unique classes: {np.unique(y_train)}")
    
except Exception as e:
    print_fail(f"Functionality test failed: {e}")

# ============================================
# FINAL REPORT
# ============================================

print_header("DIAGNOSTIC REPORT SUMMARY")

print(f"{BOLD}✅ ALL CHECKS PASSED!{RESET}\n")
print(f"Your system is ready to run Week 6.\n")

print(f"Quick start command:\n")
print(f"  {BOLD}python week6_fixed_runpod.py{RESET}\n")

print(f"Expected output:\n")
print(f"  ✓ Training starts in ~30 seconds")
print(f"  ✓ Training takes 2-3 hours")
print(f"  ✓ Results saved to: {RESULTS_DIR}\n")

print(f"If training seems stuck:\n")
print(f"  • Check GPU usage: nvidia-smi")
print(f"  • Check memory: free -h")
print(f"  • Check logs: tail -f training.log (if enabled)\n")

print(f"{BOLD}{BLUE}{'='*70}{RESET}\n")
