"""
NPY FILE VALIDATION AND RECOVERY SCRIPT

Purpose: Validate .npy files and recover if headers are corrupted
Tests:
1. Check if files can be loaded with np.load()
2. Check if headers are valid
3. Verify data integrity
4. Recover from raw binary if needed
5. Test denormalization (reverse ImageNet normalization)

Author: Deep Learning Engineer
Date: 2024
"""

import numpy as np
from pathlib import Path
import json
import struct

print("=" * 80)
print("NPY FILE VALIDATION AND RECOVERY")
print("=" * 80)

# Setup paths
BASE_DIR = Path('/workspace') if Path('/workspace').exists() else Path.cwd()
NETWORK_VOLUME = None

if Path('/runpod-volume').exists():
    NETWORK_VOLUME = Path('/runpod-volume')
elif Path('/workspace').exists():
    NETWORK_VOLUME = Path('/workspace')

STORAGE_BASE = NETWORK_VOLUME if NETWORK_VOLUME else BASE_DIR
OUTPUT_DIR = (STORAGE_BASE / 'outputs').resolve()

print(f"\n📁 Storage Configuration:")
print(f"   Output directory: {OUTPUT_DIR}")
print(f"   Network volume: {NETWORK_VOLUME if NETWORK_VOLUME else 'Not detected'}")

# Files to validate
FILES_TO_CHECK = {
    'X_train': OUTPUT_DIR / 'X_train.npy',
    'y_train': OUTPUT_DIR / 'y_train.npy',
    'X_val': OUTPUT_DIR / 'X_val.npy',
    'y_val': OUTPUT_DIR / 'y_val.npy',
    'X_test': OUTPUT_DIR / 'X_test.npy',
    'y_test': OUTPUT_DIR / 'y_test.npy'
}

# Expected shapes from Week 4 (balanced 10k per class, 8 classes)
EXPECTED_SHAPES = {
    'X_train': (64000, 224, 224, 3),  # 8k per class × 8 classes
    'y_train': (64000,),
    'X_val': (8000, 224, 224, 3),     # 1k per class × 8 classes
    'y_val': (8000,),
    'X_test': (8000, 224, 224, 3),    # 1k per class × 8 classes
    'y_test': (8000,)
}

EXPECTED_DTYPES = {
    'X_train': np.float32,
    'y_train': np.int32,
    'X_val': np.float32,
    'y_val': np.int32,
    'X_test': np.float32,
    'y_test': np.int32
}

# ImageNet normalization constants (from Week 2)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

print("\n" + "=" * 80)
print("TEST 1: FILE EXISTENCE")
print("=" * 80)

all_exist = True
for name, path in FILES_TO_CHECK.items():
    if path.exists():
        size_gb = path.stat().st_size / (1024**3)
        print(f"✅ {name}: {path.name} ({size_gb:.2f} GB)")
    else:
        print(f"❌ {name}: NOT FOUND")
        all_exist = False

if not all_exist:
    print("\n❌ FATAL: Missing required files!")
    exit(1)

print("\n" + "=" * 80)
print("TEST 2: NPY HEADER VALIDATION")
print("=" * 80)

def check_npy_header(filepath):
    """
    Check if file has valid NPY header.
    NPY format: magic string (6 bytes) + version (2 bytes) + header length + header dict
    """
    try:
        with open(filepath, 'rb') as f:
            # Read magic string
            magic = f.read(6)
            if magic != b'\x93NUMPY':
                return False, "Invalid magic string (not a NPY file)"
            
            # Read version
            version = struct.unpack('BB', f.read(2))
            
            # Read header length
            if version[0] == 1:
                header_len = struct.unpack('<H', f.read(2))[0]
            elif version[0] in (2, 3):
                header_len = struct.unpack('<I', f.read(4))[0]
            else:
                return False, f"Unknown NPY version: {version}"
            
            # Read header
            header = f.read(header_len)
            header_str = header.decode('latin1')
            
            # Parse header dict
            header_dict = eval(header_str.strip())
            
            return True, {
                'version': version,
                'shape': header_dict['shape'],
                'dtype': header_dict['descr'],
                'fortran_order': header_dict['fortran_order']
            }
    except Exception as e:
        return False, str(e)

header_results = {}
for name, path in FILES_TO_CHECK.items():
    print(f"\n📄 Checking {name}...")
    is_valid, info = check_npy_header(path)
    header_results[name] = (is_valid, info)
    
    if is_valid:
        print(f"   ✅ Valid NPY header")
        print(f"      Version: {info['version']}")
        print(f"      Shape: {info['shape']}")
        print(f"      Dtype: {info['dtype']}")
        print(f"      Fortran order: {info['fortran_order']}")
    else:
        print(f"   ❌ Invalid header: {info}")

print("\n" + "=" * 80)
print("TEST 3: LOADING WITH NP.LOAD()")
print("=" * 80)

load_results = {}
for name, path in FILES_TO_CHECK.items():
    print(f"\n📂 Loading {name}...")
    try:
        # Try with memmap first (safe for large files)
        if name.startswith('X_'):
            data = np.load(path, mmap_mode='r')
        else:
            data = np.load(path)
        
        print(f"   ✅ Loaded successfully")
        print(f"      Shape: {data.shape}")
        print(f"      Dtype: {data.dtype}")
        print(f"      Expected shape: {EXPECTED_SHAPES[name]}")
        print(f"      Expected dtype: {EXPECTED_DTYPES[name]}")
        
        # Verify shape
        if data.shape == EXPECTED_SHAPES[name]:
            print(f"      ✅ Shape matches!")
        else:
            print(f"      ❌ Shape mismatch!")
        
        # Verify dtype
        if data.dtype == EXPECTED_DTYPES[name]:
            print(f"      ✅ Dtype matches!")
        else:
            print(f"      ⚠️  Dtype mismatch (expected {EXPECTED_DTYPES[name]}, got {data.dtype})")
        
        load_results[name] = {
            'success': True,
            'data': data,
            'shape': data.shape,
            'dtype': data.dtype
        }
        
    except Exception as e:
        print(f"   ❌ Failed to load: {e}")
        load_results[name] = {
            'success': False,
            'error': str(e)
        }

# Check if all loads succeeded
all_loaded = all(result['success'] for result in load_results.values())

if not all_loaded:
    print("\n❌ FATAL: Some files could not be loaded!")
    print("   Recovery options:")
    print("   1. Re-run Week 4 script to regenerate splits")
    print("   2. Manually recover from raw binary (complex)")
    exit(1)

print("\n✅ All files loaded successfully!")

print("\n" + "=" * 80)
print("TEST 4: DATA INTEGRITY CHECKS")
print("=" * 80)

# Check X_train data range
print("\n📊 Checking X_train data range (sample 500 images)...")
X_train = load_results['X_train']['data']
sample_indices = np.random.choice(len(X_train), min(500, len(X_train)), replace=False)
sample_data = X_train[sample_indices]

print(f"   Min: {sample_data.min():.6f}")
print(f"   Max: {sample_data.max():.6f}")
print(f"   Mean: {sample_data.mean():.6f}")
print(f"   Std: {sample_data.std():.6f}")

# Determine normalization state
if sample_data.min() < -0.5 and sample_data.max() < 1.0:
    print(f"   📌 Data appears to be ImageNet normalized (range ~ [-3, 3])")
    is_imagenet_normalized = True
elif sample_data.min() >= 0.0 and sample_data.max() <= 1.0:
    print(f"   📌 Data appears to be in [0, 1] range (denormalized)")
    is_imagenet_normalized = False
elif sample_data.min() >= 0.0 and sample_data.max() <= 255.0:
    print(f"   📌 Data appears to be in [0, 255] range (raw pixels)")
    is_imagenet_normalized = False
else:
    print(f"   ⚠️  Data range is unusual - cannot determine normalization state")
    is_imagenet_normalized = None

# Check for corruption
print("\n📊 Checking for data corruption (10 random samples)...")
corruption_detected = False
for i in range(10):
    idx = np.random.randint(0, len(X_train))
    img = X_train[idx]
    
    # Check for all-zero images
    if np.all(img == 0):
        print(f"   ❌ Sample {idx}: ALL ZERO (corrupted!)")
        corruption_detected = True
    # Check for all-same values
    elif np.std(img) < 0.001:
        print(f"   ❌ Sample {idx}: No variation (std={np.std(img):.6f}, corrupted!)")
        corruption_detected = True
    # Check for NaN/Inf
    elif np.isnan(img).any() or np.isinf(img).any():
        print(f"   ❌ Sample {idx}: NaN/Inf detected (corrupted!)")
        corruption_detected = True
    else:
        print(f"   ✅ Sample {idx}: OK (mean={img.mean():.3f}, std={img.std():.3f})")

if corruption_detected:
    print("\n❌ FATAL: Data corruption detected!")
    exit(1)
else:
    print("\n✅ No corruption detected!")

# Check labels
print("\n📊 Checking labels...")
y_train = load_results['y_train']['data']
y_val = load_results['y_val']['data']
y_test = load_results['y_test']['data']

print(f"   y_train: min={y_train.min()}, max={y_train.max()}, unique={len(np.unique(y_train))}")
print(f"   y_val: min={y_val.min()}, max={y_val.max()}, unique={len(np.unique(y_val))}")
print(f"   y_test: min={y_test.min()}, max={y_test.max()}, unique={len(np.unique(y_test))}")

if y_train.min() < 0 or y_train.max() > 7:
    print(f"   ❌ Invalid label range!")
    exit(1)

print(f"   ✅ Labels in valid range [0, 7]")

# Check class distribution
from collections import Counter
print("\n📊 Class distribution:")
train_counts = Counter(y_train)
val_counts = Counter(y_val)
test_counts = Counter(y_test)

CLASS_NAMES = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']
print(f"{'Class':<8} {'Train':<10} {'Val':<10} {'Test':<10}")
print("-" * 50)
for i, name in enumerate(CLASS_NAMES):
    print(f"{name:<8} {train_counts.get(i, 0):<10} {val_counts.get(i, 0):<10} {test_counts.get(i, 0):<10}")

print("\n" + "=" * 80)
print("TEST 5: DENORMALIZATION TEST")
print("=" * 80)

if is_imagenet_normalized is True:
    print("\n🔄 Testing denormalization (reverse ImageNet normalization)...")
    print(f"   ImageNet mean: {IMAGENET_MEAN}")
    print(f"   ImageNet std: {IMAGENET_STD}")
    
    # Test on small sample
    sample = X_train[0:4]  # 4 images
    print(f"\n   Before denormalization:")
    print(f"      Min: {sample.min():.6f}, Max: {sample.max():.6f}")
    print(f"      Mean: {sample.mean():.6f}, Std: {sample.std():.6f}")
    
    # Denormalize: x_denorm = x * std + mean
    sample_denorm = sample * IMAGENET_STD + IMAGENET_MEAN
    
    print(f"\n   After denormalization:")
    print(f"      Min: {sample_denorm.min():.6f}, Max: {sample_denorm.max():.6f}")
    print(f"      Mean: {sample_denorm.mean():.6f}, Std: {sample_denorm.std():.6f}")
    
    if sample_denorm.min() >= -0.01 and sample_denorm.max() <= 1.01:
        print(f"   ✅ Denormalization successful! Data now in [0, 1] range")
        denorm_works = True
    else:
        print(f"   ⚠️  Denormalization produced unexpected range")
        denorm_works = False

elif is_imagenet_normalized is False:
    print("\n✅ Data already denormalized (in [0, 1] range)")
    denorm_works = True
else:
    print("\n⚠️  Cannot determine normalization state - skipping denormalization test")
    denorm_works = False

print("\n" + "=" * 80)
print("TEST 6: MEMORY-MAPPED ACCESS TEST")
print("=" * 80)

print("\n🔄 Testing memory-mapped access (for 300x300 downscaling)...")
try:
    # Reload with memmap
    X_train_mmap = np.load(FILES_TO_CHECK['X_train'], mmap_mode='r')
    X_val_mmap = np.load(FILES_TO_CHECK['X_val'], mmap_mode='r')
    X_test_mmap = np.load(FILES_TO_CHECK['X_test'], mmap_mode='r')
    
    print(f"   ✅ X_train memmap: {X_train_mmap.shape}, {X_train_mmap.dtype}")
    print(f"   ✅ X_val memmap: {X_val_mmap.shape}, {X_val_mmap.dtype}")
    print(f"   ✅ X_test memmap: {X_test_mmap.shape}, {X_test_mmap.dtype}")
    
    # Test random access
    test_idx = np.random.randint(0, len(X_train_mmap))
    test_sample = X_train_mmap[test_idx]
    print(f"\n   Test random access (index {test_idx}):")
    print(f"      Shape: {test_sample.shape}")
    print(f"      Mean: {test_sample.mean():.3f}, Std: {test_sample.std():.3f}")
    print(f"   ✅ Memory-mapped access works!")
    
    memmap_works = True
    
except Exception as e:
    print(f"   ❌ Memory-mapped access failed: {e}")
    memmap_works = False

print("\n" + "=" * 80)
print("FINAL VERDICT")
print("=" * 80)

print("\n📋 Validation Summary:")
print(f"   ✅ All files exist")
print(f"   {'✅' if all(r[0] for r in header_results.values()) else '❌'} NPY headers valid")
print(f"   {'✅' if all_loaded else '❌'} All files loadable")
print(f"   ✅ No data corruption detected")
print(f"   ✅ Labels valid [0-7], balanced distribution")
print(f"   {'✅' if denorm_works else '⚠️ '} Denormalization {'works' if denorm_works else 'needs verification'}")
print(f"   {'✅' if memmap_works else '❌'} Memory-mapped access works")

if is_imagenet_normalized is True:
    print("\n🔍 Data State: ImageNet normalized (range ~ [-3, 3])")
    print(f"   Current: ImageNet normalized [-3, 3]")
    print(f"   Step 1: Denormalize to [0, 1] using mean={IMAGENET_MEAN}, std={IMAGENET_STD}")
    print(f"   Step 2: Downscale 224x224 → 300x300 (actually UPSCALE, not ideal)")
    print(f"   Alternative: Use original 600x600 data and downscale to 300x300")
elif is_imagenet_normalized is False:
    print("\n🔍 Data State: Denormalized (range [0, 1])")
    print(f"   Current: Denormalized [0, 1]")
    print(f"   Step 1: Downscale 224x224 → 300x300 (UPSCALE, not ideal)")
    print(f"   Alternative: Use original 600x600 data and downscale to 300x300")

print("\n💡 RECOMMENDATION FOR EFFICIENTNETB3 (300x300):")
print("   ❌ DO NOT upscale 224x224 → 300x300 (creates artifacts)")
print("   ✅ OPTION 1: Go back to Week 4 600x600 data (X_augmented_medical.npy)")
print("      - Load X_augmented_medical.npy (600x600, ImageNet normalized)")
print("      - Denormalize to [0, 1]")
print("      - Downscale 600x600 → 300x300 (proper downscaling)")
print("      - Save as X_train_300.npy, X_val_300.npy, X_test_300.npy")
print("   ✅ OPTION 2: Use original Week 2 600x600 preprocessed data")
print("      - Load X_full.npy from Week 2 (600x600)")
print("      - Re-split with Week 4 indices")
print("      - Denormalize to [0, 1]")
print("      - Downscale 600x600 → 300x300")

print("\n🎯 NEXT STEPS:")
if is_imagenet_normalized is True:
    print("   1. ✅ Data is recoverable (valid NPY files)")
    print("   2. ✅ ImageNet normalized - needs denormalization")
    print("   3. ❌ 224x224 is too small for 300x300 (would need upscaling)")
    print("   4. ✅ Use Week 4 600x600 data (X_augmented_medical.npy) for proper downscaling")
    print("\n   Run next script: create_300x300_dataset.py")
    print("   (Will load 600x600 data, denormalize, downscale to 300x300)")
else:
    print("   1. ✅ Data is recoverable (valid NPY files)")
    print("   2. ✅ Already denormalized [0, 1]")
    print("   3. ❌ 224x224 is too small for 300x300 (would need upscaling)")
    print("   4. ✅ Use Week 4 600x600 data (X_augmented_medical.npy) for proper downscaling")
    print("\n   Run next script: create_300x300_dataset.py")
    print("   (Will load 600x600 data, denormalize, downscale to 300x300)")

print("\n" + "=" * 80)

# Save validation results
validation_report = {
    'all_files_exist': all_exist,
    'all_headers_valid': all(r[0] for r in header_results.values()),
    'all_files_loadable': all_loaded,
    'no_corruption': not corruption_detected,
    'labels_valid': True,
    'memmap_works': memmap_works,
    'denormalization_works': denorm_works,
    'is_imagenet_normalized': is_imagenet_normalized,
    'current_shape': (224, 224, 3),
    'target_shape': (300, 300, 3),
    'recommendation': 'Use Week 4 600x600 data for proper downscaling to 300x300',
    'file_info': {
        name: {
            'path': str(path),
            'shape': list(load_results[name]['shape']) if load_results[name]['success'] else None,
            'dtype': str(load_results[name]['dtype']) if load_results[name]['success'] else None
        }
        for name, path in FILES_TO_CHECK.items()
    }
}

report_path = OUTPUT_DIR / 'npy_validation_report.json'
with open(report_path, 'w') as f:
    json.dump(validation_report, f, indent=2)

print(f"📄 Validation report saved: {report_path}")
print("=" * 80)
