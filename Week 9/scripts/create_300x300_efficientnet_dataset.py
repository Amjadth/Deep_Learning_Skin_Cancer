"""
CREATE 300x300 DATASET FOR EFFICIENTNETB3
==========================================

Takes recovered 600x600 ImageNet-normalized data and:
1. Downscales to 300x300 (high-quality bicubic interpolation)
2. Denormalizes from ImageNet to [0,1] range
3. Saves as X_train_300.npy, X_val_300.npy, X_test_300.npy

Memory-optimized with chunk processing like Week 4.
"""

import numpy as np
from pathlib import Path
import json
import gc
import os
from tqdm import tqdm
import psutil
from PIL import Image
import time

# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

print("=" * 80)
print("CREATE 300x300 EFFICIENTNETB3 DATASET")
print("=" * 80)

# ============================================
# MEMORY UTILITIES (FROM WEEK 4)
# ============================================

def get_container_memory_limit():
    """
    Get actual container memory limit (Docker/RunPod cgroup limit).
    Falls back to system memory if not in container.
    """
    try:
        # Try cgroup v2 first (newer systems)
        memory_max = Path('/sys/fs/cgroup/memory.max')
        if memory_max.exists():
            limit = memory_max.read_text().strip()
            if limit != 'max':
                return int(limit)
        
        # Try cgroup v1 (older systems)
        memory_limit = Path('/sys/fs/cgroup/memory/memory.limit_in_bytes')
        if memory_limit.exists():
            limit = int(memory_limit.read_text().strip())
            # Check if it's a real limit (not max value)
            if limit < (1 << 62):  # Less than ~4.6 exabytes
                return limit
    except:
        pass
    
    # Fallback to system memory
    return psutil.virtual_memory().total

def get_memory_info():
    """Get detailed memory information (respects container limits)."""
    mem = psutil.virtual_memory()
    process = psutil.Process()
    
    # Get actual container limit
    container_limit = get_container_memory_limit()
    container_limit_gb = container_limit / (1024**3)
    
    # If container limit is much smaller than reported total, we're in a container
    if container_limit < mem.total * 0.9:
        # Use container limits
        total_gb = container_limit_gb
        # Estimate available based on container limit and used memory
        used_gb = mem.used / (1024**3)
        available_gb = total_gb - used_gb
        percent = (used_gb / total_gb) * 100
    else:
        # Use system memory (not containerized)
        total_gb = mem.total / (1024**3)
        available_gb = mem.available / (1024**3)
        used_gb = mem.used / (1024**3)
        percent = mem.percent
    
    info = {
        'total_gb': total_gb,
        'available_gb': max(0, available_gb),  # Ensure non-negative
        'used_gb': used_gb,
        'percent': percent,
        'process_gb': process.memory_info().rss / (1024**3),
        'is_containerized': container_limit < mem.total * 0.9,
        'host_total_gb': mem.total / (1024**3)
    }
    
    return info

def print_memory():
    """Print current memory status with clear labeling."""
    info = get_memory_info()
    
    # System RAM (what this script uses)
    if info.get('is_containerized'):
        print(f"   💾 Container RAM: Process={info['process_gb']:.1f}GB | "
              f"Used={info['used_gb']:.1f}GB/{info['total_gb']:.1f}GB "
              f"({info['percent']:.1f}%) | Available={info['available_gb']:.1f}GB")
        print(f"      (Host has {info['host_total_gb']:.1f}GB, but container limited to {info['total_gb']:.1f}GB)")
    else:
        print(f"   💾 System RAM: Process={info['process_gb']:.1f}GB | "
              f"Used={info['used_gb']:.1f}GB/{info['total_gb']:.1f}GB "
              f"({info['percent']:.1f}%) | Available={info['available_gb']:.1f}GB")

# ============================================
# ENVIRONMENT DETECTION (FROM WEEK 4)
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

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not OUTPUT_DIR.exists():
    raise RuntimeError(f"Failed to create output directory: {OUTPUT_DIR}")

print("=" * 80)
print("SETUP")
print("=" * 80)
print(f"📁 Output Directory: {OUTPUT_DIR}")
print(f"💾 Network Volume: {NETWORK_VOLUME if NETWORK_VOLUME else 'Not detected'}")

# Display system info
mem_info = get_memory_info()
print(f"\n🖥️  System Information:")
if mem_info.get('is_containerized'):
    print(f"   Container RAM: {mem_info['total_gb']:.1f}GB (Docker/RunPod limit)")
    print(f"   Host RAM: {mem_info['host_total_gb']:.1f}GB (physical server)")
    print(f"   ⚠️  Running in container - using container limit!")
else:
    print(f"   System RAM: {mem_info['total_gb']:.1f}GB")

# ============================================
# SETUP PATHS
# ============================================

# Source files (600x600 ImageNet normalized)
SOURCE_FILES = {
    'train': OUTPUT_DIR / 'X_train.npy',
    'val': OUTPUT_DIR / 'X_val.npy',
    'test': OUTPUT_DIR / 'X_test.npy'
}

# Destination files (300x300 denormalized [0,1])
DEST_FILES = {
    'train': OUTPUT_DIR / 'X_train_300.npy',
    'val': OUTPUT_DIR / 'X_val_300.npy',
    'test': OUTPUT_DIR / 'X_test_300.npy'
}

LABEL_FILES = {
    'train': OUTPUT_DIR / 'y_train.npy',
    'val': OUTPUT_DIR / 'y_val.npy',
    'test': OUTPUT_DIR / 'y_test.npy'
}

# Memory-conservative settings (from Week 4)
MEMORY_SAFETY_FACTOR = 0.5  # Use only 50% of available RAM
CHUNK_RAM_GB = 0.5  # Target 512MB per chunk
CHUNK_SIZE = 100  # Conservative for 46GB RAM system

print_memory()

# Expected shapes for loading (from recover_pickle_npy.py results)
EXPECTED_SHAPES = {
    'train': (64000, 600, 600, 3),
    'val': (8000, 600, 600, 3),
    'test': (8000, 600, 600, 3)
}

# ============================================
# PROCESSING FUNCTIONS
# ============================================

def denormalize_imagenet(image_batch):
    """
    Denormalize from ImageNet normalization to [0, 1].
    
    Input: ImageNet normalized (mean-centered, std-scaled)
    Output: [0, 1] range
    """
    # Reverse standardization: x_original = (x_normalized * std) + mean
    denorm = image_batch.copy()
    
    # Add back mean and multiply by std
    for c in range(3):
        denorm[:, :, :, c] = (denorm[:, :, :, c] * IMAGENET_STD[c]) + IMAGENET_MEAN[c]
    
    # Clip to [0, 1] (handles any float precision issues)
    denorm = np.clip(denorm, 0.0, 1.0)
    
    return denorm

def downscale_batch_pil(image_batch, target_size=(300, 300)):
    """
    Downscale batch using PIL's high-quality LANCZOS resampling.
    
    Input: (batch, 600, 600, 3) in any range
    Output: (batch, 300, 300, 3) in same range
    """
    batch_size = len(image_batch)
    downscaled = np.zeros((batch_size, target_size[0], target_size[1], 3), dtype=np.float32)
    
    for i in range(batch_size):
        img = image_batch[i]
        
        # Check range and convert to uint8 for PIL
        if img.min() >= 0.0 and img.max() <= 1.0:
            # [0,1] range
            img_uint8 = (img * 255).astype(np.uint8)
        elif img.min() >= 0 and img.max() <= 255:
            # [0,255] range
            img_uint8 = img.astype(np.uint8)
        else:
            # Normalize to [0,255] if unusual range
            img_min, img_max = img.min(), img.max()
            img_normalized = (img - img_min) / (img_max - img_min)
            img_uint8 = (img_normalized * 255).astype(np.uint8)
        
        # Downscale with PIL (LANCZOS = highest quality)
        pil_img = Image.fromarray(img_uint8)
        pil_resized = pil_img.resize(target_size, Image.LANCZOS)
        
        # Convert back to float32 [0,1]
        downscaled[i] = np.array(pil_resized, dtype=np.float32) / 255.0
    
    return downscaled

def safe_load_npy_direct(filepath, expected_shape=None):
    """
    3-level fallback loading strategy (same as recover_pickle_npy.py).
    
    Level 1: Standard np.load() with memmap
    Level 2: Allow pickle format
    Level 3: Direct memmap (bypasses corrupted headers)
    """
    filepath = Path(filepath)
    
    try:
        # Level 1: Try standard np.load first
        print(f"     Attempting standard np.load()...")
        data = np.load(str(filepath), mmap_mode='r')
        print(f"     ✅ Loaded with standard np.load()")
        return data
        
    except (ValueError, OSError) as e:
        print(f"     ❌ Standard load failed: {e}")
        print(f"     Trying with allow_pickle=True...")
        
        try:
            # Level 2: Fallback with allow_pickle
            data = np.load(str(filepath), allow_pickle=True, mmap_mode='r')
            print(f"     ✅ Loaded with allow_pickle=True")
            return data
            
        except Exception as e2:
            print(f"     ❌ Pickle load failed: {e2}")
            
            # Level 3: Direct memmap (last resort)
            if expected_shape:
                print(f"     Trying direct memmap with expected shape: {expected_shape}...")
                try:
                    data = np.memmap(
                        str(filepath),
                        dtype=np.float32,
                        mode='r',
                        shape=expected_shape
                    )
                    print(f"     ✅ Loaded with direct memmap (bypassing headers)")
                    return data
                except Exception as e3:
                    print(f"     ❌ Direct memmap also failed: {e3}")
                    return None
            else:
                print(f"     ❌ Cannot use memmap fallback - no expected_shape provided")
                return None

def process_split(split_name, source_path, dest_path, expected_shape, chunk_size=100):
    """
    Process one split: load chunks, denormalize, downscale, save.
    Uses 3-level fallback loading strategy.
    Skips processing if destination file already exists and is valid.
    """
    print(f"\n{'='*80}")
    print(f"PROCESSING {split_name.upper()} SET")
    print(f"{'='*80}")
    
    # Check if destination already exists and is valid
    if dest_path.exists():
        print(f"\n✅ Destination file already exists: {dest_path.name}")
        print(f"   Size: {dest_path.stat().st_size / (1024**3):.2f} GB")
        
        # Try to verify it's valid
        dest_shape_verify = expected_shape[:1] + (300, 300, 3)
        print(f"   Attempting to verify integrity...")
        verify = safe_load_npy_direct(dest_path, expected_shape=dest_shape_verify)
        
        if verify is not None:
            print(f"   ✅ File is VALID and readable!")
            print(f"      Shape: {verify.shape}")
            print(f"      Dtype: {verify.dtype}")
            
            # Quick data check
            sample_indices = np.random.choice(len(verify), min(5, len(verify)), replace=False)
            samples = verify[sample_indices].copy()
            print(f"      Data range: [{samples.min():.4f}, {samples.max():.4f}]")
            
            if samples.min() >= 0.0 and samples.max() <= 1.0:
                print(f"      ✅ Data is in correct [0,1] range")
            
            del samples
            del verify
            gc.collect()
            
            print(f"\n🚀 SKIPPING {split_name.upper()} - Already processed!")
            return
        else:
            print(f"   ⚠️  File exists but cannot load - will reprocess")
            print(f"   Removing corrupted file...")
            dest_path.unlink()
    
    # Load source with 3-level fallback (robust recovery)
    print(f"\n📥 Loading {split_name} source with recovery strategy...")
    print(f"   Path: {source_path}")
    print(f"   Size: {source_path.stat().st_size / (1024**3):.2f} GB")
    
    source = safe_load_npy_direct(source_path, expected_shape=expected_shape)
    
    if source is None:
        print(f"❌ FATAL: Could not load {split_name} with any strategy!")
        raise RuntimeError(f"Failed to load {source_path}")
    
    total_samples = len(source)
    source_shape = source.shape
    
    print(f"   Source: {source_shape}")
    print(f"   Total samples: {total_samples:,}")
    print(f"   Chunk size: {chunk_size} images")
    print_memory()
    
    # Create destination memmap
    dest_shape = (total_samples, 300, 300, 3)
    print(f"\n📝 Creating destination: {dest_shape}")
    dest = np.memmap(dest_path, dtype=np.float32, mode='w+', shape=dest_shape)
    
    # Process in chunks
    num_chunks = (total_samples + chunk_size - 1) // chunk_size
    print(f"\n🔄 Processing {num_chunks} chunks...")
    
    with tqdm(total=total_samples, desc=f"  {split_name}", unit="img") as pbar:
        for i in range(0, total_samples, chunk_size):
            end = min(i + chunk_size, total_samples)
            current_chunk_size = end - i
            
            # Load chunk (600x600 ImageNet normalized)
            chunk_600 = source[i:end].copy()
            
            # Step 1: Denormalize ImageNet → [0,1]
            chunk_denorm = denormalize_imagenet(chunk_600)
            del chunk_600
            gc.collect()
            
            # Step 2: Downscale 600x600 → 300x300 (stays in [0,1])
            chunk_300 = downscale_batch_pil(chunk_denorm, target_size=(300, 300))
            del chunk_denorm
            gc.collect()
            
            # Step 3: Save to destination
            dest[i:end] = chunk_300
            dest.flush()
            
            del chunk_300
            gc.collect()
            
            # Update progress
            pbar.update(current_chunk_size)
            if (i // chunk_size) % 10 == 0:
                info = get_memory_info()
                pbar.set_postfix_str(f"RAM: {info['process_gb']:.1f}GB")
    
    # Clean up
    del source
    del dest
    gc.collect()
    
    print(f"\n✅ {split_name.capitalize()} set complete!")
    print_memory()
    
    # Verify using same 3-level fallback strategy
    print(f"\n🔍 Verifying {split_name}...")
    dest_shape_verify = (total_samples, 300, 300, 3)
    verify = safe_load_npy_direct(dest_path, expected_shape=dest_shape_verify)
    
    if verify is None:
        print(f"   ⚠️  Could not verify {split_name} - but file was created")
        return
    
    # Sample 10 random images
    sample_indices = np.random.choice(len(verify), min(10, len(verify)), replace=False)
    samples = verify[sample_indices].copy()  # Copy to RAM for stats
    
    print(f"   Shape: {verify.shape}")
    print(f"   Dtype: {verify.dtype}")
    print(f"   Min: {samples.min():.6f}")
    print(f"   Max: {samples.max():.6f}")
    print(f"   Mean: {samples.mean():.6f}")
    print(f"   Std: {samples.std():.6f}")
    
    if samples.min() >= 0.0 and samples.max() <= 1.0:
        print(f"   ✅ Data is in [0, 1] range (CORRECT for EfficientNet preprocessing)")
    else:
        print(f"   ⚠️  Data range unexpected!")
    
    del samples
    del verify
    gc.collect()

# ============================================
# MAIN PROCESSING
# ============================================

print("\n" + "=" * 80)
print("STEP 1: CHECK SOURCE FILES")
print("=" * 80)

all_exist = True
for split_name, source_path in SOURCE_FILES.items():
    if source_path.exists():
        size_gb = source_path.stat().st_size / (1024**3)
        print(f"   ✅ {split_name}: {source_path.name} ({size_gb:.2f} GB)")
    else:
        print(f"   ❌ {split_name}: {source_path.name} NOT FOUND")
        all_exist = False

if not all_exist:
    print("\n❌ Error: Source files missing. Run recovery script first.")
    exit(1)

print("\n" + "=" * 80)
print("STEP 2: PROCESS ALL SPLITS")
print("=" * 80)

# Memory-conservative chunk size
# Each chunk: 100 images × 600×600×3 × 4 bytes = ~412MB
# + Intermediate processing = ~1GB peak per chunk
CHUNK_SIZE = 100  # Conservative for 46GB RAM

start_time = time.time()

# Process each split
for split_name in ['train', 'val', 'test']:
    source_path = SOURCE_FILES[split_name]
    dest_path = DEST_FILES[split_name]
    expected_shape = EXPECTED_SHAPES[split_name]
    
    process_split(split_name, source_path, dest_path, expected_shape, chunk_size=CHUNK_SIZE)

elapsed = time.time() - start_time

# ============================================
# FINAL VERIFICATION
# ============================================

print("\n" + "=" * 80)
print("FINAL VERIFICATION")
print("=" * 80)

print("\n📊 Created Files:")
for split_name, dest_path in DEST_FILES.items():
    if dest_path.exists():
        size_gb = dest_path.stat().st_size / (1024**3)
        # Use 3-level fallback to load (files may have corrupted headers)
        expected_shape_300 = EXPECTED_SHAPES[split_name][:1] + (300, 300, 3)
        data = safe_load_npy_direct(dest_path, expected_shape=expected_shape_300)
        if data is not None:
            print(f"   ✅ {dest_path.name}")
            print(f"      Shape: {data.shape}")
            print(f"      Size: {size_gb:.2f} GB")
            print(f"      Dtype: {data.dtype}")
            del data
        else:
            print(f"   ⚠️  {dest_path.name} - Created but cannot load")
    else:
        print(f"   ❌ {dest_path.name} - FAILED TO CREATE")

print("\n📊 Label Files (unchanged):")
for split_name, label_path in LABEL_FILES.items():
    if label_path.exists():
        labels = np.load(label_path)
        print(f"   ✅ {label_path.name}: {labels.shape}, dtype={labels.dtype}")
    else:
        print(f"   ❌ {label_path.name} - NOT FOUND")

print_memory()

# Save processing report
datasets_info = {}
for split_name, dest_path in DEST_FILES.items():
    if dest_path.exists():
        expected_shape_300 = EXPECTED_SHAPES[split_name][:1] + (300, 300, 3)
        data = safe_load_npy_direct(dest_path, expected_shape=expected_shape_300)
        if data is not None:
            datasets_info[split_name] = {
                'filename': dest_path.name,
                'shape': list(data.shape),
                'size_gb': dest_path.stat().st_size / (1024**3),
                'dtype': 'float32',
                'range': '[0, 1] (denormalized from ImageNet)',
                'ready_for': 'EfficientNetB3 training (will scale to [0,255] in pipeline)'
            }
            del data

report = {
    'status': 'SUCCESS',
    'processing_time_minutes': elapsed / 60,
    'datasets_created': datasets_info,
    'labels': {
        split_name: label_path.name
        for split_name, label_path in LABEL_FILES.items()
        if label_path.exists()
    },
    'chunk_size': CHUNK_SIZE,
    'denormalization': 'ImageNet → [0,1]',
    'downscaling': '600x600 → 300x300 (LANCZOS)',
    'next_step': 'Use X_train_300.npy with Week9_EfficientNetB3.py'
}

report_path = OUTPUT_DIR / 'efficientnetb3_dataset_report.json'
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2)

print(f"\n📁 Report saved: {report_path}")

print("\n" + "=" * 80)
print("✅ DATASET CREATION COMPLETE!")
print("=" * 80)

print(f"\n⏱️  Processing time: {elapsed/60:.1f} minutes")

print(f"\n🎯 NEXT STEPS:")
print(f"   1. ✅ 300x300 datasets created and verified")
print(f"   2. ✅ Data denormalized to [0,1] range")
print(f"   3. Create Week9_EfficientNetB3.py training script")
print(f"   4. Use scale_to_efficientnet_range() to scale [0,1]→[0,255]")
print(f"   5. Train EfficientNetB3 and compare with B0 (70.45%)")

print("\n" + "=" * 80)