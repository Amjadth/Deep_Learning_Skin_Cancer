# =================================================================================================
# WEEK 4 — MEMORY-OPTIMIZED TRAIN/VAL/TEST SPLIT (CPU-ONLY)
# LARGE-SCALE HIGH-RES DATASET SPLITTING WITH MEMMAP + MULTIPROCESSING
# =================================================================================================
#
# Purpose:
#   Create balanced, train/val/test splits from a very large, high-resolution augmented dataset
#   (e.g., ISIC 2019 at 600×600×3) in a way that is safe under strict container RAM limits
#   (RunPod / Docker). Uses disk-backed arrays and controlled multiprocessing instead of
#   loading everything into memory.
#
# Environment:
#   • Intended for RunPod / containerized environments with constrained RAM
#   • GPU: Optional (this script is CPU-only; GPU is only monitored, not used)
#   • Uses NumPy memmap + multiprocessing + psutil for memory-aware operations
#
# Highlights:
#   • Pure CPU/RAM workflow — no GPU dependency for splitting
#   • Works directly on disk-backed arrays:
#       - X_augmented_medical.npy — high-res images (memmap, read-only)
#       - y_augmented_medical.npy — labels (small, standard load)
#   • Balanced per-class sampling with configurable per-class quotas
#   • Deterministic splits via fixed RANDOM_SEED
#   • Automatically reuses existing split files if already generated
#
# Memory & Performance Strategy:
#   • Detects real container memory limits via cgroups (RunPod/Docker-aware)
#   • Uses conservative MEMORY_SAFETY_FACTOR to cap usable RAM
#   • Chunk-based copying with bounded CHUNK_RAM_GB per worker
#   • Controlled multiprocessing (MAX_WORKERS) based on available RAM and CPU count
#   • Aggressive garbage collection and memmap lifecycle management in workers
#   • Periodic memory status logging for debugging and monitoring
#
# Splitting Logic:
#   • Class-wise sampling up to SAMPLES_PER_CLASS (e.g., 10,000 per class)
#   • Per-class split targets (e.g., 8k / 1k / 1k -> train / val / test)
#   • Respects classes with fewer samples by using all available indices
#   • Shuffles indices after per-class splitting to avoid ordering bias
#   • Saves an indices checkpoint: week4_split_indices.npz
#
# Outputs (saved to persistent volume / OUTPUT_DIR):
#   • X_train.npy, y_train.npy
#   • X_val.npy,   y_val.npy
#   • X_test.npy,  y_test.npy
#   • week4_split_indices.npz — split indices for reproducibility/debugging
#
# RunPod / Storage Behavior:
#   • Detects BASE_DIR and NETWORK_VOLUME (/workspace, /notebooks, /runpod-volume)
#   • Ensures outputs/ exists both on the network volume and local workspace
#   • Creates a symlink from local outputs/ → network outputs/ when appropriate
#   • Prints a concise system summary (container RAM vs host RAM, GPU VRAM if present)
#
# Verification:
#   • Prints class distribution across train/val/test splits
#   • Reports final sample counts per split and total
#   • Provides memory snapshot at the end for sanity checking
#
# Prerequisites:
#   • Augmented dataset files already generated:
#       - X_augmented_medical.npy
#       - y_augmented_medical.npy
#   • Sufficient disk space for new split files (tens of GB for 600×600×3 float32)
#   • Container RAM and CPU sufficient for chosen CHUNK_RAM_GB and MAX_WORKERS
#
# Version: 2.0 (2025)
# =================================================================================================
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter
import json
import os
import shutil
from tqdm import tqdm
import time
from multiprocessing import Pool, cpu_count
import gc
import psutil

try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except:
    GPU_AVAILABLE = False

# ============================================
# MEMORY UTILITIES
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
    
    # Add GPU info if available
    if GPU_AVAILABLE:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            info['gpu_total_gb'] = gpu_mem.total / (1024**3)
            info['gpu_used_gb'] = gpu_mem.used / (1024**3)
            info['gpu_free_gb'] = gpu_mem.free / (1024**3)
        except:
            pass
    
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
    
    # GPU VRAM (not used by this script, just for monitoring)
    if 'gpu_total_gb' in info:
        gpu_percent = (info['gpu_used_gb'] / info['gpu_total_gb']) * 100
        print(f"   🎮 GPU VRAM: {info['gpu_used_gb']:.1f}GB/{info['gpu_total_gb']:.1f}GB "
              f"({gpu_percent:.1f}%) - NOT USED (this is CPU-only script)")

def check_memory_safe(required_gb, operation="operation"):
    """Check if we have enough memory for an operation."""
    info = get_memory_info()
    if info['available_gb'] < required_gb * 1.2:  # 20% safety margin
        print(f"⚠️  Warning: Low memory for {operation}")
        print(f"   Required: {required_gb:.2f}GB, Available: {info['available_gb']:.2f}GB")
        print(f"   Running garbage collection...")
        gc.collect()
        time.sleep(1)
        info = get_memory_info()
        if info['available_gb'] < required_gb:
            raise MemoryError(f"Insufficient memory for {operation}")
    return True    

# ============================================
# CONFIGURATION
# ============================================

# Memory-conservative settings (TUNABLE FOR YOUR 48GB SYSTEM)
MEMORY_SAFETY_FACTOR = 0.5  # Use only 50% of available RAM (conservative for 48GB)
CHUNK_RAM_GB = 0.5  # Target 512MB per chunk (conservative for 48GB system)
MAX_WORKERS = 2  # Only 2 workers for 48GB system (each can use ~0.5GB)

# Split ratios
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10
RANDOM_SEED = 42

CLASS_NAMES = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']

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
WORKSPACE_OUTPUT_DIR = (BASE_DIR / 'outputs').resolve()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
WORKSPACE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not OUTPUT_DIR.exists():
    raise RuntimeError(f"Failed to create output directory: {OUTPUT_DIR}")

# Create symlink
if NETWORK_VOLUME and OUTPUT_DIR != WORKSPACE_OUTPUT_DIR:
    try:
        if WORKSPACE_OUTPUT_DIR.exists() and not WORKSPACE_OUTPUT_DIR.is_symlink():
            backup_dir = BASE_DIR / 'outputs_backup'
            if not backup_dir.exists():
                shutil.move(str(WORKSPACE_OUTPUT_DIR), str(backup_dir))
        
        if not WORKSPACE_OUTPUT_DIR.exists() or not WORKSPACE_OUTPUT_DIR.is_symlink():
            if WORKSPACE_OUTPUT_DIR.exists():
                WORKSPACE_OUTPUT_DIR.rmdir()
            os.symlink(str(OUTPUT_DIR), str(WORKSPACE_OUTPUT_DIR))
    except Exception as e:
        OUTPUT_DIR = WORKSPACE_OUTPUT_DIR

print("=" * 70)
print("WEEK 4: MEMORY-OPTIMIZED TRAIN/VAL/TEST SPLIT")
print("=" * 70)
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
if 'gpu_total_gb' in mem_info:
    print(f"   GPU VRAM: {mem_info['gpu_total_gb']:.1f}GB (NOT used - this is CPU-only)")
else:
    print(f"   GPU: Not detected (this is CPU-only script anyway)")
print(f"   Note: Week 4 is pure CPU/RAM operations (no GPU needed)")
print(f"   GPU will be used in Week 5+ for model training")

AUGMENTED_DATA_PATH = OUTPUT_DIR / 'X_augmented_medical.npy'
AUGMENTED_LABELS_PATH = OUTPUT_DIR / 'y_augmented_medical.npy'



# ============================================
# OPTIMIZED WORKER FUNCTION
# ============================================

def read_chunk_worker(args):
    """
    Worker function with aggressive memory cleanup.
    Returns chunk data immediately, then cleans up.
    """
    chunk_start, chunk_indices, source_path, source_shape = args
    
    try:
        # Open memmap (read-only, no memory copy)
        source = np.memmap(source_path, dtype=np.float32, mode='r', shape=source_shape)
        
        # Read chunk (this is the memory allocation)
        chunk = source[chunk_indices].copy()  # Explicit copy to own memory
        
        # CRITICAL: Clean up immediately
        del source
        gc.collect()
        
        return (chunk_start, chunk)
        
    except Exception as e:
        print(f"❌ Worker error at position {chunk_start}: {e}")
        gc.collect()
        return None

# ============================================
# OPTIMIZED COPY FUNCTION
# ============================================

def copy_with_memory_control(source_path, source_shape, dest_path, indices, 
                              split_name, image_shape):
    """
    Memory-controlled sequential copy with monitoring.
    """
    print(f"\n🔄 Copying {split_name} set...")
    
    # Calculate memory requirements
    total_images = len(indices)
    bytes_per_image = np.prod(image_shape) * 4  # float32
    gb_per_image = bytes_per_image / (1024**3)
    
    # Conservative chunk size (1GB target)
    chunk_size = max(1, int(CHUNK_RAM_GB / gb_per_image))
    
    # Memory check
    mem_info = get_memory_info()
    available_gb = mem_info['available_gb'] * MEMORY_SAFETY_FACTOR
    max_safe_workers = max(1, int(available_gb / CHUNK_RAM_GB))
    num_workers = min(MAX_WORKERS, max_safe_workers, cpu_count())
    
    print(f"   Images: {total_images:,}")
    print(f"   Chunk size: {chunk_size} images (~{CHUNK_RAM_GB:.1f}GB per chunk)")
    print(f"   Workers: {num_workers} (each worker loads 1 chunk at a time)")
    print(f"   Max memory usage: ~{num_workers * CHUNK_RAM_GB:.1f}GB ({num_workers} workers × {CHUNK_RAM_GB:.1f}GB)")
    print(f"   System RAM available: {mem_info['available_gb']:.1f}GB")
    print_memory()
    
    # Create destination memmap
    dest_shape = (total_images,) + image_shape
    dest = np.memmap(dest_path, dtype=np.float32, mode='w+', shape=dest_shape)
    
    # Prepare chunks
    tasks = []
    for i in range(0, total_images, chunk_size):
        end = min(i + chunk_size, total_images)
        chunk_indices = indices[i:end]
        tasks.append((i, chunk_indices, source_path, source_shape))
    
    print(f"   Total chunks: {len(tasks)}")
    
    # Process with pool
    try:
        with Pool(processes=num_workers) as pool:
            pbar = tqdm(total=len(tasks), desc=f"  {split_name}", unit="chunk")
            
            # Use imap_unordered for efficiency
            for result in pool.imap_unordered(read_chunk_worker, tasks, chunksize=1):
                if result is not None:
                    chunk_start, chunk_data = result
                    chunk_end = chunk_start + len(chunk_data)
                    
                    # Write immediately
                    dest[chunk_start:chunk_end] = chunk_data
                    dest.flush()
                    
                    # Clean up
                    del chunk_data
                    del result
                    gc.collect()
                    
                    # Update progress
                    pbar.update(1)
                    if pbar.n % 10 == 0:  # Update memory every 10 chunks
                        info = get_memory_info()
                        pbar.set_postfix_str(f"SysRAM: {info['process_gb']:.1f}GB (total used: {info['used_gb']:.1f}GB)")
            
            pbar.close()
            
    except Exception as e:
        print(f"❌ Error during copy: {e}")
        raise
    finally:
        del dest
        gc.collect()
    
    print(f"   ✓ Completed {split_name}")
    print_memory()

# ============================================
# MAIN SCRIPT
# ============================================

print("\n" + "=" * 70)
print("STEP 1: LOADING DATASET")
print("=" * 70)

if not AUGMENTED_DATA_PATH.exists() or not AUGMENTED_LABELS_PATH.exists():
    raise FileNotFoundError(f"Augmented dataset not found in {OUTPUT_DIR}")

# Load with memmap (no RAM usage)
print("Loading images (memmap)...")
X_augmented = np.load(AUGMENTED_DATA_PATH, mmap_mode='r')
print(f"✓ Shape: {X_augmented.shape}")

# Load labels (small)
print("Loading labels...")
y_augmented = np.load(AUGMENTED_LABELS_PATH)
if y_augmented.dtype != np.int32:
    y_augmented = y_augmented.astype(np.int32)
print(f"✓ Labels: {y_augmented.shape}")

total_samples = len(X_augmented)
image_shape = X_augmented.shape[1:]

print(f"\n📊 Dataset Info:")
print(f"   Total samples: {total_samples:,}")
print(f"   Image shape: {image_shape}")
print(f"   Disk size: ~{total_samples * np.prod(image_shape) * 4 / (1024**3):.1f}GB")
print_memory()

# ============================================
# STEP 2: CREATE SPLITS
# ============================================

print("\n" + "=" * 70)
print("STEP 2: CREATING BALANCED SPLITS")
print("=" * 70)

split_files = [
    OUTPUT_DIR / 'X_train.npy', OUTPUT_DIR / 'y_train.npy',
    OUTPUT_DIR / 'X_val.npy', OUTPUT_DIR / 'y_val.npy',
    OUTPUT_DIR / 'X_test.npy', OUTPUT_DIR / 'y_test.npy'
]

if all(p.exists() for p in split_files):
    print("\n✅ Split files already exist. Loading...")
    y_train = np.load(OUTPUT_DIR / 'y_train.npy')
    y_val = np.load(OUTPUT_DIR / 'y_val.npy')
    y_test = np.load(OUTPUT_DIR / 'y_test.npy')
    print(f"   Train: {len(y_train):,}")
    print(f"   Val: {len(y_val):,}")
    print(f"   Test: {len(y_test):,}")
    
else:
    print("\n🔄 Creating per-class balanced splits (10k per class)...")
    
    SAMPLES_PER_CLASS = 10000
    TRAIN_PER_CLASS = 8000
    VAL_PER_CLASS = 1000
    TEST_PER_CLASS = 1000
    
    # Sample indices (no data loading)
    print("\n📊 Sampling indices per class...")
    selected_per_class = {}
    
    for class_idx, class_name in enumerate(CLASS_NAMES):
        mask = (y_augmented == class_idx)
        class_indices = np.where(mask)[0]
        available = len(class_indices)
        
        if available >= SAMPLES_PER_CLASS:
            np.random.seed(RANDOM_SEED + class_idx)
            selected = np.random.choice(class_indices, size=SAMPLES_PER_CLASS, replace=False)
            selected_per_class[class_idx] = np.sort(selected)
            print(f"   {class_name}: {SAMPLES_PER_CLASS:,} from {available:,}")
        else:
            selected_per_class[class_idx] = np.sort(class_indices)
            print(f"   {class_name}: {available:,} (using all)")
    
    # Split per class
    print("\n📊 Splitting per class (8k/1k/1k)...")
    train_idx_list = []
    val_idx_list = []
    test_idx_list = []
    
    for class_idx, class_name in enumerate(CLASS_NAMES):
        indices = selected_per_class[class_idx]
        n = len(indices)
        
        n_train = min(TRAIN_PER_CLASS, int(n * 0.8))
        n_val = min(VAL_PER_CLASS, int(n * 0.1))
        n_test = min(TEST_PER_CLASS, n - n_train - n_val)
        
        train_idx_list.append(indices[:n_train])
        val_idx_list.append(indices[n_train:n_train+n_val])
        test_idx_list.append(indices[n_train+n_val:n_train+n_val+n_test])
        
        print(f"   {class_name}: {n_train}/{n_val}/{n_test}")
    
    # Combine and shuffle
    train_indices = np.concatenate(train_idx_list)
    val_indices = np.concatenate(val_idx_list)
    test_indices = np.concatenate(test_idx_list)
    
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    np.random.shuffle(test_indices)
    
    print(f"\n✓ Split indices created:")
    print(f"   Train: {len(train_indices):,}")
    print(f"   Val: {len(val_indices):,}")
    print(f"   Test: {len(test_indices):,}")
    
    # Save indices checkpoint
    np.savez_compressed(OUTPUT_DIR / 'week4_split_indices.npz',
                       train_indices=train_indices,
                       val_indices=val_indices,
                       test_indices=test_indices)
    print(f"   ✓ Saved checkpoint: week4_split_indices.npz")
    
    # Copy data with memory control
    print("\n" + "=" * 70)
    print("STEP 3: COPYING DATA (MEMORY-CONTROLLED)")
    print("=" * 70)
    
    source_path = AUGMENTED_DATA_PATH
    source_shape = X_augmented.shape
    
    # Train
    copy_with_memory_control(source_path, source_shape, 
                            OUTPUT_DIR / 'X_train.npy',
                            train_indices, "Training", image_shape)
    y_train = y_augmented[train_indices].astype(np.int32)
    np.save(OUTPUT_DIR / 'y_train.npy', y_train)
    
    # Val
    copy_with_memory_control(source_path, source_shape,
                            OUTPUT_DIR / 'X_val.npy',
                            val_indices, "Validation", image_shape)
    y_val = y_augmented[val_indices].astype(np.int32)
    np.save(OUTPUT_DIR / 'y_val.npy', y_val)
    
    # Test
    copy_with_memory_control(source_path, source_shape,
                            OUTPUT_DIR / 'X_test.npy',
                            test_indices, "Test", image_shape)
    y_test = y_augmented[test_indices].astype(np.int32)
    np.save(OUTPUT_DIR / 'y_test.npy', y_test)
    
    print("\n✅ All splits saved successfully!")

# ============================================
# VERIFICATION
# ============================================

print("\n" + "=" * 70)
print("VERIFICATION")
print("=" * 70)

train_counts = Counter(y_train)
val_counts = Counter(y_val)
test_counts = Counter(y_test)

print("\nClass distribution:")
print(f"{'Class':<8} {'Train':<10} {'Val':<10} {'Test':<10}")
print("-" * 50)
for i, name in enumerate(CLASS_NAMES):
    print(f"{name:<8} {train_counts[i]:<10} {val_counts[i]:<10} {test_counts[i]:<10}")

print(f"\n✅ COMPLETE!")
print(f"   Train: {len(y_train):,} samples")
print(f"   Val: {len(y_val):,} samples")
print(f"   Test: {len(y_test):,} samples")
print(f"   Total: {len(y_train) + len(y_val) + len(y_test):,} samples")
print_memory()
print("=" * 70)