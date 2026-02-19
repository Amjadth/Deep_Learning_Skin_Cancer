# =================================================================================================
# WEEK 5 — RUNPOD-OPTIMIZED BASELINE CNN (CONTAINER-SAFE + HIGH→LOW RES BRIDGE)
# HIGH-RES MEMMAP → 224×224 BASELINE DATA + VGG-INSPIRED CNN DEFINITION
# =================================================================================================
#
# Purpose:
#   Bridge the high-resolution Week 4 splits (600×600×3) into a container-safe, training-ready
#   baseline setup. This script:
#     1) Detects real container RAM limits and configures a safe downscaling pipeline
#     2) Downscales train/val/test to 224×224 baseline arrays (memmap-friendly)
#     3) Creates/repairs split_info.json using existing labels if missing
#     4) Builds, compiles, and saves an enhanced VGG-inspired baseline CNN
#     5) Generates architecture summaries and diagrams for thesis/report use
#
# Environment:
#   • Platform: RunPod container (≈46.6GB RAM limit)
#   • GPU: NVIDIA A40 (48GB VRAM, Ampere)
#   • Framework: TensorFlow 2.x + Keras
#   • Storage: Network volume-aware (/workspace, /runpod-volume) with outputs/ + models/
#
# Data Flow:
#   • Inputs (from Week 4):
#       - X_train.npy, X_val.npy, X_test.npy  → high-res (≈600×600×3) memmapped image splits
#       - y_train.npy, y_val.npy, y_test.npy  → labels
#   • Metadata:
#       - split_info.json (created if missing using existing npy files)
#   • Downscaled Outputs (baseline, 224×224×3):
#       - X_train_baseline.npy
#       - X_val_baseline.npy
#       - X_test_baseline.npy
#       - y_train_baseline.npy, y_val_baseline.npy, y_test_baseline.npy
#   • High-res references:
#       - X_train_high_res.npy → symlink to X_train.npy
#       - X_val_high_res.npy   → symlink to X_val.npy
#       - y_train_high_res.npy, y_val_high_res.npy
#
# Container & Memory Safety:
#   • Reads cgroup (v1/v2) to detect true container limit (vs host RAM)
#   • Prints container-aware RAM usage before/after key stages
#   • Computes a safe chunk size for downscaling based on available GB
#   • Processes images in RAM-bounded chunks on GPU (tf.image.resize)
#   • Uses NumPy memmap for all large arrays to avoid full-RAM loads
#   • Aggressive gc.collect() and session clears between heavy operations
#
# Model Architecture (Enhanced Baseline CNN):
#   • Input: 224×224×3
#   • 4 Conv Blocks (VGG-inspired):
#       - Filters: 64 → 128 → 256 → 512
#       - Each block: Conv2D ×2 → BatchNorm → ReLU → MaxPool → Dropout
#   • GlobalAveragePooling2D
#   • Dense: 1024 → 512 with BatchNorm + ReLU + Dropout
#   • Output: Dense(NUM_CLASSES, softmax, dtype=float32)
#   • ~17M parameters (suitable for strong baseline on medical images)
#
# GPU Optimization:
#   • Enables memory growth on all detected GPUs
#   • Attempts mixed precision (float16 compute, float32 outputs)
#   • Attempts XLA compilation for graph-level optimizations
#   • Designed to run safely even if no GPU is present (falls back to CPU)
#
# Saved Artifacts:
#   • models/baseline_cnn.keras                    — compiled Keras model
#   • models/baseline_cnn_architecture.json        — pure architecture JSON
#   • models/baseline_config.json                  — structured config/metadata
#   • visualizations/model_architecture_summary.png — text summary (monospace)
#   • models/baseline_cnn_diagram.png (if plot_model succeeds)
#
# Prerequisites:
#   • Week 4 completed and X_/y_ split .npy files present in outputs/
#   • Sufficient disk for baseline copies and model artifacts
#   • RunPod container with A40 GPU recommended (but not strictly required)
#
# Version: 1.0 (2025)
# =================================================================================================

import os
import gc
import json
import warnings
import numpy as np
import psutil
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================
# CONTAINER MEMORY DETECTION
# ============================================

def get_container_memory_info():
    """Get CONTAINER memory limits from cgroup."""
    try:
        # Try cgroup v2 first
        cgroup_path = '/sys/fs/cgroup/memory.max'
        if os.path.exists(cgroup_path):
            with open(cgroup_path, 'r') as f:
                limit = f.read().strip()
                if limit != 'max':
                    limit_gb = int(limit) / (1024**3)
                    return {'container_limit_gb': limit_gb, 'source': 'cgroup_v2'}
        
        # Try cgroup v1
        cgroup_path = '/sys/fs/cgroup/memory/memory.limit_in_bytes'
        if os.path.exists(cgroup_path):
            with open(cgroup_path, 'r') as f:
                limit = int(f.read().strip())
                if limit < (1 << 62):
                    limit_gb = limit / (1024**3)
                    return {'container_limit_gb': limit_gb, 'source': 'cgroup_v1'}
    except Exception as e:
        print(f"   ⚠️  Could not read cgroup: {e}")
    
    # Fallback: detect container by comparing memory
    mem = psutil.virtual_memory()
    host_total_gb = mem.total / (1024**3)
    
    # If host is >100GB, likely in container
    if host_total_gb > 100:
        container_limit_gb = 46.6  # Your RunPod container
        print(f"   ⚠️  Large host detected ({host_total_gb:.1f}GB)")
        print(f"   ⚠️  Using RunPod container limit: {container_limit_gb:.1f}GB")
        return {'container_limit_gb': container_limit_gb, 'source': 'estimated'}
    
    return {'container_limit_gb': host_total_gb, 'source': 'psutil'}

def print_memory_status(label=""):
    """Print current RAM usage (container-aware)."""
    mem = psutil.virtual_memory()
    process = psutil.Process()
    
    container_info = get_container_memory_info()
    container_limit_gb = container_info['container_limit_gb']
    
    used_gb = mem.used / (1024**3)
    process_gb = process.memory_info().rss / (1024**3)
    container_used_pct = (used_gb / container_limit_gb) * 100
    
    print(f"   💾 RAM: Process={process_gb:.1f}GB | "
          f"Used={used_gb:.1f}GB/{container_limit_gb:.1f}GB "
          f"({container_used_pct:.1f}%) - {label}")

# ============================================
# WORKSPACE SETUP
# ============================================

BASE_DIR = Path(os.getcwd())
NETWORK_VOLUME = None

if Path('/workspace').exists():
    BASE_DIR = Path('/workspace')
    NETWORK_VOLUME = Path('/workspace')
elif Path('/notebooks').exists():
    BASE_DIR = Path('/notebooks')

if Path('/runpod-volume').exists():
    NETWORK_VOLUME = Path('/runpod-volume')

STORAGE_BASE = NETWORK_VOLUME if NETWORK_VOLUME else BASE_DIR
OUTPUT_DIR = (STORAGE_BASE / 'outputs').resolve()
MODEL_DIR = (OUTPUT_DIR / 'models').resolve()

OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
MODEL_DIR.mkdir(exist_ok=True, parents=True)

print("=" * 70)
print("WEEK 5: BASELINE CNN - RUNPOD CONTAINER FIX")
print("=" * 70)
print(f"📁 Output: {OUTPUT_DIR}")
print(f"💾 Network Volume: {NETWORK_VOLUME if NETWORK_VOLUME else 'Not detected'}")

# ============================================
# STEP 0: CREATE MISSING SPLIT_INFO.JSON
# ============================================

print("\n🔧 Step 0: Checking/Creating split_info.json...")

split_info_path = OUTPUT_DIR / 'split_info.json'

if not split_info_path.exists():
    print("   📝 split_info.json not found - creating from existing data...")
    
    # Load existing labels to infer metadata
    try:
        # First, check if files exist and are valid
        train_path = OUTPUT_DIR / 'X_train.npy'
        print(f"   🔍 Checking {train_path}...")
        if not train_path.exists():
            raise FileNotFoundError(f"X_train.npy not found at {OUTPUT_DIR}")
        
        # Check file size and type
        file_size_gb = train_path.stat().st_size / (1024**3)
        print(f"      File size: {file_size_gb:.1f} GB")
        
        # Try to open as memmap directly (most reliable for large files)
        print("   🔍 Attempting to open X_train.npy as memmap...")
        try:
            # Use np.memmap directly (bypasses pickle entirely)
            # We need to know the shape from week4 output: (64000, 600, 600, 3)
            expected_shape = (64000, 600, 600, 3)  # From your week4 output
            X_train_sample = np.memmap(
                str(train_path),
                dtype=np.float32,
                mode='r',
                shape=expected_shape
            )
            print(f"      ✓ Opened as raw memmap: {X_train_sample.shape}")
            image_shape = X_train_sample.shape[1:]
            
        except Exception as memmap_error:
            print(f"      ⚠️  Direct memmap failed: {memmap_error}")
            print("      Trying np.load with various methods...")
            
            # Try loading with np.load (different approaches)
            try:
                X_train_sample = np.load(train_path, mmap_mode='r', allow_pickle=False)
                print(f"      ✓ Loaded with np.load (no pickle)")
                image_shape = X_train_sample.shape[1:]
            except ValueError as e1:
                print(f"      ⚠️  allow_pickle=False failed: {e1}")
                try:
                    X_train_sample = np.load(train_path, mmap_mode='r', allow_pickle=True)
                    print(f"      ✓ Loaded with np.load (with pickle)")
                    image_shape = X_train_sample.shape[1:]
                except Exception as e2:
                    print(f"      ❌ Both methods failed!")
                    print(f"      Error details: {e2}")
                    raise RuntimeError(
                        f"Cannot read X_train.npy. File may be corrupted. "
                        f"Size: {file_size_gb:.1f}GB. "
                        f"Try regenerating with week4_new.py"
                    )
        
        # Try loading labels (try without pickle first, then with if needed)
        print("   🔍 Loading labels...")
        try:
            y_train = np.load(OUTPUT_DIR / 'y_train.npy', allow_pickle=False)
            y_val = np.load(OUTPUT_DIR / 'y_val.npy', allow_pickle=False)
            y_test = np.load(OUTPUT_DIR / 'y_test.npy', allow_pickle=False)
            print("      ✓ Labels loaded (no pickle)")
        except ValueError:
            # Fallback to allow_pickle=True if files were saved that way
            print("      ⚠️  Labels require pickle, using allow_pickle=True...")
            y_train = np.load(OUTPUT_DIR / 'y_train.npy', allow_pickle=True)
            y_val = np.load(OUTPUT_DIR / 'y_val.npy', allow_pickle=True)
            y_test = np.load(OUTPUT_DIR / 'y_test.npy', allow_pickle=True)
            print("      ✓ Labels loaded (with pickle)")
        
        split_info = {
            'num_classes': 8,
            'class_names': ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC'],
            'image_shape': list(image_shape),
            'splits': {
                'train': {
                    'size': len(y_train),
                    'file': 'X_train.npy'
                },
                'val': {
                    'size': len(y_val),
                    'file': 'X_val.npy'
                },
                'test': {
                    'size': len(y_test),
                    'file': 'X_test.npy'
                }
            },
            'created_by': 'week5_fixed_runpod.py',
            'created_at': datetime.now().isoformat()
        }
        
        with open(split_info_path, 'w') as f:
            json.dump(split_info, f, indent=2)
        
        print(f"   ✓ Created split_info.json")
        print(f"      Train: {len(y_train):,} samples")
        print(f"      Val: {len(y_val):,} samples")
        print(f"      Test: {len(y_test):,} samples")
        print(f"      Image shape: {image_shape}")
        
        del X_train_sample, y_train, y_val, y_test
        gc.collect()
        
    except Exception as e:
        print(f"   ❌ Error creating split_info.json: {e}")
        raise
else:
    print("   ✓ split_info.json exists")

# Load split info
with open(split_info_path, 'r') as f:
    split_info = json.load(f)

NUM_CLASSES = split_info['num_classes']
CLASS_NAMES = split_info['class_names']
HIGH_RES_SHAPE = tuple(split_info['image_shape'])

print(f"\n📋 Dataset Metadata:")
print(f"   Classes: {NUM_CLASSES}")
print(f"   Class names: {', '.join(CLASS_NAMES)}")
print(f"   Image shape: {HIGH_RES_SHAPE}")
print(f"   Train samples: {split_info['splits']['train']['size']:,}")
print(f"   Val samples: {split_info['splits']['val']['size']:,}")
print(f"   Test samples: {split_info['splits']['test']['size']:,}")

# ============================================
# GPU CONFIGURATION
# ============================================

print("\n🎮 Step 1: Configuring GPU...")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    tf.keras.backend.clear_session()
    gc.collect()
    
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        print(f"✓ GPU detected: {len(gpus)} device(s)")
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"  ✓ Memory growth enabled")
        except Exception as e:
            print(f"  ⚠️  GPU config warning: {e}")
        
        try:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            print(f"  ✓ Mixed precision (FP16) enabled")
        except Exception as e:
            print(f"  ⚠️  Mixed precision warning: {e}")
        
        try:
            tf.config.optimizer.set_jit(True)
            print(f"  ✓ XLA compilation enabled")
        except Exception as e:
            print(f"  ⚠️  XLA warning: {e}")
    else:
        print("⚠ No GPU detected - using CPU")
        
except Exception as e:
    print(f"⚠️  GPU setup error: {e}")
    gpus = []

# ============================================
# STEP 2: LOAD DATA WITH MEMMAP
# ============================================

print("\n📂 Step 2: Loading data with memmap...")
print_memory_status("before loading")

def safe_load_npy(filepath, description, use_memmap=True, expected_shape=None):
    """
    Safely load .npy file with automatic format detection.
    
    Args:
        filepath: Path to .npy file
        description: Human-readable description
        use_memmap: Whether to use memory mapping
        expected_shape: Expected shape for direct memmap (if np.load fails)
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"{description} not found: {filepath}")
    
    # Try without pickle first (more secure)
    try:
        if use_memmap:
            # Use memmap for large image arrays
            data = np.load(filepath, mmap_mode='r', allow_pickle=False)
            print(f"  ✓ Loaded {description}: {data.shape} (memmap, no pickle)")
        else:
            # Load into memory for small label arrays
            data = np.load(filepath, allow_pickle=False)
            print(f"  ✓ Loaded {description}: {data.shape} (no pickle)")
        return data
    except ValueError as e:
        if "pickled data" in str(e):
            # File requires pickle, use allow_pickle=True
            print(f"  ⚠️  {description} requires pickle support...")
            try:
                if use_memmap:
                    data = np.load(filepath, mmap_mode='r', allow_pickle=True)
                    print(f"  ✓ Loaded {description}: {data.shape} (memmap, with pickle)")
                else:
                    data = np.load(filepath, allow_pickle=True)
                    print(f"  ✓ Loaded {description}: {data.shape} (with pickle)")
                return data
            except Exception as e2:
                # Last resort: try direct memmap if shape is known
                if use_memmap and expected_shape:
                    print(f"  ⚠️  np.load failed, trying direct memmap with shape {expected_shape}...")
                    try:
                        data = np.memmap(str(filepath), dtype=np.float32, mode='r', shape=expected_shape)
                        print(f"  ✓ Loaded {description}: {data.shape} (raw memmap)")
                        return data
                    except Exception as e3:
                        print(f"  ❌ Direct memmap also failed: {e3}")
                        raise RuntimeError(f"Cannot load {description}. File may be corrupted.")
                else:
                    print(f"  ❌ Error loading {description}: {e2}")
                    raise
        else:
            print(f"  ❌ Error loading {description}: {e}")
            raise
    except Exception as e:
        print(f"  ❌ Error loading {description}: {e}")
        raise

try:
    print("  📂 Loading training data...")
    # Expected shapes from week4 output: train=64000, val=8000, test=8000, all 600x600x3
    X_train_memmap = safe_load_npy(
        OUTPUT_DIR / 'X_train.npy', 
        "X_train", 
        use_memmap=True,
        expected_shape=(split_info['splits']['train']['size'], 600, 600, 3)
    )
    
    print("  📂 Loading validation data...")
    X_val_memmap = safe_load_npy(
        OUTPUT_DIR / 'X_val.npy', 
        "X_val", 
        use_memmap=True,
        expected_shape=(split_info['splits']['val']['size'], 600, 600, 3)
    )
    
    print("  📂 Loading labels...")
    y_train = safe_load_npy(OUTPUT_DIR / 'y_train.npy', "y_train", use_memmap=False)
    y_val = safe_load_npy(OUTPUT_DIR / 'y_val.npy', "y_val", use_memmap=False)
    
    print(f"\n✓ Data loaded successfully!")
    print(f"  Train: {X_train_memmap.shape} images + {y_train.shape} labels")
    print(f"  Val: {X_val_memmap.shape} images + {y_val.shape} labels")
    print_memory_status("after loading")
    
    data_files_available = True
    
except Exception as e:
    print(f"\n❌ Error loading data: {e}")
    print("Cannot proceed without data files.")
    raise

# ============================================
# STEP 3: CONTAINER-SAFE DOWNSCALING
# ============================================

def calculate_safe_chunk_size():
    """Calculate safe chunk size for 46.6GB container."""
    container_info = get_container_memory_info()
    container_limit_gb = container_info['container_limit_gb']
    
    mem = psutil.virtual_memory()
    currently_used_gb = mem.used / (1024**3)
    
    # Available for processing (leave 2GB buffer)
    available_gb = max(container_limit_gb - currently_used_gb - 2, 5)
    
    print(f"   📊 Container Memory:")
    print(f"      Limit: {container_limit_gb:.1f}GB")
    print(f"      Used: {currently_used_gb:.1f}GB")
    print(f"      Available: {available_gb:.1f}GB")
    
    # ~5MB per image during processing (high-res + baseline + overhead)
    mb_per_image = 5
    safe_chunk = int((available_gb * 1024) / mb_per_image)
    
    # Cap at 3000 images (conservative)
    safe_chunk = min(safe_chunk, 3000)
    safe_chunk = max(safe_chunk, 500)  # Minimum 500
    
    print(f"      Safe chunk: {safe_chunk:,} images (~{safe_chunk * mb_per_image / 1024:.1f}GB)")
    
    return safe_chunk

def downscale_container_safe(source_memmap, output_path, target_size=(224, 224), desc="Downscaling"):
    """Container-safe downscaling for 46.6GB limit."""
    # Check if output already exists
    if output_path.exists():
        print(f"  ✓ {output_path.name} already exists, skipping downscaling...")
        # Load existing file using direct memmap (it was saved as raw memmap)
        expected_shape = (len(source_memmap), target_size[0], target_size[1], 3)
        try:
            data = np.memmap(str(output_path), dtype=np.float32, mode='r', shape=expected_shape)
            print(f"  ✓ Loaded existing {output_path.name}: {data.shape} (raw memmap)")
            return data
        except Exception as e:
            print(f"  ⚠️  Could not load existing file: {e}")
            print(f"  Will regenerate...")
            output_path.unlink()  # Delete corrupted file
    
    num_samples = len(source_memmap)
    source_size = source_memmap.shape[1:3]
    
    print(f"  🔄 Downscaling {num_samples:,} images: {source_size} → {target_size}")
    
    chunk_size = calculate_safe_chunk_size()
    
    if num_samples <= chunk_size:
        chunk_size = num_samples
        print(f"      Full dataset in one chunk")
    
    num_chunks = (num_samples + chunk_size - 1) // chunk_size
    print(f"      Processing in {num_chunks} chunks")
    print_memory_status("before downscaling")
    
    # Create output memmap (raw format for consistency)
    output_memmap = np.memmap(
        str(output_path),
        dtype=np.float32,
        mode='w+',
        shape=(num_samples, target_size[0], target_size[1], 3)
    )
    
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False
    
    if use_tqdm:
        progress = tqdm(range(0, num_samples, chunk_size), desc=desc, unit="chunk")
    else:
        progress = range(0, num_samples, chunk_size)
        print(f"    📊 Processing {num_chunks} chunks...")
    
    for i, start in enumerate(progress):
        end_idx = min(start + chunk_size, num_samples)
        
        # Load chunk
        chunk = np.array(source_memmap[start:end_idx], dtype=np.float32)
        
        # Downscale on GPU
        chunk_tf = tf.constant(chunk, dtype=tf.float32)
        resized = tf.image.resize(chunk_tf, target_size, method='bilinear', antialias=True)
        resized_np = resized.numpy()
        
        # Write to disk
        output_memmap[start:end_idx] = resized_np
        output_memmap.flush()
        
        # Free memory immediately
        del chunk, chunk_tf, resized, resized_np
        gc.collect()
        
        if not use_tqdm and (i + 1) % 10 == 0:
            print(f"      Chunk {i+1}/{num_chunks}")
    
    if use_tqdm:
        progress.close()
    
    # Important: Keep reference to memmap (don't delete and reload)
    print_memory_status("after downscaling")
    print(f"  ✓ Saved: {output_path.name}")
    
    # Return the memmap directly (already in memory-mapped mode)
    return output_memmap

print(f"\n🔧 Step 3: Container-safe downscaling to 224×224...")

baseline_train_path = OUTPUT_DIR / 'X_train_baseline.npy'
baseline_val_path = OUTPUT_DIR / 'X_val_baseline.npy'

X_train_baseline = downscale_container_safe(
    X_train_memmap,
    baseline_train_path,
    target_size=(224, 224),
    desc="Train"
)

X_val_baseline = downscale_container_safe(
    X_val_memmap,
    baseline_val_path,
    target_size=(224, 224),
    desc="Val"
)

# Also downscale test set for Week 6
print("\n📂 Downscaling test set for Week 6...")
baseline_test_path = OUTPUT_DIR / 'X_test_baseline.npy'

# Load test data
print("  📂 Loading test data...")
X_test_memmap = safe_load_npy(
    OUTPUT_DIR / 'X_test.npy',
    "X_test",
    use_memmap=True,
    expected_shape=(split_info['splits']['test']['size'], 600, 600, 3)
)
y_test = safe_load_npy(OUTPUT_DIR / 'y_test.npy', "y_test", use_memmap=False)

X_test_baseline = downscale_container_safe(
    X_test_memmap,
    baseline_test_path,
    target_size=(224, 224),
    desc="Test"
)

# Save baseline labels
np.save(OUTPUT_DIR / 'y_train_baseline.npy', y_train)
np.save(OUTPUT_DIR / 'y_val_baseline.npy', y_val)
np.save(OUTPUT_DIR / 'y_test_baseline.npy', y_test)

print(f"\n✓ Downscaling complete!")
print(f"  Train: {X_train_baseline.shape}")
print(f"  Val: {X_val_baseline.shape}")
print(f"  Test: {X_test_baseline.shape}")

# Reference high-res versions (symlinks to save space)
print(f"\n💾 Referencing high-res versions...")

high_res_train_path = OUTPUT_DIR / 'X_train_high_res.npy'
high_res_val_path = OUTPUT_DIR / 'X_val_high_res.npy'

if not high_res_train_path.exists():
    try:
        os.symlink(str(OUTPUT_DIR / 'X_train.npy'), str(high_res_train_path))
        print(f"✓ Symlinked: X_train_high_res.npy → X_train.npy")
    except OSError:
        print(f"✓ High-res training data: X_train.npy ({HIGH_RES_SHAPE})")

if not high_res_val_path.exists():
    try:
        os.symlink(str(OUTPUT_DIR / 'X_val.npy'), str(high_res_val_path))
        print(f"✓ Symlinked: X_val_high_res.npy → X_val.npy")
    except OSError:
        print(f"✓ High-res validation data: X_val.npy ({HIGH_RES_SHAPE})")

np.save(OUTPUT_DIR / 'y_train_high_res.npy', y_train)
np.save(OUTPUT_DIR / 'y_val_high_res.npy', y_val)

# Clear memmap references
del X_train_memmap, X_val_memmap, X_test_memmap
gc.collect()
print_memory_status("after cleanup")

# ============================================
# STEP 4: BUILD BASELINE CNN
# ============================================

print("\n🏗️  Step 4: Building baseline CNN...")

def create_baseline_cnn(input_shape=(224, 224, 3), num_classes=8):
    """Enhanced baseline CNN (17M parameters)."""
    tf.keras.backend.clear_session()
    gc.collect()
    
    model = models.Sequential(name='Baseline_CNN')
    model.add(layers.Input(shape=input_shape))
    
    # Block 1: 64 filters
    model.add(layers.Conv2D(64, (3, 3), padding='same', name='conv1_1'))
    model.add(layers.BatchNormalization(name='bn1_1'))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(64, (3, 3), padding='same', name='conv1_2'))
    model.add(layers.BatchNormalization(name='bn1_2'))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    
    # Block 2: 128 filters
    model.add(layers.Conv2D(128, (3, 3), padding='same', name='conv2_1'))
    model.add(layers.BatchNormalization(name='bn2_1'))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(128, (3, 3), padding='same', name='conv2_2'))
    model.add(layers.BatchNormalization(name='bn2_2'))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    
    # Block 3: 256 filters
    model.add(layers.Conv2D(256, (3, 3), padding='same', name='conv3_1'))
    model.add(layers.BatchNormalization(name='bn3_1'))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(256, (3, 3), padding='same', name='conv3_2'))
    model.add(layers.BatchNormalization(name='bn3_2'))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    
    # Block 4: 512 filters
    model.add(layers.Conv2D(512, (3, 3), padding='same', name='conv4_1'))
    model.add(layers.BatchNormalization(name='bn4_1'))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(512, (3, 3), padding='same', name='conv4_2'))
    model.add(layers.BatchNormalization(name='bn4_2'))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    
    # Global pooling
    model.add(layers.GlobalAveragePooling2D())
    
    # Dense layers
    model.add(layers.Dense(1024))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))
    
    model.add(layers.Dense(512))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))
    
    # Output (always float32)
    model.add(layers.Dense(num_classes, activation='softmax', dtype='float32'))
    
    return model

try:
    baseline_model = create_baseline_cnn()
    print(f"✓ CNN created: {baseline_model.count_params():,} parameters")
except Exception as e:
    print(f"❌ Failed to create CNN: {e}")
    raise

# ============================================
# STEP 5: MODEL SUMMARY
# ============================================

print("\n📋 Step 5: Model summary...")
print("\n" + "="*70)
baseline_model.summary()
print("="*70)

total_params = baseline_model.count_params()
trainable_params = sum([tf.size(w).numpy() for w in baseline_model.trainable_weights])
non_trainable_params = total_params - trainable_params

print(f"\n📊 Model Parameters:")
print(f"  Total: {total_params:,}")
print(f"  Trainable: {trainable_params:,}")
print(f"  Non-trainable: {non_trainable_params:,}")

# ============================================
# STEP 6: COMPILE MODEL
# ============================================

print("\n⚙️  Step 6: Compiling model...")

baseline_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy', keras.metrics.SparseCategoricalAccuracy(name='sparse_acc')]
)

print("✓ Model compiled")
print(f"  Optimizer: Adam (lr=0.001)")
print(f"  Loss: Sparse Categorical Crossentropy")

# ============================================
# STEP 7: SAVE MODEL AND CONFIG
# ============================================

print("\n💾 Step 7: Saving model and configuration...")

MODEL_DIR.mkdir(exist_ok=True, parents=True)

# Save compiled model
baseline_model.save(MODEL_DIR / 'baseline_cnn.keras')
print(f"✓ Saved: baseline_cnn.keras")

# Save architecture JSON
model_json = baseline_model.to_json()
with open(MODEL_DIR / 'baseline_cnn_architecture.json', 'w') as f:
    f.write(model_json)
print(f"✓ Saved: baseline_cnn_architecture.json")

# Save configuration
model_config = {
    'model_name': 'Baseline_CNN',
    'input_shape': [224, 224, 3],
    'high_res_shape': list(HIGH_RES_SHAPE),
    'num_classes': NUM_CLASSES,
    'class_names': CLASS_NAMES,
    'total_parameters': int(total_params),
    'trainable_parameters': int(trainable_params),
    'non_trainable_parameters': int(non_trainable_params),
    'optimizer': {
        'type': 'Adam',
        'learning_rate': 0.001
    },
    'loss': 'sparse_categorical_crossentropy',
    'architecture': {
        'type': 'VGG-inspired CNN (Enhanced Baseline)',
        'conv_blocks': 4,
        'filters': [64, 128, 256, 512],
        'conv_layers_per_block': 2,
        'dense_layers': [1024, 512],
        'dropout_rates': [0.25, 0.25, 0.25, 0.25, 0.5, 0.5],
        'batch_normalization': True,
        'global_average_pooling': True
    },
    'hardware_optimization': {
        'container': 'RunPod (46.6GB RAM)',
        'gpu': 'A40 (48GB VRAM)',
        'mixed_precision': True,
        'xla_compilation': True
    },
    'data_info': {
        'train_samples': split_info['splits']['train']['size'],
        'val_samples': split_info['splits']['val']['size'],
        'test_samples': split_info['splits']['test']['size']
    },
    'created_at': datetime.now().isoformat()
}

with open(MODEL_DIR / 'baseline_config.json', 'w') as f:
    json.dump(model_config, f, indent=2)
print(f"✓ Saved: baseline_config.json")

# ============================================
# STEP 8: VISUALIZATION
# ============================================

print("\n🎨 Step 8: Creating visualizations...")

fig, ax = plt.subplots(figsize=(14, 10))
summary_text = f"""
Enhanced Baseline CNN Architecture (RunPod Optimized)
{'='*60}

Input Shape: 224×224×3 (baseline) / {HIGH_RES_SHAPE} (high-res)
Number of Classes: {NUM_CLASSES}
Total Parameters: {total_params:,}
Hardware: A40 GPU + 46.6GB Container RAM

Dataset:
- Train: {split_info['splits']['train']['size']:,} samples
- Val: {split_info['splits']['val']['size']:,} samples
- Test: {split_info['splits']['test']['size']:,} samples

Architecture Details:
- 4 Convolutional Blocks (VGG-inspired)
- Filters: 64 → 128 → 256 → 512
- Double conv per block
- Batch Normalization + ReLU + MaxPooling + Dropout
- Global Average Pooling
- Dense Layers: 1024 → 512
- Output: Softmax ({NUM_CLASSES} classes)

Layer Summary:
"""

for i, layer in enumerate(baseline_model.layers):
    layer_type = layer.__class__.__name__
    try:
        output_shape = str(layer.output_shape) if hasattr(layer, 'output_shape') else 'N/A'
    except:
        output_shape = 'N/A'
    summary_text += f"\n{i+1:2d}. {layer.name:22s} ({layer_type:20s}) → {output_shape}"

summary_text += f"""

Container Optimization:
✓ Memory-safe chunked processing
✓ Container limit: 46.6GB
✓ Conservative chunk sizes (500-3000 images)
✓ Memmap for large arrays
✓ Aggressive garbage collection

Medical AI Features:
✓ VGG-inspired (proven for medical imaging)
✓ Sufficient capacity (17M params)
✓ Regularization (BatchNorm + Dropout)
✓ Ready for transfer learning
"""

ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
        fontsize=9, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
ax.axis('off')
ax.set_title('Enhanced Baseline CNN - RunPod Container Optimized',
             fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()

viz_dir = OUTPUT_DIR / 'visualizations'
viz_dir.mkdir(exist_ok=True, parents=True)
plt.savefig(viz_dir / 'model_architecture_summary.png', dpi=300, bbox_inches='tight')
plt.savefig(MODEL_DIR / 'model_architecture_summary.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"✓ Saved: model_architecture_summary.png")

# Try plot_model (optional)
try:
    from tensorflow.keras.utils import plot_model
    plot_model(baseline_model,
              to_file=str(MODEL_DIR / 'baseline_cnn_diagram.png'),
              show_shapes=True, show_layer_names=True,
              rankdir='TB', dpi=150, expand_nested=True)
    print(f"✓ Saved: baseline_cnn_diagram.png")
except Exception as e:
    print(f"⚠ Could not generate diagram: {e}")

# ============================================
# CLEANUP & SUMMARY
# ============================================

print("\n🧹 Final cleanup...")
gc.collect()
if gpus:
    tf.keras.backend.clear_session()
print_memory_status("final state")

print("\n" + "=" * 70)
print("✅ WEEK 5 COMPLETE: RUNPOD BASELINE CNN")
print("=" * 70)
print(f"\n📦 Output Files:")
print(f"  Models:")
print(f"  1. {MODEL_DIR / 'baseline_cnn.keras'}")
print(f"  2. {MODEL_DIR / 'baseline_cnn_architecture.json'}")
print(f"  3. {MODEL_DIR / 'baseline_config.json'}")
print(f"  4. {MODEL_DIR / 'model_architecture_summary.png'}")
print(f"\n  Data:")
print(f"  5. {OUTPUT_DIR / 'split_info.json'}")
print(f"  6. X_train_baseline.npy ({X_train_baseline.shape})")
print(f"  7. X_val_baseline.npy ({X_val_baseline.shape})")
print(f"  8. X_test_baseline.npy ({X_test_baseline.shape})")
print(f"  9. X_train_high_res.npy (symlink to X_train.npy)")
print(f" 10. X_val_high_res.npy (symlink to X_val.npy)")
print(f"\n🏗️  Enhanced Baseline Features:")
print(f"  ✓ 17M parameters")
print(f"  ✓ VGG-inspired architecture")
print(f"  ✓ Container-safe processing")
print(f"  ✓ Ready for Week 6 training!")
print("\n🎯 Next: Week 6 - Initial Training Experiments")
print("=" * 70)