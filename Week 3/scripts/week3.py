# =================================================================================================
# WEEK 3 — TIERED MEDICAL DATA AUGMENTATION (RUNPOD + NVIDIA A40)
# LARGE-SCALE, CPU-ONLY AUGMENTATION WITH MEMMAP, CHECKPOINT/RESUME & VISUALIZATIONS
# =================================================================================================
#
# Purpose:
#   Perform large-scale, dermatology-appropriate data augmentation on the high-resolution
#   preprocessed ISIC 2019 dataset from Week 2. Uses a tiered strategy to heavily boost
#   underrepresented classes while modestly expanding abundant classes, all in a way that
#   is safe under strict container RAM and storage constraints.
#
# Environment:
#   • Platform: RunPod (containerized) with optional NVIDIA A40 GPU
#   • GPU: NVIDIA A40 (48GB VRAM, Ampere) — monitored, NOT used for augmentation
#   • Processing: CPU-only, OpenCV + NumPy + multiprocessing
#   • Storage: Network volume-aware (e.g., /workspace, /runpod-volume)
#
# Highlights:
#   • Tiered augmentation:
#       - Tier 1 (abundant classes): standard 4× augmentation (5× total incl. originals)
#       - Tier 2 (rare classes): boosted up to a fixed target per class (e.g., 10,000)
#   • Works directly on Week 2 outputs:
#       - X_full.npy — preprocessed images (memmap, 600×600×3, float32)
#       - y_full.npy — class labels
#   • Streaming, memory-safe design:
#       - Memmapped input and output arrays (no full dataset in RAM)
#       - Chunked writing with bounded buffers
#       - Parallel augmentations via multiprocessing workers
#   • Robust checkpoint/resume:
#       - JSON checkpoint with per-class offsets and write pointer
#       - Resumes safely after interruptions (e.g., Spot/Pod restarts)
#       - Validates memmap contents when resuming
#
# Storage & RunPod Behavior:
#   • Detects /workspace and distinguishes network volume vs container disk
#   • Supports alternative network volume paths (/runpod-volume, markers, size heuristics)
#   • Creates/updates outputs/ on the persistent volume
#   • Optionally symlinks workspace outputs/ → network outputs/ for convenience
#   • Verifies available disk space before writing large augmented arrays
#
# Medical Augmentation Pipeline:
#   • Geometric transforms:
#       - Random rotation, flip (H/V), zoom, translation (shifts)
#       - Elastic deformation (dermatology-appropriate warping)
#   • Photometric transforms:
#       - Brightness and contrast adjustments
#       - Gamma correction
#       - Color jitter in HSV space
#   • Regularization:
#       - Cutout-style occlusion
#       - Low-variance Gaussian noise
#   • ImageNet-aware:
#       - Optional denormalize → augment → renormalize if ImageNet stats are detected
#
# Outputs (saved to persistent OUTPUT_DIR on network volume if available):
#   • X_augmented_medical.npy  — Final augmented image tensor
#   • y_augmented_medical.npy  — Corresponding labels
#   • augmentation_config_medical.json
#   • visualizations/           — Report-ready PNGs (300 DPI):
#       - medical_augmentation_examples.png
#       - augmentation_class_distribution_comparison.png
#       - augmentation_statistics_summary.png
#
# Safety & Monitoring:
#   • Validates storage requirements vs free disk space (psutil)
#   • Option to abort if storage is insufficient
#   • Frequent flushing of memmaps and checkpoint writing
#   • Detailed class distribution and augmentation factor reporting
#
# Prerequisites:
#   • Week 2 completed successfully:
#       - X_full.npy and y_full.npy present in OUTPUT_DIR
#   • Sufficient disk space for augmented dataset (tens to hundreds of GB)
#   • Stable CPU resources for multiprocessing
#
# Version: 1.0 (2025)
# =================================================================================================

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import json
from tqdm import tqdm
import random
from collections import Counter
import os
import time
import shutil
import warnings
from multiprocessing import Pool, cpu_count # <-- ADD THIS
warnings.filterwarnings('ignore')

print("=" * 70)
print("WEEK 3: TIERED DATA AUGMENTATION - RUNPOD A40")
print("=" * 70)
print("🎮 GPU: NVIDIA A40 (48GB VRAM)")
print("💻 Processing: CPU-based augmentation (OpenCV)")
print("🔧 Note: A40 specs identical to A6000 for CPU preprocessing")
print("=" * 70)

# --------------------------------------------
# CRITICAL FIX: Network Volume Detection with /workspace Mount
# --------------------------------------------
BASE_DIR = Path(os.getcwd())
NETWORK_VOLUME = None
IS_NETWORK_VOLUME_WORKSPACE = False

print(f"\n🔍 Detecting storage configuration...")
print(f"   Current directory: {BASE_DIR}")

# CRITICAL: Check if /workspace is network volume or container storage
# Network volume will have specific markers or be larger than container disk
workspace_path = Path('/workspace')

if workspace_path.exists():
    print(f"   Found /workspace directory")
    
    # Check for RunPod network volume indicators
    # Network volumes typically have .runpod or specific markers
    network_volume_markers = [
        workspace_path / '.network_volume',  # Create this marker in your volume
        workspace_path / 'NETWORK_VOLUME_MARKER',  # Alternative marker
    ]
    
    # Check if any marker exists
    has_marker = any(marker.exists() for marker in network_volume_markers)
    
    # Check storage size (network volumes are typically much larger)
    try:
        import psutil
        workspace_disk = psutil.disk_usage(str(workspace_path))
        workspace_size_gb = workspace_disk.total / (1e9)
        
        # Network volumes are typically 100GB+ (yours is 1500GB)
        # Container disk is typically 50-100GB
        is_large_volume = workspace_size_gb > 200  # Likely network volume if >200GB
        
        print(f"   /workspace total size: {workspace_size_gb:.1f} GB")
        
        if is_large_volume or has_marker:
            NETWORK_VOLUME = workspace_path
            IS_NETWORK_VOLUME_WORKSPACE = True
            print(f"   ✓ Detected: Network volume mounted at /workspace")
            print(f"   ✓ Storage type: Persistent (survives pod restarts)")
            
            # Create marker file if it doesn't exist
            marker_file = workspace_path / '.network_volume'
            if not marker_file.exists():
                try:
                    marker_file.touch()
                    print(f"   ✓ Created network volume marker: {marker_file}")
                except Exception:
                    pass
        else:
            BASE_DIR = workspace_path
            print(f"   ⚠ /workspace appears to be container storage (not network volume)")
            print(f"   ⚠ Storage type: Temporary (may be lost on pod stop)")
            print(f"   💡 If this is incorrect, create marker: touch /workspace/.network_volume")
    except Exception as e:
        print(f"   ⚠ Could not determine storage type: {e}")
        print(f"   💡 Assuming /workspace is network volume")
        NETWORK_VOLUME = workspace_path
        IS_NETWORK_VOLUME_WORKSPACE = True
elif Path('/notebooks').exists():
    BASE_DIR = Path('/notebooks')
    print(f"   Using Jupyter workspace: {BASE_DIR}")
else:
    print(f"   Using current directory: {BASE_DIR}")

# Check for alternative network volume locations
if not NETWORK_VOLUME:
    if Path('/runpod-volume').exists():
        NETWORK_VOLUME = Path('/runpod-volume')
        print(f"   ✓ Found network volume: {NETWORK_VOLUME}")
    elif Path('/workspace/.runpod').exists():
        NETWORK_VOLUME = Path('/workspace/.runpod')
        print(f"   ✓ Found network volume: {NETWORK_VOLUME}")

# Configuration
if IS_NETWORK_VOLUME_WORKSPACE:
    # Network volume is /workspace - use it directly
    STORAGE_BASE = NETWORK_VOLUME
    OUTPUT_DIR = (STORAGE_BASE / 'outputs').resolve()
    WORKSPACE_OUTPUT_DIR = OUTPUT_DIR  # Same location
    print(f"\n📁 Storage Configuration (Network Volume at /workspace):")
    print(f"   Network volume: {NETWORK_VOLUME} (Persistent)")
    print(f"   Output directory: {OUTPUT_DIR} (Persistent)")
    print(f"   ✓ All outputs saved directly to network volume")
else:
    # Traditional setup: workspace and network volume are separate
    STORAGE_BASE = NETWORK_VOLUME if NETWORK_VOLUME else BASE_DIR
    OUTPUT_DIR = (STORAGE_BASE / 'outputs').resolve()
    WORKSPACE_OUTPUT_DIR = (BASE_DIR / 'outputs').resolve()
    
    print(f"\n📁 Storage Configuration:")
    print(f"   Base directory: {BASE_DIR}")
    print(f"   Network volume: {NETWORK_VOLUME if NETWORK_VOLUME else 'Not detected'}")
    print(f"   Output directory: {OUTPUT_DIR}")
    
    # Create symlink from workspace to network volume if needed
    if NETWORK_VOLUME and OUTPUT_DIR != WORKSPACE_OUTPUT_DIR:
        try:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            
            if WORKSPACE_OUTPUT_DIR.exists() and not WORKSPACE_OUTPUT_DIR.is_symlink():
                backup_dir = BASE_DIR / f'outputs_backup_{int(time.time())}'
                shutil.move(str(WORKSPACE_OUTPUT_DIR), str(backup_dir))
                print(f"   ⚠ Moved existing workspace outputs to: {backup_dir}")
            
            if not WORKSPACE_OUTPUT_DIR.exists():
                os.symlink(str(OUTPUT_DIR), str(WORKSPACE_OUTPUT_DIR))
                print(f"   ✓ Created symlink: {WORKSPACE_OUTPUT_DIR} -> {OUTPUT_DIR}")
            elif WORKSPACE_OUTPUT_DIR.is_symlink():
                current_target = WORKSPACE_OUTPUT_DIR.resolve()
                if current_target != OUTPUT_DIR:
                    WORKSPACE_OUTPUT_DIR.unlink()
                    os.symlink(str(OUTPUT_DIR), str(WORKSPACE_OUTPUT_DIR))
                    print(f"   ✓ Updated symlink: {WORKSPACE_OUTPUT_DIR} -> {OUTPUT_DIR}")
        except Exception as e:
            print(f"   ⚠ Could not create symlink: {e}")
            OUTPUT_DIR = WORKSPACE_OUTPUT_DIR

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROCESSED_DATA_PATH = OUTPUT_DIR / 'X_full.npy'
LABELS_PATH = OUTPUT_DIR / 'y_full.npy'

# Augmentation settings
AUGMENTATION_MULTIPLIER = 4
RANDOM_SEED = 42

# Tiered sampling strategy
TIERED_SAMPLING = True
ABUNDANT_CLASSES = ['NV', 'MEL', 'BCC', 'BKL']
RARE_CLASSES = ['AK', 'SCC', 'VASC', 'DF']
RARE_CLASS_TARGET = 10000

# Persistence & resume
CHECKPOINT_PATH = OUTPUT_DIR / 'week3_checkpoint.json'
X_MEMMAP_PATH = OUTPUT_DIR / 'X_aug_memmap.dat'
Y_MEMMAP_PATH = OUTPUT_DIR / 'y_aug_memmap.dat'

print("\n🔧 Augmentation Configuration:")
print(f"   Augmentation multiplier: {AUGMENTATION_MULTIPLIER}x")
print(f"   Tier 1 (Abundant): {', '.join(ABUNDANT_CLASSES)}")
print(f"   Tier 2 (Rare): {', '.join(RARE_CLASSES)} → {RARE_CLASS_TARGET:,} each")
print(f"   Random seed: {RANDOM_SEED}")
print(f"   Checkpoint support: ✓ (saves every 1000 images)")

# Set random seeds
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ============================================
# STEP 1: Load Preprocessed Data
# ============================================
print("\n" + "=" * 70)
print("STEP 1: LOADING PREPROCESSED DATA")
print("=" * 70)

if not PROCESSED_DATA_PATH.exists():
    print(f"❌ Error: {PROCESSED_DATA_PATH} not found!")
    print(f"\nAvailable files in {OUTPUT_DIR}:")
    for file in OUTPUT_DIR.glob("*.npy"):
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"  - {file.name} ({size_mb:.1f} MB)")
    print("\n💡 Make sure to run Week 2 preprocessing first!")
    raise FileNotFoundError(f"Processed data not found: {PROCESSED_DATA_PATH}")

if not LABELS_PATH.exists():
    print(f"❌ Error: {LABELS_PATH} not found!")
    raise FileNotFoundError(f"Labels not found: {LABELS_PATH}")

print(f"📂 Loading data from: {OUTPUT_DIR}")
print("   Loading X_original via memmap (RAM-efficient)...")
X_original = np.load(PROCESSED_DATA_PATH, mmap_mode='r')
print("   Loading y_original via memmap (RAM-efficient)...")
y_original = np.load(LABELS_PATH, mmap_mode='r')  # Use memmap for labels too
print("   ✓ Memory-map load complete.")

print(f"✓ Loaded images: {X_original.shape}")
print(f"✓ Loaded labels: {y_original.shape}")
print(f"✓ Original dataset: {len(X_original):,} images")
print(f"✓ Data type: {X_original.dtype}")
print(f"✓ Value range: [{X_original.min():.3f}, {X_original.max():.3f}]")

# Detect ImageNet normalization
IS_IMAGENET_NORMALIZED = X_original.min() < -1.0 or X_original.std() > 0.5
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

if IS_IMAGENET_NORMALIZED:
    print(f"✓ Detected ImageNet normalization")
    print(f"  Will denormalize → augment → renormalize")
else:
    print(f"✓ Images in [0, 1] range (no ImageNet normalization)")

# ============================================
# STEP 2: Calculate Storage Requirements
# ============================================
print("\n" + "=" * 70)
print("STEP 2: STORAGE PLANNING & VALIDATION")
print("=" * 70)

CLASS_NAMES = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']

# Convert y_original to array for counting (small, ~100KB for 25k labels)
y_original_array = np.array(y_original) if hasattr(y_original, '__array__') else y_original
class_counts = {i: int((y_original_array == i).sum()) for i in range(len(CLASS_NAMES))}

print(f"\n📊 Original Class Distribution:")
for i, name in enumerate(CLASS_NAMES):
    count = class_counts[i]
    percentage = (count / len(X_original)) * 100
    tier = "Tier 1" if name in ABUNDANT_CLASSES else "Tier 2" if name in RARE_CLASSES else "Default"
    print(f"  {name}: {count:5,} images ({percentage:5.2f}%) - {tier}")

# Calculate final augmented sizes
final_counts = {}
total_final = 0

print(f"\n🎯 Planned Augmented Distribution:")
for i, name in enumerate(CLASS_NAMES):
    if name in ABUNDANT_CLASSES:
        # Tier 1: original + 4x augmentation = 5x total
        final_counts[i] = class_counts[i] * (1 + AUGMENTATION_MULTIPLIER)
        print(f"  {name}: {class_counts[i]:,} → {final_counts[i]:,} (5x: original + 4x augmented)")
    elif name in RARE_CLASSES:
        # Tier 2: boost to 10,000
        final_counts[i] = max(RARE_CLASS_TARGET, class_counts[i])
        multiplier = final_counts[i] / class_counts[i] if class_counts[i] > 0 else 0
        print(f"  {name}: {class_counts[i]:,} → {final_counts[i]:,} ({multiplier:.1f}x: boosted to target)")
    else:
        # Default: 5x (shouldn't happen with current classes)
        final_counts[i] = class_counts[i] * (1 + AUGMENTATION_MULTIPLIER)
        print(f"  {name}: {class_counts[i]:,} → {final_counts[i]:,} (5x: standard)")
    total_final += final_counts[i]

print(f"\n📦 Storage Requirements (600×600×3, float32):")
bytes_per_image = X_original.shape[1] * X_original.shape[2] * X_original.shape[3] * 4  # float32 = 4 bytes
original_size_gb = len(X_original) * bytes_per_image / (1024**3)
augmented_size_gb = total_final * bytes_per_image / (1024**3)

print(f"  Original dataset: ~{original_size_gb:.1f} GB")
print(f"  Augmented dataset: ~{augmented_size_gb:.1f} GB")
print(f"  Total samples: {total_final:,} images")
print(f"  Augmentation factor: {total_final / len(X_original):.2f}x")

# Validate storage availability
try:
    import psutil
    disk_usage = psutil.disk_usage(str(OUTPUT_DIR))
    available_gb = disk_usage.free / (1024**3)
    total_gb = disk_usage.total / (1024**3)
    
    print(f"\n💾 Storage Availability:")
    print(f"  Total: {total_gb:.1f} GB")
    print(f"  Free: {available_gb:.1f} GB")
    print(f"  Required: ~{augmented_size_gb:.1f} GB")
    
    if available_gb < augmented_size_gb:
        print(f"  ⚠️ WARNING: Insufficient storage!")
        print(f"  Shortfall: {augmented_size_gb - available_gb:.1f} GB")
        print(f"  💡 Consider:")
        print(f"     - Cleaning up old files")
        print(f"     - Reducing AUGMENTATION_MULTIPLIER")
        print(f"     - Using lower resolution")
        response = input(f"\nContinue anyway? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            raise SystemExit("Aborted due to insufficient storage")
    else:
        remaining = available_gb - augmented_size_gb
        print(f"  ✓ Sufficient storage ({remaining:.1f} GB remaining after augmentation)")
except ImportError:
    print(f"  ⚠️ psutil not available - cannot verify storage")
    print(f"  Proceeding with caution...")

# ============================================
# STEP 3: Prepare Memory-Mapped Arrays
# ============================================
print("\n" + "=" * 70)
print("STEP 3: PREPARING MEMORY-MAPPED ARRAYS")
print("=" * 70)

# Calculate per-class offsets
class_write_offsets = {}
running_offset = 0
for i in range(len(CLASS_NAMES)):
    class_write_offsets[i] = running_offset
    running_offset += final_counts[i]

print(f"📍 Class Memory Offsets:")
for i, name in enumerate(CLASS_NAMES):
    start = class_write_offsets[i]
    end = start + final_counts[i]
    print(f"  {name}: [{start:,} - {end:,}] ({final_counts[i]:,} samples)")

# Check for resume
resume = False
write_ptr = 0
processed_classes = set()

if CHECKPOINT_PATH.exists() and X_MEMMAP_PATH.exists() and Y_MEMMAP_PATH.exists():
    try:
        with open(CHECKPOINT_PATH, 'r') as f:
            ckpt = json.load(f)
        
        last_completed_class = ckpt.get('last_completed_class', -1)
        current_class_idx = ckpt.get('current_class_idx', None)
        processed_classes = set(ckpt.get('processed_classes', []))
        saved_offsets = {int(k): v for k, v in ckpt.get('class_offsets', {}).items()}
        write_ptr = ckpt.get('write_ptr', 0)
        
        # Validate checkpoint integrity
        if saved_offsets == class_write_offsets:
            resume = True
            print(f"\n🔄 Resume from checkpoint:")
            print(f"  Last completed class: {CLASS_NAMES[last_completed_class] if last_completed_class >= 0 else 'None'}")
            print(f"  Processed classes: {[CLASS_NAMES[i] for i in sorted(processed_classes)]}")
            if current_class_idx is not None:
                print(f"  Partially processed: {CLASS_NAMES[current_class_idx]} (will restart)")
                processed_classes.discard(current_class_idx)
            print(f"  Images written: {write_ptr:,}")
            
            # Open existing memmaps
            X_aug_mm = np.memmap(X_MEMMAP_PATH, dtype=np.float32, mode='r+', 
                               shape=(total_final, *X_original.shape[1:]))
            y_aug_mm = np.memmap(Y_MEMMAP_PATH, dtype=np.int32, mode='r+', 
                               shape=(total_final,))
            print(f"  ✓ Opened existing memmap files")
        else:
            print(f"  ⚠️ Checkpoint offsets mismatch - starting fresh")
            resume = False
    except Exception as e:
        print(f"  ⚠️ Failed to load checkpoint: {e}")
        print(f"  Starting fresh...")
        resume = False

if not resume:
    print(f"\n🆕 Creating new memmap files:")
    print(f"  X_aug_memmap.dat: {augmented_size_gb:.2f} GB")
    print(f"  y_aug_memmap.dat: {total_final * 4 / (1024**3):.2f} GB")
    
    X_aug_mm = np.memmap(X_MEMMAP_PATH, dtype=np.float32, mode='w+', 
                        shape=(total_final, *X_original.shape[1:]))
    y_aug_mm = np.memmap(Y_MEMMAP_PATH, dtype=np.int32, mode='w+', 
                        shape=(total_final,))
    write_ptr = 0
    processed_classes = set()
    print(f"  ✓ Created new memmap files")

def save_checkpoint(last_completed_class, processed_classes_set, class_positions, 
                   total_written, current_class_idx=None):
    """Save comprehensive checkpoint for resume capability."""
    data = {
        'last_completed_class': int(last_completed_class),
        'processed_classes': [int(i) for i in processed_classes_set],
        'current_class_idx': int(current_class_idx) if current_class_idx is not None else None,
        'class_offsets': {int(k): int(v) for k, v in class_positions.items()},
        'write_ptr': int(total_written),
        'timestamp': time.time(),
        'total_planned': int(total_final),
        'augmentation_config': {
            'multiplier': AUGMENTATION_MULTIPLIER,
            'rare_class_target': RARE_CLASS_TARGET,
            'tiered_sampling': TIERED_SAMPLING
        },
        'gpu_config': {
            'type': 'A40',
            'vram_gb': 48,
            'architecture': 'Ampere'
        }
    }
    try:
        with open(CHECKPOINT_PATH, 'w') as f:
            json.dump(data, f, indent=2)
        X_aug_mm.flush()
        y_aug_mm.flush()
    except Exception as e:
        print(f"⚠️ Failed to write checkpoint: {e}")

# ============================================
# STEP 4: Medical Augmentation Functions
# ============================================
print("\n" + "=" * 70)
print("STEP 4: MEDICAL AUGMENTATION PIPELINE")
print("=" * 70)

class MedicalImageAugmentor:
    """Professional medical image augmentation for dermatology."""
    
    def __init__(self, rotation_range=20, zoom_range=0.15, shift_range=0.1,
                 horizontal_flip=True, vertical_flip=False, brightness_range=0.2,
                 contrast_range=0.2, elastic_alpha=1, elastic_sigma=50):
        self.rotation_range = rotation_range
        self.zoom_range = zoom_range
        self.shift_range = shift_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
    
    def rotate(self, image, angle=None):
        if angle is None:
            angle = np.random.uniform(-self.rotation_range, self.rotation_range)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT,
                                flags=cv2.INTER_LINEAR)
        return rotated
    
    def flip(self, image, horizontal=None, vertical=None):
        result = image.copy()
        if horizontal is None:
            horizontal = self.horizontal_flip and np.random.random() > 0.5
        if vertical is None:
            vertical = self.vertical_flip and np.random.random() > 0.5
        if horizontal:
            result = cv2.flip(result, 1)
        if vertical:
            result = cv2.flip(result, 0)
        return result
    
    def zoom(self, image, zoom_factor=None):
        if zoom_factor is None:
            zoom_factor = np.random.uniform(1 - self.zoom_range, 1 + self.zoom_range)
        h, w = image.shape[:2]
        new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        if zoom_factor > 1:
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            result = resized[start_h:start_h+h, start_w:start_w+w]
        else:
            result = np.zeros_like(image)
            start_h = (h - new_h) // 2
            start_w = (w - new_w) // 2
            result[start_h:start_h+new_h, start_w:start_w+new_w] = resized
        return result
    
    def shift(self, image, shift_x=None, shift_y=None):
        h, w = image.shape[:2]
        if shift_x is None:
            shift_x = np.random.uniform(-self.shift_range, self.shift_range) * w
        if shift_y is None:
            shift_y = np.random.uniform(-self.shift_range, self.shift_range) * h
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        shifted = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        return shifted
    
    def adjust_brightness(self, image, factor=None):
        if factor is None:
            factor = np.random.uniform(1 - self.brightness_range, 1 + self.brightness_range)
        adjusted = np.clip(image * factor, 0, 1)
        return adjusted.astype(np.float32)
    
    def adjust_contrast(self, image, factor=None):
        if factor is None:
            factor = np.random.uniform(1 - self.contrast_range, 1 + self.contrast_range)
        mean = np.mean(image)
        adjusted = np.clip((image - mean) * factor + mean, 0, 1)
        return adjusted.astype(np.float32)
    
    def add_gaussian_noise(self, image, std=0.005):
        noise = np.random.normal(0, std, image.shape)
        noisy = np.clip(image + noise, 0, 1)
        return noisy.astype(np.float32)
    
    def elastic_transform(self, image, alpha=None, sigma=None):
        if alpha is None:
            alpha = self.elastic_alpha
        if sigma is None:
            sigma = self.elastic_sigma
        h, w = image.shape[:2]
        dx = np.random.uniform(-1, 1, (h, w)) * alpha
        dy = np.random.uniform(-1, 1, (h, w)) * alpha
        dx = cv2.GaussianBlur(dx, (0, 0), sigma)
        dy = cv2.GaussianBlur(dy, (0, 0), sigma)
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        x_new = np.clip(x + dx, 0, w-1).astype(np.float32)
        y_new = np.clip(y + dy, 0, h-1).astype(np.float32)
        if len(image.shape) == 3:
            result = np.zeros_like(image)
            for c in range(image.shape[2]):
                result[:, :, c] = cv2.remap(image[:, :, c], x_new, y_new,
                                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        else:
            result = cv2.remap(image, x_new, y_new, cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT)
        return result
    
    def gamma_correction(self, image, gamma=None):
        if gamma is None:
            gamma = np.random.uniform(0.8, 1.2)
        corrected = np.power(image, gamma)
        return np.clip(corrected, 0, 1).astype(np.float32)
    
    def color_jitter(self, image, hue_shift=0.1, sat_shift=0.1, val_shift=0.1):
        hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv.astype(np.float32))
        h = (h + np.random.uniform(-hue_shift, hue_shift) * 180) % 180
        s = np.clip(s + np.random.uniform(-sat_shift, sat_shift) * 255, 0, 255)
        v = np.clip(v + np.random.uniform(-val_shift, val_shift) * 255, 0, 255)
        hsv_shifted = cv2.merge([h, s, v]).astype(np.uint8)
        rgb_shifted = cv2.cvtColor(hsv_shifted, cv2.COLOR_HSV2RGB)
        return (rgb_shifted / 255.0).astype(np.float32)
    
    def cutout(self, image, num_holes=1, hole_size=0.1):
        h, w = image.shape[:2]
        result = image.copy()
        for _ in range(num_holes):
            hole_h = int(h * hole_size * np.random.uniform(0.5, 1.5))
            hole_w = int(w * hole_size * np.random.uniform(0.5, 1.5))
            y = np.random.randint(0, h - hole_h)
            x = np.random.randint(0, w - hole_w)
            mean_val = np.mean(image)
            result[y:y+hole_h, x:x+hole_w] = mean_val
        return result
    
    def denormalize_imagenet(self, image, mean, std):
        denormalized = image * std + mean
        return np.clip(denormalized, 0, 1).astype(np.float32)
    
    def renormalize_imagenet(self, image, mean, std):
        normalized = (image - mean) / std
        return normalized.astype(np.float32)
    
    def augment(self, image, num_augmentations=1, is_imagenet_normalized=False,
                imagenet_mean=None, imagenet_std=None):
        """Apply professional medical augmentations."""
        augmented_images = []
        
        if is_imagenet_normalized and imagenet_mean is not None and imagenet_std is not None:
            working_image = self.denormalize_imagenet(image, imagenet_mean, imagenet_std)
        else:
            working_image = image.copy()
        
        for _ in range(num_augmentations):
            aug_img = working_image.copy()
            
            # Geometric transformations
            if np.random.random() > 0.3:
                aug_img = self.rotate(aug_img)
            if np.random.random() > 0.4:
                aug_img = self.flip(aug_img)
            if np.random.random() > 0.4:
                aug_img = self.zoom(aug_img)
            if np.random.random() > 0.4:
                aug_img = self.shift(aug_img)
            
            # Advanced medical augmentations
            if np.random.random() > 0.6:
                aug_img = self.elastic_transform(aug_img)
            
            # Color transformations
            if np.random.random() > 0.5:
                aug_img = self.adjust_brightness(aug_img)
            if np.random.random() > 0.5:
                aug_img = self.adjust_contrast(aug_img)
            if np.random.random() > 0.7:
                aug_img = self.gamma_correction(aug_img)
            if np.random.random() > 0.8:
                aug_img = self.color_jitter(aug_img)
            
            # Regularization
            if np.random.random() > 0.7:
                aug_img = self.cutout(aug_img)
            if np.random.random() > 0.9:
                aug_img = self.add_gaussian_noise(aug_img)
            
            # Renormalize if needed
            if is_imagenet_normalized and imagenet_mean is not None and imagenet_std is not None:
                aug_img = self.renormalize_imagenet(aug_img, imagenet_mean, imagenet_std)
            
            augmented_images.append(aug_img)
        
        return augmented_images

# ============================================
# WORKER FUNCTION FOR PARALLELISM (MOVED OUTSIDE CLASS)
# ============================================
def worker_augment_image(args):
    """
    Worker function for multiprocessing.
    Accepts image index instead of image data to avoid loading all images into RAM.
    Creates augmentor in worker process for compatibility with 'spawn' multiprocessing.
    """
    img_index, num_augs, class_idx, is_norm, mean, std, x_original_path, x_shape = args
    
    try:
        # Load image from memmap in worker process (memory-efficient)
        # Each worker opens its own memmap view
        x_mm = np.memmap(x_original_path, dtype=np.float32, mode='r', 
                        shape=x_shape)
        img = np.array(x_mm[img_index])  # Copy single image to worker memory
        
        # Create augmentor in worker process (safe for 'spawn' multiprocessing)
        # This ensures compatibility across different multiprocessing backends
        worker_augmentor = MedicalImageAugmentor(
            rotation_range=20,
            zoom_range=0.15,
            shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=False,
            brightness_range=0.2,
            contrast_range=0.2,
            elastic_alpha=1,
            elastic_sigma=50
        )
        
        aug_images = worker_augmentor.augment(
            img,
            num_augmentations=num_augs,
            is_imagenet_normalized=is_norm,
            imagenet_mean=mean if is_norm else None,
            imagenet_std=std if is_norm else None
        )
        aug_labels = [class_idx] * len(aug_images)
        return (aug_images, aug_labels)
    except Exception as e:
        print(f"Error augmenting image {img_index}: {e}")
        return ([], [])
    
# Initialize augmentor
augmentor = MedicalImageAugmentor(
    rotation_range=20,
    zoom_range=0.15,
    shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False,
    brightness_range=0.2,
    contrast_range=0.2,
    elastic_alpha=1,
    elastic_sigma=50
)

print(f"✓ Medical augmentation pipeline initialized")
print(f"  Rotation: ±{augmentor.rotation_range}°")
print(f"  Zoom: ±{augmentor.zoom_range*100}%")
print(f"  Shift: ±{augmentor.shift_range*100}%")
print(f"  Horizontal flip: {augmentor.horizontal_flip}")
print(f"  Vertical flip: {augmentor.vertical_flip}")

# ============================================
# STEP 5: Apply Tiered Augmentation
# ============================================
print("\n" + "=" * 70)
print("STEP 5: APPLYING TIERED AUGMENTATION")
print("=" * 70)

def apply_tiered_augmentation(X_original, y_original, y_original_array):
    """Apply tiered augmentation with streaming to disk (PARALLELIZED)."""
    global write_ptr, processed_classes
    
    # Create a global progress bar for total images written (original + augmented)
    total_progress = tqdm(total=total_final, initial=write_ptr, desc="Total augmentation", unit="img")
    
    # Determine starting class
    start_class_idx = 0
    if resume and processed_classes:
        for i in range(len(CLASS_NAMES)):
            if i not in processed_classes:
                start_class_idx = i
                break
        else:
            print("✓ All classes already processed!")
            total_progress.close()
            return

    BUFFER_SIZE = 1000  # We will write to memmap every 1000 *augmented* images
    
    # Setup multiprocessing pool
    num_workers = cpu_count()
    print(f"\n⚡ Initializing augmentation pool with {num_workers} CPU workers...")

    # We use 'with' to ensure the pool is closed correctly
    with Pool(processes=num_workers) as pool:
        for class_idx in range(start_class_idx, len(CLASS_NAMES)):
            class_name = CLASS_NAMES[class_idx]
            
            if class_idx in processed_classes:
                print(f"⏭️  Skipping {class_name} (already processed)")
                continue
            
            # Use y_original_array (already loaded) for efficient indexing
            class_indices = np.where(y_original_array == class_idx)[0]
            
            print(f"\n📊 Processing {class_name} class ({len(class_indices):,} images)...")
            
            if len(class_indices) == 0:
                print(f"  ⚠ No images found for {class_name}. Skipping.")
                processed_classes.add(class_idx)
                save_checkpoint(class_idx, processed_classes, class_write_offsets, write_ptr, current_class_idx=None)
                continue

            # Determine target counts
            if class_name in ABUNDANT_CLASSES:
                target_count = len(class_indices) * (1 + AUGMENTATION_MULTIPLIER)
                num_augs_per_img = AUGMENTATION_MULTIPLIER
                include_originals = True
                print(f"  Tier 1: Target {target_count:,} ({len(class_indices):,} orig + {len(class_indices) * num_augs_per_img:,} aug)")
            elif class_name in RARE_CLASSES:
                target_count = RARE_CLASS_TARGET
                include_originals = True
                # Calculate needed augmentations
                needed = target_count - len(class_indices)
                num_augs_per_img = int(np.ceil(needed / len(class_indices))) if len(class_indices) > 0 else 0
                if num_augs_per_img <= 0:
                    print(f"  Tier 2: Original ({len(class_indices)}) already meets target ({target_count}). No augmentation.")
                    num_augs_per_img = 0
                    target_count = len(class_indices) # Cap at original count
                else:
                    print(f"  Tier 2: Target {target_count:,} (boosting from {len(class_indices):,})")
                    print(f"  Generating ~{num_augs_per_img} augmentations per original image...")
            else:
                # Default case (shouldn't be hit, but safe)
                target_count = len(class_indices) * (1 + AUGMENTATION_MULTIPLIER)
                num_augs_per_img = AUGMENTATION_MULTIPLIER
                include_originals = True
                print(f"  Default: Target {target_count:,}")

            start_offset = class_write_offsets[class_idx]
            end_offset = start_offset + target_count
            
            # RESUME FIX: Check if this class was partially processed
            # Use checkpoint write_ptr and verify against memmap
            existing_samples = 0  # Initialize to 0 (no existing samples)
            skip_originals = False
            
            if resume:
                # Always check memmap for current class (memmap is source of truth, checkpoint may be stale)
                # Since checkpoint may be stale (not updated due to crash), check the entire class range
                max_check = min(start_offset + target_count, len(y_aug_mm))
                # Check from class start to class end (or memmap end) to find all existing data
                # This handles cases where checkpoint is stale but data was written
                check_end = max_check
                
                # Quick check: does the first sample of this class exist?
                if check_end > start_offset:
                    first_label = y_aug_mm[start_offset]
                    if first_label == class_idx:
                        # Class data exists - find how many consecutive valid samples
                        print(f"  🔍 Checking memmap for existing samples (checking up to position {check_end:,})...")
                        
                        # Efficient check: use larger chunks, stop at first invalid sample
                        chunk_size = min(10000, target_count // 5)  # Larger chunks for speed
                        existing_samples = 0
                        
                        # Check chunks sequentially
                        for chunk_start in range(start_offset, check_end, chunk_size):
                            chunk_end_check = min(chunk_start + chunk_size, check_end)
                            chunk_labels = y_aug_mm[chunk_start:chunk_end_check]
                            
                            # Find where valid labels end
                            valid_mask = (chunk_labels == class_idx)
                            if np.all(valid_mask):
                                # Entire chunk is valid
                                existing_samples = chunk_end_check - start_offset
                            else:
                                # Found invalid - find last valid position
                                valid_positions = np.where(valid_mask)[0]
                                if len(valid_positions) > 0:
                                    # There are some valid samples in this chunk
                                    last_valid = valid_positions[-1]
                                    existing_samples = (chunk_start - start_offset) + last_valid + 1
                                # Stop checking - found invalid data
                                break
                        
                        if existing_samples > 0:
                            print(f"  🔄 Detected {existing_samples:,} existing samples in memmap")
                            
                            # Quick spot-check: verify a few random samples
                            num_verify = min(5, existing_samples // 1000)  # Check 5 samples or 1 per 1000
                            if num_verify > 0:
                                verify_positions = np.linspace(start_offset, start_offset + existing_samples - 1, num_verify, dtype=int)
                                verify_labels = y_aug_mm[verify_positions]
                                if np.all(verify_labels == class_idx):
                                    print(f"  ✓ Verified {num_verify} random samples - data looks valid")
                                else:
                                    # Some verification failed - find the last completely valid position
                                    print(f"  ⚠️  Verification found inconsistencies, being conservative...")
                                    # Find the last position where all samples up to that point are valid
                                    # (simplified: just reduce by a safety margin)
                                    existing_samples = max(0, existing_samples - chunk_size)
            
            # Determine starting position based on existing samples
            if existing_samples > 0:
                # Use existing samples
                current_pos = start_offset + existing_samples
                samples_generated = existing_samples
                total_progress.update(existing_samples)
                skip_originals = (existing_samples >= len(class_indices))
                print(f"  ✓ Resuming from {existing_samples:,} existing samples")
            else:
                # Start fresh
                current_pos = start_offset
                samples_generated = 0
                skip_originals = False
            
            try:
                # === 1. Write Original Images (if needed) - MEMORY-EFFICIENT CHUNKED COPY ===
                if include_originals and not skip_originals:
                    orig_count = len(class_indices)
                    # Check if we need to write originals or if they're already there
                    if existing_samples >= orig_count:
                        print(f"  ✓ Originals already written ({orig_count:,} images)")
                        # Verify they're correct (quick check)
                        if current_pos == start_offset + orig_count:
                            print(f"  ✓ Originals verified, skipping copy")
                        else:
                            # Need to adjust position
                            current_pos = start_offset + orig_count
                            samples_generated = orig_count
                            total_progress.update(orig_count)
                    else:
                        # Write remaining originals if any
                        origs_to_write = orig_count - existing_samples
                        if origs_to_write > 0:
                            print(f"  📋 Copying {origs_to_write:,} remaining original images (chunked to avoid RAM spike)...")
                            
                            # Copy originals in chunks to avoid loading all into RAM at once
                            COPY_CHUNK_SIZE = 100  # Copy 100 images at a time
                            write_start_idx = existing_samples
                            for chunk_start in range(write_start_idx, orig_count, COPY_CHUNK_SIZE):
                                chunk_end = min(chunk_start + COPY_CHUNK_SIZE, orig_count)
                                chunk_indices = class_indices[chunk_start:chunk_end]
                                
                                # Load chunk from memmap (small chunks = low RAM usage)
                                chunk_images = X_original[chunk_indices].astype(np.float32)
                                
                                # Write chunk to output memmap
                                write_start = start_offset + chunk_start
                                write_end = start_offset + chunk_end
                                X_aug_mm[write_start:write_end] = chunk_images
                                y_aug_mm[write_start:write_end] = class_idx
                                
                                # Clear chunk from memory
                                del chunk_images
                            
                            current_pos = start_offset + orig_count
                            samples_generated = orig_count
                            total_progress.update(origs_to_write)
                            print(f"  ✓ Copied {origs_to_write:,} original images")
                        else:
                            current_pos = start_offset + orig_count
                            samples_generated = orig_count
                elif not include_originals:
                    # Rare classes might not include originals in some cases
                    pass

                # === 2. Generate and Write Augmentations (in Parallel) - MEMORY-EFFICIENT ===
                if num_augs_per_img > 0:
                    # Calculate how many more samples are needed
                    remaining_needed = target_count - samples_generated
                    
                    if remaining_needed <= 0:
                        print(f"  ✓ All augmentations already complete ({samples_generated:,}/{target_count:,})")
                    else:
                        print(f"  📊 Generating {remaining_needed:,} remaining augmentations...")
                        
                        # Prepare tasks for the multiprocessing pool
                        # MEMORY FIX: Pass indices instead of images to avoid loading all images into RAM
                        tasks = []
                        x_original_path_str = str(PROCESSED_DATA_PATH)  # Pass path as string for worker processes
                        x_original_shape = X_original.shape
                        
                        # For RARE classes, we need to cycle through originals
                        if class_name in RARE_CLASSES:
                            # Generate exactly the number needed
                            for i in range(remaining_needed):
                                # Pick a random original image index for this augmentation
                                img_index = class_indices[i % len(class_indices)]
                                # Pass index, not image data (worker will load from memmap)
                                tasks.append((img_index, 1, class_idx, IS_IMAGENET_NORMALIZED, IMAGENET_MEAN, IMAGENET_STD, 
                                             x_original_path_str, x_original_shape))
                        else:
                            # For ABUNDANT classes: calculate how many images need augmentation
                            # Each image needs num_augs_per_img augmentations
                            # If we've already written some, we need to figure out which images are done
                            
                            # Simple approach: if samples_generated == orig_count, all originals are written, generate all augs
                            # If samples_generated > orig_count, some augs are written
                            orig_count = len(class_indices)
                            augs_written = max(0, samples_generated - orig_count)
                            
                            if augs_written == 0:
                                # No augmentations written yet, generate all
                                for img_index in class_indices:
                                    tasks.append((img_index, num_augs_per_img, class_idx, IS_IMAGENET_NORMALIZED, IMAGENET_MEAN, IMAGENET_STD,
                                                 x_original_path_str, x_original_shape))
                            else:
                                # Some augmentations already written
                                # Calculate how many more augmentations we need
                                total_augs_needed = orig_count * num_augs_per_img
                                augs_remaining = total_augs_needed - augs_written
                                
                                if augs_remaining > 0:
                                    # Some augmentations already written, but we don't know which specific ones
                                    # Since augmentations are written out of order (parallel processing),
                                    # we can't precisely track which images are fully augmented
                                    # Safe approach: generate remaining augmentations one at a time,
                                    # cycling through images to ensure diversity
                                    for i in range(remaining_needed):
                                        # Cycle through images to ensure all images contribute to remaining augs
                                        img_idx = class_indices[i % len(class_indices)]
                                        # Generate 1 augmentation at a time to match remaining count exactly
                                        tasks.append((img_idx, 1, class_idx, IS_IMAGENET_NORMALIZED, IMAGENET_MEAN, IMAGENET_STD,
                                                     x_original_path_str, x_original_shape))
                        
                        if len(tasks) > 0:
                            aug_buffer = []
                            aug_labels = []
                            MAX_BUFFER_SIZE = BUFFER_SIZE * 2  # Allow buffer to grow to 2x before forcing write
                            
                            # Use imap_unordered for speed. It's a parallel generator.
                            # Results come back as soon as they're done, not in order.
                            pbar_desc = f"Augmenting {class_name} (Parallel)"
                            for aug_img_list, aug_label_list in tqdm(pool.imap_unordered(worker_augment_image, tasks), total=len(tasks), desc=pbar_desc):
                                
                                aug_buffer.extend(aug_img_list)
                                aug_labels.extend(aug_label_list)

                                # MEMORY FIX: Write to memmap when buffer reaches threshold to prevent RAM growth
                                # Force write if buffer gets too large, even if not at BUFFER_SIZE
                                while len(aug_buffer) >= BUFFER_SIZE or len(aug_buffer) > MAX_BUFFER_SIZE:
                                    write_count = min(len(aug_buffer), BUFFER_SIZE, end_offset - current_pos)
                                    if write_count <= 0:
                                        aug_buffer = [] # Reached target, clear buffer
                                        aug_labels = []
                                        break
                                    
                                    # Convert buffer slice to numpy array and write to memmap
                                    X_aug_mm[current_pos : current_pos + write_count] = np.array(aug_buffer[:write_count], dtype=np.float32)
                                    y_aug_mm[current_pos : current_pos + write_count] = np.array(aug_labels[:write_count], dtype=np.int32)
                                    
                                    current_pos += write_count
                                    samples_generated += write_count
                                    total_progress.update(write_count)
                                    write_ptr = current_pos # Update global write_ptr
                                    
                                    # Remove written items from buffer
                                    aug_buffer = aug_buffer[write_count:]
                                    aug_labels = aug_labels[write_count:]
                                    
                                    # Save checkpoint periodically (every 1000 images = every buffer write)
                                    # This ensures frequent checkpoints for crash recovery
                                    if samples_generated % BUFFER_SIZE == 0:  # Every buffer write (1000 images)
                                        prev_completed = max([i for i in processed_classes] + [-1])
                                        save_checkpoint(prev_completed, processed_classes, class_write_offsets, write_ptr, current_class_idx=class_idx)
                                
                                # Early exit if we've reached the target
                                if samples_generated >= target_count:
                                    break
                            
                            # Write any remaining images from the buffer
                            if len(aug_buffer) > 0:
                                write_count = min(len(aug_buffer), end_offset - current_pos, target_count - samples_generated)
                                if write_count > 0:
                                    X_aug_mm[current_pos : current_pos + write_count] = np.array(aug_buffer[:write_count], dtype=np.float32)
                                    y_aug_mm[current_pos : current_pos + write_count] = np.array(aug_labels[:write_count], dtype=np.int32)
                                    current_pos += write_count
                                    samples_generated += write_count
                                    total_progress.update(write_count)
                                    write_ptr = current_pos
                                    
                                    # Clear buffer
                                    aug_buffer = []
                                    aug_labels = []
            except Exception as e:
                print(f"  ❌ Error processing {class_name}: {e}")
                print(f"  💾 Checkpoint saved - can resume")
                raise
            else:
                # Mark class as fully processed
                processed_classes.add(class_idx)
                write_ptr = current_pos  # Final update for this class
                save_checkpoint(class_idx, processed_classes, class_write_offsets, write_ptr, current_class_idx=None)

                print(f"  ✅ {class_name}: {samples_generated:,} samples written")
    
    total_progress.close()

# Apply augmentation
start_time = time.time()
apply_tiered_augmentation(X_original, y_original, y_original_array)
elapsed = time.time() - start_time

print(f"\n✅ Augmentation complete!")
print(f"  Total: {total_final:,} images")
print(f"  Factor: {total_final / len(X_original):.2f}x")
print(f"  Time: {elapsed/60:.2f} minutes")
print(f"  Rate: {total_final / elapsed:.1f} images/second")

# ============================================
# STEP 6: Save Final Dataset (MEMORY-EFFICIENT)
# ============================================
print("\n" + "=" * 70)
print("STEP 6: SAVING AUGMENTED DATASET")
print("=" * 70)

X_aug_output_path = OUTPUT_DIR / 'X_augmented_medical.npy'
y_aug_output_path = OUTPUT_DIR / 'y_augmented_medical.npy'

# Check if files already exist
if X_aug_output_path.exists() and y_aug_output_path.exists():
    try:
        X_existing = np.load(X_aug_output_path, mmap_mode='r')
        y_existing = np.load(y_aug_output_path, mmap_mode='r')
        if len(X_existing) == total_final and len(y_existing) == total_final:
            print(f"✓ Output files already exist with correct size ({total_final:,} images)")
            print(f"  Skipping save step (files are already saved)")
            X_aug_view = X_existing
            y_aug_view = y_existing
        else:
            print(f"⚠ Output files exist but size mismatch. Re-saving...")
            X_aug_output_path.unlink(missing_ok=True)
            y_aug_output_path.unlink(missing_ok=True)
            raise FileNotFoundError("Files need to be recreated")
    except Exception as e:
        print(f"⚠ Error checking existing files: {e}")
        print(f"  Re-saving output files...")
        X_aug_output_path.unlink(missing_ok=True)
        y_aug_output_path.unlink(missing_ok=True)

# Save files if they don't exist or were deleted
if not X_aug_output_path.exists() or not y_aug_output_path.exists():
    X_aug_view_mm = np.memmap(X_MEMMAP_PATH, dtype=np.float32, mode='r', shape=(total_final, *X_original.shape[1:]))
    y_aug_view_mm = np.memmap(Y_MEMMAP_PATH, dtype=np.int32, mode='r', shape=(total_final,))
    
    print(f"💾 Saving to network volume (memory-efficient, may take 10-30 minutes)...")
    print(f"   Saving X_augmented_medical.npy from memmap...")
    
    # MEMORY FIX: Save directly from memmap (np.save handles memmaps efficiently internally)
    # This will read from disk in chunks and write to .npy file without loading entire array into RAM
    start_save_time = time.time()
    np.save(X_aug_output_path, X_aug_view_mm)
    save_time = time.time() - start_save_time
    print(f"   ✓ Saved X_augmented_medical.npy in {save_time/60:.2f} minutes")
    
    # Save y (much smaller, fast)
    print(f"   Saving y_augmented_medical.npy...")
    np.save(y_aug_output_path, y_aug_view_mm)
    print(f"   ✓ Saved y_augmented_medical.npy")
    
    # Load for visualizations (memory-mapped, read-only)
    X_aug_view = np.load(X_aug_output_path, mmap_mode='r')
    y_aug_view = np.load(y_aug_output_path, mmap_mode='r')
    
    del X_aug_view_mm, y_aug_view_mm  # Clean up memmap views
else:
    # Files already exist, load for visualizations
    X_aug_view = np.load(X_aug_output_path, mmap_mode='r')
    y_aug_view = np.load(y_aug_output_path, mmap_mode='r')

aug_config = {
    'original_size': len(X_original),
    'augmented_size': int(total_final),
    'augmentation_multiplier': AUGMENTATION_MULTIPLIER,
    'augmentation_params': {
        'rotation_range': augmentor.rotation_range,
        'zoom_range': augmentor.zoom_range,
        'shift_range': augmentor.shift_range,
        'horizontal_flip': augmentor.horizontal_flip,
        'vertical_flip': augmentor.vertical_flip,
        'brightness_range': augmentor.brightness_range,
        'contrast_range': augmentor.contrast_range
    },
    'tiered_sampling': {
        'tier1_classes': ABUNDANT_CLASSES,
        'tier2_classes': RARE_CLASSES,
        'rare_class_target': RARE_CLASS_TARGET
    },
    'gpu_config': {
        'type': 'A40',
        'vram_gb': 48,
        'architecture': 'Ampere',
        'note': 'CPU-based preprocessing'
    },
    'storage_config': {
        'network_volume': str(NETWORK_VOLUME) if NETWORK_VOLUME else None,
        'output_dir': str(OUTPUT_DIR),
        'is_persistent': NETWORK_VOLUME is not None
    },
    'random_seed': RANDOM_SEED,
    'medical_appropriate': True
}

with open(OUTPUT_DIR / 'augmentation_config_medical.json', 'w') as f:
    json.dump(aug_config, f, indent=2)

print(f"✓ Saved: X_augmented_medical.npy")
print(f"✓ Saved: y_augmented_medical.npy")
print(f"✓ Saved: augmentation_config_medical.json")

# ============================================
# STEP 7: Create Visualizations
# ============================================
print("\n" + "=" * 70)
print("STEP 7: CREATING VISUALIZATIONS")
print("=" * 70)

viz_dir = OUTPUT_DIR / 'visualizations'
viz_dir.mkdir(exist_ok=True, parents=True)
print(f"✓ Visualizations directory: {viz_dir}")

def denormalize_for_display(img):
    if IS_IMAGENET_NORMALIZED:
        denormalized = img * IMAGENET_STD + IMAGENET_MEAN
        return np.clip(denormalized, 0, 1)
    return np.clip(img, 0, 1)

# Visualization 1: Single image augmentation examples
print("  🖼️  Creating augmentation examples...")
sample_idx = np.random.randint(0, len(X_original))
sample_image = np.array(X_original[sample_idx])  # Load single image from memmap
sample_label = CLASS_NAMES[y_original_array[sample_idx]]  # Use y_original_array

aug_examples = augmentor.augment(
    sample_image, num_augmentations=8,
    is_imagenet_normalized=IS_IMAGENET_NORMALIZED,
    imagenet_mean=IMAGENET_MEAN if IS_IMAGENET_NORMALIZED else None,
    imagenet_std=IMAGENET_STD if IS_IMAGENET_NORMALIZED else None
)

fig, axes = plt.subplots(3, 3, figsize=(18, 18))
fig.suptitle(f'Medical Augmentation Examples - Class: {sample_label}', fontsize=18, fontweight='bold')

axes[0, 0].imshow(denormalize_for_display(sample_image))
axes[0, 0].set_title('Original', fontsize=14, fontweight='bold')
axes[0, 0].axis('off')

for i, (ax, aug_img) in enumerate(zip(axes.flat[1:], aug_examples)):
    ax.imshow(denormalize_for_display(aug_img))
    ax.set_title(f'Augmented {i+1}', fontsize=12, fontweight='bold')
    ax.axis('off')

plt.tight_layout()
plt.savefig(viz_dir / 'medical_augmentation_examples.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"    ✓ Saved: medical_augmentation_examples.png")

# Visualization 2: Class distribution comparison
print("  📊 Creating class distribution comparison...")
# Use y_original_array (already loaded in Step 2) for counting
original_counts = Counter(y_original_array)
# MEMORY FIX: Load labels into memory for counting (small, ~320 KB for 80k labels)
y_aug_labels_array = np.array(y_aug_view) if hasattr(y_aug_view, '__array__') else y_aug_view
augmented_counts = Counter(y_aug_labels_array)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

class_names_sorted = [CLASS_NAMES[i] for i in sorted(original_counts.keys())]
original_counts_sorted = [original_counts[i] for i in sorted(original_counts.keys())]
augmented_counts_sorted = [augmented_counts.get(i, 0) for i in sorted(original_counts.keys())]
colors = plt.cm.Set3(np.linspace(0, 1, len(original_counts)))

bars1 = ax1.bar(class_names_sorted, original_counts_sorted, color=colors, edgecolor='black', linewidth=1.5)
ax1.set_xlabel('Class', fontsize=14, fontweight='bold')
ax1.set_ylabel('Number of Images', fontsize=14, fontweight='bold')
ax1.set_title('Before Augmentation', fontsize=16, fontweight='bold')
ax1.grid(axis='y', alpha=0.3, linestyle='--')
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height, f'{int(height):,}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

bars2 = ax2.bar(class_names_sorted, augmented_counts_sorted, color=colors, edgecolor='black', linewidth=1.5)
ax2.set_xlabel('Class', fontsize=14, fontweight='bold')
ax2.set_ylabel('Number of Images', fontsize=14, fontweight='bold')
ax2.set_title('After Tiered Augmentation', fontsize=16, fontweight='bold')
ax2.grid(axis='y', alpha=0.3, linestyle='--')
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height, f'{int(height):,}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig(viz_dir / 'augmentation_class_distribution_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"    ✓ Saved: augmentation_class_distribution_comparison.png")

# Visualization 3: Statistics summary
print("  📄 Creating statistics summary...")
fig, ax = plt.subplots(figsize=(16, 10))
fig.patch.set_facecolor('white')

stats_text = f"""
ISIC 2019 - Tiered Data Augmentation Summary (A40 GPU)
{'='*60}

GPU Configuration:
  • Type: NVIDIA A40
  • VRAM: 48GB
  • Architecture: Ampere
  • Processing: CPU-based augmentation (OpenCV)

Original Dataset:
  • Total Images: {len(X_original):,}
  • Classes: {len(CLASS_NAMES)}
  • Resolution: {X_original.shape[1]}×{X_original.shape[2]} pixels

Augmentation Strategy:
  • Tier 1 (Abundant): {', '.join(ABUNDANT_CLASSES)} - 4x augmentation
  • Tier 2 (Rare): {', '.join(RARE_CLASSES)} - Boost to 10,000 each

Results:
  • Augmented Size: {len(y_aug_view):,} images
  • Augmentation Factor: {len(y_aug_view) / len(X_original):.2f}x
  • Processing Time: {elapsed/60:.2f} minutes
  • Rate: {len(y_aug_view) / elapsed:.1f} images/second

Class Distribution (After):
"""
for class_idx, class_name in enumerate(CLASS_NAMES):
    orig = original_counts.get(class_idx, 0)
    aug = augmented_counts.get(class_idx, 0)
    increase = ((aug / orig) - 1) * 100 if orig > 0 else 0
    stats_text += f"  • {class_name}: {orig:,} → {aug:,} ({increase:+.1f}%)\n"

stats_text += f"""
Storage:
  • Network Volume: {NETWORK_VOLUME if NETWORK_VOLUME else 'Not detected'}
  • Output Directory: {OUTPUT_DIR}
  • Persistent: {'✓' if NETWORK_VOLUME else '✗'}
  • Total Size: ~{augmented_size_gb:.1f} GB
"""

ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
ax.axis('off')
ax.set_title('Augmentation Statistics Summary', fontsize=18, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(viz_dir / 'augmentation_statistics_summary.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"    ✓ Saved: augmentation_statistics_summary.png")

print(f"\n✅ All visualizations saved to: {viz_dir}")

# ============================================
# SUMMARY
# ============================================
print("\n" + "=" * 70)
print("✅ WEEK 3 COMPLETE: MEDICAL DATA AUGMENTATION (A40)")
print("=" * 70)

print(f"\n📦 Output Files:")
print(f"  1. X_augmented_medical.npy ({X_aug_view.shape})")
print(f"  2. y_augmented_medical.npy ({y_aug_view.shape})")
print(f"  3. augmentation_config_medical.json")
print(f"  4. visualizations/ (3 report-ready images, 300 DPI)")

print(f"\n📊 Statistics:")
print(f"  Original: {len(X_original):,} images")
print(f"  Augmented: {len(y_aug_view):,} images")
print(f"  Factor: {len(y_aug_view) / len(X_original):.2f}x")
print(f"  Storage: ~{augmented_size_gb:.1f} GB")
print(f"  Time: {elapsed/60:.2f} minutes")

print(f"\n💾 Storage Information:")
print(f"  Network Volume: {NETWORK_VOLUME if NETWORK_VOLUME else 'Not detected'}")
print(f"  Output Directory: {OUTPUT_DIR}")
print(f"  Persistent: {'✓ Yes' if NETWORK_VOLUME else '✗ No (workspace only)'}")

print(f"\n🎮 GPU Configuration:")
print(f"  Type: NVIDIA A40")
print(f"  VRAM: 48GB")
print(f"  Architecture: Ampere")
print(f"  Note: CPU preprocessing (GPU not used for augmentation)")

print(f"\n🎯 Next Step: Week 4 - Train/Validation/Test Split")
print("=" * 70)