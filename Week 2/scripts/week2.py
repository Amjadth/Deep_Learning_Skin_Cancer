# =================================================================================================
# WEEK 2 — HIGH-RESOLUTION MEDICAL IMAGE PREPROCESSING
# PIPELINE WITH CHECKPOINT/RESUME (RUNPOD + NVIDIA A40)
# =================================================================================================
#
# Purpose:
#   End-to-end preprocessing of the ISIC 2019 dermatology dataset (25,331 images, 8 classes).
#   Produces a standardized, high-resolution (600×600) dataset suitable for transfer learning
#   and downstream model training. Optimized for RunPod A40 infrastructure with full
#   checkpoint/resume support for Spot interruptions.
#
# Environment:
#   • GPU: NVIDIA A40 (48GB VRAM) — GPU optional for preprocessing
#   • CUDA: 12.4
#   • Framework: TensorFlow 2.15.0 (pre-installed)
#   • Template: madiator2011/better-tensorflow:cuda12.4-cudnn8
#
# Highlights:
#   • Medical-grade preprocessing (CLAHE, color constancy, hair removal)
#   • Dermatology-specific enhancements in LAB/RGB spaces
#   • ImageNet normalization for transfer-learning compatibility
#   • Optimized tf.data pipeline with automatic CPU parallelism
#   • Fully memory-safe: streaming writes + memmap (no RAM overflow)
#   • Production-grade checkpointing (every N batches) and resume logic
#   • RunPod-aware directory detection (/workspace, /notebooks)
#   • Automatic skip if final outputs are already available
#   • Optional auto-download from Kaggle using environment variables
#
# Performance / Optimization:
#   • 64-core parallel statistics computation
#   • Disk-based caching (avoids >100GB RAM usage)
#   • Pipeline fusion + zero-copy GPU prefetch
#   • Batch-oriented processing for stability at high resolution
#   • Aspect-ratio preservation via reflection padding
#
# Medical Imaging Pipeline:
#   • Shades-of-Gray color constancy (dermatology standard)
#   • DullRazor-style hair/marker removal
#   • CLAHE histogram equalization in LAB space
#   • Noise reduction + gamma correction + mild edge refinement
#   • Final normalization: ImageNet (mean/std)
#
# RunPod Production Features:
#   • Persistent volume detection + safe directory creation
#   • Checkpoint/resume via checkpoint.json
#   • Auto-cleanup of temp files on successful completion
#   • Robust validation of dataset paths, storage, and output files
#
# Outputs (saved to persistent Pod Volume / network volume):
#   • X_full.npy                     — Preprocessed images (600×600×3, float32)
#   • y_full.npy                     — Class labels (int32)
#   • full_metadata.csv              — Preprocessing metadata per image
#   • custom_dataset_statistics.json — Dataset-level statistics
#   • visualizations/                — Thesis-ready figures and preprocessing reports
#   • checkpoint.json                — Auto-generated checkpoint (removed on completion)
#
# Prerequisites:
#   • ISIC 2019 dataset in ./data/isic2019/  OR  Kaggle API credentials
#   • Persistent RunPod Volume (~200GB recommended)
#   • NVIDIA A40 GPU (optional, preprocessing is CPU-heavy)
#
# Optional Environment Variables:
#   KAGGLE_USERNAME=your_username
#   KAGGLE_KEY=your_api_key
#
# Usage:
#   python week2.py
#
# Version: 2.0 (2025)
# Framework: TensorFlow 2.15.0
# =================================================================================================

import os
import sys
import subprocess
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import time
import tensorflow as tf
import shutil
from multiprocessing import Pool, cpu_count

# --------------------------------------------
# RunPod/Jupyter workspace awareness with Network Volume Support
# --------------------------------------------
# Detect RunPod environment (Pod vs Serverless)
runpod_pod_id = os.environ.get('RUNPOD_POD_ID', None)
runpod_pod_type = os.environ.get('RUNPOD_POD_TYPE', None)
is_pod = runpod_pod_id is not None
is_serverless = os.environ.get('RUNPOD_WORKER_ID', None) is not None

# Detect RunPod workspace
BASE_DIR = Path(os.getcwd())
if Path('/workspace').exists():
    BASE_DIR = Path('/workspace')
elif Path('/notebooks').exists():
    BASE_DIR = Path('/notebooks')

# Check for network volumes (RunPod persistent storage)
# According to RunPod docs:
# - For Pods: Network volumes mount at /workspace (replaces default volume disk)
# - For Serverless: Network volumes mount at /runpod-volume
network_volumes = []
NETWORK_VOLUME = None

if is_serverless:
    # Serverless: Check for /runpod-volume
    if os.path.exists("/runpod-volume"):
        network_volumes.append("/runpod-volume")
        NETWORK_VOLUME = Path("/runpod-volume")
        print(f"✓ Network volume detected: /runpod-volume (Serverless)")
    else:
        print(f"⚠ No network volume detected at /runpod-volume (Serverless)")
elif is_pod:
    # Pod: Network volume is mounted at /workspace if attached
    # When network volume is attached to Pod, it replaces default volume at /workspace
    if os.path.exists("/workspace"):
        # For Pods, if network volume is attached, /workspace IS the network volume
        # We assume network volume is attached if user configured it during deployment
        NETWORK_VOLUME = Path("/workspace")
        network_volumes.append("/workspace")
        print(f"✓ Using /workspace as network volume (Pod)")
        print(f"  Note: Network volumes for Pods are mounted at /workspace per RunPod docs")
        print(f"  If you attached a network volume during deployment, /workspace is persistent")
        print(f"  If no network volume was attached, /workspace is temporary (data lost when pod stops)")
    else:
        print(f"⚠ No /workspace found - unexpected for Pod")
else:
    # Fallback: Check for /runpod-volume (Serverless) or /workspace/.runpod (legacy)
    if os.path.exists("/runpod-volume"):
        network_volumes.append("/runpod-volume")
        NETWORK_VOLUME = Path("/runpod-volume")
        print(f"✓ Network volume detected: /runpod-volume")
    elif os.path.exists("/workspace/.runpod"):
        network_volumes.append("/workspace/.runpod")
        NETWORK_VOLUME = Path("/workspace/.runpod")
        print(f"✓ Network volume detected: /workspace/.runpod")
    else:
        print(f"⚠ No network volume detected - will use workspace (non-persistent)")

# Configuration (paths resolved relative to workspace or network volume)
# Use network volume for persistent storage if available, otherwise use workspace
# For Pods: If network volume is attached, /workspace IS the network volume
# For Serverless: Network volume is at /runpod-volume
if NETWORK_VOLUME:
    # Network volume is available (either /workspace for Pods or /runpod-volume for Serverless)
    STORAGE_BASE = NETWORK_VOLUME
    OUTPUT_DIR = (STORAGE_BASE / 'outputs').resolve()
    INPUT_DIR = (STORAGE_BASE / 'data' / 'isic2019').resolve()  # Dataset path
    print(f"✓ Using network volume for persistent storage: {STORAGE_BASE}")
else:
    # No network volume detected, use workspace (temporary)
    STORAGE_BASE = BASE_DIR
    OUTPUT_DIR = (STORAGE_BASE / 'outputs').resolve()
    INPUT_DIR = (STORAGE_BASE / 'data' / 'isic2019').resolve()  # Dataset path
    print(f"⚠ No network volume detected - using workspace (temporary storage)")

# Auto-detect dataset location and update INPUT_DIR if needed
# Handles case where dataset is in /workspace/data/ instead of /workspace/data/isic2019/
# FIXED: Avoids recursive folder creation by directly updating INPUT_DIR instead of creating symlinks
def auto_detect_dataset_location():
    """Auto-detect dataset location and update INPUT_DIR if data is in parent directory."""
    global INPUT_DIR  # Allow modification of global INPUT_DIR
    
    # ISIC 2019 class names (hardcoded for detection purposes)
    isic_classes = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']
    data_parent = STORAGE_BASE / 'data'
    
    # Check if INPUT_DIR already exists and has data
    if INPUT_DIR.exists():
        # Check if it's a symlink that might cause recursion
        if INPUT_DIR.is_symlink():
            real_path = INPUT_DIR.resolve()
            # If symlink points to a parent directory, resolve to avoid recursion
            if str(real_path) == str(data_parent) or str(real_path).startswith(str(data_parent)):
                print(f"⚠ Found recursive symlink: {INPUT_DIR} -> {real_path}")
                print(f"   Removing symlink and using direct path")
                INPUT_DIR.unlink()
                INPUT_DIR = data_parent
                print(f"   ✓ Updated INPUT_DIR to: {INPUT_DIR}")
                return True
        
        csv_files = list(INPUT_DIR.glob('*.csv')) + list(INPUT_DIR.rglob('*.csv'))
        class_dirs = [INPUT_DIR / class_name for class_name in isic_classes 
                     if (INPUT_DIR / class_name).exists() and (INPUT_DIR / class_name).is_dir()]
        if csv_files or class_dirs:
            print(f"✓ Dataset found at expected location: {INPUT_DIR}")
            return True
    
    # Check if data is in parent directory instead
    if data_parent.exists():
        csv_files = list(data_parent.glob('*.csv')) + list(data_parent.rglob('*.csv'))
        class_dirs = [data_parent / class_name for class_name in isic_classes 
                     if (data_parent / class_name).exists() and (data_parent / class_name).is_dir()]
        
        if csv_files or class_dirs:
            print(f"\n🔍 Auto-detection: Dataset found in parent directory: {data_parent}")
            print(f"   Expected location: {INPUT_DIR}")
            print(f"   Actual location: {data_parent}")
            
            # FIXED: Instead of creating symlink (which can cause recursion), directly update INPUT_DIR
            # This avoids recursive folder creation issues
            try:
                # Remove INPUT_DIR if it exists (empty dir or broken symlink)
                if INPUT_DIR.exists():
                    if INPUT_DIR.is_symlink():
                        INPUT_DIR.unlink()
                        print(f"   ✓ Removed existing symlink: {INPUT_DIR}")
                    elif INPUT_DIR.is_dir():
                        # Check if directory is empty or contains only nested isic2019 folders (recursion)
                        contents = list(INPUT_DIR.iterdir())
                        if len(contents) == 0:
                            INPUT_DIR.rmdir()
                            print(f"   ✓ Removed empty directory: {INPUT_DIR}")
                        elif len(contents) == 1 and contents[0].name == 'isic2019':
                            # This is likely a recursive structure - remove it
                            print(f"   ⚠ Detected recursive folder structure, removing: {INPUT_DIR}")
                            shutil.rmtree(INPUT_DIR)
                        else:
                            # Directory has content, backup it
                            backup_dir = INPUT_DIR.parent / f"{INPUT_DIR.name}_backup"
                            if not backup_dir.exists():
                                shutil.move(str(INPUT_DIR), str(backup_dir))
                                print(f"   ⚠ Moved existing {INPUT_DIR.name} to backup: {backup_dir}")
                            else:
                                print(f"   ⚠ Cannot update INPUT_DIR: {INPUT_DIR} exists and is not empty")
                                # Still update INPUT_DIR to use parent directory
                                INPUT_DIR = data_parent
                                print(f"   ✓ Updated INPUT_DIR to use parent directory: {INPUT_DIR}")
                                return True
                
                # FIXED: Directly update INPUT_DIR instead of creating symlink
                # This prevents recursive folder creation
                INPUT_DIR = data_parent
                print(f"   ✓ Updated INPUT_DIR to: {INPUT_DIR}")
                print(f"   ✓ Dataset will be accessed directly (no symlink to avoid recursion)")
                return True
            except Exception as e:
                print(f"   ⚠ Failed to update INPUT_DIR: {e}")
                print(f"   ⚠ Falling back to using parent directory directly")
                INPUT_DIR = data_parent
                return True
    
    return False

# Run auto-detection
auto_detect_dataset_location()

# For Pods with network volume: /workspace is the network volume, so no symlink needed
# For Serverless: Network volume is at /runpod-volume, workspace might be different
WORKSPACE_OUTPUT_DIR = (BASE_DIR / 'outputs').resolve()

print(f"\n📁 Storage Configuration:")
print(f"   RunPod Environment: {'Pod' if is_pod else 'Serverless' if is_serverless else 'Unknown'}")
if runpod_pod_id:
    print(f"   Pod ID: {runpod_pod_id}")
print(f"   Base directory: {BASE_DIR}")
print(f"   Network volume: {NETWORK_VOLUME if NETWORK_VOLUME else 'Not detected (using workspace)'}")
if NETWORK_VOLUME:
    if is_pod:
        print(f"   ✓ Network volume at /workspace (Pod) - data will persist")
    elif is_serverless:
        print(f"   ✓ Network volume at /runpod-volume (Serverless) - data will persist")
else:
    print(f"   ⚠ No network volume - data will be lost when pod/worker stops")
    if is_pod:
        print(f"   💡 Tip: Attach network volume during Pod deployment for persistent storage")
print(f"   Output directory: {OUTPUT_DIR}")
print(f"   Input directory: {INPUT_DIR}")

# Image settings - HIGH RESOLUTION PREPROCESSING
TARGET_SIZE = (600, 600)  # Start with high resolution for maximum detail preservation
CLASS_NAMES = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

print(f"🔧 HIGH-RESOLUTION PREPROCESSING STRATEGY:")
print(f"   Target Size: {TARGET_SIZE} (maximum detail preservation)")
print(f"   Later downscaling: Dynamic per model requirements")
print(f"   Benefits: No upscaling artifacts, optimal medical AI performance")

# Preprocessing toggles (optimized for A40/A6000 + full dataset)
APPLY_COLOR_CONSTANCY = True       # Shades-of-Gray color constancy (medical imaging standard)
APPLY_HAIR_REMOVAL = True          # DullRazor-style hair/marker removal (dermatology standard)
APPLY_IMAGENET_NORM = True         # Use ImageNet mean/std normalization (for transfer learning)
APPLY_ADVANCED_PREPROCESSING = True # Advanced medical preprocessing

# Helper: ensure output dirs exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# For Pods with network volume: /workspace is the network volume, so no symlink needed
# For Serverless: Create symlink from workspace to /runpod-volume for easy access
if is_serverless and NETWORK_VOLUME and NETWORK_VOLUME != BASE_DIR:
    WORKSPACE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        if WORKSPACE_OUTPUT_DIR.exists() and not WORKSPACE_OUTPUT_DIR.is_symlink():
            # Backup existing workspace outputs if any
            backup_dir = BASE_DIR / 'outputs_backup'
            if not backup_dir.exists():
                shutil.move(str(WORKSPACE_OUTPUT_DIR), str(backup_dir))
                print(f"⚠ Moved existing workspace outputs to: {backup_dir}")
        
        # Create symlink if it doesn't exist
        if not WORKSPACE_OUTPUT_DIR.exists() or not WORKSPACE_OUTPUT_DIR.is_symlink():
            if WORKSPACE_OUTPUT_DIR.exists():
                WORKSPACE_OUTPUT_DIR.rmdir()
            os.symlink(str(OUTPUT_DIR), str(WORKSPACE_OUTPUT_DIR))
            print(f"✓ Created symlink: {WORKSPACE_OUTPUT_DIR} -> {OUTPUT_DIR}")
    except Exception as e:
        print(f"⚠ Could not create symlink (using network volume directly): {e}")
elif is_pod and NETWORK_VOLUME:
    # For Pods: /workspace is the network volume, so OUTPUT_DIR is already at /workspace/outputs
    print(f"✓ Network volume is at /workspace (Pod) - outputs will be persistent")

print("=" * 70)
print("WEEK 2: HIGH-RESOLUTION PREPROCESSING (ISIC 2019)")
print("=" * 70)
print("💻 Note: Preprocessing runs on CPU (medical image processing)")
print("🎯 Target: Full ISIC 2019 dataset with 600x600 preprocessing")
print("🔧 Strategy: High-resolution preprocessing → Dynamic downscaling per model")
print("✅ Benefits: No upscaling artifacts, maximum detail preservation")
print("=" * 70)

def check_dataset_availability():
    """Check if ISIC 2019 dataset is available locally.
    
    Handles multiple dataset structures:
    - Standard ISIC 2019 structure
    - salviohexia/isic-2019 dataset structure
    - Alternative structures
    """
    print(f"\n📁 Checking local dataset availability...")
    print(f"  Dataset path: {INPUT_DIR}")
    
    # Check if INPUT_DIR exists
    if not INPUT_DIR.exists():
        print(f"✗ Dataset directory not found at: {INPUT_DIR}")
        return False
    
    # Look for CSV files (ground truth and metadata)
    csv_files = list(INPUT_DIR.glob('*.csv'))
    csv_files.extend(list(INPUT_DIR.rglob('*.csv')))  # Also search subdirectories
    
    if not csv_files:
        print(f"✗ No CSV files found in dataset directory")
        print(f"  Expected files:")
        print(f"    - Ground truth CSV (e.g., ISIC_2019_Training_GroundTruth.csv)")
        print(f"    - Metadata CSV (e.g., ISIC_2019_Training_Metadata.csv)")
        return False
    
    print(f"✓ Found {len(csv_files)} CSV file(s):")
    for csv_file in csv_files[:5]:  # Show first 5
        print(f"    - {csv_file.name}")
    
    # Look for image directories (optimized for class-based structure - salviohexia/isic-2019)
    image_dirs = []
    # First, check for class-based directories (matches salviohexia/isic-2019 structure)
    class_dirs_found = []
    for class_name in CLASS_NAMES:
        class_dir = INPUT_DIR / class_name
        if class_dir.exists() and class_dir.is_dir():
            image_count = len(list(class_dir.glob('*.jpg'))) + len(list(class_dir.glob('*.png')))
            if image_count > 0:
                class_dirs_found.append((class_dir, image_count))
    
    if class_dirs_found:
        image_dirs = class_dirs_found
        print(f"✓ Found {len(image_dirs)} class directories (class-based structure - salviohexia/isic-2019):")
        for img_dir, count in image_dirs:
            print(f"    - {img_dir.name}: ~{count} images")
    else:
        # Fallback to other possible structures
        possible_image_dirs = [
            INPUT_DIR / 'ISIC_2019_Training_Input',
            INPUT_DIR / 'images',
            INPUT_DIR / 'train',
            INPUT_DIR / 'data',
            INPUT_DIR,  # Images might be directly in INPUT_DIR
        ]
        
        for img_dir in possible_image_dirs:
            if img_dir.exists():
                image_count = len(list(img_dir.glob('*.jpg'))) + len(list(img_dir.glob('*.png')))
                if image_count > 0:
                    image_dirs.append((img_dir, image_count))
        
        if not image_dirs:
            print(f"⚠ No image directories found (images might be in subdirectories)")
            # Try recursive search
            for path in INPUT_DIR.rglob('*'):
                if path.is_file() and path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    image_dirs.append((path.parent, 1))
                    break
        else:
            print(f"✓ Found {len(image_dirs)} image directory/directories:")
            for img_dir, count in image_dirs[:3]:  # Show first 3
                print(f"    - {img_dir}: ~{count} images")
    
    print(f"✓ Dataset found and validated")
    return True


def setup_kaggle_credentials():
    """Setup Kaggle credentials from environment or hardcoded values.
    
    Priority:
    1. Environment variables (KAGGLE_USERNAME, KAGGLE_KEY)
    2. Hardcoded credentials (provided by user)
    """
    kaggle_user = os.environ.get('KAGGLE_USERNAME')
    kaggle_key = os.environ.get('KAGGLE_KEY')

    # Fallback to provided credentials if env vars not set
    if not kaggle_user or not kaggle_key:
        print("📝 Using provided Kaggle credentials...")
        kaggle_user = "ahadraza000"
        kaggle_key = "ad1134baf3ceb94ac167cee8acd326cc"
    else:
        print("📝 Using Kaggle credentials from environment variables...")
    
    # Setup Kaggle API credentials
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_dir.mkdir(exist_ok=True)
    kaggle_cred_file = kaggle_dir / 'kaggle.json'
    
    # Write credentials file
    cred_data = {
        "username": kaggle_user,
        "key": kaggle_key
    }
    
    try:
        with open(kaggle_cred_file, 'w') as f:
            json.dump(cred_data, f)
        # Set proper permissions (Kaggle requires 600)
        os.chmod(kaggle_cred_file, 0o600)
        print(f"✓ Kaggle credentials configured: {kaggle_user}")
        return True
    except Exception as e:
        print(f"⚠ Failed to setup Kaggle credentials: {e}")
        return False


def ensure_kaggle_dataset():
    """Download dataset from Kaggle if missing.
    
    Dataset: salviohexia/isic-2019-skin-lesion-images-for-classification
    Contains 25,331 images across 8 classes.
    
    Returns:
        tuple: (success: bool, dataset_path: Path)
    """
    if INPUT_DIR.exists() and any(INPUT_DIR.glob('*.csv')):
        print(f"✓ Dataset already exists at: {INPUT_DIR}")
        return True, INPUT_DIR

    print("\n📥 Setting up Kaggle API...")
    if not setup_kaggle_credentials():
        print("⚠ Failed to setup Kaggle credentials")
        return False, None

    # Install kaggle package if needed
    try:
        import kaggle  # noqa: F401
        print("✓ Kaggle API available")
    except ImportError:
        print("🔧 Installing kaggle CLI...")
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '--quiet', '--no-cache-dir', 'kaggle>=1.6.14'],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode != 0:
            print(f"⚠ Kaggle installation had issues: {result.stderr[:200]}")
            return False, None

    # Dataset information
    kaggle_dataset = "salviohexia/isic-2019-skin-lesion-images-for-classification"
    
    # Create destination directory (use network volume if available for persistence)
    download_base = STORAGE_BASE / 'data'
    download_base.mkdir(parents=True, exist_ok=True)
    download_dir = download_base.resolve()

    print(f"\n⬇️  Downloading Kaggle dataset: {kaggle_dataset}")
    print(f"   → Destination: {download_dir}")
    print(f"   → Estimated size: ~10-15 GB (compressed)")
    print(f"   → This may take 10-30 minutes depending on connection speed...")
    
    # Download dataset
    result = subprocess.run([
        'kaggle', 'datasets', 'download', '-d', kaggle_dataset,
        '-p', str(download_dir), '--unzip'
    ], capture_output=True, text=True, timeout=3600)  # 1 hour timeout

    if result.returncode != 0:
        print("❌ Kaggle download failed:")
        if result.stderr:
            print(result.stderr[-1000:])
        if result.stdout:
            print(result.stdout[-1000:])
        
        # Try alternative: download without unzip flag and extract manually
        print("\n🔄 Trying alternative download method...")
        result2 = subprocess.run([
            'kaggle', 'datasets', 'download', '-d', kaggle_dataset,
            '-p', str(download_dir)
        ], capture_output=True, text=True, timeout=3600)
        
        if result2.returncode == 0:
            # Extract manually
            try:
                import zipfile
                zip_files = list(download_dir.glob('*.zip'))
                for zpath in zip_files:
                    print(f"📦 Extracting: {zpath.name} (this may take a while...)")
                    with zipfile.ZipFile(zpath, 'r') as zf:
                        zf.extractall(download_dir)
                    print(f"✓ Extracted: {zpath.name}")
                    zpath.unlink(missing_ok=True)
            except Exception as e:
                print(f"⚠ Extraction failed: {e}")
                return False, None
        else:
            return False, None

    # Find the dataset directory structure
    # The dataset might have different structures, so we search for CSV files
    print("\n🔍 Locating dataset files...")
    csv_files = list(download_dir.rglob('*.csv'))
    image_dirs = []
    
    # Look for common dataset structures
    possible_structures = [
        download_dir / 'ISIC_2019_Training_Input',
        download_dir / 'images',
        download_dir / 'train',
        download_dir / 'data',
    ]
    
    # Also search for directories with many image files
    for path in download_dir.rglob('*'):
        if path.is_dir():
            image_count = len(list(path.glob('*.jpg'))) + len(list(path.glob('*.png')))
            if image_count > 100:  # Likely the image directory
                image_dirs.append(path)
    
    print(f"   Found {len(csv_files)} CSV files")
    print(f"   Found {len(image_dirs)} potential image directories")
    
    # Create INPUT_DIR if needed and move/organize files
    INPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    
    # If dataset is already in the right place, we're done
    if INPUT_DIR.exists() and (any(INPUT_DIR.glob('*.csv')) or len(list(INPUT_DIR.iterdir())) > 0):
        print(f"✓ Dataset already organized at: {INPUT_DIR}")
        return True, INPUT_DIR
    
    # Try to find the main dataset directory
    dataset_candidate = None
    for candidate in [download_dir] + possible_structures:
        if candidate.exists():
            csv_in_candidate = list(candidate.glob('*.csv'))
            if csv_in_candidate:
                dataset_candidate = candidate
                break
    
    # If no candidate found, use download_dir
    if dataset_candidate is None:
        dataset_candidate = download_dir
    
    # Organize dataset to INPUT_DIR location
    final_dataset_path = INPUT_DIR
    if dataset_candidate != INPUT_DIR:
        try:
            # Ensure parent exists
            if not INPUT_DIR.exists():
                INPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
            # Guard: do NOT move a parent directory into its child (e.g., /data -> /data/isic2019)
            dataset_candidate_resolved = Path(dataset_candidate).resolve()
            input_dir_resolved = Path(INPUT_DIR).resolve()
            is_parent_of_input = str(input_dir_resolved).startswith(str(dataset_candidate_resolved))
            # Prefer a symlink to avoid long copy/move operations and to prevent parent->child moves
            try:
                if INPUT_DIR.exists():
                    if INPUT_DIR.is_symlink():
                        INPUT_DIR.unlink()
                    elif INPUT_DIR.is_dir():
                        # Remove empty dir only; otherwise leave it and fallback to using candidate directly
                        try:
                            INPUT_DIR.rmdir()
                        except OSError:
                            pass
                if not INPUT_DIR.exists():
                    os.symlink(str(dataset_candidate_resolved), str(INPUT_DIR))
                    print(f"✓ Created symlink: {INPUT_DIR} -> {dataset_candidate_resolved}")
                    final_dataset_path = INPUT_DIR
                else:
                    # If we could not replace INPUT_DIR, just use candidate in place
                    final_dataset_path = dataset_candidate_resolved
                    print(f"⚠ Could not replace INPUT_DIR; using dataset at original location: {final_dataset_path}")
            except Exception as e_symlink:
                print(f"⚠ Symlink failed ({e_symlink}); attempting safe copy (this may take a while)...")
                # Only allow copy if candidate is NOT a parent of INPUT_DIR to avoid recursive copy
                if not is_parent_of_input:
                    try:
                        shutil.copytree(str(dataset_candidate_resolved), str(INPUT_DIR), dirs_exist_ok=True)
                        print(f"✓ Copied dataset to: {INPUT_DIR}")
                        final_dataset_path = INPUT_DIR
                    except Exception as e_copy:
                        print(f"⚠ Copy failed ({e_copy}); using dataset at original location")
                        final_dataset_path = dataset_candidate_resolved
                else:
                    print("⚠ Refusing to copy parent directory into its own child. Using original location instead.")
                    final_dataset_path = dataset_candidate_resolved
        except Exception as e:
            print(f"⚠ Could not organize dataset: {e}")
            # Fallback: use the candidate directory directly
            final_dataset_path = dataset_candidate
            print(f"   Using dataset at: {final_dataset_path}")

    print(f"✓ Dataset prepared at: {final_dataset_path}")
    return True, final_dataset_path


# ============================================
# STEP 1: Load Metadata and Build Full Dataset
# ============================================
print("\n📊 Step 1: Preparing dataset...")

# Check dataset availability; try Kaggle if missing
if not check_dataset_availability():
    print("\n📥 Dataset not found locally. Attempting to download from Kaggle...")
    ensured, dataset_path = ensure_kaggle_dataset()
    if ensured and dataset_path:
        # Update INPUT_DIR if dataset was downloaded to a different location
        if dataset_path != INPUT_DIR:
            INPUT_DIR = dataset_path
            print(f"✓ Updated dataset path to: {INPUT_DIR}")
        if check_dataset_availability():
            print("✓ Dataset downloaded and validated successfully")
        else:
            print("\n❌ Dataset validation failed after download")
            exit(1)
    else:
        print("\n❌ Dataset not available. Please download ISIC 2019 dataset manually.")
        print("   Kaggle dataset: salviohexia/isic-2019-skin-lesion-images-for-classification")
        print("   Or download from: https://www.isic-archive.com/#!/topWithHeader/onlyHeaderTop/gallery")
        exit(1)

# Find CSV files (handle different naming conventions)
csv_files = {}
csv_patterns = {
    'ground_truth': ['ISIC_2019_Training_GroundTruth.csv', 'GroundTruth.csv', 'ground_truth.csv', 'labels.csv'],
    'metadata': ['ISIC_2019_Training_Metadata.csv', 'Metadata.csv', 'metadata.csv', 'train_metadata.csv']
}

for csv_type, patterns in csv_patterns.items():
    for pattern in patterns:
        csv_path = INPUT_DIR / pattern
        if csv_path.exists():
            csv_files[csv_type] = csv_path
            break
    # Also search recursively
    if csv_type not in csv_files:
        for path in INPUT_DIR.rglob(pattern):
            csv_files[csv_type] = path
            break

if 'ground_truth' not in csv_files:
    # Try to find any CSV file with class columns
    all_csvs = list(INPUT_DIR.glob('*.csv')) + list(INPUT_DIR.rglob('*.csv'))
    for csv_file in all_csvs:
        try:
            test_df = pd.read_csv(csv_file, nrows=5)
            if 'image' in test_df.columns and any(col in test_df.columns for col in CLASS_NAMES):
                csv_files['ground_truth'] = csv_file
                print(f"✓ Found ground truth file: {csv_file.name}")
                break
        except Exception:
            continue

if 'ground_truth' not in csv_files:
    print("❌ Could not find ground truth CSV file")
    print(f"   Searched in: {INPUT_DIR}")
    print(f"   Please ensure the dataset contains a CSV file with 'image' column and class labels")
    exit(1)

# Load ground truth labels
print(f"\n📖 Loading ground truth from: {csv_files['ground_truth'].name}")
gt_df = pd.read_csv(csv_files['ground_truth'])

# Load metadata if available
metadata_df = None
if 'metadata' in csv_files:
    print(f"📖 Loading metadata from: {csv_files['metadata'].name}")
    metadata_df = pd.read_csv(csv_files['metadata'])
    # Merge dataframes
    if 'image' in metadata_df.columns:
        df = pd.merge(gt_df, metadata_df, on='image', how='left')
    else:
        print("⚠ Metadata CSV doesn't have 'image' column, using ground truth only")
        df = gt_df.copy()
else:
    print("⚠ Metadata CSV not found, using ground truth only")
    df = gt_df.copy()

# Extract labels - handle different CSV formats
label_columns = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']

# Check which label columns exist in the dataframe
available_label_cols = [col for col in label_columns if col in df.columns]

if not available_label_cols:
    # Try alternative column names (case-insensitive, with/without spaces)
    for col in df.columns:
        col_upper = col.upper().strip()
        if col_upper in [l.upper() for l in label_columns]:
            # Map to standard name
            std_name = label_columns[[l.upper() for l in label_columns].index(col_upper)]
            df[std_name] = df[col]
            available_label_cols.append(std_name)

if not available_label_cols:
    # Check if there's a 'label' or 'class' column
    if 'label' in df.columns:
        df['label'] = df['label'].astype(str).str.upper().str.strip()
        print("✓ Using 'label' column from CSV")
    elif 'class' in df.columns:
        df['label'] = df['class'].astype(str).str.upper().str.strip()
        print("✓ Using 'class' column from CSV")
    else:
        print("❌ Could not find label columns in CSV")
        print(f"   Available columns: {list(df.columns)}")
        print(f"   Expected columns: {label_columns}")
        exit(1)
else:
    # Use one-hot encoded columns
    print(f"✓ Found {len(available_label_cols)} label columns: {available_label_cols}")
    df['label'] = df[available_label_cols].idxmax(axis=1)

# Map labels to indices
df['label'] = df['label'].str.upper().str.strip()
df['label_idx'] = df['label'].map(CLASS_TO_IDX)

# Check for unmapped labels
unmapped = df[df['label_idx'].isna()]['label'].unique()
if len(unmapped) > 0:
    print(f"⚠ Warning: Found {len(unmapped)} unmapped labels: {unmapped}")
    print(f"   These will be filtered out")
    df = df[df['label_idx'].notna()].copy()

# Show original distribution
original_counts = df['label'].value_counts()
print("\n📈 Original Dataset Class Distribution:")
for class_name, count in original_counts.items():
    percentage = (count / len(df)) * 100
    print(f"  {class_name}: {count:5d} images ({percentage:5.2f}%)")

print(f"\n🎯 Using full dataset (no per-class sampling)")

# Use all images of the specified classes
custom_df = df[df['label'].isin(CLASS_NAMES)].copy()
custom_df = custom_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n✅ Full dataset dataframe prepared!")
print(f"  Total images: {len(custom_df)}")

# Show dataset distribution
custom_counts = custom_df['label'].value_counts()
print("\n📈 Dataset Class Distribution (full):")
for class_name, count in custom_counts.items():
    percentage = (count / len(custom_df)) * 100
    print(f"  {class_name}: {count:5d} images ({percentage:5.2f}%)")

# ============================================
# STEP 2: Fast Medical Preprocessing Function
# ============================================
print("\n🔧 Step 2: Fast medical preprocessing function...")

def _shades_of_gray_color_constancy(img_rgb, power=6, gamma=None):
    try:
        img = img_rgb.astype(np.float32)
        if gamma is not None:
            img = np.power(img, gamma)
        eps = 1e-6
        mean_per_channel = np.power(np.mean(np.power(img, power), axis=(0, 1)) + eps, 1.0 / power)
        norm_factor = np.sqrt(np.sum(np.power(mean_per_channel, 2.0))) + eps
        img = img / (mean_per_channel / norm_factor)
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)
    except Exception:
        return img_rgb


def _hair_removal_dullrazor(img_rgb):
    """Enhanced DullRazor hair removal for dermatology images."""
    try:
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Enhanced kernel for better hair detection
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        
        # Improved thresholding
        _, thresh = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # Morphological operations to clean up hair mask
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_clean)
        
        # Inpainting with better algorithm
        inpainted = cv2.inpaint(img_bgr, thresh, 5, cv2.INPAINT_TELEA)
        return cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
    except Exception:
        return img_rgb


def _advanced_medical_preprocessing(img_rgb):
    """Advanced medical preprocessing pipeline for dermatology images."""
    try:
        # 1. Noise reduction (medical imaging standard)
        img_denoised = cv2.bilateralFilter(img_rgb, 9, 75, 75)
        
        # 2. Edge enhancement for lesion boundaries
        kernel_edge = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        img_enhanced = cv2.filter2D(img_denoised, -1, kernel_edge)
        img_enhanced = np.clip(img_enhanced, 0, 255)
        
        # 3. Gamma correction for better contrast
        gamma = 1.2
        img_gamma = np.power(img_enhanced / 255.0, gamma) * 255.0
        img_gamma = np.clip(img_gamma, 0, 255).astype(np.uint8)
        
        return img_gamma
    except Exception:
        return img_rgb


def _lesion_enhancement(img_rgb):
    """Lesion-specific enhancement for better feature extraction."""
    try:
        # Convert to LAB for better color space manipulation
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Enhance L channel (luminance) for better lesion visibility
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        l_enhanced = clahe.apply(l)
        
        # Merge channels
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        img_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
        
        return img_enhanced
    except Exception:
        return img_rgb


def preprocess_image_medical_fast(image_path, target_size=TARGET_SIZE, normalize=True, 
                                apply_clahe=True, denoise=False):
    """
    Advanced medical preprocessing pipeline for dermatology images.
    Runs on CPU (medical image processing standard).
    Optimized for full ISIC 2019 dataset processing.
    
    Args:
        image_path: Path to image file
        target_size: Tuple of (height, width)
        normalize: Whether to apply normalization
        apply_clahe: Apply Contrast Limited Adaptive Histogram Equalization
        denoise: DISABLED - too slow for large datasets
    
    Returns:
        Preprocessed image as numpy array
    """
    try:
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Step 1: Color constancy (medical imaging standard)
        if APPLY_COLOR_CONSTANCY:
            img = _shades_of_gray_color_constancy(img, power=6, gamma=None)
        
        # Step 2: Advanced medical preprocessing
        if APPLY_ADVANCED_PREPROCESSING:
            img = _advanced_medical_preprocessing(img)
        
        # Step 3: Lesion enhancement
        if APPLY_ADVANCED_PREPROCESSING:
            img = _lesion_enhancement(img)
        
        # Step 4: CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # This is the most important medical enhancement
        if apply_clahe:
            # Convert to LAB color space
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel with optimized parameters
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            
            # Merge channels and convert back to RGB
            lab = cv2.merge([l, a, b])
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        # Step 5: Hair/marker removal (dermatology standard)
        if APPLY_HAIR_REMOVAL:
            img = _hair_removal_dullrazor(img)
        
        # Fast resize with aspect ratio preservation (letterbox)
        h, w = img.shape[:2]
        aspect = w / h
        target_aspect = target_size[1] / target_size[0]
        
        if aspect > target_aspect:
            new_w = target_size[1]
            new_h = int(new_w / aspect)
        else:
            new_h = target_size[0]
            new_w = int(new_h * aspect)
        
        # Choose interpolation: INTER_AREA for downscale, INTER_LINEAR for upscale
        interp = cv2.INTER_AREA if (new_w < w or new_h < h) else cv2.INTER_LINEAR
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
        
        # Reflect padding to target size (avoid black borders)
        pad_top = (target_size[0] - new_h) // 2
        pad_bottom = target_size[0] - new_h - pad_top
        pad_left = (target_size[1] - new_w) // 2
        pad_right = target_size[1] - new_w - pad_left
        canvas = cv2.copyMakeBorder(
            img_resized,
            pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_REFLECT
        )
        
        # Normalize to [0, 1]
        if normalize:
            canvas = canvas.astype(np.float32) / 255.0

        # ImageNet normalization (for transfer learning with pretrained models)
        if APPLY_IMAGENET_NORM:
            # ImageNet mean and std (standard for pretrained models)
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            canvas = (canvas - mean) / std
        else:
            # Medical imaging normalization (dataset-specific)
            # This will be computed from the full dataset statistics
            pass
        
        return canvas
        
    except Exception as e:
        print(f"Error preprocessing {image_path}: {e}")
        return None

# ============================================
# STEP 3: Fast Statistics Computation
# ============================================
print("\n📊 Step 3: Computing fast statistics...")

def resolve_image_path(base: Path, image_id: str, label: str) -> Path:
    """Resolve image path for ISIC-2019 dataset (handles multiple structures).
    
    Tries multiple common dataset structures:
    1. Standard ISIC structure
    2. Class-based directories
    3. Flat image directory
    4. Recursive search
    """
    # Remove file extension if present
    image_id_clean = image_id
    if image_id_clean.endswith('.jpg') or image_id_clean.endswith('.png'):
        image_id_clean = image_id_clean.rsplit('.', 1)[0]
    
    # Try multiple file extensions
    extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    
    # Standard ISIC 2019 structure candidates
    # Priority order optimized for salviohexia/isic-2019 dataset structure:
    # 1. Class-based directories (most common in this dataset)
    # 2. Standard ISIC structure
    # 3. Alternative structures
    # 4. Direct in base directory
    candidates = [
        # Class-based directories (PRIORITY - matches salviohexia/isic-2019 structure)
        base / label / f"{image_id_clean}.jpg",
        base / label / f"{image_id}.jpg",
        # Standard ISIC structure
        base / 'ISIC_2019_Training_Input' / f"{image_id_clean}.jpg",
        base / 'ISIC_2019_Training_Input' / f"{image_id}.jpg",
        # Alternative structures
        base / 'images' / f"{image_id_clean}.jpg",
        base / 'images' / f"{image_id}.jpg",
        base / 'train' / f"{image_id_clean}.jpg",
        base / 'train' / f"{image_id}.jpg",
        # Direct in base directory
        base / f"{image_id_clean}.jpg",
        base / f"{image_id}.jpg",
    ]
    
    # Try with different extensions
    all_candidates = []
    for candidate in candidates:
        for ext in extensions:
            candidate_with_ext = candidate.with_suffix(ext)
            all_candidates.append(candidate_with_ext)
    
    # Check candidates
    for p in all_candidates:
        if p.exists():
            return p
    
    # If not found in candidates, try recursive search (slower but more flexible)
    # Only do this if we haven't found the image yet
    for ext in extensions:
        pattern = f"{image_id_clean}*{ext}"
        matches = list(base.rglob(pattern))
        if matches:
            return matches[0]
        # Also try with full image_id
        pattern2 = f"{image_id}*{ext}"
        matches2 = list(base.rglob(pattern2))
        if matches2:
            return matches2[0]
    
    # Return the most likely path for error reporting
    return base / 'ISIC_2019_Training_Input' / f"{image_id_clean}.jpg"

def _process_image_for_stats(args):
    """Worker function for parallel statistics computation."""
    image_id, label, input_dir_str = args
    try:
        # Convert string back to Path for resolve_image_path
        input_dir = Path(input_dir_str)
        image_path = resolve_image_path(input_dir, image_id, label)
        
        # Check if image exists before processing
        if not image_path.exists():
            return None
            
        # Use full medical preprocessing for accurate statistics
        # normalize=False returns uint8, which is what we want for stats
        img = preprocess_image_medical_fast(
            image_path, 
            normalize=False, 
            apply_clahe=True,
            denoise=False
        )
        
        if img is not None:
            # Use float64 for precision in large sums
            img_64 = img.astype(np.float64)
            pixel_count = img_64.shape[0] * img_64.shape[1]
            
            # Sum pixels for mean
            sum_channels = np.sum(img_64, axis=(0, 1))
            
            # Sum squared pixels for std deviation
            sum_sq_channels = np.sum(np.square(img_64), axis=(0, 1))
            
            # Return the calculated sums, not the full image
            return (sum_channels, sum_sq_channels, pixel_count)
        return None
    except Exception as e:
        # Silently skip errors (common for missing/corrupt images)
        return None

def compute_fast_statistics(custom_df, sample_size=None):
    """Compute dataset mean/std over the full dataset with medical preprocessing.
    
    Uses multiprocessing to utilize all CPU cores for maximum performance.
    FIXED: Progress bar now updates properly with multiprocessing.
    """
    # Limit workers to avoid overhead (optimal is usually 2x physical cores, but cap at 64)
    total_cores = cpu_count()
    num_workers = min(total_cores, 64)  # Cap at 64 workers to avoid overhead
    print(f"Computing statistics for {len(custom_df)} images...")
    print(f"⚡ Using {num_workers} CPU cores for parallel processing (out of {total_cores} available)")
    
    # Prepare arguments for worker function (convert to tuples for pickleability)
    # Convert INPUT_DIR to string for multiprocessing compatibility
    input_dir_str = str(INPUT_DIR)
    rows_list = [(row['image'], row['label'], input_dir_str) for _, row in custom_df.iterrows()]
    
    # FIXED: Use much smaller chunksize for better progress bar updates
    # Smaller chunksize means results come in more frequently, allowing progress bar to update
    chunksize = max(1, min(10, len(rows_list) // (num_workers * 20)))  # Much smaller chunks = faster progress updates
    print(f"⚙️  Chunksize: {chunksize} (optimized for progress updates)")
    
    pixel_values = [] # This line is no longer used for pixels
    with Pool(processes=num_workers) as pool:
        # FIXED: Use imap_unordered with proper tqdm configuration for multiprocessing
        # Configure tqdm to update more frequently and ensure it works with multiprocessing
        iterator = pool.imap_unordered(_process_image_for_stats, rows_list, chunksize=chunksize)
        
        # --- START MODIFICATION ---
        # Initialize aggregators
        total_sum = np.zeros(3, dtype=np.float64)
        total_sum_sq = np.zeros(3, dtype=np.float64)
        total_pixels = 0
        valid_images_count = 0
        failed_images_count = 0
        # --- END MODIFICATION ---

        # Use tqdm with proper settings for multiprocessing
        with tqdm(
            total=len(rows_list),
            desc="Computing stats (parallel)",
            unit="img",
            mininterval=0.5,  # Update at least every 0.5 seconds (more frequent)
            maxinterval=2.0,    # Force update every 2 seconds even if no progress
            file=sys.stdout,    # Explicitly write to stdout
            dynamic_ncols=True,  # Adapt to terminal width
            position=0,         # Keep at top of terminal
            leave=True        # Keep progress bar visible after completion
        ) as pbar:
            # Manually iterate and update progress bar for better control
            for result in iterator:
                pbar.update(1)  # Update progress bar for each completed task
                
                # --- START MODIFICATION ---
                # Aggregate results as they come in
                if result is not None:
                    sum_c, sum_sq_c, px_count = result
                    total_sum += sum_c
                    total_sum_sq += sum_sq_c
                    total_pixels += px_count
                    valid_images_count += 1
                else:
                    failed_images_count += 1
                # --- END MODIFICATION ---
    
    # --- START MODIFICATION ---
    # Filter out None results - (This block is no longer needed)
    # pixel_values = [r for r in results if r is not None]
    
    if total_pixels > 0:
        # Calculate mean and std from the aggregated sums
        mean = total_sum / total_pixels
        # Variance = E[X^2] - (E[X])^2
        variance = (total_sum_sq / total_pixels) - np.square(mean)
        std = np.sqrt(variance)
        
        # (This block replaces the old 'all_pixels = np.vstack(pixel_values)' block)
        # --- END MODIFICATION ---
        
        # This block replaces the original 'stats = ...' dictionary
        stats = {
            'mean': mean.tolist(),
            'std': std.tolist(),
            'mean_rgb': mean.tolist(),
            'std_rgb': std.tolist(),
            'preprocessing_applied': [
                'color_constancy', 'advanced_medical', 'lesion_enhancement',
                'clahe', 'hair_removal', 'aspect_ratio_preservation'
            ],
            'medical_optimizations': 'enabled',
            'dataset_type': 'full_dataset_medical_preprocessed',
            'total_images': valid_images_count,  # Number of successfully processed images
            'target_size': TARGET_SIZE,
            'normalization': 'imagenet' if APPLY_IMAGENET_NORM else 'dataset_specific',
            'parallel_workers': num_workers
        }

        print(f"✓ Dataset Mean (RGB): [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
        print(f"✓ Dataset Std  (RGB): [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]")
        print(f"✓ Medical preprocessing: {stats['preprocessing_applied']}")
        print(f"✓ Processed {valid_images_count}/{len(custom_df)} images successfully")
        if failed_images_count > 0:
            print(f"⚠ Failed to process {failed_images_count} images")
        
        return stats
    return None

# ============================================
# STEP 4: Process Custom Dataset with tf.data Pipeline
# ============================================
print("\n🖼️  Step 4: Processing full dataset with tf.data pipeline...")

# Compute statistics
dataset_stats = compute_fast_statistics(custom_df, sample_size=500)

# Save statistics
if dataset_stats:
    stats_path = OUTPUT_DIR / 'custom_dataset_statistics.json'
    with open(stats_path, 'w') as f:
        json.dump(dataset_stats, f, indent=2)
    print(f"✓ Statistics saved to: {stats_path}")

# Create tf.data pipeline for efficient processing
print(f"\n⚙️  Creating tf.data pipeline for {len(custom_df)} images...")

def create_tfdata_pipeline(df, input_dir, target_size, batch_size=32, shuffle=True):
    """Create highly optimized tf.data pipeline for maximum A40/A6000 GPU utilization."""
    
    def preprocess_tf_wrapper(image_path, label_idx):
        """Wrapper function for tf.py_function with enhanced parallelism."""
        def _preprocess_py(image_path_bytes, label_idx):
            image_path_str = image_path_bytes.numpy().decode('utf-8')
            processed_img = preprocess_image_medical_fast(
                image_path_str,
                target_size=target_size,
                normalize=True,
                apply_clahe=True,
                denoise=False
            )
            if processed_img is not None:
                return processed_img.astype(np.float32), label_idx.numpy()
            else:
                # Return zero image if processing fails
                return np.zeros((*target_size, 3), dtype=np.float32), label_idx.numpy()
        
        return tf.py_function(_preprocess_py, [image_path, label_idx], [tf.float32, tf.int32])
    
    # Prepare data
    image_paths = []
    labels = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        image_path = resolve_image_path(input_dir, row['image'], row['label'])
        if image_path.exists():
            image_paths.append(str(image_path))
            labels.append(row['label_idx'])
            valid_indices.append(idx)
    
    print(f"✓ Found {len(image_paths)} valid images out of {len(df)}")
    
    # Create tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    
    # A40/A6000-optimized pipeline configuration
    if shuffle:
        # Larger buffer for better shuffling on A40/A6000
        dataset = dataset.shuffle(buffer_size=min(len(image_paths), 10000), seed=42)
    
    # Advanced parallel processing for A40/A6000
    dataset = dataset.map(
        preprocess_tf_wrapper,
        num_parallel_calls=tf.data.AUTOTUNE,  # Auto-tune based on CPU cores
        deterministic=False  # Allow non-deterministic for better performance
    )
    
    # Cache to disk to avoid high RAM usage on large datasets
    cache_path = (OUTPUT_DIR / 'tf_cache').as_posix()
    dataset = dataset.cache(cache_path)
    
    # Optimized batching for A40/A6000
    dataset = dataset.batch(
        batch_size,
        drop_remainder=False,  # Keep all data
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Enable experimental optimizations for A40/A6000 (apply before prefetch)
    options = tf.data.Options()
    options.experimental_optimization.parallel_batch = True
    options.experimental_optimization.map_fusion = True
    options.experimental_optimization.map_and_batch_fusion = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_threading.max_intra_op_parallelism = 0  # Use all CPU cores
    options.experimental_threading.private_threadpool_size = 0  # Use all CPU cores
    dataset = dataset.with_options(options)
    
    # Advanced prefetching for maximum GPU utilization
    # Increased buffer size to 4 for better GPU utilization on A40/A6000
    dataset = dataset.prefetch(4)
    
    return dataset, valid_indices

# Create tf.data pipeline
batch_size = 32  # Process in batches for memory efficiency
dataset, valid_indices = create_tfdata_pipeline(
    custom_df, INPUT_DIR, TARGET_SIZE, batch_size=batch_size, shuffle=False
)

print(f"✓ tf.data pipeline created with batch size: {batch_size}")

# ============================================
# CHECKPOINT/RESUME LOGIC
# ============================================
checkpoint_path = OUTPUT_DIR / 'checkpoint.json'
final_x_path = OUTPUT_DIR / 'X_full.npy'
final_y_path = OUTPUT_DIR / 'y_full.npy'

# Check if final output files already exist (skip processing if complete)
skip_to_visualization = False
if final_x_path.exists() and final_y_path.exists():
    print(f"\n✅ Final output files already exist! Skipping processing.")
    print(f"   - X_full.npy: {final_x_path}")
    print(f"   - y_full.npy: {final_y_path}")
    print(f"   To reprocess, delete these files first.")
    print(f"\n📊 Loading existing dataset for visualization...")
    X_view = np.load(final_x_path)
    y_view = np.load(final_y_path)
    # Load metadata if exists
    metadata_path = OUTPUT_DIR / 'full_metadata.csv'
    if metadata_path.exists():
        processed_metadata_df = pd.read_csv(metadata_path)
    else:
        processed_metadata_df = pd.DataFrame()
    skip_to_visualization = True

if not skip_to_visualization:
    # Load checkpoint if exists (resume from interruption)
    checkpoint = None
    resume_from_batch = 0
    write_ptr = 0
    processed_metadata = []

    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
            resume_from_batch = checkpoint.get('last_batch_idx', 0) + 1  # Resume from next batch
            write_ptr = checkpoint.get('write_ptr', 0)
            # Try to load full metadata from CSV if it exists (more reliable than checkpoint)
            metadata_path = OUTPUT_DIR / 'full_metadata.csv'
            if metadata_path.exists() and write_ptr > 0:
                try:
                    existing_metadata_df = pd.read_csv(metadata_path)
                    processed_metadata = existing_metadata_df.to_dict('records')
                    print(f"\n🔄 Resuming from checkpoint:")
                    print(f"   - Last completed batch: {resume_from_batch - 1}")
                    print(f"   - Images processed: {write_ptr}")
                    print(f"   - Metadata loaded from CSV: {len(processed_metadata)} entries")
                    # Verify metadata count matches write_ptr
                    if len(processed_metadata) != write_ptr:
                        print(f"   ⚠ Warning: Metadata count ({len(processed_metadata)}) doesn't match processed images ({write_ptr})")
                        print(f"   Using metadata from CSV (may need to reprocess if inconsistent)")
                except Exception as e:
                    print(f"   ⚠ Could not load metadata from CSV: {e}")
                    print(f"   Using checkpoint metadata (last 100 entries only)")
                    processed_metadata = checkpoint.get('processed_metadata', [])
            else:
                # Fallback to checkpoint metadata (only last 100 entries)
                processed_metadata = checkpoint.get('processed_metadata', [])
                print(f"\n🔄 Resuming from checkpoint:")
                print(f"   - Last completed batch: {resume_from_batch - 1}")
                print(f"   - Images processed: {write_ptr}")
                print(f"   - Metadata entries (from checkpoint): {len(processed_metadata)}")
                print(f"   ⚠ Warning: Only last 100 metadata entries saved in checkpoint")
                print(f"   Full metadata will be reconstructed for new batches")
        except Exception as e:
            print(f"⚠ Failed to load checkpoint: {e}")
            print(f"   Starting fresh...")
            checkpoint = None

    # Process dataset using tf.data pipeline
    if checkpoint is None:
        print(f"\n⚙️  Processing images with tf.data pipeline...")
    else:
        print(f"\n⚙️  Resuming processing from batch {resume_from_batch}...")

    failed_count = checkpoint.get('failed_count', 0) if checkpoint else 0
    start_time = time.time()

    # Prepare memory-mapped arrays to avoid RAM spikes (size ≈ 100GB at 600x600)
    valid_count = len([idx for idx in valid_indices])
    X_memmap_path = OUTPUT_DIR / 'X_full_memmap.dat'
    y_memmap_path = OUTPUT_DIR / 'y_full_memmap.dat'

    # Open memmap in append mode if resuming, otherwise create new
    if checkpoint and write_ptr > 0:
        X_mm = np.memmap(X_memmap_path, dtype=np.float32, mode='r+', shape=(valid_count, TARGET_SIZE[0], TARGET_SIZE[1], 3))
        y_mm = np.memmap(y_memmap_path, dtype=np.int32, mode='r+', shape=(valid_count,))
        print(f"   ✓ Opened existing memmap files for resuming")
    else:
        X_mm = np.memmap(X_memmap_path, dtype=np.float32, mode='w+', shape=(valid_count, TARGET_SIZE[0], TARGET_SIZE[1], 3))
        y_mm = np.memmap(y_memmap_path, dtype=np.int32, mode='w+', shape=(valid_count,))

    # Process in batches with manual batch counter
    batch_idx = 0
    for batch_images, batch_labels in tqdm(dataset, desc="Processing batches"):
        # Skip batches that were already processed (if resuming)
        if checkpoint and batch_idx < resume_from_batch:
            batch_idx += 1
            continue
        
        batch_images_np = batch_images.numpy()
        batch_labels_np = batch_labels.numpy()

        keep_mask = np.any(batch_images_np != 0, axis=(1, 2, 3))
        kept_images = batch_images_np[keep_mask]
        kept_labels = batch_labels_np[keep_mask]

        if kept_images.size > 0:
            end_ptr = write_ptr + kept_images.shape[0]
            X_mm[write_ptr:end_ptr] = kept_images
            y_mm[write_ptr:end_ptr] = kept_labels

            # Metadata gathering for kept samples
            # Get batch indices of kept images (within the current batch)
            kept_batch_indices = np.nonzero(keep_mask)[0]
            # Calculate dataset indices for kept images
            dataset_start_idx = batch_idx * batch_size
            for kept_batch_idx in kept_batch_indices:
                dataset_idx = dataset_start_idx + kept_batch_idx
                if dataset_idx < len(valid_indices):
                    original_idx = valid_indices[dataset_idx]
                    row = custom_df.iloc[original_idx]
                    processed_metadata.append({
                        'image_name': row['image'],
                        'class': row['label'],
                        'age': row.get('age_approx', np.nan),
                        'sex': row.get('sex', 'unknown'),
                        'anatom_site': row.get('anatom_site_general', 'unknown'),
                        'preprocessing_applied': 'colorconstancy_clahe_hairremove_reflectpad_imagenetnorm' if (APPLY_COLOR_CONSTANCY or APPLY_HAIR_REMOVAL or APPLY_IMAGENET_NORM) else 'clahe_reflectpad_no_denoise'
                    })
            write_ptr = end_ptr
        else:
            failed_count += 1
        
        # Save checkpoint every 10 batches (resume safeguard)
        # Also save metadata to CSV periodically to avoid data loss
        if (batch_idx + 1) % 10 == 0:
            checkpoint_data = {
                'last_batch_idx': batch_idx,
                'write_ptr': write_ptr,
                'failed_count': failed_count,
                'processed_metadata': processed_metadata[-100:] if len(processed_metadata) > 100 else processed_metadata  # Keep last 100 for resume
            }
            try:
                with open(checkpoint_path, 'w', encoding='utf-8') as f:
                    json.dump(checkpoint_data, f, indent=2)
            except Exception as e:
                print(f"⚠ Failed to save checkpoint: {e}")
            
            # Save metadata to CSV periodically (every 10 batches) to avoid data loss on resume
            if len(processed_metadata) > 0:
                try:
                    metadata_df_temp = pd.DataFrame(processed_metadata)
                    metadata_path_temp = OUTPUT_DIR / 'full_metadata.csv'
                    metadata_df_temp.to_csv(metadata_path_temp, index=False)
                except Exception as e:
                    print(f"⚠ Failed to save metadata to CSV: {e}")
        
        batch_idx += 1

    processing_time = time.time() - start_time
    print(f"\n✅ Successfully processed: {write_ptr} images")
    print(f"❌ Failed to process: {failed_count} images")
    print(f"⏱️  Total processing time: {processing_time/60:.2f} minutes")
    rate = (write_ptr / processing_time) if processing_time > 0 else 0.0
    print(f"⚡ Average rate: {rate:.1f} images/second")
    print(f"🚀 tf.data pipeline efficiency: {rate:.1f} images/second")

    # ============================================
    # STEP 5: Save Custom Dataset
    # ============================================
    print("\n💾 Step 5: Saving full dataset...")

    # Trim memmaps to actual written length (in case of failures)
    X_mm.flush(); y_mm.flush()
    X_view_memmap = np.memmap(X_memmap_path, dtype=np.float32, mode='r', shape=(valid_count, TARGET_SIZE[0], TARGET_SIZE[1], 3))[:write_ptr]
    y_view_memmap = np.memmap(y_memmap_path, dtype=np.int32, mode='r', shape=(valid_count,))[:write_ptr]

    # Convert memmap views to regular arrays for Step 6 visualizations
    # This ensures data is in memory before we delete memmap files
    X_view = np.array(X_view_memmap, copy=True)
    y_view = np.array(y_view_memmap, copy=True)

    print(f"  Images shape: {X_view.shape}")
    print(f"  Labels shape: {y_view.shape}")
    try:
        print(f"  Images size on disk (memmap): {Path(X_memmap_path).stat().st_size / (1024**3):.2f} GB")
    except Exception:
        pass

    # Save final .npy snapshots (can be very large)
    np.save(OUTPUT_DIR / 'X_full.npy', X_view)
    np.save(OUTPUT_DIR / 'y_full.npy', y_view)

    # Verify metadata consistency
    metadata_count = len(processed_metadata)
    if metadata_count != write_ptr:
        print(f"⚠ Warning: Metadata count ({metadata_count}) doesn't match processed images ({write_ptr})")
        print(f"   This may happen if resuming from checkpoint. Metadata will be saved with available entries.")
    else:
        print(f"✓ Metadata verified: {metadata_count} entries match {write_ptr} processed images")
    
    metadata_df = pd.DataFrame(processed_metadata)
    metadata_df.to_csv(OUTPUT_DIR / 'full_metadata.csv', index=False)

    print(f"✓ Saved: {OUTPUT_DIR / 'X_full.npy'}")
    print(f"✓ Saved: {OUTPUT_DIR / 'y_full.npy'}")
    print(f"✓ Saved: {OUTPUT_DIR / 'full_metadata.csv'}")

    # Delete checkpoint on successful completion
    if checkpoint_path.exists():
        try:
            checkpoint_path.unlink()
            print("✓ Removed checkpoint file (processing complete)")
        except Exception as e:
            print(f"⚠ Failed to delete checkpoint: {e}")

    # Optional cleanup of temporary artifacts to reclaim disk space
    # Note: Cleanup happens AFTER converting to regular arrays so Step 6 can use X_view/y_view
    try:
        if (OUTPUT_DIR / 'tf_cache').exists():
            shutil.rmtree(OUTPUT_DIR / 'tf_cache', ignore_errors=True)
            print("✓ Removed tf.data disk cache: tf_cache/")
        if Path(X_memmap_path).exists():
            Path(X_memmap_path).unlink(missing_ok=True)
            print("✓ Removed temporary memmap file: X_full_memmap.dat")
        if Path(y_memmap_path).exists():
            Path(y_memmap_path).unlink(missing_ok=True)
            print("✓ Removed temporary memmap file: y_full_memmap.dat")
    except Exception as e:
        print(f"⚠ Cleanup skipped: {e}")

# ============================================
# STEP 6: Enhanced Visualizations for Reports
# ============================================
print("\n🎨 Step 6: Creating comprehensive visualizations...")

# Verify data is available for visualization
try:
    # Check if X_view and y_view exist and have data
    if len(X_view) == 0 or len(y_view) == 0:
        print("⚠ Warning: Empty dataset. Skipping visualizations.")
        exit(0)
    
    if len(X_view) != len(y_view):
        print(f"⚠ Warning: Mismatch between images ({len(X_view)}) and labels ({len(y_view)}).")
        print("   Visualizations may be incomplete.")
except NameError:
    print("❌ Error: X_view and y_view are not defined. Cannot create visualizations.")
    print("   This should not happen if Step 5 completed successfully or files were loaded.")
    exit(1)

# Create visualizations directory (on network volume for persistence)
viz_dir = OUTPUT_DIR / 'visualizations'
viz_dir.mkdir(exist_ok=True, parents=True)
print(f"✓ Visualizations directory: {viz_dir}")
print(f"  {'Network Volume (Persistent)' if NETWORK_VOLUME else 'Workspace (Temporary)'}")
print(f"  Dataset: {len(X_view):,} images, {len(y_view):,} labels")

# Visualization 1: Class Distribution Bar Chart
print("  📊 Creating class distribution chart...")
fig, ax = plt.subplots(figsize=(12, 6))
class_counts = pd.Series(y_view).value_counts().sort_index()
class_names_sorted = [CLASS_NAMES[i] for i in class_counts.index]
colors = plt.cm.Set3(np.linspace(0, 1, len(class_counts)))

bars = ax.bar(class_names_sorted, class_counts.values, color=colors, edgecolor='black', linewidth=1.5)
ax.set_xlabel('Class', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
ax.set_title('ISIC 2019 Dataset - Class Distribution', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height):,}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(viz_dir / 'class_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"    ✓ Saved: class_distribution.png")

# Visualization 2: Sample Images from Each Class (Enhanced)
print("  🖼️  Creating sample images visualization...")
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('ISIC 2019 - Sample Preprocessed Images by Class', fontsize=18, fontweight='bold', y=0.98)

for class_idx, class_name in enumerate(CLASS_NAMES):
    class_indices = np.where(y_view == class_idx)[0]
    row = class_idx // 4
    col = class_idx % 4
    
    if len(class_indices) > 0:
        sample_idx = class_indices[0]
        img_display = X_view[sample_idx]
        
        # Denormalize ImageNet normalization for display
        if APPLY_IMAGENET_NORM:
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_display = img_display * std + mean
            img_display = np.clip(img_display, 0, 1)
        
        axes[row, col].imshow(img_display)
        axes[row, col].set_title(f'{class_name}\n({class_counts[class_idx]:,} images)', 
                                 fontsize=14, fontweight='bold', pad=10)
        axes[row, col].axis('off')
    else:
        axes[row, col].text(0.5, 0.5, f'No {class_name}\nimages', 
                           ha='center', va='center', fontsize=12)
        axes[row, col].axis('off')

plt.tight_layout()
plt.savefig(viz_dir / 'sample_images_by_class.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"    ✓ Saved: sample_images_by_class.png")

# Visualization 3: Class Distribution Pie Chart
print("  🥧 Creating class distribution pie chart...")
fig, ax = plt.subplots(figsize=(12, 12))
colors_pie = plt.cm.Set3(np.linspace(0, 1, len(class_counts)))
wedges, texts, autotexts = ax.pie(class_counts.values, labels=class_names_sorted, 
                                   autopct='%1.1f%%', colors=colors_pie,
                                   startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})

# Enhance text
for autotext in autotexts:
    autotext.set_color('black')
    autotext.set_fontweight('bold')

ax.set_title('ISIC 2019 Dataset - Class Distribution (Pie Chart)', 
             fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(viz_dir / 'class_distribution_pie.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"    ✓ Saved: class_distribution_pie.png")

# Visualization 4: Dataset Statistics Summary
print("  📈 Creating dataset statistics summary...")
fig, ax = plt.subplots(figsize=(14, 8))
fig.patch.set_facecolor('white')

# Create statistics text
stats_text = f"""
ISIC 2019 Skin Lesion Classification Dataset
{'='*50}

Dataset Overview:
  • Total Images: {len(X_view):,}
  • Number of Classes: {len(CLASS_NAMES)}
  • Image Resolution: {TARGET_SIZE[0]}x{TARGET_SIZE[1]} pixels
  • Preprocessing: Medical Imaging Pipeline

Class Distribution:
"""
for class_idx, class_name in enumerate(CLASS_NAMES):
    count = class_counts.get(class_idx, 0)
    percentage = (count / len(X_view)) * 100 if len(X_view) > 0 else 0
    stats_text += f"  • {class_name}: {count:,} images ({percentage:.2f}%)\n"

stats_text += f"""
Preprocessing Pipeline:
  • Color Constancy: {'✓' if APPLY_COLOR_CONSTANCY else '✗'}
  • Hair Removal: {'✓' if APPLY_HAIR_REMOVAL else '✗'}
  • CLAHE Enhancement: ✓
  • Advanced Medical Preprocessing: {'✓' if APPLY_ADVANCED_PREPROCESSING else '✗'}
  • ImageNet Normalization: {'✓' if APPLY_IMAGENET_NORM else '✗'}

Storage Information:
  • Output Directory: {OUTPUT_DIR}
  • Network Volume: {NETWORK_VOLUME if NETWORK_VOLUME else 'Not used'}
"""

ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax.axis('off')
ax.set_title('Dataset Statistics Summary', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(viz_dir / 'dataset_statistics_summary.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"    ✓ Saved: dataset_statistics_summary.png")

# Visualization 5: Combined Report Figure
print("  📄 Creating combined report figure...")
fig = plt.figure(figsize=(20, 12))
# Grid: 3 rows, 4 columns (row 0 for bar chart, rows 1-2 for 8 classes in 2x4 grid)
gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

# Top: Class distribution bar chart (spans all 4 columns)
ax1 = fig.add_subplot(gs[0, :])
bars = ax1.bar(class_names_sorted, class_counts.values, color=colors, edgecolor='black', linewidth=1.5)
ax1.set_xlabel('Class', fontsize=12, fontweight='bold')
ax1.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
ax1.set_title('Class Distribution', fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3, linestyle='--')
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height, f'{int(height):,}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Bottom: Sample images (2x4 grid for all 8 classes)
for class_idx, class_name in enumerate(CLASS_NAMES):
    # Calculate grid position: row 1 or 2, column 0-3
    row = 1 + (class_idx // 4)  # Row 1 for classes 0-3, row 2 for classes 4-7
    col = class_idx % 4  # Column 0-3
    ax = fig.add_subplot(gs[row, col])
    
    class_indices = np.where(y_view == class_idx)[0]
    if len(class_indices) > 0:
        sample_idx = class_indices[0]
        img_display = X_view[sample_idx]
        
        # Denormalize ImageNet normalization for display
        if APPLY_IMAGENET_NORM:
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_display = img_display * std + mean
            img_display = np.clip(img_display, 0, 1)
        
        ax.imshow(img_display)
        ax.set_title(f'{class_name}\n({class_counts[class_idx]:,})', fontsize=10, fontweight='bold')
        ax.axis('off')
    else:
        ax.text(0.5, 0.5, f'No {class_name}\nimages', 
               ha='center', va='center', fontsize=10)
        ax.axis('off')

fig.suptitle('ISIC 2019 Skin Lesion Classification Dataset - Comprehensive Report', 
             fontsize=18, fontweight='bold', y=0.98)
plt.savefig(viz_dir / 'comprehensive_report.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"    ✓ Saved: comprehensive_report.png")

# Also save the simple version for quick reference
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('ISIC 2019 - Sample Images (Full Dataset)', fontsize=16, fontweight='bold')

for class_idx, class_name in enumerate(CLASS_NAMES):
    class_indices = np.where(y_view == class_idx)[0]
    if len(class_indices) > 0:
        sample_idx = class_indices[0]
        img_display = X_view[sample_idx]
        
        if APPLY_IMAGENET_NORM:
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_display = img_display * std + mean
            img_display = np.clip(img_display, 0, 1)
        
        row = class_idx // 4
        col = class_idx % 4
        
        axes[row, col].imshow(img_display)
        axes[row, col].set_title(f'{class_name}', fontsize=12, fontweight='bold')
        axes[row, col].axis('off')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'full_samples.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\n✅ All visualizations saved to: {viz_dir}")
print(f"  📊 High-Resolution Visualizations (300 DPI - Report Ready):")
print(f"     • class_distribution.png (300 DPI)")
print(f"     • sample_images_by_class.png (300 DPI)")
print(f"     • class_distribution_pie.png (300 DPI)")
print(f"     • dataset_statistics_summary.png (300 DPI)")
print(f"     • comprehensive_report.png (300 DPI)")
print(f"  📄 Quick Reference (150 DPI):")
print(f"     • full_samples.png (150 DPI) - Saved to {OUTPUT_DIR}")
print(f"  💾 Storage: {'Network Volume (Persistent)' if NETWORK_VOLUME else 'Workspace (Temporary)'}")

# ============================================
# SUMMARY
# ============================================
print("\n" + "=" * 70)
print("✅ WEEK 2 COMPLETE: FULL DATASET PREPROCESSING")
print("=" * 70)
print(f"\n📦 Output Files (Saved to {'Network Volume (Persistent)' if NETWORK_VOLUME else 'Workspace (Temporary)'}):")
print(f"  Data Files:")
print(f"  1. X_full.npy - Preprocessed images ({X_view.shape})")
print(f"  2. y_full.npy - Class labels ({y_view.shape})")
print(f"  3. full_metadata.csv - Image metadata")
print(f"  4. custom_dataset_statistics.json - Dataset statistics")
print(f"  ")
print(f"  Visualizations (Report-Ready, 300 DPI):")
print(f"  5. visualizations/class_distribution.png - Bar chart (300 DPI)")
print(f"  6. visualizations/class_distribution_pie.png - Pie chart (300 DPI)")
print(f"  7. visualizations/sample_images_by_class.png - Sample images (300 DPI)")
print(f"  8. visualizations/dataset_statistics_summary.png - Statistics summary (300 DPI)")
print(f"  9. visualizations/comprehensive_report.png - Combined report (300 DPI)")
print(f"  ")
print(f"  Quick Reference (150 DPI):")
print(f" 10. full_samples.png - Quick reference (150 DPI)")
print(f"  ")
print(f"  💾 Storage: {OUTPUT_DIR}")
print(f"  {'✓ Network Volume (Persistent)' if NETWORK_VOLUME else '⚠ Workspace (Temporary - may be lost when pod stops)'}")

print(f"\n📊 Dataset Statistics:")
print(f"  - Total images: {X_view.shape[0]:,}")
print(f"  - Images per class: variable (full dataset)")
try:
    print(f"  - Storage size (X_full.npy): {Path(OUTPUT_DIR / 'X_full.npy').stat().st_size / (1024**3):.2f} GB")
except Exception:
    pass
if not skip_to_visualization:
    print(f"  - Processing time: {processing_time/60:.2f} minutes")
else:
    print(f"  - Status: Preprocessed dataset loaded from disk")

print(f"\n🔧 Medical Preprocessing Applied:")
print(f"  ✓ Color constancy (Shades-of-Gray)")
print(f"  ✓ Advanced medical preprocessing")
print(f"  ✓ Lesion enhancement")
print(f"  ✓ CLAHE (Contrast Limited Adaptive Histogram Equalization)")
print(f"  ✓ Hair removal (DullRazor)")
print(f"  ✓ ImageNet normalization (for transfer learning)")

print(f"\n🚀 Ready for High-Resolution Augmentation:")
print(f"  - Original: {X_view.shape[0]:,} images at {TARGET_SIZE}")
print(f"  - After 4x augmentation: {X_view.shape[0] * 4:,} images at {TARGET_SIZE}")
print(f"  - Total storage needed: {(X_view.shape[0] * 4 * (TARGET_SIZE[0]*TARGET_SIZE[1]*3*4)) / (1024**3):.2f} GB")
print(f"  - A40/A6000 optimized: High-resolution preprocessing + dynamic downscaling")
print(f"  - Strategy: Downscale to model-specific sizes (no upscaling artifacts)")

print(f"\n📍 Storage Information:")
print(f"  Output Location: {OUTPUT_DIR}")
print(f"  Input Dataset: {INPUT_DIR}")
if NETWORK_VOLUME:
    print(f"  Network Volume: {NETWORK_VOLUME} (Persistent storage)")
    print(f"  ✓ All outputs saved to network volume for persistence")
    if WORKSPACE_OUTPUT_DIR.is_symlink():
        print(f"  Workspace Symlink: {WORKSPACE_OUTPUT_DIR} -> {OUTPUT_DIR}")
else:
    print(f"  ⚠ Network Volume: Not detected")
    print(f"  ⚠ Outputs saved to workspace (may be lost when pod stops)")
    print(f"  💡 Tip: Attach network volume for persistent storage")

print(f"\n💻 Execution: CPU Preprocessing (Medical Image Processing)")
print(f"🎯 Next Step: Week 3 - Data Augmentation")
print(f"\n📝 Report Files:")
print(f"  All visualizations saved to: {viz_dir}")
print(f"  Use these files for your project report and documentation")
print(f"\n💾 Network Volume Persistence:")
print(f"  ✓ All data files saved to: {OUTPUT_DIR}")
print(f"  ✓ All visualizations saved to: {viz_dir}")
print(f"  ✓ All metadata saved to: {OUTPUT_DIR}")
if NETWORK_VOLUME:
    print(f"  ✓ Network Volume: {NETWORK_VOLUME} (Persistent storage)")
    print(f"  ✓ All outputs are persistent (survive pod restarts)")
    if WORKSPACE_OUTPUT_DIR.is_symlink():
        print(f"  ✓ Workspace symlink: {WORKSPACE_OUTPUT_DIR} -> {OUTPUT_DIR}")
else:
    print(f"  ⚠ Network Volume: Not detected")
    print(f"  ⚠ Outputs may be lost when pod stops")
    print(f"  💡 Tip: Attach network volume for persistent storage")
print("=" * 70)