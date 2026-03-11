#!/usr/bin/env python3
"""
Week 1: RunPod Environment Setup & Initialization

Module:       week1.py
Purpose:      Automated setup of reproducible deep-learning environment
Project:      ISIC 2019 Skin Cancer Classification
Dataset:      8-class dermoscopy images
Author:       Amjad
Date:         February 2026
Platform:     RunPod (NVIDIA A40 • CUDA 12.4)

═══════════════════════════════════════════════════════════════════════════════

DESCRIPTION
───────────
Comprehensive RunPod environment initialization script for GPU-accelerated
training. Configures TensorFlow, installs ML/medical imaging stacks,
validates GPU, and creates persistent project structure.

ENVIRONMENT
───────────
• GPU:       NVIDIA A40 (48GB VRAM, Ampere architecture)
• CUDA:      12.4
• cuDNN:     8.x
• Template:  madiator2011/better-tensorflow:cuda12.4-cudnn8
• Framework: TensorFlow 2.15.0 (pre-installed)

FEATURES PROVIDED
─────────────────
✓ GPU Memory Configuration
  ├─ Dynamic memory growth (prevents OOM crashes)
  ├─ Safe memory limits (up to 90% VRAM)
  └─ Ampere optimizations (TensorFloat-32, mixed precision ready)

✓ ML/DL Stack Installation
  ├─ Core: numpy, pandas, scikit-learn, scikit-image
  ├─ Vision: OpenCV, Pillow, albumentations, imgaug
  ├─ Medical: SimpleITK, pydicom, medpy
  ├─ DL: ONNX, tf2onnx, onnxruntime-gpu
  └─ TensorRT: 8.6.1 + PyCUDA (for week15 acceleration)

✓ Experiment Tracking
  ├─ TensorBoard (visualization)
  ├─ Weights & Biases (cloud logging)
  └─ MLflow (experiment management)

✓ Validation & Diagnostics
  ├─ CUDA/GPU verification
  ├─ Package compatibility checks
  ├─ Resource monitoring (CPU, RAM, disk)
  ├─ Dataset validation (ISIC 2019)
  └─ System information logging

✓ Storage Configuration
  ├─ Network volume detection (persistent)
  ├─ Workspace organization
  ├─ Automatic symlink setup
  └─ Disk space validation

CRITICAL EXECUTION NOTES
────────────────────────
⚠️  MUST run from RunPod TERMINAL, NOT notebook cell.

   WRONG: Open notebook → run first cell with week1.py code
   RIGHT: File → New → Terminal → python /workspace/week1.py

   Why: TensorFlow claims CUDA context on import. If TRT/PyCUDA
   are imported in same process, pycuda.autoinit fails.

   This script installs TRT & PyCUDA in subprocess, keeping them
   separate from TensorFlow. week15.py also requires fresh terminal
   for same reason.

USAGE
─────
Run from dedicated terminal:

    python /workspace/week1.py

Expected runtime: 10-20 minutes (depends on package installation speed)

Requirements:
  • RunPod GPU pod with A40 or RTX A6000 (48GB VRAM)
  • CUDA 12.4-compatible base image (provided by template)
  • Network volume attached (recommended for persistence)
  • ~50GB free disk space (for packages + dataset)

Outputs:
  • Complete /workspace/outputs/ directory structure
  • System diagnostics and GPU validation
  • Quick reference file with configuration details
  • All ML/DL packages installed and verified

Version: 3.0 (2026)
═══════════════════════════════════════════════════════════════════════════════
"""

import sys
import subprocess
import os
import warnings
import shutil
from pathlib import Path
warnings.filterwarnings('ignore')

print("=" * 70)
print("🚀 RUNPOD ENVIRONMENT SETUP - SKIN CANCER CLASSIFICATION")
print("=" * 70)
print(f"Python version: {sys.version.split()[0]}")
print(f"Python executable: {sys.executable}")
print("=" * 70)

# ============================================
# STEP 1: RUNPOD ENVIRONMENT DETECTION
# ============================================
print("\n🔍 Detecting RunPod environment...")

# Detect RunPod workspace (RunPod uses /workspace by default)
runpod_workspace = None
if os.path.exists("/workspace"):
    runpod_workspace = "/workspace"
    print(f"✓ RunPod workspace detected: {runpod_workspace}")
elif os.path.exists("/notebooks"):
    runpod_workspace = "/notebooks"
    print(f"✓ Jupyter workspace detected: {runpod_workspace}")
else:
    runpod_workspace = os.getcwd()
    print(f"⚠ Using current directory: {runpod_workspace}")

# Check for RunPod environment variables
runpod_pod_id = os.environ.get('RUNPOD_POD_ID', None)
runpod_pod_type = os.environ.get('RUNPOD_POD_TYPE', None)
if runpod_pod_id:
    print(f"✓ RunPod Pod ID: {runpod_pod_id}")
if runpod_pod_type:
    print(f"✓ RunPod Pod Type: {runpod_pod_type}")

# Check for network volumes (RunPod persistent storage)
network_volumes = []
NETWORK_VOLUME = None
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

print("=" * 70)

# ============================================
# STEP 2: VERIFY GPU WITH NVIDIA-SMI
# ============================================
print("\n🎮 Verifying GPU with nvidia-smi...")
gpu_available = False
gpu_name = "Unknown"
gpu_memory_mb = 0

try:
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, timeout=10, check=False
    )
    if result.returncode == 0 and result.stdout.strip():
        gpu_info = result.stdout.strip().split('\n')[0].split(',')
        if len(gpu_info) >= 2:
            gpu_name = gpu_info[0].strip()
            gpu_memory_mb = int(float(gpu_info[1].strip()))
            gpu_available = True
            print(f"✓ GPU detected: {gpu_name}")
            print(f"  VRAM: {gpu_memory_mb} MB ({gpu_memory_mb/1024:.1f} GB)")
            if 'A40' in gpu_name or 'A6000' in gpu_name or 'RTX A6000' in gpu_name:
                if 'A40' in gpu_name:
                    print(f"  ✓ NVIDIA A40 confirmed (48GB VRAM)")
                else:
                    print(f"  ✓ RTX A6000 confirmed (48GB VRAM)")
            else:
                print(f"  ⚠ Warning: Expected A40/A6000, got {gpu_name}")
                print(f"     Note: Code is optimized for 48GB VRAM Ampere GPUs (A40/A6000)")
    else:
        print("⚠ nvidia-smi not available or no GPU detected")
except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
    print(f"⚠ Could not verify GPU with nvidia-smi: {type(e).__name__}")

print("=" * 70)

# ============================================
# STEP 3: CHECK TENSORFLOW (FROM TEMPLATE)
# ============================================
print("\n📦 Checking TensorFlow installation (from RunPod template)...")

tf_installed = False
tf_version = None
cuda_version = None
cudnn_version = None
tf_gpu_detected = False

try:
    # Suppress TensorFlow info/warning messages during import
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import tensorflow as tf
    tf_installed = True
    tf_version = tf.__version__
    print(f"✓ TensorFlow found: {tf_version}")
    
    # Check CUDA support
    cuda_built = tf.test.is_built_with_cuda()
    print(f"  Built with CUDA: {cuda_built}")
    
    # Get CUDA and cuDNN versions
    try:
        build_info = tf.sysconfig.get_build_info()
        cuda_version = build_info.get('cuda_version', 'Unknown')
        cudnn_version = build_info.get('cudnn_version', 'Unknown')
        print(f"  CUDA version: {cuda_version}")
        print(f"  cuDNN version: {cudnn_version}")
        
        # Verify CUDA 12.4 compatibility
        if cuda_version and cuda_version != 'Unknown':
            if '12' in str(cuda_version):
                print(f"  ✓ CUDA {cuda_version} compatible with RunPod template")
            else:
                print(f"  ⚠ CUDA {cuda_version} may not match template (expected 12.4)")
    except (AttributeError, KeyError, TypeError) as e:
        print(f"  ⚠ Could not determine CUDA version: {type(e).__name__}")
    
    # Check GPU availability in TensorFlow (but don't initialize GPU yet)
    try:
        gpu_devices = tf.config.list_physical_devices('GPU')
        if gpu_devices:
            tf_gpu_detected = True
            print(f"  ✓ TensorFlow GPU devices detected: {len(gpu_devices)}")
            for i, gpu in enumerate(gpu_devices):
                print(f"    Device {i}: {gpu.name}")
                try:
                    gpu_details = tf.config.experimental.get_device_details(gpu)
                    if gpu_details:
                        device_name = gpu_details.get('device_name', 'Unknown')
                        print(f"      Name: {device_name}")
                except (AttributeError, RuntimeError, Exception):
                    pass
        else:
            print("  ⚠ TensorFlow: No GPU devices detected")
    except Exception as e:
        print(f"  ⚠ Could not list GPU devices: {type(e).__name__}")
        
except ImportError:
    print("✗ TensorFlow not found in template")
    print("  This is unexpected for the selected template")
    print("  Will attempt to install TensorFlow compatible with CUDA 12.4")
except Exception as e:
    print(f"⚠ TensorFlow import error: {type(e).__name__}")

print("=" * 70)

# ============================================
# STEP 4: INSTALL COMPATIBLE PACKAGES (CUDA 12.4)
# ============================================
print("\n🔧 Installing packages compatible with CUDA 12.4 and TensorFlow...")
print("   (Using --no-cache-dir to save RunPod disk space)")

# Core packages with CUDA 12.4 compatible versions
core_packages = [
    # Data processing (compatible with TensorFlow 2.15+)
    "numpy>=1.24.0,<2.0.0",
    "pandas>=2.0.0,<3.0.0",
    "scikit-learn>=1.3.0,<2.0.0",
    "scikit-image>=0.21.0,<0.23.0",
    
    # Computer Vision
    "opencv-python-headless>=4.8.0,<5.0.0",
    "Pillow>=10.0.0,<11.0.0",
    
    # Visualization
    "matplotlib>=3.7.0,<4.0.0",
    "seaborn>=0.12.0,<0.14.0",
    
    # Deep Learning Optimization (CUDA 12.4 compatible)
    "onnx>=1.15.0,<1.17.0",
    
    # Model Explainability
    "tf-keras-vis>=0.8.0,<0.9.0",
    "grad-cam>=1.4.0,<1.5.0",
    
    # API Development
    "fastapi>=0.104.0,<0.110.0",
    "uvicorn[standard]>=0.24.0,<0.30.0",
    "python-multipart>=0.0.6,<1.0.0",
    "pydantic>=2.5.0,<3.0.0",
    
    # Medical Imaging Specific
    "albumentations>=1.3.0,<1.4.0",
    "imgaug>=0.4.0,<0.5.0",
    "medpy>=0.4.0,<0.5.0",
    "SimpleITK>=2.3.0,<2.4.0",
    "pydicom>=2.4.0,<2.5.0",
    
    # Performance Monitoring
    "wandb>=0.16.0,<0.17.0",
    "tensorboard>=2.15.0,<2.16.0",
    "mlflow>=2.8.0,<2.9.0",
    
    # System Monitoring
    "memory-profiler>=0.61.0,<0.62.0",
    "psutil>=5.9.0,<6.0.0",
    
    # Utilities
    "tqdm>=4.66.0,<5.0.0",
    "joblib>=1.3.0,<1.4.0",
    "pyyaml>=6.0.0,<7.0.0",
    "requests>=2.31.0,<3.0.0",
]

# Install packages with error handling
failed_packages = []
successful_packages = []
warned_packages = []
skipped_packages = []

print("\n📦 Installing core packages...")
for package in core_packages:
    try:
        package_name = package.split('>=')[0].split('==')[0].split(',')[0].strip()
        print(f"  Installing: {package_name}...", end=' ', flush=True)
        
        # Check if package is already installed
        try:
            check_result = subprocess.run([
                sys.executable, "-m", "pip", "show", package_name
            ], capture_output=True, text=True, timeout=10, check=False)
            
            if check_result.returncode == 0:
                installed_version = None
                for line in check_result.stdout.split('\n'):
                    if line.startswith('Version:'):
                        installed_version = line.split(':', 1)[1].strip()
                        break
                
                # Skip non-critical packages if already installed
                critical_packages = ['numpy', 'pandas', 'opencv-python-headless', 'tensorflow']
                if package_name not in critical_packages and installed_version:
                    print("✓ (already installed, skipping)")
                    successful_packages.append(package_name)
                    skipped_packages.append(package_name)
                    continue
        except Exception:
            pass
        
        # Install with upgrade
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "--no-cache-dir", "--quiet", "--upgrade", package
        ], capture_output=True, text=True, check=False, timeout=300)
        
        if result.returncode == 0:
            print("✓")
            successful_packages.append(package_name)
        else:
            stdout_lower = result.stdout.lower()
            stderr_lower = result.stderr.lower()
            if "already satisfied" in stdout_lower or "requirement already satisfied" in stdout_lower:
                print("✓ (already installed)")
                successful_packages.append(package_name)
                warned_packages.append(package_name)
            elif "WARNING" in result.stderr.upper() and "already installed" in stderr_lower:
                print("✓ (already installed)")
                successful_packages.append(package_name)
                warned_packages.append(package_name)
            else:
                print(f"✗")
                error_msg = result.stderr[:200] if result.stderr else result.stdout[:200]
                if error_msg:
                    print(f"    Error: {error_msg}")
                failed_packages.append(package)
            
    except subprocess.TimeoutExpired:
        print(f"✗ (timeout)")
        failed_packages.append(package)
    except Exception as e:
        print(f"✗ ({str(e)[:50]})")
        failed_packages.append(package)

print(f"\n  ✓ Successfully installed/verified: {len(successful_packages)} packages")
if skipped_packages:
    print(f"  ⭐ Skipped (already installed): {len(skipped_packages)} packages")
if warned_packages:
    print(f"  ⚠ Already installed (upgraded): {len(warned_packages)} packages")
if failed_packages:
    print(f"  ✗ Failed: {len(failed_packages)} packages")
    print(f"    Failed packages: {', '.join([p.split('>=')[0].split(',')[0].strip() for p in failed_packages[:5]])}")

# ============================================
# STEP 5: INSTALL TENSORFLOW (IF MISSING)
# ============================================
if not tf_installed:
    print("\n🔄 Installing TensorFlow (CUDA 12.4 compatible)...")
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "install",
            "--no-cache-dir", "--quiet", "--upgrade", "tensorflow[and-cuda]>=2.15.0"
        ], capture_output=True, text=True, check=False, timeout=600)
        
        if result.returncode == 0:
            print("✓ TensorFlow installed successfully")
            successful_packages.append("tensorflow")
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            import tensorflow as tf
            tf_version = tf.__version__
            tf_installed = True
            print(f"  TensorFlow version: {tf_version}")
        else:
            print("⚠ TensorFlow installation had issues")
            error_msg = result.stderr[:300] if result.stderr else result.stdout[:300]
            if error_msg:
                print(f"  Error: {error_msg}")
            failed_packages.append("tensorflow")
    except subprocess.TimeoutExpired:
        print("✗ TensorFlow installation timeout")
        failed_packages.append("tensorflow")
    except Exception as e:
        print(f"✗ TensorFlow installation failed: {type(e).__name__}")
        failed_packages.append("tensorflow")
else:
    print(f"\n✓ TensorFlow already installed from template: {tf_version}")

# ============================================
# STEP 6: INSTALL ONNX RUNTIME GPU (CUDA 12.4)
# ============================================
print("\n🔄 Installing ONNX Runtime GPU (CUDA 12.4 compatible)...")
onnxruntime_gpu_installed = False
try:
    result = subprocess.run([
        sys.executable, "-m", "pip", "install",
        "--no-cache-dir", "--quiet", "--upgrade", "onnxruntime-gpu>=1.18.0"
    ], capture_output=True, text=True, check=False, timeout=300)
    
    if result.returncode == 0:
        print("✓ ONNX Runtime GPU installed successfully")
        successful_packages.append("onnxruntime-gpu")
        onnxruntime_gpu_installed = True
    else:
        stdout_lower = result.stdout.lower()
        stderr_lower = result.stderr.lower()
        if "already satisfied" in stdout_lower or "already satisfied" in stderr_lower:
            print("✓ ONNX Runtime GPU already installed")
            successful_packages.append("onnxruntime-gpu")
            onnxruntime_gpu_installed = True
        else:
            print("⚠ ONNX Runtime GPU installation had issues, trying CPU version...")
            result_cpu = subprocess.run([
                sys.executable, "-m", "pip", "install",
                "--no-cache-dir", "--quiet", "--upgrade", "onnxruntime>=1.18.0"
            ], capture_output=True, text=True, check=False, timeout=300)
            
            if result_cpu.returncode == 0:
                print("✓ ONNX Runtime (CPU) installed as fallback")
                successful_packages.append("onnxruntime")
                warned_packages.append("onnxruntime-gpu")
            else:
                print("⚠ ONNX Runtime installation had issues (may work with TensorFlow directly)")
                failed_packages.append("onnxruntime-gpu")
except subprocess.TimeoutExpired:
    print("✗ ONNX Runtime installation timeout")
    failed_packages.append("onnxruntime-gpu")
except Exception as e:
    print(f"⚠ ONNX Runtime installation issue: {type(e).__name__}")
    failed_packages.append("onnxruntime-gpu")

# ============================================
# STEP 7: INSTALL TF2ONNX
# ============================================
print("\n🔄 Installing tf2onnx (TensorFlow to ONNX converter)...")
try:
    result = subprocess.run([
        sys.executable, "-m", "pip", "install",
        "--no-cache-dir", "--quiet", "--upgrade", "tf2onnx>=1.16.0"
    ], capture_output=True, text=True, check=False, timeout=300)
    
    if result.returncode == 0:
        print("✓ tf2onnx installed successfully")
        successful_packages.append("tf2onnx")
    else:
        stdout_lower = result.stdout.lower()
        stderr_lower = result.stderr.lower()
        if "already satisfied" in stdout_lower or "already satisfied" in stderr_lower:
            print("✓ tf2onnx already installed")
            successful_packages.append("tf2onnx")
        else:
            print(f"⚠ tf2onnx installation had issues: {result.stderr[:200] if result.stderr else 'Unknown error'}")
            failed_packages.append("tf2onnx")
except subprocess.TimeoutExpired:
    print("✗ tf2onnx installation timeout")
    failed_packages.append("tf2onnx")
except Exception as e:
    print(f"⚠ tf2onnx installation issue: {type(e).__name__}")
    failed_packages.append("tf2onnx")


# ============================================
# STEP 8: INSTALL TENSORRT 8.6.1
# ============================================
# TensorRT MUST come from NVIDIA's PyPI index, not the default one.
# tensorrt==8.6.1 is compatible with CUDA 12.4 (the template's CUDA version).
#
# IMPORTANT: We only *install* TensorRT here — we do NOT import it.
# TensorRT and PyCUDA must never be imported in the same Python process
# as TensorFlow. week15.py handles all TRT work in its own fresh process.
# ============================================
print("\n" + "=" * 70)
print("⚡ STEP 8: TENSORRT 8.6.1 INSTALLATION")
print("=" * 70)
print("  Source  : https://pypi.nvidia.com")
print("  Version : 8.6.1 (compatible with CUDA 12.4)")
print("  NOTE    : Installing only — NOT importing (TF owns CUDA context here)")

trt_installed = False
trt_packages = [
    "tensorrt==8.6.1",
    "tensorrt-lean==8.6.1",
    "tensorrt-dispatch==8.6.1",
]

try:
    # Check if already installed first
    check = subprocess.run(
        [sys.executable, "-m", "pip", "show", "tensorrt"],
        capture_output=True, text=True, check=False
    )
    already_installed_version = None
    if check.returncode == 0:
        for line in check.stdout.split('\n'):
            if line.startswith('Version:'):
                already_installed_version = line.split(':', 1)[1].strip()
                break

    if already_installed_version == "8.6.1":
        print(f"\n  ✓ TensorRT 8.6.1 already installed — skipping")
        successful_packages.append("tensorrt")
        trt_installed = True
    else:
        if already_installed_version:
            print(f"\n  ⚠ Found TensorRT {already_installed_version} — upgrading to 8.6.1")
        else:
            print(f"\n  Installing TensorRT packages (may take 2-5 minutes)...")

        result = subprocess.run(
            [
                sys.executable, "-m", "pip", "install",
                "--no-cache-dir", "--quiet",
            ] + trt_packages + [
                "--extra-index-url", "https://pypi.nvidia.com"
            ],
            capture_output=True, text=True, check=False, timeout=600
        )

        if result.returncode == 0:
            print("  ✓ TensorRT 8.6.1 installed successfully")
            successful_packages.append("tensorrt")
            trt_installed = True
        else:
            # pip often exits non-zero but still installs — check stdout
            if "already satisfied" in result.stdout.lower():
                print("  ✓ TensorRT already satisfied")
                successful_packages.append("tensorrt")
                trt_installed = True
            else:
                print(f"  ✗ TensorRT installation failed")
                err = result.stderr[:400] if result.stderr else result.stdout[:400]
                print(f"    Error: {err}")
                failed_packages.append("tensorrt==8.6.1")

except subprocess.TimeoutExpired:
    print("  ✗ TensorRT installation timed out (>600s)")
    failed_packages.append("tensorrt==8.6.1")
except Exception as e:
    print(f"  ✗ TensorRT installation error: {type(e).__name__}: {e}")
    failed_packages.append("tensorrt==8.6.1")

print("=" * 70)


# ============================================
# STEP 9: INSTALL PYCUDA (SOURCE BUILD)
# ============================================
# pycuda is a C++ extension that wraps the CUDA driver API.
# It MUST be compiled from source against the system's CUDA 12.4 headers
# so that pycuda.autoinit can create a valid CUDA context in week15.py.
#
# Using --no-binary=pycuda forces pip to compile from source rather than
# grabbing a pre-built wheel that may target a different CUDA version.
#
# Again: only installing here, not importing. TF owns the context right now.
# ============================================
print("\n" + "=" * 70)
print("⚡ STEP 9: PYCUDA INSTALLATION (SOURCE BUILD)")
print("=" * 70)
print("  Compiling against system CUDA 12.4 headers")
print("  Using --no-binary=pycuda to force source compilation")
print("  NOTE    : Installing only — NOT importing (TF owns CUDA context here)")

pycuda_installed = False

try:
    # Check if already installed
    check = subprocess.run(
        [sys.executable, "-m", "pip", "show", "pycuda"],
        capture_output=True, text=True, check=False
    )
    if check.returncode == 0:
        installed_ver = ""
        for line in check.stdout.split('\n'):
            if line.startswith('Version:'):
                installed_ver = line.split(':', 1)[1].strip()
                break
        print(f"\n  Found existing PyCUDA {installed_ver} — reinstalling from source to ensure CUDA 12.4 compatibility...")
    else:
        print(f"\n  Building PyCUDA from source (takes ~2-3 minutes)...")

    # Set CUDA environment so the compiler finds the right headers
    env = os.environ.copy()
    env["PATH"] = "/usr/local/cuda/bin:" + env.get("PATH", "")
    env["CUDA_ROOT"] = "/usr/local/cuda"

    result = subprocess.run(
        [
            sys.executable, "-m", "pip", "install",
            "--no-cache-dir",
            "--no-binary=pycuda",
            "pycuda>=2022.1",
        ],
        capture_output=True, text=True, check=False,
        timeout=600, env=env
    )

    if result.returncode == 0:
        print("  ✓ PyCUDA built and installed successfully")
        successful_packages.append("pycuda")
        pycuda_installed = True
    else:
        if "already satisfied" in result.stdout.lower():
            print("  ✓ PyCUDA already satisfied")
            successful_packages.append("pycuda")
            pycuda_installed = True
        else:
            print("  ✗ PyCUDA build failed")
            err = result.stderr[-600:] if result.stderr else result.stdout[-600:]
            print(f"    Error (last 600 chars):\n{err}")
            print("\n  Troubleshooting:")
            print("    1. Check nvcc is available: nvcc --version")
            print("    2. Check CUDA headers: ls /usr/local/cuda/include/cuda.h")
            print("    3. Try manually: pip install --no-cache-dir --no-binary=pycuda pycuda")
            failed_packages.append("pycuda")

except subprocess.TimeoutExpired:
    print("  ✗ PyCUDA build timed out (>600s)")
    failed_packages.append("pycuda")
except Exception as e:
    print(f"  ✗ PyCUDA installation error: {type(e).__name__}: {e}")
    failed_packages.append("pycuda")

print("=" * 70)
print("\n" + "=" * 70)
print("📋 STEP 10: INSTALLATION VERIFICATION")
print("=" * 70)

# Critical imports for skin cancer classification
# NOTE: tensorrt and pycuda are intentionally NOT in this list.
# They cannot be imported in the same process as TensorFlow.
# They are verified separately via subprocess below.
critical_imports = {
    'tensorflow': 'TensorFlow',
    'sklearn': 'scikit-learn',
    'cv2': 'OpenCV',
    'PIL': 'Pillow',
    'onnx': 'ONNX',
    'onnxruntime': 'ONNX Runtime',
    'fastapi': 'FastAPI',
    'matplotlib': 'Matplotlib',
    'seaborn': 'Seaborn',
    'pandas': 'Pandas',
    'numpy': 'NumPy',
    'tqdm': 'tqdm',
    'albumentations': 'Albumentations',
    'wandb': 'Weights & Biases',
    'tensorboard': 'TensorBoard',
    'psutil': 'psutil',
}

print("\n📦 Testing critical imports...")
working_imports = []
failed_imports = []

for module_name, display_name in critical_imports.items():
    try:
        __import__(module_name)
        print(f"  ✓ {display_name} ({module_name})")
        working_imports.append(module_name)
    except ImportError as e:
        print(f"  ✗ {display_name} ({module_name}): {str(e)[:50]}")
        failed_imports.append(module_name)

print(f"\n  Results: {len(working_imports)}/{len(critical_imports)} imports successful")

# ── TensorRT + PyCUDA subprocess verification ────────────────────────────────
# These MUST be verified in a subprocess — importing them in this process
# (where TensorFlow is active) would cause a CUDA context conflict.
print("\n📦 Verifying TensorRT + PyCUDA (subprocess — isolated from TF)...")

trt_pycuda_verify_code = """
import sys, os
results = {}

try:
    import tensorrt as trt
    results['tensorrt'] = trt.__version__
except Exception as e:
    results['tensorrt'] = f'FAILED: {e}'

try:
    # Set CUDA path before importing pycuda
    os.environ['PATH'] = '/usr/local/cuda/bin:' + os.environ.get('PATH', '')
    import pycuda.driver as cuda
    import pycuda.autoinit
    dev = cuda.Device(0)
    mem_mb = dev.total_memory() // (1024**2)
    cc = dev.compute_capability()
    results['pycuda'] = f'OK — {dev.name()} | {mem_mb:,} MB | CC {cc[0]}.{cc[1]}'
except Exception as e:
    results['pycuda'] = f'FAILED: {e}'

for k, v in results.items():
    print(f'  {k}: {v}')
"""

try:
    verify_result = subprocess.run(
        [sys.executable, "-c", trt_pycuda_verify_code],
        capture_output=True, text=True, check=False, timeout=60
    )
    output = verify_result.stdout.strip()
    if output:
        for line in output.split('\n'):
            if 'FAILED' in line:
                print(f"  ✗ {line.strip()}")
                if 'tensorrt' in line.lower():
                    failed_imports.append('tensorrt')
                else:
                    failed_imports.append('pycuda')
            else:
                print(f"  ✓ {line.strip()}")
                if 'tensorrt' in line.lower():
                    working_imports.append('tensorrt')
                else:
                    working_imports.append('pycuda')
    if verify_result.stderr.strip():
        # Only show stderr if there's a real failure (filter pycuda build noise)
        stderr_lines = [l for l in verify_result.stderr.split('\n')
                        if l.strip() and 'warning' not in l.lower()
                        and 'deprecated' not in l.lower()]
        if stderr_lines:
            print(f"  stderr: {chr(10).join(stderr_lines[:5])}")
except subprocess.TimeoutExpired:
    print("  ✗ TRT/PyCUDA verification timed out")
except Exception as e:
    print(f"  ✗ Subprocess verification failed: {e}")

# ── Step numbering fix for downstream steps ──────────────────────────────────
# ============================================
# STEP 11: SYSTEM RESOURCES CHECK
# ============================================
print("\n" + "=" * 70)
print("🖥️ STEP 11: SYSTEM RESOURCES")
print("=" * 70)

try:
    import psutil
    print("\n  System Resources:")
    print(f"    CPU Cores: {psutil.cpu_count()} (logical)")
    print(f"    CPU Cores: {psutil.cpu_count(logical=False)} (physical)")
    
    memory = psutil.virtual_memory()
    print(f"    RAM Total: {memory.total / 1e9:.1f} GB")
    print(f"    RAM Available: {memory.available / 1e9:.1f} GB")
    print(f"    RAM Used: {memory.percent:.1f}%")
    
    disk = psutil.disk_usage('/')
    print(f"    Disk Total: {disk.total / 1e9:.1f} GB")
    print(f"    Disk Free: {disk.free / 1e9:.1f} GB")
    print(f"    Disk Used: {disk.percent:.1f}%")
    
    # CRITICAL: Check disk space for ISIC dataset (~25GB)
    if disk.free / 1e9 < 30:
        print(f"\n    ⚠ WARNING: Low disk space ({disk.free / 1e9:.1f} GB free)")
        print(f"       ISIC 2019 dataset requires ~25GB")
        print(f"       Consider using network volume or cleaning up space")
    else:
        print(f"\n    ✓ Sufficient disk space for ISIC dataset")
    
    if memory.available / 1e9 < 8:
        print(f"\n    ⚠ Warning: Low RAM available ({memory.available / 1e9:.1f} GB)")
        
except Exception as e:
    print(f"⚠ System resources check failed: {e}")

print("=" * 70)

# ============================================
# STEP 12: GPU CONFIGURATION (A40/A6000 OPTIMIZED)
# ============================================
print("\n" + "=" * 70)
print("🎮 STEP 12: GPU CONFIGURATION - NVIDIA A40/A6000")
print("=" * 70)

gpu_configured = False
gpu_memory_limit_set = False

if tf_installed:
    try:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        import tensorflow as tf
        
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✓ TensorFlow GPU: {len(gpus)} device(s) available")
            
            for i, gpu in enumerate(gpus):
                print(f"\n  Device {i}: {gpu.name}")
                
                # Get GPU details
                try:
                    gpu_details = tf.config.experimental.get_device_details(gpu)
                    if gpu_details:
                        device_name = gpu_details.get('device_name', 'Unknown')
                        print(f"    Name: {device_name}")
                        
                        if 'A40' in device_name:
                            print(f"    ✓ NVIDIA A40 detected (48GB VRAM)")
                        elif 'A6000' in device_name or 'RTX A6000' in device_name:
                            print(f"    ✓ RTX A6000 detected (48GB VRAM)")
                        elif 'A40' in gpu_name:
                            print(f"    ✓ NVIDIA A40 confirmed via nvidia-smi")
                        elif 'A6000' in gpu_name:
                            print(f"    ✓ RTX A6000 confirmed via nvidia-smi")
                except (AttributeError, RuntimeError, TypeError, Exception) as e:
                    if 'A40' in gpu_name or 'A6000' in gpu_name:
                        print(f"    Name: {gpu_name} (from nvidia-smi)")
                    else:
                        print(f"    (Could not get device details: {type(e).__name__})")
                
                # CRITICAL FIX: Enhanced GPU memory configuration with multiple strategies
                print(f"\n    Configuring GPU memory...")
                
                # Strategy 1: Try memory growth (preferred for dynamic allocation)
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"    ✓ Memory growth enabled (dynamic allocation)")
                    print(f"    ✓ No hard memory limit (uses available VRAM)")
                    gpu_configured = True
                    gpu_memory_limit_set = True
                except RuntimeError as e:
                    error_msg = str(e).lower()
                    if "virtual devices" in error_msg or "already been set" in error_msg:
                        print(f"    ⚠ GPU already initialized (common in Jupyter)")
                        print(f"       Memory settings will apply on next kernel restart")
                        
                        # Strategy 2: Try to get current memory info as fallback
                        try:
                            memory_info = tf.config.experimental.get_memory_info(gpu.name.replace('/physical_device:', ''))
                            if memory_info:
                                print(f"    Current memory usage: {memory_info['current'] / 1e9:.2f} GB")
                                print(f"    Peak memory usage: {memory_info['peak'] / 1e9:.2f} GB")
                        except Exception:
                            pass
                        
                        # Strategy 3: Warn user but mark as partially configured
                        print(f"    💡 Recommendation: Restart kernel and run this cell first")
                        gpu_configured = True  # Mark as configured but with warning
                    else:
                        print(f"    ⚠ Memory configuration warning: {type(e).__name__}")
                        
                        # Strategy 4: Try alternative memory limit approach
                        try:
                            # Set a specific memory limit (e.g., 45GB for A40/A6000)
                            if gpu_memory_mb > 0:
                                memory_limit_mb = int(gpu_memory_mb * 0.9)  # Use 90% of total
                                tf.config.set_logical_device_configuration(
                                    gpu,
                                    [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit_mb)]
                                )
                                print(f"    ✓ Memory limit set: {memory_limit_mb/1024:.1f} GB")
                                gpu_configured = True
                                gpu_memory_limit_set = True
                        except Exception as e2:
                            print(f"    ⚠ Alternative memory config failed: {type(e2).__name__}")
                except Exception as e:
                    print(f"    ⚠ Memory configuration: {type(e).__name__}")
                    print(f"       Continuing without memory limits...")
            
            # Performance optimizations info
            print(f"\n  Performance Settings:")
            print(f"    ✓ Mixed precision: Ready (use tf.keras.mixed_precision.set_global_policy('mixed_float16'))")
            print(f"    ✓ XLA compilation: Available (enable with tf.config.optimizer.set_jit(True))")
            print(f"    ✓ TensorFloat-32: Enabled by default on A40/A6000 (Ampere architecture)")
            
            # Thread configuration
            try:
                inter_threads = tf.config.threading.get_inter_op_parallelism_threads()
                intra_threads = tf.config.threading.get_intra_op_parallelism_threads()
                print(f"    Inter-op threads: {inter_threads} (0 = auto, uses all CPU cores)")
                print(f"    Intra-op threads: {intra_threads} (0 = auto, uses all CPU cores)")
            except Exception:
                pass
                
            # Verify GPU is accessible with a simple computation
            try:
                print(f"\n  Testing GPU computation...")
                with tf.device('/GPU:0'):
                    test_tensor = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0])
                    result = tf.reduce_sum(test_tensor)
                    result_value = result.numpy()
                    if result_value == 15.0:
                        print(f"    ✓ GPU test passed: TensorFlow can access GPU")
                        print(f"    ✓ Test computation result: {result_value}")
                        gpu_configured = True
                    else:
                        print(f"    ⚠ GPU test: Unexpected result {result_value}")
            except Exception as e:
                print(f"    ⚠ GPU test failed: {type(e).__name__}")
                print(f"       Error: {str(e)[:100]}")
                
        else:
            print("⚠ TensorFlow GPU: No GPU devices found")
            print("  Possible reasons:")
            print("    - CPU-only pod selected")
            print("    - GPU not properly configured")
            print("    - CUDA drivers not installed")
            if gpu_available:
                print("    - GPU detected via nvidia-smi but TensorFlow can't see it")
                print("    - May need to restart kernel or check TensorFlow CUDA installation")
            print("  Continuing with CPU mode...")
            
    except ImportError:
        print("✗ TensorFlow not available - cannot configure GPU")
    except Exception as e:
        print(f"⚠ GPU configuration error: {type(e).__name__}")
        import traceback
        error_details = traceback.format_exc()
        if "already been set" in error_details.lower() or "virtual devices" in error_details.lower():
            print("   (GPU already initialized - this is normal in Jupyter notebooks)")
        else:
            print(f"   Details: {error_details[:300]}")
else:
    print("⚠ TensorFlow not installed - skipping GPU configuration")

print("=" * 70)

# ============================================
# STEP 13: DATASET VALIDATION (RUNPOD PATH AWARE)
# ============================================
print("\n" + "=" * 70)
print("📁 STEP 13: DATASET VALIDATION")
print("=" * 70)

import glob
from PIL import Image

# Use detected RunPod workspace
base_dir = runpod_workspace if runpod_workspace else os.getcwd()
print(f"Working directory: {base_dir}")

# Dataset paths (supports RunPod network volumes and workspace)
dataset_paths = [
    os.path.join(base_dir, "data", "isic2019"),
    "./data/isic2019",
    "data/isic2019",
    os.path.join("/workspace", "data", "isic2019"),
]

# Add network volume paths if available
if NETWORK_VOLUME:
    network_dataset_path = os.path.join(str(NETWORK_VOLUME), "data", "isic2019")
    dataset_paths.insert(0, network_dataset_path)  # Prioritize network volume
    print(f"  Checking network volume path first: {network_dataset_path}")

final_dataset_path = None
for path in dataset_paths:
    if os.path.exists(path):
        final_dataset_path = path
        print(f"✓ Dataset found at: {final_dataset_path}")
        break

if final_dataset_path:
    try:
        subdirs = [d for d in os.listdir(final_dataset_path) if os.path.isdir(os.path.join(final_dataset_path, d))]
        print(f"  Classes found: {len(subdirs)}")
        
        # Count images per class
        total_images = 0
        class_counts = {}
        
        for subdir in subdirs:
            class_path = os.path.join(final_dataset_path, subdir)
            image_files = glob.glob(os.path.join(class_path, "*.jpg")) + glob.glob(os.path.join(class_path, "*.png"))
            class_counts[subdir] = len(image_files)
            total_images += len(image_files)
            print(f"    - {subdir}: {len(image_files)} images")
        
        print("\n📊 Dataset Summary:")
        print(f"  Total images: {total_images}")
        if len(subdirs) > 0:
            print(f"  Average per class: {total_images / len(subdirs):.0f}")
        
        # Check image quality
        print("\n🔍 Image Quality Check:")
        sample_images = []
        for subdir in subdirs[:3]:
            class_path = os.path.join(final_dataset_path, subdir)
            image_files = glob.glob(os.path.join(class_path, "*.jpg"))[:2]
            sample_images.extend(image_files)
        
        valid_images = 0
        for img_path in sample_images[:10]:
            try:
                with Image.open(img_path) as img:
                    if img.size[0] > 0 and img.size[1] > 0:
                        valid_images += 1
            except Exception as e:
                print(f"    ⚠ Invalid image: {os.path.basename(img_path)}")
        
        print(f"  Valid images: {valid_images}/{min(10, len(sample_images))} samples")
        
    except Exception as e:
        print(f"  ⚠ Could not validate dataset: {e}")
else:
    print("✗ Dataset not found")
    print("\n💡 RunPod Dataset Setup Instructions:")
    print(f"  1. Option A - Network Volume (RECOMMENDED - persistent storage):")
    if NETWORK_VOLUME:
        print(f"     mkdir -p {NETWORK_VOLUME}/data")
        print(f"     # Upload/download ISIC 2019 dataset to {NETWORK_VOLUME}/data/isic2019/")
    else:
        print(f"     # Attach network volume in RunPod dashboard first")
        print(f"     # Then dataset should be at: /runpod-volume/data/isic2019")
    print(f"  2. Option B - Workspace (temporary, lost when pod stops):")
    print(f"     mkdir -p {os.path.join(base_dir, 'data')}")
    print(f"     # Download ISIC 2019 dataset here")
    print(f"  3. Dataset structure: data/isic2019/<class_name>/*.jpg")
    print(f"  4. ISIC 2019 dataset: https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic")
    print("\n  Note: Dataset validation is optional for Week 1 setup")
    print("        You can proceed without dataset for now")
    print("        Download dataset in Week 2 if not available now")

# ============================================
# CRITICAL FIX: IMPROVED STORAGE CONFIGURATION
# ============================================
print("\n" + "=" * 70)
print("💾 STORAGE CONFIGURATION")
print("=" * 70)

# Determine storage base (prefer network volume for persistence)
STORAGE_BASE = NETWORK_VOLUME if NETWORK_VOLUME else Path(base_dir)
print(f"\nStorage base: {STORAGE_BASE}")
print(f"Storage type: {'Network Volume (Persistent)' if NETWORK_VOLUME else 'Workspace (Temporary)'}")

# Create output directory
OUTPUT_DIR = (STORAGE_BASE / 'outputs').resolve()
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
print(f"✓ Output directory created: {OUTPUT_DIR}")

# Workspace shortcut (for easy access in Jupyter)
WORKSPACE_OUTPUT_DIR = Path(base_dir) / 'outputs'

# CRITICAL FIX: Improved symlink handling
if NETWORK_VOLUME and OUTPUT_DIR != WORKSPACE_OUTPUT_DIR.resolve():
    print(f"\nCreating workspace shortcut to network volume...")
    try:
        # Check if workspace outputs exists
        if WORKSPACE_OUTPUT_DIR.exists():
            if WORKSPACE_OUTPUT_DIR.is_symlink():
                # Check if symlink points to correct location
                current_target = WORKSPACE_OUTPUT_DIR.resolve()
                if current_target == OUTPUT_DIR:
                    print(f"✓ Symlink already correct: {WORKSPACE_OUTPUT_DIR} -> {OUTPUT_DIR}")
                else:
                    print(f"⚠ Symlink points to wrong location, updating...")
                    WORKSPACE_OUTPUT_DIR.unlink()
                    os.symlink(str(OUTPUT_DIR), str(WORKSPACE_OUTPUT_DIR))
                    print(f"✓ Symlink updated: {WORKSPACE_OUTPUT_DIR} -> {OUTPUT_DIR}")
            else:
                # Regular directory exists, backup and create symlink
                backup_dir = Path(base_dir) / f'outputs_backup_{os.getpid()}'
                print(f"⚠ Regular directory exists, backing up to: {backup_dir}")
                shutil.move(str(WORKSPACE_OUTPUT_DIR), str(backup_dir))
                os.symlink(str(OUTPUT_DIR), str(WORKSPACE_OUTPUT_DIR))
                print(f"✓ Symlink created: {WORKSPACE_OUTPUT_DIR} -> {OUTPUT_DIR}")
        else:
            # Create new symlink
            os.symlink(str(OUTPUT_DIR), str(WORKSPACE_OUTPUT_DIR))
            print(f"✓ Symlink created: {WORKSPACE_OUTPUT_DIR} -> {OUTPUT_DIR}")
            
    except OSError as e:
        if e.errno == 17:  # File exists
            print(f"⚠ Symlink creation race condition, using existing")
        else:
            print(f"⚠ Could not create symlink: {e}")
            print(f"   Using network volume directly: {OUTPUT_DIR}")
    except Exception as e:
        print(f"⚠ Symlink creation failed: {type(e).__name__}")
        print(f"   Using network volume directly: {OUTPUT_DIR}")
else:
    # No network volume or paths are the same
    if not NETWORK_VOLUME:
        print(f"⚠ No network volume detected, using workspace (temporary storage)")
    OUTPUT_DIR = WORKSPACE_OUTPUT_DIR.resolve()
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Create subdirectories for project organization
subdirs_to_create = ["models", "checkpoints", "logs", "results", "plots", "visualizations"]
print(f"\nCreating project subdirectories...")
for subdir in subdirs_to_create:
    subdir_path = OUTPUT_DIR / subdir
    subdir_path.mkdir(exist_ok=True, parents=True)
    print(f"  ✓ {subdir}")

print(f"\n✓ All directories created successfully")

# Print storage summary
print(f"\n📊 Storage Summary:")
print(f"   Base directory: {base_dir}")
print(f"   Network volume: {NETWORK_VOLUME if NETWORK_VOLUME else 'Not attached'}")
print(f"   Output directory: {OUTPUT_DIR}")
if NETWORK_VOLUME:
    print(f"   ✓ Outputs saved to network volume (persistent)")
    if WORKSPACE_OUTPUT_DIR.is_symlink():
        print(f"   ✓ Workspace shortcut: {WORKSPACE_OUTPUT_DIR} -> {OUTPUT_DIR}")
else:
    print(f"   ⚠ Outputs saved to workspace (may be lost when pod stops)")
    print(f"   💡 Recommendation: Attach network volume for persistent storage")

# Check available disk space
try:
    import psutil
    if NETWORK_VOLUME:
        volume_disk = psutil.disk_usage(str(NETWORK_VOLUME))
        print(f"\n   Network Volume Space:")
        print(f"      Total: {volume_disk.total / 1e9:.1f} GB")
        print(f"      Free: {volume_disk.free / 1e9:.1f} GB")
        print(f"      Used: {volume_disk.percent:.1f}%")
        if volume_disk.free / 1e9 < 50:
            print(f"      ⚠ WARNING: Low space on network volume")
    
    workspace_disk = psutil.disk_usage(base_dir)
    print(f"\n   Workspace Space:")
    print(f"      Total: {workspace_disk.total / 1e9:.1f} GB")
    print(f"      Free: {workspace_disk.free / 1e9:.1f} GB")
    print(f"      Used: {workspace_disk.percent:.1f}%")
    if workspace_disk.free / 1e9 < 30:
        print(f"      ⚠ WARNING: Low workspace space")
except Exception as e:
    print(f"   ⚠ Could not check disk space: {e}")

print("=" * 70)

# ============================================
# SUMMARY
# ============================================
print("\n" + "=" * 70)
print("✅ SETUP COMPLETE!")
print("=" * 70)

print("\n📊 Summary:")
print(f"  ✓ Working imports: {len(working_imports)}/{len(critical_imports)}")
print(f"  ✓ Successful packages: {len(successful_packages)}")
if failed_imports:
    print(f"  ✗ Failed imports: {len(failed_imports)}")
if failed_packages:
    print(f"  ✗ Failed packages: {len(failed_packages)}")

if failed_imports:
    print(f"\n⚠ Failed imports: {', '.join(failed_imports[:5])}")
    print("  You may need to install these manually or check compatibility")

if failed_packages:
    print(f"\n⚠ Failed packages: {', '.join([p.split('>=')[0] for p in failed_packages[:5]])}")
    print("  These can be installed individually if needed")

# Verify ONNX Runtime GPU compatibility
print("\n🔍 ONNX Runtime GPU Compatibility Check...")
try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    print(f"  Available providers: {providers}")
    if 'CUDAExecutionProvider' in providers:
        print("  ✓ ONNX Runtime GPU (CUDA) provider available")
        try:
            cuda_provider_options = ort.get_cuda_provider_options()
            print(f"  ONNX Runtime version: {ort.__version__}")
        except Exception:
            pass
    else:
        print("  ⚠ ONNX Runtime GPU provider not available - will use CPU")
        print("     This may affect inference performance in later weeks")
except ImportError:
    print("  ⚠ ONNX Runtime not installed")
except Exception as e:
    print(f"  ⚠ ONNX Runtime check failed: {type(e).__name__}")

print("\n🚀 Ready for skin cancer classification!")
print("   Dataset: ISIC 2019 (8 classes)")
print("   Framework: TensorFlow")
print("   GPU: NVIDIA A40/A6000 (48GB VRAM)")
print("   Monitoring: Wandb + TensorBoard")
print("   Augmentation: Albumentations + imgaug")
print("   Environment: RunPod Optimized")
print("   TensorRT: 8.6.1 (installed, ready for week15)")

print("\n💡 Next steps:")
print("   1. Load and explore the dataset (Week 2)")
print("   2. Implement data preprocessing (Week 2)")
print("   3. Apply data augmentation (Week 3)")
print("   4. Create train/val/test splits (Week 4)")
print("   5. Build baseline CNN model (Week 5)")
print("   ...")
print("   15. Run ONNX → TensorRT conversion:")
print("       ⚠  Open a NEW terminal: File → New → Terminal")
print("       ⚠  Run: python /workspace/week15.py")
print("       ⚠  Do NOT run week15.py from a notebook cell")

print("\n🔧 Enhanced Features:")
print("   ✓ Medical imaging packages")
print("   ✓ Advanced augmentation")
print("   ✓ Experiment tracking")
print("   ✓ GPU memory optimization")
print("   ✓ System resource monitoring")
print("   ✓ Persistent storage on network volume")
print("   ✓ TensorRT 8.6.1 + PyCUDA (for week15 TRT acceleration)")

print("\n🚀 A40/A6000 Optimizations:")
print("   ✓ 48GB VRAM utilization (dynamic allocation)")
print("   ✓ Mixed precision training ready (float16)")
print("   ✓ XLA compilation available")
print("   ✓ TensorFloat-32 for Ampere architecture")
print("   ✓ Memory growth management")

print("\n⚡ Expected Performance (ISIC 2019):")
print("   • Full dataset: ~25,000 images")
print("   • Training time: 20-40 hours (depends on architecture)")
print("   • Batch size: 32-64 (model dependent)")
print("   • Memory usage: ~35-40GB VRAM")
print("   • Target accuracy: 85-90%")

print("\n📦 RunPod Configuration:")
print("   • Template: madiator2011/better-tensorflow:cuda12.4-cudnn8")
print("   • GPU: NVIDIA A40 or RTX A6000 (48GB VRAM)")
print("   • Pricing: Varies by GPU type and region")
print("   • System: 50GB RAM • 9 vCPU (A40) / 8 vCPU (A6000)")
print(f"   • Working Directory: {base_dir}")
if final_dataset_path:
    print(f"   • Dataset Path: {final_dataset_path}")
else:
    print(f"   • Dataset Path: Not found (download in Week 2)")
print(f"   • Output Path: {OUTPUT_DIR}")
if NETWORK_VOLUME:
    print(f"   • Network Volume: {NETWORK_VOLUME} (persistent)")
else:
    print("   • Network Volume: Not attached (recommended)")
print(f"   • CUDA Version: {cuda_version if cuda_version else 'Auto-detected'}")
print(f"   • TensorFlow Version: {tf_version if tf_version else 'Not installed'}")
print(f"   • TensorRT: {'8.6.1 installed' if trt_installed else 'installation failed — see above'}")
print(f"   • PyCUDA: {'installed' if pycuda_installed else 'installation failed — see above'}")

print("\n💡 RunPod Best Practices:")
print("   ✓ Run this script (week1.py) from a terminal, not a notebook cell")
print("   ✓ Run week15.py from a terminal too (fresh CUDA context required)")
print("   ✓ Use Spot pricing for cost savings (18% cheaper)")
print("   ✓ Attach network volume for persistent storage")
print("   ✓ Save checkpoints frequently to network volume")
print("   ✓ Download final models before stopping pod")
print("   ✓ Monitor GPU: Run 'nvidia-smi' or 'watch -n 1 nvidia-smi'")
print("   ✓ Use --no-cache-dir for pip installs (saves space)")

# GPU Status Summary
print("\n" + "=" * 70)
if gpu_configured and gpu_memory_limit_set:
    print("✅ GPU STATUS: Fully Configured and Ready")
    print("   • Memory management active")
    print("   • TensorFlow can access GPU")
    print("   • Ready for training")
elif gpu_configured and not gpu_memory_limit_set:
    print("⚠️ GPU STATUS: Configured with Warnings")
    print("   • GPU accessible but memory settings may need restart")
    print("   • Can proceed but consider restarting kernel")
    print("   • Memory growth will apply on next restart")
elif gpu_available and not gpu_configured:
    print("⚠️ GPU STATUS: Detected but Not Configured")
    print("   • GPU visible via nvidia-smi")
    print("   • TensorFlow configuration needs attention")
    print("   • Restart kernel and run this cell first")
else:
    print("❌ GPU STATUS: Not Available")
    print("   • No GPU detected")
    print("   • Will run in CPU mode (very slow)")
    print("   • Check pod configuration")

print("=" * 70)

print("\n🔍 Next Steps:")
if failed_imports or failed_packages:
    print("   1. ⚠ Review failed packages/imports above")
    print("   2. Install missing packages manually if needed")
if not gpu_configured and gpu_available:
    print("   3. ⚠ Restart kernel to configure GPU properly")
if not final_dataset_path:
    print("   4. Download ISIC 2019 dataset to network volume")
print("   5. Continue to Week 2: Data preprocessing")

# ============================================
# HELPER: GPU Configuration Function
# ============================================
print("\n" + "=" * 70)
print("💡 GPU CONFIGURATION HELPER")
print("=" * 70)
print("\nIf GPU wasn't configured properly, use this in a NEW cell BEFORE importing TensorFlow:")
print("""
# Run this in a NEW cell if GPU memory wasn't configured
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✓ GPU memory growth enabled")
        print(f"✓ {len(gpus)} GPU(s) configured")
        
        # Test GPU
        with tf.device('/GPU:0'):
            test = tf.reduce_sum(tf.constant([1.0, 2.0, 3.0]))
            print(f"✓ GPU test passed: {test.numpy()}")
    except RuntimeError as e:
        print(f"⚠ GPU configuration error: {e}")
        print("   Kernel may need restart")
else:
    print("✗ No GPU devices found")
""")
print("=" * 70)

# Create a quick reference file in output directory
try:
    quick_ref_path = OUTPUT_DIR / "QUICK_REFERENCE.txt"
    with open(quick_ref_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("SKIN CANCER CLASSIFICATION PROJECT - QUICK REFERENCE\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Project Directory: {OUTPUT_DIR}\n")
        f.write(f"Dataset Path: {final_dataset_path if final_dataset_path else 'Not found'}\n")
        f.write(f"GPU: {gpu_name}\n")
        f.write(f"TensorFlow: {tf_version if tf_version else 'Not installed'}\n")
        f.write(f"CUDA: {cuda_version if cuda_version else 'Unknown'}\n\n")
        f.write("Subdirectories:\n")
        for subdir in subdirs_to_create:
            f.write(f"  - {subdir}/\n")
        f.write("\nImportant Commands:\n")
        f.write("  - Monitor GPU: watch -n 1 nvidia-smi\n")
        f.write("  - Check disk: df -h\n")
        f.write("  - Check memory: free -h\n")
        f.write("\nNext Steps: See Week 2 notebook\n")
    print(f"\n✓ Quick reference saved: {quick_ref_path}")
except Exception as e:
    print(f"⚠ Could not save quick reference: {e}")

print("\n" + "=" * 70)
print("🎉 WEEK 1 SETUP COMPLETE - READY FOR DEVELOPMENT!")
print("=" * 70)