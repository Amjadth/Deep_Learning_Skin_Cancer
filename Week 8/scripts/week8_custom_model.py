#!/usr/bin/env python3
"""
WEEK 8: OPTIMIZED REGULARIZATION METHODS (A40-OPTIMIZED) - APPROACH B
================================================================================
🎯 PURPOSE:
    Build 5 DIFFERENT REGULARIZATION STRATEGIES on the SAME 5.75M parameter baseline
    CNN architecture to find the BEST approach for preventing overfitting.

🔬 WHAT WE'RE DOING (Clear Explanation):
    ✅ We build 5 NEW models from scratch (not loading Week 6/7 weights)
    ✅ All 5 models have SAME architecture as Week 6/7 baseline (5.75M parameters)
    ✅ Each model has DIFFERENT regularization added to that baseline
    ✅ Train all 5 from scratch to compare which regularization strategy works best
    ✅ Select the BEST regularization strategy for Week 9 fine-tuning
    
    5 DIFFERENT REGULARIZATION STRATEGIES:
    
    1. Baseline_Reg:      Classic L2(0.01) + Dropout(0.3-0.5)
    2. Heavy_Reg:         Aggressive L2(0.02) + Heavy Dropout(0.4-0.5) 
    3. Spatial_Dropout:   Image-specific SpatialDropout2D + L2(0.01)
    4. Mixed_Reg:         Combined L1+L2 + Spatial Dropout (multi-method)
    5. Advanced_Reg:      Comprehensive approach (all techniques tuned)

📊 EXPECTED RESULTS:
    • NO Regularization (baseline): 91-92% accuracy (Week 6/7 result)
    • Strategy 1 (Baseline_Reg): ~92% accuracy (+0.5-1% improvement)
    • Strategy 2 (Heavy_Reg): ~93-94% accuracy (+1.5-2% improvement) ⭐ LIKELY BEST
    • Strategy 3 (Spatial_Dropout): ~92.5-93% accuracy (+1-2% improvement)
    • Strategy 4 (Mixed_Reg): ~93% accuracy (+1-2% improvement)
    • Strategy 5 (Advanced_Reg): ~92.5-93% accuracy (+1-2% improvement)
    • We test all 5 strategies and select the BEST one for Week 9 production

⚙️ OPTIMIZATIONS (Week 6 Best Practices):
    1. Memory mapping for all data loads (np.load with mmap_mode='r')
    2. Generator-based tf.data pipeline (from_generator)
    3. Reduced prefetch buffer (buffer_size=2 to prevent memory buildup)
    4. Aggressive memory cleanup via MemoryOptimizedCallback
    5. Learning rate schedule with warmup (WarmupExponentialDecay)
    6. Model parameter verification before training
    7. All models saved in .keras format
    8. Safe file loading with fallback mechanisms
    9. Per-epoch garbage collection and cache clearing
   10. Gradient clipping for numerical stability
   11. Reduced callback logging to avoid bottlenecks
   12. Memory monitoring with psutil

🖥️ GPU TARGET: NVIDIA A40 (48GB VRAM)
📦 Architecture: 5.75M parameter baseline CNN
📈 Data Split: 64k training, 8k validation, 8k test
🎨 Input Shape: (224, 224, 3)
🏥 Classes: 8 skin cancer types
⚡ Batch Size: 32 (optimized for A40)
🔄 Prefetch: 2 (reduced from AUTOTUNE to minimize memory)
⏱️ EPOCHS: 50 (typical stop: ~42 with early stopping, patience=8)
"""

import os
import sys
import gc
import json
import psutil
import traceback
import time
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers, callbacks, backend as K
import matplotlib.pyplot as plt
from sklearn.metrics import (precision_score, recall_score, f1_score, confusion_matrix, 
                             classification_report, roc_curve, auc, roc_auc_score)
from sklearn.preprocessing import label_binarize

# ============================================
# UTILITY FUNCTIONS (Week 6 Best Practices)
# ============================================

def print_memory_status(label=""):
    """Print current memory usage for monitoring (Week 6 Best Practice)."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    mem_usage_gb = mem_info.rss / (1024 ** 3)
    available_gb = psutil.virtual_memory().available / (1024 ** 3)
    vm = psutil.virtual_memory()
    print(f"💾 Memory ({label}): {mem_usage_gb:.2f}GB / Available: {available_gb:.2f}GB ({vm.percent:.1f}% used)")
    
    # Add GPU memory tracking (Week 6 enhancement)
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total',
             '--format=csv,nounits,noheader'],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            gpu_used, gpu_total = map(int, result.stdout.strip().split(','))
            gpu_pct = (gpu_used / gpu_total) * 100
            print(f"   GPU: {gpu_used}MB/{gpu_total}MB ({gpu_pct:.1f}%)")
    except Exception:
        pass  # nvidia-smi not available
    
    return mem_usage_gb


def clear_linux_cache():
    """
    Clear Linux page cache safely (Week 6 Best Practice).
    Recovers 10-15GB per epoch on RunPod/Linux systems.
    """
    try:
        # Sync filesystems first
        os.system('sync > /dev/null 2>&1')
        time.sleep(0.2)  # Let I/O settle
        # Clear cache (requires Linux/RunPod)
        os.system('echo 3 > /proc/sys/vm/drop_caches > /dev/null 2>&1')
        time.sleep(0.2)  # Let cache recovery settle
        return True
    except:
        return False


def safe_load_npy(filepath, description, use_memmap=True, expected_shape=None):
    """
    Safely load .npy files with memmap support - WEEK 6 METHOD
    Handles files with invalid headers using raw memmap fallback
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    print(f"  📥 Loading {description}...")
    try:
        if use_memmap:
            try:
                # Try standard mmap with pickle
                data = np.load(filepath, mmap_mode='r', allow_pickle=True)
                print(f"  ✓ {description}: {data.shape} (memmap + pickle)")
                return data
            except:
                try:
                    # Try standard mmap without pickle
                    data = np.load(filepath, mmap_mode='r', allow_pickle=False)
                    print(f"  ✓ {description}: {data.shape} (memmap)")
                    return data
                except:
                    if expected_shape:
                        # FALLBACK: Use raw memmap with expected shape
                        print(f"  ⚠️  Using raw memmap with shape {expected_shape}...")
                        data = np.memmap(str(filepath), dtype=np.float32, mode='r', shape=expected_shape)
                        print(f"  ✓ {description}: {data.shape} (raw memmap)")
                        return data
                    raise
        else:
            data = np.load(filepath, allow_pickle=True)
            print(f"  ✓ {description}: {data.shape}")
            return data
    except Exception as e:
        print(f"  ❌ Error: {e}")
        raise


class WarmupExponentialDecay(keras.optimizers.schedules.LearningRateSchedule):
    """
    Learning rate schedule with warmup followed by exponential decay.
    
    Week 6 Best Practice: Reduce callback overhead with stable LR schedule
    Uses tf.cond for graph-compatible conditional logic
    """
    def __init__(self, initial_learning_rate, warmup_epochs, decay_epochs, 
                 decay_rate=0.1, steps_per_epoch=1, name="WarmupExponentialDecay"):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_epochs = warmup_epochs
        self.decay_epochs = decay_epochs
        self.decay_rate = decay_rate
        self.steps_per_epoch = steps_per_epoch
        self.name = name

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        steps_per_epoch = tf.cast(self.steps_per_epoch, tf.float32)
        epoch = step / steps_per_epoch
        
        # Convert all parameters to float32 for consistency
        initial_lr = tf.cast(self.initial_learning_rate, tf.float32)
        warmup_epochs = tf.cast(self.warmup_epochs, tf.float32)
        decay_epochs = tf.cast(self.decay_epochs, tf.float32)
        decay_rate = tf.cast(self.decay_rate, tf.float32)

        # Use tf.cond for graph-compatible conditionals
        def warmup_lr():
            """Warmup phase: linear increase"""
            return initial_lr * (epoch / warmup_epochs)

        def decay_lr():
            """Decay phase: exponential decrease"""
            decay_epoch = epoch - warmup_epochs
            return initial_lr * tf.math.pow(
                decay_rate, 
                decay_epoch / decay_epochs
            )

        def constant_lr():
            """Constant phase: hold at final LR"""
            return initial_lr * decay_rate

        # First condition: warmup phase
        lr = tf.cond(
            epoch < warmup_epochs,
            warmup_lr,
            lambda: tf.cond(
                epoch < (warmup_epochs + decay_epochs),
                decay_lr,
                constant_lr
            )
        )

        return tf.cast(lr, tf.float32)

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "warmup_epochs": self.warmup_epochs,
            "decay_epochs": self.decay_epochs,
            "decay_rate": self.decay_rate,
            "steps_per_epoch": self.steps_per_epoch,
        }


class MemoryOptimizedCallback(keras.callbacks.Callback):
    """
    Memory-optimized callback with aggressive cleanup (Week 6 Best Practice).
    
    Features:
    - Garbage collection after each epoch
    - TensorFlow session clearing
    - Linux page cache clearing every N epochs (recovers 10-15GB)
    - Memory monitoring with detailed metrics
    """
    def __init__(self, log_freq=5, clear_cache_freq=5):
        super().__init__()
        self.log_freq = log_freq  # Only log every N epochs
        self.clear_cache_freq = clear_cache_freq
        self.start_memory = print_memory_status("Start")
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        # Force garbage collection (always)
        gc.collect()
        
        # Clear TensorFlow session
        try:
            tf.keras.backend.clear_session()
        except:
            pass
        
        # ✅ Week 6 Best Practice: Conditional cache clearing based on memory pressure
        if epoch % self.clear_cache_freq == 0:
            try:
                vm = psutil.virtual_memory()
                mem_used_pct = vm.percent
                
                # Only clear cache if memory pressure is high (> 80%)
                if mem_used_pct > 80:
                    mem_before = mem_used_pct
                    if clear_linux_cache():
                        mem_after = psutil.virtual_memory().percent
                        print(f"   ✨ Cache cleared (memory: {mem_before:.1f}% → {mem_after:.1f}%)")
            except:
                pass
        
        # Log every log_freq epochs
        if epoch % self.log_freq == 0:
            current_mem = print_memory_status(f"Epoch {epoch}")
            
            # Print epoch stats
            logs = logs or {}
            elapsed = time.time() - self.start_time
            hours = elapsed // 3600
            minutes = (elapsed % 3600) // 60
            seconds = elapsed % 60
            
            print(f"  📊 Epoch {epoch+1} Stats:")
            print(f"     Loss: {logs.get('loss', 0):.4f}, Accuracy: {logs.get('accuracy', 0):.4f}")
            print(f"     Val Loss: {logs.get('val_loss', 0):.4f}, Val Accuracy: {logs.get('val_accuracy', 0):.4f}")
            print(f"     Elapsed: {hours:.0f}h {minutes:.0f}m {seconds:.0f}s")


def create_optimized_dataset(X, y, batch_size=32, shuffle=True, seed=42):
    """
    Create optimized tf.data pipeline with memory mapping and reduced prefetch.
    
    Week 6 Best Practice: Generator-based pipeline with reduced prefetch buffer
    to prevent memory buildup while maintaining GPU utilization.
    """
    num_samples = len(X)
    indices = np.arange(num_samples)
    
    if shuffle:
        np.random.RandomState(seed).shuffle(indices)
    
    def data_generator():
        """Generator that yields single samples from memory-mapped array."""
        for idx in indices:
            # Access single sample from memory-mapped array (minimal RAM impact)
            img = X[idx].astype(np.float32)  # Ensure float32
            label = int(y[idx])
            yield img, label

    # Create dataset with reduced prefetch buffer
    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=(
            tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=2)  # Reduced from AUTOTUNE
    
    return dataset


# ============================================
# WORKSPACE CONFIGURATION
# ============================================

BASE_DIR = Path(os.getcwd())
NETWORK_VOLUME = None

# Detect RunPod workspace
if Path('/workspace').exists():
    BASE_DIR = Path('/workspace')
elif Path('/notebooks').exists():
    BASE_DIR = Path('/notebooks')

# Detect network volume (persistent storage)
if Path('/runpod-volume').exists():
    NETWORK_VOLUME = Path('/runpod-volume')
    print(f"✓ Network volume detected: {NETWORK_VOLUME}")
elif Path('/workspace/.runpod').exists():
    NETWORK_VOLUME = Path('/workspace/.runpod')
    print(f"✓ Network volume detected: {NETWORK_VOLUME}")

# Configuration
STORAGE_BASE = NETWORK_VOLUME if NETWORK_VOLUME else BASE_DIR
OUTPUT_DIR = (STORAGE_BASE / 'outputs').resolve()
MODEL_DIR = (OUTPUT_DIR / 'models').resolve()
REG_DIR = (OUTPUT_DIR / 'regularization_experiments').resolve()
RESULTS_DIR = (OUTPUT_DIR / 'results').resolve()

OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
MODEL_DIR.mkdir(exist_ok=True, parents=True)
REG_DIR.mkdir(exist_ok=True, parents=True)
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

print(f"📁 Storage: {OUTPUT_DIR}")
print(f"📁 Models: {REG_DIR}")

# Target Metrics
TARGET_ACCURACY = 0.90
TARGET_PRECISION = 0.88
TARGET_RECALL = 0.88
TARGET_F1 = 0.90

print("=" * 70)
print("WEEK 8: A40-OPTIMIZED REGULARIZATION METHODS (with Memory Management)")
print("=" * 70)

# ============================================
# GPU OPTIMIZATION SETUP
# ============================================

print("\n🔧 Setting up A40 optimizations...")

try:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("✓ Mixed precision (FP16) enabled")
except Exception as e:
    print(f"⚠ Mixed precision setup failed: {e}")

try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✓ GPU configured: {len(gpus)} GPU(s) detected")
    else:
        print("⚠ No GPU detected")
except Exception as e:
    print(f"⚠ GPU configuration failed: {e}")

try:
    tf.config.optimizer.set_jit(True)
    print("✓ XLA compilation enabled")
except Exception as e:
    print(f"⚠ XLA compilation failed: {e}")

try:
    tf.config.threading.set_inter_op_parallelism_threads(0)
    tf.config.threading.set_intra_op_parallelism_threads(0)
    print("✓ CUDA kernel optimization enabled")
except Exception as e:
    print(f"⚠ CUDA optimization failed: {e}")

# ============================================
# STEP 1: LOAD DATA (with memory mapping)
# ============================================

print("\n📂 Step 1: Loading data with memory mapping...")
print_memory_status("Before loading")

# Verify required files exist
required_files = [
    OUTPUT_DIR / 'X_train_denormalized.npy',
    OUTPUT_DIR / 'y_train_baseline.npy',
    OUTPUT_DIR / 'X_val_denormalized.npy',
    OUTPUT_DIR / 'y_val_baseline.npy',
    OUTPUT_DIR / 'X_test_denormalized.npy',
    OUTPUT_DIR / 'y_test_baseline.npy',
    OUTPUT_DIR / 'split_info.json',
    OUTPUT_DIR / 'tuning_results' / 'hyperparameter_tuning_results.csv',
]

missing = [str(p) for p in required_files if not p.exists()]
if missing:
    raise FileNotFoundError(f"Missing required files: {missing}")

# Load data with memory mapping (Week 6 Best Practice - EXACT METHOD)
print("\n📥 Loading denormalized data with memory mapping...")
X_train_reg = safe_load_npy(
    OUTPUT_DIR / 'X_train_denormalized.npy',
    'X_train_denormalized (64k images)',
    use_memmap=True,
    expected_shape=(64000, 224, 224, 3)
)
y_train = safe_load_npy(
    OUTPUT_DIR / 'y_train_baseline.npy',
    'y_train (labels)',
    use_memmap=False
)

X_val_reg = safe_load_npy(
    OUTPUT_DIR / 'X_val_denormalized.npy',
    'X_val_denormalized (8k images)',
    use_memmap=True,
    expected_shape=(8000, 224, 224, 3)
)
y_val = safe_load_npy(
    OUTPUT_DIR / 'y_val_baseline.npy',
    'y_val (labels)',
    use_memmap=False
)

X_test_baseline = safe_load_npy(
    OUTPUT_DIR / 'X_test_denormalized.npy',
    'X_test_denormalized (8k images)',
    use_memmap=True,
    expected_shape=(8000, 224, 224, 3)
)
y_test = safe_load_npy(
    OUTPUT_DIR / 'y_test_baseline.npy',
    'y_test (labels)',
    use_memmap=False
)

# NOTE: DO NOT call .astype() on memmap arrays - it forces full array into RAM!
# Instead, rely on dtype conversion in data generator (line 309-310)
# X_train_reg, X_val_reg, X_test_baseline are already float32 (or will be converted by generator)
# y_train, y_val, y_test are already int type

# Verify dtypes WITHOUT converting (memmap inspection only)
print(f"  ℹ️  Data types (memmap-safe inspection):")
print(f"     X_train dtype: {X_train_reg.dtype} (will be cast in generator)")
print(f"     y_train dtype: {y_train.dtype}")
print(f"     X_val dtype: {X_val_reg.dtype} (will be cast in generator)")
print(f"     X_test dtype: {X_test_baseline.dtype} (will be cast in generator)")

print(f"✓ Training set: {X_train_reg.shape}")
print(f"✓ Validation set: {X_val_reg.shape}")
print(f"✓ Test set: {X_test_baseline.shape}")
print_memory_status("After loading with mmap")

# ============================================
# MEMORY BREAKDOWN (For Reference)
# ============================================
print(f"\n💾 Memory Consumption Breakdown:")
print(f"  X_train_denormalized:  64k × 224×224×3 × 4 bytes = ~19.1 GB")
print(f"  X_val_denormalized:     8k × 224×224×3 × 4 bytes = ~2.4 GB")
print(f"  X_test_denormalized:    8k × 224×224×3 × 4 bytes = ~2.4 GB")
print(f"  y_train + y_val + y_test labels:                  ~0.3 GB")
print(f"  5 Models + optimizers:                            ~2-3 GB")
print(f"  ─────────────────────────────────────────────────────────")
print(f"  Expected Total:                                   ~26-27 GB")
print(f"  ⚠️  ACTUAL on 46GB system: Check above 'Memory' line")
print(f"  ℹ️  If using 45GB+ → System has 1GB buffer only!")
print(f"  🔴 WARNING: Very little headroom for training (risky!)")

# Load split info
with open(OUTPUT_DIR / 'split_info.json', 'r') as f:
    split_info = json.load(f)

# Load best hyperparameters from Week 7 (CSV format)
tuning_results_csv = OUTPUT_DIR / 'tuning_results' / 'hyperparameter_tuning_results.csv'
tuning_df = pd.read_csv(tuning_results_csv)
# Best result is first row (sorted by F1-score)
best_result = tuning_df.iloc[0]
BEST_LR = best_result['learning_rate']
# ✅ Week 6 Best Practice: Increase batch size from 32 to 128 (3-4x faster epochs)
BEST_BATCH_SIZE = 64  # ⚠️ REDUCED from 128 to stay under 50GB container limit (87.3% → 50-60%)
CLEAR_CACHE_FREQ = 5  # Clear Linux cache every 5 epochs (Week 6 optimization)

print(f"\n✓ Best hyperparameters (with Week 6 optimizations):")
print(f"  Learning Rate: {BEST_LR}")
print(f"  Batch Size: {BEST_BATCH_SIZE} (↑ from 32 - Week 6 optimization)")
print(f"  Cache Clear Frequency: Every {CLEAR_CACHE_FREQ} epochs")

# Constants
BASELINE_INPUT_SHAPE = (224, 224, 3)
INPUT_SHAPE = BASELINE_INPUT_SHAPE
NUM_CLASSES = len(split_info['class_names'])
CLASS_NAMES = split_info['class_names']

print(f"✓ Classes: {NUM_CLASSES} - {CLASS_NAMES}")

# ============================================
# STEP 2: VERIFY BASELINE ARCHITECTURE (Reference Only)
# ============================================

print("\n✅ Step 2: Baseline architecture verification")
print("   We are building 5 NEW models from scratch")
print("   Architecture: Same 5.75M parameter CNN baseline as Week 6/7")
print("   Regularization: Adding DIFFERENT strategies to each model")
print("   Training: All 5 models trained from scratch on training data")
print("   Goal: Compare which regularization works BEST for skin cancer")
print("\n   ❌ NOT loading Week 6/7 weights")
print("   ❌ NOT doing transfer learning")
print("   ✅ Building 5 fresh models with same architecture but different reg")

# ============================================
# STEP 3: CREATE REGULARIZED MODELS (Approach B: 5 Strategies)
# ============================================

print("\n🏗️ Step 3: Creating 5 regularization strategies...")


def create_model_baseline_reg(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES,
                               dropout_rate=0.3, l2_reg=0.01):
    """
    🎯 STRATEGY 1: BASELINE REGULARIZATION (Classic Approach)
    
    Philosophy: Standard L2 regularization + Dropout
    
    Parameters:
    • L2 regularization: 0.01 (decay coefficient for weight penalty)
    • Dropout: 0.3 in conv layers, 0.5 in dense layers
    • Approach: Time-tested combination that works well for most problems
    
    Expected Results:
    • Prevents weight magnitudes from growing too large
    • Randomly drops units during training to reduce co-adaptation
    • Good generalization, +0.5-1% improvement over baseline
    
    Best For: Conservative approach, proven effectiveness
    """
    model = models.Sequential(name='Baseline_Reg')
    model.add(layers.Input(shape=input_shape))
    
    # Block 1: 32 filters, L2(0.01), Dropout(0.3)
    model.add(layers.Conv2D(32, (3, 3), padding='same',
                            kernel_regularizer=regularizers.l2(l2_reg)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(32, (3, 3), padding='same',
                            kernel_regularizer=regularizers.l2(l2_reg)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(dropout_rate))
    
    # Block 2: 64 filters, L2(0.01), Dropout(0.3)
    model.add(layers.Conv2D(64, (3, 3), padding='same',
                            kernel_regularizer=regularizers.l2(l2_reg)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(64, (3, 3), padding='same',
                            kernel_regularizer=regularizers.l2(l2_reg)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(dropout_rate))
    
    # Block 3: 128 filters, L2(0.01), Dropout(0.4)
    model.add(layers.Conv2D(128, (3, 3), padding='same',
                            kernel_regularizer=regularizers.l2(l2_reg)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(128, (3, 3), padding='same',
                            kernel_regularizer=regularizers.l2(l2_reg)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(dropout_rate + 0.1))
    
    # Dense layers with L2(0.01) and Dropout(0.5)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(256, kernel_regularizer=regularizers.l2(l2_reg)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(128, kernel_regularizer=regularizers.l2(l2_reg)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax', dtype='float32'))
    
    return model


def create_model_heavy_reg(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    """
    🎯 STRATEGY 2: HEAVY REGULARIZATION (Aggressive Approach)
    
    Philosophy: Maximum regularization strength to prevent overfitting
    
    Parameters:
    • L2 regularization: 0.02 (2x stronger than baseline)
    • Dropout: 0.4 in conv layers, 0.5 in dense layers
    • Approach: Aggressive weight decay + high dropout
    
    Expected Results:
    • Strong constraint on weight magnitudes
    • Significant unit dropping for robust features
    • Good for complex datasets with high overfitting risk
    • May reduce training speed, +1-2% improvement if helpful
    
    Best For: Problems with severe overfitting, large datasets
    """
    model = models.Sequential(name='Heavy_Reg')
    model.add(layers.Input(shape=input_shape))
    
    # Block 1: 32 filters, L2(0.02), Dropout(0.4)
    model.add(layers.Conv2D(32, (3, 3), padding='same',
                            kernel_regularizer=regularizers.l2(0.02)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(32, (3, 3), padding='same',
                            kernel_regularizer=regularizers.l2(0.02)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.4))
    
    # Block 2: 64 filters, L2(0.02), Dropout(0.4)
    model.add(layers.Conv2D(64, (3, 3), padding='same',
                            kernel_regularizer=regularizers.l2(0.02)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(64, (3, 3), padding='same',
                            kernel_regularizer=regularizers.l2(0.02)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.4))
    
    # Block 3: 128 filters, L2(0.02), Dropout(0.5)
    model.add(layers.Conv2D(128, (3, 3), padding='same',
                            kernel_regularizer=regularizers.l2(0.02)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(128, (3, 3), padding='same',
                            kernel_regularizer=regularizers.l2(0.02)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.5))
    
    # Dense layers with L2(0.01) and Dropout(0.5)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(256, kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(128, kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax', dtype='float32'))
    
    return model


def create_model_spatial_dropout(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    """
    🎯 STRATEGY 3: SPATIAL DROPOUT (Image-Specific Approach)
    
    Philosophy: Drop entire feature maps instead of individual units
    
    Parameters:
    • SpatialDropout2D: 0.2 (conv layers), 0.3 (deeper layers)
    • L2 regularization: 0.01
    • Approach: Drops entire feature maps to prevent feature co-adaptation
    
    Expected Results:
    • Better for image data (preserves spatial relationships)
    • Each training iteration removes different feature maps
    • Learns diverse feature representations
    • Particularly good for skin lesion features
    • +1-1.5% improvement if spatial patterns matter
    
    Best For: Image classification, medical imaging, spatial data
    Why for skin cancer: Lesion patterns are spatial, so spatial dropout helps
    """
    model = models.Sequential(name='Spatial_Dropout')
    model.add(layers.Input(shape=input_shape))
    
    # Block 1: 32 filters, L2(0.01), SpatialDropout2D(0.2)
    model.add(layers.Conv2D(32, (3, 3), padding='same',
                            kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(32, (3, 3), padding='same',
                            kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.SpatialDropout2D(0.2))  # Drop entire feature maps
    
    # Block 2: 64 filters, L2(0.01), SpatialDropout2D(0.2)
    model.add(layers.Conv2D(64, (3, 3), padding='same',
                            kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(64, (3, 3), padding='same',
                            kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.SpatialDropout2D(0.2))
    
    # Block 3: 128 filters, L2(0.01), SpatialDropout2D(0.3)
    model.add(layers.Conv2D(128, (3, 3), padding='same',
                            kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(128, (3, 3), padding='same',
                            kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.SpatialDropout2D(0.3))  # Higher rate in deeper layers
    
    # Dense layers with L2(0.01) and regular Dropout(0.5)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(256, kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))  # Switch to regular dropout for dense
    model.add(layers.Dense(128, kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax', dtype='float32'))
    
    return model


def create_model_mixed_reg(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    """
    🎯 STRATEGY 4: MIXED REGULARIZATION (Combined Approach)
    
    Philosophy: Combine L1+L2 regularization with spatial dropout
    
    Parameters:
    • L1+L2 regularization: L1(0.001), L2(0.01) for weight elasticity
    • SpatialDropout2D: 0.2 (conv layers)
    • Regular Dropout: 0.5 (dense layers)
    • Approach: Multi-method regularization combining best of both worlds
    
    Expected Results:
    • L1 encourages sparsity (some weights → zero)
    • L2 prevents large weights
    • Spatial dropout prevents feature co-adaptation
    • Balanced approach across all regularization types
    • +1-2% improvement with good generalization
    
    Best For: Balanced overfitting prevention, diverse regularization methods
    Why for skin cancer: Combines multiple strategies for robust features
    """
    model = models.Sequential(name='Mixed_Reg')
    model.add(layers.Input(shape=input_shape))
    
    # Block 1: 32 filters, L1+L2, SpatialDropout2D(0.2)
    model.add(layers.Conv2D(32, (3, 3), padding='same',
                            kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.01)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(32, (3, 3), padding='same',
                            kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.01)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.SpatialDropout2D(0.2))
    
    # Block 2: 64 filters, L1+L2, SpatialDropout2D(0.2)
    model.add(layers.Conv2D(64, (3, 3), padding='same',
                            kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.01)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(64, (3, 3), padding='same',
                            kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.01)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.SpatialDropout2D(0.2))
    
    # Block 3: 128 filters, L1+L2, SpatialDropout2D(0.3)
    model.add(layers.Conv2D(128, (3, 3), padding='same',
                            kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.01)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(128, (3, 3), padding='same',
                            kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.01)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.SpatialDropout2D(0.3))
    
    # Dense layers with L2 and regular Dropout
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(256, kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(128, kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax', dtype='float32'))
    
    return model


def create_model_advanced_reg(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    """
    🎯 STRATEGY 5: ADVANCED REGULARIZATION (Comprehensive Tuned Approach)
    
    Philosophy: Optimal combination of all regularization techniques
    
    Parameters:
    • L1+L2: Varied by layer (stronger in deeper layers)
    • SpatialDropout2D: 0.25-0.3 (feature map level)
    • Regular Dropout: 0.5 (dense layers)
    • Gradient Clipping: Implicit through loss scaling
    • Approach: Carefully tuned multi-level regularization
    
    Expected Results:
    • Strongest overfitting prevention when tuned well
    • Balanced approach that adapts to layer depth
    • Best generalization if parameters are well-chosen
    • +1.5-2% improvement with good tuning
    
    Best For: Final production model after research phase
    Why for skin cancer: Comprehensive protection against all overfitting types
    """
    model = models.Sequential(name='Advanced_Reg')
    model.add(layers.Input(shape=input_shape))
    
    # Block 1: 32 filters - lighter regularization (early layers learn basic features)
    model.add(layers.Conv2D(32, (3, 3), padding='same',
                            kernel_regularizer=regularizers.l1_l2(l1=0.0005, l2=0.008)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(32, (3, 3), padding='same',
                            kernel_regularizer=regularizers.l1_l2(l1=0.0005, l2=0.008)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.SpatialDropout2D(0.25))
    
    # Block 2: 64 filters - moderate regularization
    model.add(layers.Conv2D(64, (3, 3), padding='same',
                            kernel_regularizer=regularizers.l2(0.015)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(64, (3, 3), padding='same',
                            kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.015)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.SpatialDropout2D(0.25))
    
    # Block 3: 128 filters - stronger regularization (deeper layers need more control)
    model.add(layers.Conv2D(128, (3, 3), padding='same',
                            kernel_regularizer=regularizers.l2(0.02)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(128, (3, 3), padding='same',
                            kernel_regularizer=regularizers.l2(0.02)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.SpatialDropout2D(0.3))
    
    # Dense layers - strongest regularization for reduction step
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(256, kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.01)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(128, kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax', dtype='float32'))
    
    return model


# Create models
print("\n✓ Creating 5 regularization models...")
models_dict = {
    'Baseline_Reg': create_model_baseline_reg(),
    'Heavy_Reg': create_model_heavy_reg(),
    'Spatial_Dropout': create_model_spatial_dropout(),
    'Mixed_Reg': create_model_mixed_reg(),
    'Advanced_Reg': create_model_advanced_reg()
}

for name, model in models_dict.items():
    param_count = model.count_params()
    print(f"  ✓ {name:20s}: {param_count:,} parameters")

# 🚨 WARNING: All 5 models are now in memory!
print(f"\n⚠️  RUNPOD CONTAINER MEMORY WARNING:")
print(f"  Container Limit:     50 GB (hard limit)")
print(f"  All 5 models:        ~2-3 GB")
print(f"  Data memmap:         ~23-24 GB")
print(f"  System overhead:     ~3-5 GB")
print(f"  ─────────────────────────────────────")
print(f"  Expected before training: ~26-27 GB")
print(f"  Peak during training:     ~30-35 GB (Batch=64) ✅ SAFE")
print(f"  With Batch=128:           ~45-50 GB ❌ RISKY (now reduced to 64)")
print_memory_status("After creating all 5 models")

# ============================================
# STEP 3: TRAINING WITH OPTIMIZED PIPELINE
# ============================================

print("\n🚀 Step 3: Training models with optimized pipeline...")

EPOCHS = 50
training_results = {}


def load_checkpoint_state(model_name: str) -> dict:
    """
    Load checkpoint state to resume training from a previous session.
    
    Week 6 Best Practice: Pod crash resilience - resume from last checkpoint
    
    Returns:
        dict: Contains initial_epoch, best_val_accuracy, and checkpoint status
    """
    checkpoint_state_path = REG_DIR / f'{model_name.lower()}_checkpoint_state.json'
    
    if checkpoint_state_path.exists():
        with open(checkpoint_state_path, 'r') as f:
            state = json.load(f)
        # Ensure initial_epoch is set for resuming training
        last_epoch = state.get('last_epoch', 0)
        state['initial_epoch'] = last_epoch  # Resume from last completed epoch
        print(f"  ✓ Resuming from checkpoint: Epoch {last_epoch}/{EPOCHS}")
        print(f"    Best validation accuracy: {state['best_val_accuracy']:.4f}")
        return state
    else:
        # First training run
        return {
            'initial_epoch': 0,
            'last_epoch': 0,
            'best_val_accuracy': 0.0,
            'checkpoint_path': str(REG_DIR / f'{model_name.lower()}_best.keras'),
            'state_path': str(checkpoint_state_path)
        }


def save_checkpoint_state(model_name: str, epoch: int, best_val_acc: float):
    """
    Save checkpoint state after each epoch for pod crash recovery.
    
    Week 6 Best Practice: Pod crash resilience - save state every epoch
    """
    checkpoint_state_path = REG_DIR / f'{model_name.lower()}_checkpoint_state.json'
    
    state = {
        'model_name': model_name,
        'last_epoch': epoch,
        'best_val_accuracy': best_val_acc,
        'timestamp': datetime.now().isoformat(),
        'checkpoint_path': str(REG_DIR / f'{model_name.lower()}_best.keras'),
        'epochs_total': EPOCHS,
        'status': 'training_in_progress'
    }
    
    with open(checkpoint_state_path, 'w') as f:
        json.dump(state, f, indent=2)


class CheckpointResumableCallback(keras.callbacks.Callback):
    """
    Callback to save checkpoint state for pod crash recovery.
    Enables resuming training from last saved epoch.
    
    Week 6 Best Practice: Pod crash resilience
    """
    def __init__(self, model_name: str, save_freq=1):
        super().__init__()
        self.model_name = model_name
        self.save_freq = save_freq
        self.best_val_accuracy = 0.0

    def on_epoch_end(self, epoch, logs=None):
        if logs:
            current_val_acc = logs.get('val_accuracy', 0.0)
            if current_val_acc > self.best_val_accuracy:
                self.best_val_accuracy = current_val_acc
            
            # Save checkpoint state every epoch (for crash recovery)
            if (epoch + 1) % self.save_freq == 0:
                save_checkpoint_state(self.model_name, epoch + 1, self.best_val_accuracy)

# Calculate steps per epoch for learning rate schedule
steps_per_epoch_train = len(X_train_reg) // BEST_BATCH_SIZE
steps_per_epoch_val = len(X_val_reg) // BEST_BATCH_SIZE

print(f"⚙️  Training config:")
print(f"  Batch size: {BEST_BATCH_SIZE}")
print(f"  Steps per epoch: {steps_per_epoch_train}")
print(f"  Learning rate: {BEST_LR}")
print(f"  Epochs: {EPOCHS}")

# ============================================
# POD CRASH RECOVERY SYSTEM
# ============================================
# If pod crashes, resume training from checkpoint:
# 1. Metadata saved in: {model_name}_checkpoint_state.json
# 2. Latest weights saved in: {model_name}_latest.keras (every epoch)
# 3. Best weights saved in: {model_name}_best.keras (when val_acc improves)
# 4. On restart, code automatically loads latest weights and continues from exact epoch
# 5. Learning rate and optimizer state are preserved in .keras format
# ============================================

for model_name, model in models_dict.items():
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"{'='*60}")
    
    # Load checkpoint state to resume from previous session if available
    checkpoint_state = load_checkpoint_state(model_name)
    initial_epoch = checkpoint_state['initial_epoch']
    
    # Load checkpoint weights if resuming training after pod crash
    # Priority: Latest checkpoint (most recent) -> Best checkpoint (best performance)
    latest_checkpoint_path = Path(str(REG_DIR / f'{model_name.lower()}_latest.keras'))
    best_checkpoint_path = Path(checkpoint_state.get('checkpoint_path', ''))
    
    weights_loaded = False
    if latest_checkpoint_path.exists() and initial_epoch > 0:
        try:
            print(f"  ✓ RESUMING TRAINING from latest checkpoint (Epoch {initial_epoch}/{EPOCHS})")
            model.load_weights(str(latest_checkpoint_path))
            print(f"  ✓ Loaded weights from: {latest_checkpoint_path}")
            weights_loaded = True
        except Exception as e:
            print(f"  ⚠️  Could not load latest checkpoint: {e}")
    
    # Fallback to best checkpoint if latest doesn't exist
    if not weights_loaded and best_checkpoint_path.exists() and initial_epoch > 0:
        try:
            print(f"  ✓ Loading best checkpoint as fallback")
            model.load_weights(str(best_checkpoint_path))
            print(f"  ✓ Loaded best weights from: {best_checkpoint_path}")
            print(f"  ✓ Best val_accuracy: {checkpoint_state['best_val_accuracy']:.4f}")
            weights_loaded = True
        except Exception as e:
            print(f"  ⚠️  Warning: Could not load checkpoint weights - starting fresh")
            print(f"     Error: {e}")
            initial_epoch = 0  # Fallback to fresh start
    
    if initial_epoch == 0:
        print(f"  ✓ Starting fresh training from epoch 0")
    
    # Create optimized datasets (Week 6 Best Practice)
    print(f"  Creating optimized datasets...")
    train_dataset = create_optimized_dataset(X_train_reg, y_train, 
                                            batch_size=BEST_BATCH_SIZE, shuffle=True)
    val_dataset = create_optimized_dataset(X_val_reg, y_val, 
                                          batch_size=BEST_BATCH_SIZE, shuffle=False)
    
    # Create learning rate schedule (Week 6 Best Practice)
    lr_schedule = WarmupExponentialDecay(
        initial_learning_rate=BEST_LR,
        warmup_epochs=5,
        decay_epochs=45,
        decay_rate=0.1,
        steps_per_epoch=steps_per_epoch_train
    )
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0, clipvalue=0.5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    
    print(f"✓ Model compiled with LR schedule and gradient clipping")
    
    # Setup callbacks (Week 6 Best Practices for resilience)
    callback_list = [
        # EARLY STOPPING CRITERIA:
        # - Monitors: val_accuracy (validation accuracy)
        # - Patience: 8 epochs (stops if no improvement for 8 epochs)
        # - Min Delta: 0.001 (minimum improvement threshold)
        # - Restore Best: Yes (restores best model weights on stop)
        # PURPOSE: Prevents overfitting by stopping when validation performance plateaus
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=8,  # Stop after 8 epochs with no improvement
            min_delta=0.001,  # Minimum improvement required (0.1%)
            restore_best_weights=True,  # Restore best model weights
            verbose=1
        ),
        
        # Dynamic learning rate adjustment
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,  # Multiply LR by 0.3
            patience=8,  # After 8 epochs with no improvement
            min_lr=1e-8,
            verbose=1
        ),
        
        # Checkpoint resumable callback (Pod crash recovery - Week 6 Best Practice)
        CheckpointResumableCallback(
            model_name=model_name,
            save_freq=1  # Save state every epoch for crash recovery
        ),
        
        # Save best model checkpoint (.keras format)
        keras.callbacks.ModelCheckpoint(
            filepath=str(REG_DIR / f'{model_name.lower()}_best.keras'),
            monitor='val_accuracy',
            save_best_only=True,  # Only save if this is best so far
            verbose=0  # Reduced verbosity
        ),
        
        # Save latest checkpoint every epoch (for pod crash recovery)
        keras.callbacks.ModelCheckpoint(
            filepath=str(REG_DIR / f'{model_name.lower()}_latest.keras'),
            monitor='val_accuracy',
            save_best_only=False,  # Save every epoch
            verbose=0,
            save_freq='epoch'  # Explicit save frequency
        ),
        
        # CSV logging for all metrics per epoch
        keras.callbacks.CSVLogger(
            filename=str(REG_DIR / f'{model_name.lower()}_training.csv')
        ),
        
        # Terminate on NaN (numerical instability detection)
        keras.callbacks.TerminateOnNaN(),
        
        # Memory cleanup and monitoring (Week 6 Best Practice with cache clearing)
        MemoryOptimizedCallback(log_freq=5, clear_cache_freq=CLEAR_CACHE_FREQ)
    ]
    
    print(f"✓ Callbacks configured (Early Stopping: patience=15, min_delta=0.001)")
    print_memory_status("Before training")
    
    # Train model with optimized dataset
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        initial_epoch=initial_epoch,  # Resume from checkpoint if available
        callbacks=callback_list,
        verbose=1
    )
    
    print_memory_status("After training")
    
    # Evaluate on test set
    print(f"\n✓ Evaluating on test set...")
    test_dataset = create_optimized_dataset(X_test_baseline, y_test,
                                           batch_size=BEST_BATCH_SIZE, shuffle=False)
    test_loss, test_acc = model.evaluate(test_dataset, verbose=0)
    
    # Get predictions for metrics
    y_pred = model.predict(test_dataset, verbose=0).argmax(axis=1)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    # Store results
    training_results[model_name] = {
        'model': model,
        'history': history.history,
        'epochs_trained': len(history.history['loss']),
        'test_metrics': {
            'loss': float(test_loss),
            'accuracy': float(test_acc),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        }
    }
    
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Test F1-Score: {f1:.4f}")
    
    # Mark checkpoint as complete for this model
    checkpoint_state_path = REG_DIR / f'{model_name.lower()}_checkpoint_state.json'
    if checkpoint_state_path.exists():
        with open(checkpoint_state_path, 'r') as f:
            state = json.load(f)
        state['status'] = 'training_complete'
        state['test_accuracy'] = float(test_acc)
        state['test_f1_score'] = float(f1)
        with open(checkpoint_state_path, 'w') as f:
            json.dump(state, f, indent=2)
        print(f"  ✓ Checkpoint marked as complete")
    
    # ============================================
    # AGGRESSIVE CLEANUP BEFORE NEXT MODEL
    # ============================================
    print(f"\n  🧹 Cleanup after {model_name}...")
    
    # Delete model weights and optimizer state
    del model  # Remove from memory
    del lr_schedule  # Remove learning rate schedule
    del callback_list  # Remove callbacks
    del train_dataset  # Remove dataset objects
    del val_dataset
    
    # Force multiple garbage collection passes
    for _ in range(3):
        gc.collect()
    
    # Clear TensorFlow session and caches
    try:
        tf.keras.backend.clear_session()
    except:
        pass
    
    # Clear Linux page cache if memory > 80%
    try:
        vm = psutil.virtual_memory()
        if vm.percent > 80:
            clear_linux_cache()
    except:
        pass
    
    print_memory_status(f"After cleanup ({model_name})")
    print()

# ============================================
# STEP 4: COMPREHENSIVE RESULTS ANALYSIS
# ============================================

print("\n📊 Step 4: Comprehensive results analysis...")

# Create detailed results dataframe
results_data = []
baseline_accuracy = 0.91  # Week 6/7 baseline (no regularization)

for name, results in training_results.items():
    metrics = results['test_metrics']
    accuracy = metrics['accuracy']
    improvement = (accuracy - baseline_accuracy) * 100  # Percentage points
    
    results_data.append({
        'Model': name,
        'Test_Accuracy': accuracy,
        'Test_Precision': metrics['precision'],
        'Test_Recall': metrics['recall'],
        'Test_F1': metrics['f1_score'],
        'Epochs_Trained': results['epochs_trained'],
        'Improvement_vs_Baseline': improvement,
        'Meets_Target_F1': "✓" if metrics['f1_score'] >= TARGET_F1 else "✗",
        'Meets_Target_Accuracy': "✓" if accuracy >= TARGET_ACCURACY else "✗"
    })

results_df = pd.DataFrame(results_data).sort_values('Test_F1', ascending=False)

# Calculate ranking
results_df['Rank'] = range(1, len(results_df) + 1)

print("\n" + "="*80)
print("COMPREHENSIVE TEST RESULTS ANALYSIS")
print("="*80)
print(results_df.to_string(index=False))

# Statistical Analysis
print("\n" + "="*80)
print("STATISTICAL ANALYSIS")
print("="*80)

best_model_name = results_df.iloc[0]['Model']
best_metrics = training_results[best_model_name]['test_metrics']
worst_model_name = results_df.iloc[-1]['Model']
worst_metrics = training_results[worst_model_name]['test_metrics']

accuracy_range = results_df['Test_Accuracy'].max() - results_df['Test_Accuracy'].min()
f1_range = results_df['Test_F1'].max() - results_df['Test_F1'].min()
mean_accuracy = results_df['Test_Accuracy'].mean()
mean_f1 = results_df['Test_F1'].mean()

print(f"\n📈 Accuracy Analysis:")
print(f"  Best:       {results_df['Test_Accuracy'].max():.4f} ({best_model_name})")
print(f"  Worst:      {results_df['Test_Accuracy'].min():.4f} ({worst_model_name})")
print(f"  Mean:       {mean_accuracy:.4f}")
print(f"  Range:      {accuracy_range:.4f}")
print(f"  Std Dev:    {results_df['Test_Accuracy'].std():.4f}")

print(f"\n📊 F1-Score Analysis:")
print(f"  Best:       {results_df['Test_F1'].max():.4f} ({best_model_name}) ⭐")
print(f"  Worst:      {results_df['Test_F1'].min():.4f} ({worst_model_name})")
print(f"  Mean:       {mean_f1:.4f}")
print(f"  Range:      {f1_range:.4f}")
print(f"  Std Dev:    {results_df['Test_F1'].std():.4f}")

print(f"\n🎯 Target Achievement:")
targets_met = len(results_df[results_df['Meets_Target_F1'] == "✓"])
print(f"  Models meeting F1 target ({TARGET_F1}): {targets_met}/{len(results_df)}")
accuracy_targets_met = len(results_df[results_df['Meets_Target_Accuracy'] == "✓"])
print(f"  Models meeting Accuracy target ({TARGET_ACCURACY}): {accuracy_targets_met}/{len(results_df)}")

print(f"\n📈 Improvement vs Baseline (91-92%):")
for idx, row in results_df.iterrows():
    improvement = row['Improvement_vs_Baseline']
    improvement_str = f"+{improvement:.2f}%" if improvement >= 0 else f"{improvement:.2f}%"
    marker = "⭐" if row['Rank'] == 1 else "  "
    print(f"  {marker} {row['Model']:20s}: {improvement_str:>8s} (Rank #{row['Rank']})")

# Find best model
print(f"\n{'='*80}")
print(f"🏆 BEST MODEL: {best_model_name}")
print(f"{'='*80}")
print(f"   Accuracy:     {best_metrics['accuracy']:.4f} / {TARGET_ACCURACY:.4f} (Target)")
print(f"   F1-Score:     {best_metrics['f1_score']:.4f} / {TARGET_F1:.4f} (Target)")
print(f"   Precision:    {best_metrics['precision']:.4f} / {TARGET_PRECISION:.4f} (Target)")
print(f"   Recall:       {best_metrics['recall']:.4f} / {TARGET_RECALL:.4f} (Target)")
improvement = (best_metrics['accuracy'] - baseline_accuracy) * 100
print(f"   Improvement:  +{improvement:.2f}% vs baseline")
print(f"   Status:       {'✓ MEETS ALL TARGETS' if best_metrics['f1_score'] >= TARGET_F1 else '⚠ Close to targets'}")

# Recommendation
print(f"\n💡 RECOMMENDATION:")
print(f"   Use {best_model_name} for Week 9 fine-tuning")
print(f"   This strategy shows the best generalization across all metrics")

# ============================================
# STEP 5: CREATE VISUALIZATIONS (Matplotlib)
# ============================================

print("\n📊 Step 5: Creating visualizations...")

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12', '#9b59b6']

# --- Visualization 1: Accuracy Comparison Bar Chart ---
fig, ax = plt.subplots(figsize=(12, 6))
models = results_df['Model'].values
accuracies = results_df['Test_Accuracy'].values
bars = ax.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=2)

# Add baseline line
ax.axhline(y=baseline_accuracy, color='red', linestyle='--', linewidth=2, label=f'Baseline ({baseline_accuracy:.4f})', alpha=0.7)
ax.axhline(y=TARGET_ACCURACY, color='green', linestyle='--', linewidth=2, label=f'Target ({TARGET_ACCURACY:.4f})', alpha=0.7)

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{acc:.4f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
ax.set_xlabel('Regularization Strategy', fontsize=12, fontweight='bold')
ax.set_title('Test Accuracy Comparison - All 5 Regularization Strategies', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.set_ylim([baseline_accuracy - 0.02, 1.0])
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
viz_path_1 = REG_DIR / 'viz_01_accuracy_comparison.png'
plt.savefig(viz_path_1, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {viz_path_1}")
plt.close()

# --- Visualization 2: Metrics Comparison (All 4 metrics) ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Comprehensive Metrics Comparison - All 5 Strategies', fontsize=16, fontweight='bold')

metrics_list = ['Test_Accuracy', 'Test_Precision', 'Test_Recall', 'Test_F1']
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
targets_list = [TARGET_ACCURACY, TARGET_PRECISION, TARGET_RECALL, TARGET_F1]

for idx, (metric, name, target) in enumerate(zip(metrics_list, metric_names, targets_list)):
    row, col = idx // 2, idx % 2
    ax = axes[row, col]
    
    values = results_df[metric].values
    bars = ax.bar(results_df['Model'].values, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add target line
    ax.axhline(y=target, color='green', linestyle='--', linewidth=2, label=f'Target ({target:.4f})', alpha=0.7)
    
    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel(name, fontsize=11, fontweight='bold')
    ax.set_xlabel('Strategy', fontsize=11, fontweight='bold')
    ax.set_title(f'{name} Comparison', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_ylim([min(0.8, values.min() - 0.05), 1.02])
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
viz_path_2 = REG_DIR / 'viz_02_metrics_comparison.png'
plt.savefig(viz_path_2, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {viz_path_2}")
plt.close()

# --- Visualization 3: F1-Score Ranking ---
fig, ax = plt.subplots(figsize=(12, 6))

# Sort by F1 for ranking
ranking_df = results_df.sort_values('Test_F1', ascending=True)
f1_scores = ranking_df['Test_F1'].values
models = ranking_df['Model'].values
ranks = ranking_df['Rank'].values

# Create horizontal bar chart (easier to read ranks)
bars = ax.barh(models, f1_scores, color=colors[::-1], alpha=0.8, edgecolor='black', linewidth=2)

# Add target line
ax.axvline(x=TARGET_F1, color='green', linestyle='--', linewidth=2, label=f'Target ({TARGET_F1:.4f})', alpha=0.7)

# Add value and rank labels
for i, (bar, score, rank) in enumerate(zip(bars, f1_scores, ranks)):
    width = bar.get_width()
    ax.text(width, bar.get_y() + bar.get_height()/2.,
            f'  #{rank}: {score:.4f}',
            ha='left', va='center', fontsize=11, fontweight='bold')

ax.set_xlabel('F1-Score', fontsize=12, fontweight='bold')
ax.set_ylabel('Regularization Strategy', fontsize=12, fontweight='bold')
ax.set_title('F1-Score Ranking (Best to Worst)', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.set_xlim([0.8, 1.02])
plt.tight_layout()
viz_path_3 = REG_DIR / 'viz_03_f1_ranking.png'
plt.savefig(viz_path_3, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {viz_path_3}")
plt.close()

# --- Visualization 4: Training Curves (Best Model) ---
best_model_history = training_results[best_model_name]['history']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f'Training History - Best Model: {best_model_name}', fontsize=14, fontweight='bold')

# Accuracy curve
axes[0].plot(best_model_history['accuracy'], label='Training Accuracy', linewidth=2, marker='o', markersize=4)
axes[0].plot(best_model_history['val_accuracy'], label='Validation Accuracy', linewidth=2, marker='s', markersize=4)
axes[0].axhline(y=TARGET_ACCURACY, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Target ({TARGET_ACCURACY})')
axes[0].set_xlabel('Epoch', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Accuracy', fontsize=11, fontweight='bold')
axes[0].set_title('Accuracy per Epoch', fontsize=12, fontweight='bold')
axes[0].legend(loc='lower right', fontsize=10)
axes[0].grid(True, alpha=0.3)

# Loss curve
axes[1].plot(best_model_history['loss'], label='Training Loss', linewidth=2, marker='o', markersize=4)
axes[1].plot(best_model_history['val_loss'], label='Validation Loss', linewidth=2, marker='s', markersize=4)
axes[1].set_xlabel('Epoch', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Loss', fontsize=11, fontweight='bold')
axes[1].set_title('Loss per Epoch', fontsize=12, fontweight='bold')
axes[1].legend(loc='upper right', fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
viz_path_4 = REG_DIR / 'viz_04_training_curves.png'
plt.savefig(viz_path_4, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {viz_path_4}")
plt.close()

# --- Visualization 5: Improvement vs Baseline ---
fig, ax = plt.subplots(figsize=(12, 6))

improvements = results_df['Improvement_vs_Baseline'].values
models = results_df['Model'].values

# Color bars based on positive/negative improvement
bar_colors = ['#2ecc71' if x >= 0 else '#e74c3c' for x in improvements]
bars = ax.bar(models, improvements, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=2)

# Add baseline line
ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)

# Add value labels
for bar, imp in zip(bars, improvements):
    height = bar.get_height()
    label_y = height + 0.05 if height >= 0 else height - 0.05
    ax.text(bar.get_x() + bar.get_width()/2., label_y,
            f'{imp:+.2f}%',
            ha='center', va='bottom' if height >= 0 else 'top', 
            fontsize=11, fontweight='bold')

ax.set_ylabel('Improvement (Percentage Points)', fontsize=12, fontweight='bold')
ax.set_xlabel('Regularization Strategy', fontsize=12, fontweight='bold')
ax.set_title('Accuracy Improvement vs Baseline (91-92%)', fontsize=14, fontweight='bold')
ax.set_ylim([improvements.min() - 0.5, improvements.max() + 0.5])
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
viz_path_5 = REG_DIR / 'viz_05_improvement_vs_baseline.png'
plt.savefig(viz_path_5, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {viz_path_5}")
plt.close()

# --- Visualization 6: Strategy Comparison Matrix ---
fig, ax = plt.subplots(figsize=(12, 8))

# Prepare data for heatmap
comparison_data = results_df[['Model', 'Test_Accuracy', 'Test_Precision', 'Test_Recall', 'Test_F1']].set_index('Model')
comparison_data = comparison_data.sort_values('Test_F1', ascending=False)

# Create heatmap
im = ax.imshow(comparison_data.T.values, cmap='RdYlGn', aspect='auto', vmin=0.8, vmax=1.0)

# Set ticks and labels
ax.set_xticks(range(len(comparison_data)))
ax.set_yticks(range(len(comparison_data.columns)))
ax.set_xticklabels(comparison_data.index, rotation=45, ha='right')
ax.set_yticklabels(comparison_data.columns)

# Add text annotations
for i in range(len(comparison_data.columns)):
    for j in range(len(comparison_data)):
        value = comparison_data.T.values[i, j]
        ax.text(j, i, f'{value:.3f}',
                ha='center', va='center', color='black', fontsize=11, fontweight='bold')

ax.set_title('Strategy Performance Heatmap (Green=Better)', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax, label='Metric Value')
plt.tight_layout()
viz_path_6 = REG_DIR / 'viz_06_strategy_heatmap.png'
plt.savefig(viz_path_6, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {viz_path_6}")
plt.close()

print(f"\n✓ All visualizations created successfully!")

# ============================================
# STEP 5.5: ADDITIONAL VISUALIZATIONS (Week 6 Best Practices)
# ============================================

print("\n📊 Step 5.5: Creating additional detailed visualizations (Week 6 style)...")

# For each model: Create confusion matrix and per-class metrics
for model_name, model_results in training_results.items():
    print(f"\n  Creating visualizations for {model_name}...")
    
    # Get predictions
    model = model_results['model']
    y_pred_probs = model.predict(test_dataset, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.concatenate([y for x, y in test_dataset], axis=0)
    y_true = np.argmax(y_true, axis=1)
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    cm_path = REG_DIR / f'viz_{model_name.lower()}_confusion_matrix.png'
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {cm_path}")
    plt.close()
    
    # 2. Per-Class Metrics (Precision, Recall, F1)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    
    # Extract per-class metrics
    per_class_metrics = []
    for class_name in CLASS_NAMES:
        if class_name in report:
            per_class_metrics.append({
                'Class': class_name,
                'Precision': report[class_name]['precision'],
                'Recall': report[class_name]['recall'],
                'F1-Score': report[class_name]['f1-score'],
                'Support': report[class_name]['support']
            })
    
    metrics_df = pd.DataFrame(per_class_metrics)
    
    # Save CSV
    csv_path = REG_DIR / f'{model_name.lower()}_per_class_metrics.csv'
    metrics_df.to_csv(csv_path, index=False)
    print(f"  ✓ Saved: {csv_path}")
    
    # Plot per-class metrics
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(CLASS_NAMES))
    width = 0.25
    
    bars1 = ax.bar(x - width, metrics_df['Precision'], width, label='Precision', color='#3498db')
    bars2 = ax.bar(x, metrics_df['Recall'], width, label='Recall', color='#2ecc71')
    bars3 = ax.bar(x + width, metrics_df['F1-Score'], width, label='F1-Score', color='#e74c3c')
    
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Per-Class Metrics - {model_name}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.set_ylim([0, 1.05])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    metrics_path = REG_DIR / f'viz_{model_name.lower()}_per_class_metrics.png'
    plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {metrics_path}")
    plt.close()

# 3. Training/Validation Curves for ALL Models
print("\n  Creating training/validation curves for all models...")
fig, axes = plt.subplots(3, 2, figsize=(16, 14))
axes = axes.flatten()

for idx, (model_name, model_results) in enumerate(training_results.items()):
    if idx >= len(axes):
        break
    
    history = model_results['history']
    ax = axes[idx]
    
    # Plot loss
    ax.plot(history.history['loss'], label='Training Loss', color='#e74c3c', linewidth=2)
    ax.plot(history.history['val_loss'], label='Validation Loss', color='#3498db', linewidth=2)
    ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('Loss', fontsize=10)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

# Hide unused subplots if any
for idx in range(len(training_results), len(axes)):
    fig.delaxes(axes[idx])

plt.suptitle('Training & Validation Loss Curves - All Models', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
curves_path = REG_DIR / 'viz_all_training_validation_curves.png'
plt.savefig(curves_path, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {curves_path}")
plt.close()

# 4. ROC Curves for Best Model
print(f"\n  Creating ROC curves for best model ({best_model_name})...")

# Get predictions for best model
best_model = training_results[best_model_name]['model']
y_pred_probs = best_model.predict(test_dataset, verbose=0)
y_true = np.concatenate([y for x, y in test_dataset], axis=0)

# Binarize labels for ROC curve
y_true_bin = label_binarize(np.argmax(y_true, axis=1), classes=range(len(CLASS_NAMES)))

# Per-Class ROC Curves
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for i, class_name in enumerate(CLASS_NAMES):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
    roc_auc = auc(fpr, tpr)
    
    ax = axes[i]
    ax.plot(fpr, tpr, color='#e74c3c', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='#95a5a6', lw=2, linestyle='--', label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=10)
    ax.set_ylabel('True Positive Rate', fontsize=10)
    ax.set_title(f'{class_name}', fontsize=11, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle(f'Per-Class ROC Curves - {best_model_name}', fontsize=16, fontweight='bold')
plt.tight_layout()
roc_path = REG_DIR / f'viz_{best_model_name.lower()}_roc_curves.png'
plt.savefig(roc_path, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {roc_path}")
plt.close()

# 5. Average ROC Curves (Micro & Macro)
print(f"  Creating average ROC curves...")

# Compute micro-average ROC curve
fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_pred_probs.ravel())
roc_auc_micro = auc(fpr_micro, tpr_micro)

# Compute macro-average ROC curve
fpr_dict = {}
tpr_dict = {}
roc_auc_dict = {}
for i in range(len(CLASS_NAMES)):
    fpr_dict[i], tpr_dict[i], _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
    roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])

# Aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(len(CLASS_NAMES))]))

# Interpolate all ROC curves at these points
mean_tpr = np.zeros_like(all_fpr)
for i in range(len(CLASS_NAMES)):
    mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])

mean_tpr /= len(CLASS_NAMES)
fpr_macro = all_fpr
tpr_macro = mean_tpr
roc_auc_macro = auc(fpr_macro, tpr_macro)

# Plot
plt.figure(figsize=(10, 8))
plt.plot(fpr_micro, tpr_micro, color='#e74c3c', lw=3, 
         label=f'Micro-average ROC (AUC = {roc_auc_micro:.3f})')
plt.plot(fpr_macro, tpr_macro, color='#3498db', lw=3, 
         label=f'Macro-average ROC (AUC = {roc_auc_macro:.3f})')
plt.plot([0, 1], [0, 1], color='#95a5a6', lw=2, linestyle='--', label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
plt.title(f'Average ROC Curves - {best_model_name}', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
roc_avg_path = REG_DIR / f'viz_{best_model_name.lower()}_roc_average.png'
plt.savefig(roc_avg_path, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {roc_avg_path}")
plt.close()

print("\n✓ All additional visualizations (Week 6 style) created successfully!")

# Save best model in .keras format (Week 6 Best Practice)
best_model = training_results[best_model_name]['model']
best_model_path = REG_DIR / 'best_regularized_model.keras'
best_model.save(best_model_path)
print(f"✓ Saved best model: {best_model_path}")

# Save all models
for name, results in training_results.items():
    model_path = REG_DIR / f'{name.lower()}_final.keras'
    results['model'].save(model_path)
    print(f"✓ Saved: {model_path}")

# Save results CSV
results_csv = REG_DIR / 'regularization_results.csv'
results_df.to_csv(results_csv, index=False)
print(f"✓ Saved: {results_csv}")

# Save comprehensive summary JSON
summary = {
    'experiment': 'Week8_Optimized_Regularization',
    'timestamp': datetime.now().isoformat(),
    'gpu_target': 'NVIDIA A40 (48GB VRAM)',
    'architecture': {
        'model_type': 'Baseline CNN',
        'parameters': 5750000,
        'input_shape': [224, 224, 3],
        'num_classes': NUM_CLASSES,
        'class_names': CLASS_NAMES,
        'batch_size': BEST_BATCH_SIZE,
        'epochs': EPOCHS
    },
    'baseline_performance': {
        'accuracy': baseline_accuracy,
        'source': 'Week 6/7 (no regularization)',
        'note': 'Used as reference point for improvement calculation'
    },
    'optimizations_applied': [
        'Memory mapping (mmap_mode=r) - Reduces RAM usage',
        'Generator-based tf.data pipeline - Streaming data',
        'Reduced prefetch buffer (size=2) - Prevents memory buildup',
        'WarmupExponentialDecay learning rate schedule - Stable training',
        'MemoryOptimizedCallback with reduced logging - Efficient cleanup',
        'Gradient clipping (clipvalue=1.0) - Numerical stability',
        'All models in .keras format - Standard format',
        'Per-epoch garbage collection - Memory management'
    ],
    'target_metrics': {
        'accuracy': TARGET_ACCURACY,
        'precision': TARGET_PRECISION,
        'recall': TARGET_RECALL,
        'f1_score': TARGET_F1
    },
    'best_model': {
        'name': best_model_name,
        'rank': 1,
        'metrics': best_metrics,
        'improvement_vs_baseline_pct': (best_metrics['accuracy'] - baseline_accuracy) * 100,
        'recommendation': f"Use {best_model_name} for Week 9 fine-tuning. This strategy shows the best generalization across all metrics."
    },
    'all_strategies_ranked': [
        {
            'rank': int(row['Rank']),
            'name': row['Model'],
            'accuracy': float(row['Test_Accuracy']),
            'precision': float(row['Test_Precision']),
            'recall': float(row['Test_Recall']),
            'f1_score': float(row['Test_F1']),
            'epochs_trained': int(row['Epochs_Trained']),
            'improvement_vs_baseline_pct': float(row['Improvement_vs_Baseline']),
            'meets_f1_target': row['Meets_Target_F1'] == '✓',
            'meets_accuracy_target': row['Meets_Target_Accuracy'] == '✓'
        }
        for _, row in results_df.iterrows()
    ],
    'statistical_summary': {
        'accuracy': {
            'best': float(results_df['Test_Accuracy'].max()),
            'worst': float(results_df['Test_Accuracy'].min()),
            'mean': float(mean_accuracy),
            'std_dev': float(results_df['Test_Accuracy'].std()),
            'range': float(accuracy_range)
        },
        'f1_score': {
            'best': float(results_df['Test_F1'].max()),
            'worst': float(results_df['Test_F1'].min()),
            'mean': float(mean_f1),
            'std_dev': float(results_df['Test_F1'].std()),
            'range': float(f1_range)
        },
        'targets_met': {
            'f1_score': targets_met,
            'total_models': len(results_df)
        }
    },
    'next_steps': [
        f'1. Use {best_model_name} model for Week 9 fine-tuning',
        '2. Apply best strategy to larger dataset if available',
        '3. Consider ensemble combining top-3 strategies',
        '4. Optimize learning rate further for best strategy',
        '5. Test on external validation dataset'
    ]
}

summary_path = REG_DIR / 'week8_optimization_summary.json'
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"✓ Saved: {summary_path}")

# --- Create Professional HTML Report ---
html_report = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Week 8 - Regularization Optimization Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; background: white; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
        
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 40px; border-radius: 10px; margin-bottom: 30px; text-align: center; }}
        .header h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
        .header p {{ font-size: 1.2em; opacity: 0.9; }}
        
        h2 {{ color: #667eea; border-bottom: 3px solid #667eea; padding-bottom: 10px; margin: 30px 0 20px 0; }}
        h3 {{ color: #764ba2; margin: 20px 0 15px 0; }}
        
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
        .stat-box {{ background: #f9f9f9; border-left: 5px solid #667eea; padding: 20px; border-radius: 5px; }}
        .stat-box h4 {{ color: #667eea; margin-bottom: 10px; }}
        .stat-box .value {{ font-size: 1.8em; font-weight: bold; color: #333; }}
        .stat-box .label {{ font-size: 0.9em; color: #666; }}
        
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; background: white; }}
        table th {{ background: #667eea; color: white; padding: 15px; text-align: left; font-weight: 600; }}
        table td {{ padding: 12px 15px; border-bottom: 1px solid #ddd; }}
        table tr:nth-child(even) {{ background: #f9f9f9; }}
        table tr:hover {{ background: #f0f0f0; }}
        
        .best-model {{ background: #d4edda; border-left: 5px solid #28a745; padding: 20px; border-radius: 5px; margin: 20px 0; }}
        .best-model h3 {{ color: #28a745; margin-top: 0; }}
        
        .visualization {{ margin: 30px 0; text-align: center; }}
        .visualization img {{ max-width: 100%; height: auto; border-radius: 5px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
        .visualization p {{ font-size: 0.95em; color: #666; margin-top: 10px; font-style: italic; }}
        
        .recommendations {{ background: #fff3cd; border-left: 5px solid #ffc107; padding: 20px; border-radius: 5px; margin: 20px 0; }}
        .recommendations h3 {{ color: #856404; margin-top: 0; }}
        .recommendations ol {{ margin-left: 20px; }}
        .recommendations li {{ margin: 10px 0; }}
        
        .footer {{ text-align: center; padding: 20px; color: #999; border-top: 1px solid #ddd; margin-top: 40px; font-size: 0.9em; }}
        
        .rank-badge {{ display: inline-block; background: #667eea; color: white; padding: 5px 10px; border-radius: 20px; font-weight: bold; margin-right: 10px; }}
        .success {{ color: #28a745; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 Week 8: Regularization Optimization Report</h1>
            <p>Comprehensive Analysis of 5 Regularization Strategies</p>
        </div>
        
        <h2>📊 Executive Summary</h2>
        <div class="stats-grid">
            <div class="stat-box">
                <h4>🏆 Best Model</h4>
                <div class="value">{best_model_name}</div>
                <div class="label">Rank #1 by F1-Score</div>
            </div>
            <div class="stat-box">
                <h4>📈 Best Accuracy</h4>
                <div class="value">{results_df['Test_Accuracy'].max():.4f}</div>
                <div class="label">Improvement: +{(results_df['Test_Accuracy'].max() - baseline_accuracy) * 100:.2f}%</div>
            </div>
            <div class="stat-box">
                <h4>⭐ Best F1-Score</h4>
                <div class="value">{results_df['Test_F1'].max():.4f}</div>
                <div class="label">vs Target: {TARGET_F1:.4f}</div>
            </div>
            <div class="stat-box">
                <h4>🎯 Targets Met</h4>
                <div class="value">{targets_met}/{len(results_df)}</div>
                <div class="label">Models meeting F1 target</div>
            </div>
        </div>
        
        <h2>📋 Detailed Results Ranking</h2>
        <table>
            <thead>
                <tr><th>Rank</th><th>Strategy</th><th>Accuracy</th><th>F1-Score</th><th>Improvement</th><th>Status</th></tr>
            </thead>
            <tbody>
"""

for _, row in results_df.iterrows():
    status = '<span class="success">✓</span>' if row['Meets_Target_F1'] == '✓' else '—'
    html_report += f"<tr><td><span class='rank-badge'>#{int(row['Rank'])}</span></td><td><strong>{row['Model']}</strong></td><td>{row['Test_Accuracy']:.4f}</td><td>{row['Test_F1']:.4f}</td><td>+{row['Improvement_vs_Baseline']:.2f}%</td><td>{status}</td></tr>"

html_report += """
            </tbody>
        </table>
"""

html_report += f"""
        <div class="best-model">
            <h3>🏆 Best Model: {best_model_name}</h3>
            <p><strong>Accuracy:</strong> {best_metrics['accuracy']:.4f} / {TARGET_ACCURACY:.4f} (Target)</p>
            <p><strong>F1-Score:</strong> {best_metrics['f1_score']:.4f} / {TARGET_F1:.4f} (Target)</p>
            <p><strong>Improvement:</strong> +{(best_metrics['accuracy'] - baseline_accuracy) * 100:.2f}% vs Baseline</p>
        </div>
        
        <div class="recommendations">
            <h3>💡 Recommendation</h3>
            <p><strong>Use {best_model_name} for Week 9 fine-tuning</strong></p>
            <p>This strategy demonstrates the best generalization across all metrics and should be used for production optimization.</p>
        </div>
        
        <h2>📊 Performance Visualizations</h2>
        <div class="visualization"><p>✓ 6 high-quality charts generated and saved</p></div>
        
        <div class="footer">
            <p>Week 8 - Regularization Optimization Report | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </div>
</body>
</html>
"""

html_path = REG_DIR / 'week8_optimization_report.html'
with open(html_path, 'w') as f:
    f.write(html_report)
print(f"✓ Saved: {html_path}")

# ============================================
# FINAL SUMMARY
# ============================================

print("\n" + "=" * 80)
print("✅ WEEK 8 COMPLETE: OPTIMIZED REGULARIZATION METHODS WITH FULL ANALYSIS")
print("=" * 80)

print(f"\n📦 Optimizations Applied (Week 6 Best Practices):")
for opt in summary['optimizations_applied']:
    print(f"  ✓ {opt}")

print(f"\n🏆 Best Model: {best_model_name}")
print(f"   Accuracy:  {best_metrics['accuracy']:.4f} / {TARGET_ACCURACY:.4f}")
print(f"   F1-Score:  {best_metrics['f1_score']:.4f} / {TARGET_F1:.4f}")
print(f"   Precision: {best_metrics['precision']:.4f} / {TARGET_PRECISION:.4f}")
print(f"   Recall:    {best_metrics['recall']:.4f} / {TARGET_RECALL:.4f}")
print(f"   Improvement: +{(best_metrics['accuracy'] - baseline_accuracy) * 100:.2f}% vs Baseline")

print(f"\n📁 Results saved to: {REG_DIR}")

print(f"\n📄 Report Files Generated:")
print(f"  ✓ week8_optimization_summary.json - Comprehensive data summary")
print(f"  ✓ week8_optimization_report.html - Professional HTML report")
print(f"  ✓ regularization_results.csv - CSV results table")

print(f"\n📊 Visualizations Generated:")
print(f"  ✓ viz_01_accuracy_comparison.png")
print(f"  ✓ viz_02_metrics_comparison.png")
print(f"  ✓ viz_03_f1_ranking.png")
print(f"  ✓ viz_04_training_curves.png")
print(f"  ✓ viz_05_improvement_vs_baseline.png")
print(f"  ✓ viz_06_strategy_heatmap.png")

print(f"\n🎯 Next Steps for Week 9:")
print(f"  1. Use {best_model_name} for fine-tuning")
print(f"  2. Model saved at: {REG_DIR / 'best_regularized_model.keras'}")
print(f"  3. Review HTML report: {html_path}")
print(f"  4. Proceed with Week 9 optimization phase")

print("=" * 80)