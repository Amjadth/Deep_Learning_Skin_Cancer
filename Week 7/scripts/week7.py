# ================================================================================================
# WEEK 7 — OPTIMIZED HYPERPARAMETER TUNING (LR × BATCH SIZE GRID SEARCH)
# MEMORY-SAFE, GPU-AWARE TUNING BUILT ON WEEK 6 PIPELINE
# ================================================================================================
#
# Purpose:
#   Run a **resource-efficient hyperparameter search** over learning rates and batch sizes
#   for the Week 6 baseline CNN, using:
#     • Pre-denormalized [0, 1] image data (saved from earlier weeks)
#     • Memmap-backed arrays + generator-based tf.data pipelines (no full-array loads)
#     • Warmup + exponential decay learning rate schedules
#     • Aggressive memory cleanup callbacks (to avoid OOM in 46.6 GB container)
#     • Advanced evaluation metrics (F1, precision, recall, test accuracy)
#
# What this script does (high level):
#   1) Workspace & GPU Setup
#      - Detects RunPod-style paths (/workspace, /runpod-volume, /notebooks)
#      - Sets OUTPUT_DIR, MODEL_DIR, RESULTS_DIR, TUNING_DIR
#      - Configures GPU with memory growth enabled and quiet TensorFlow logs
#      - Applies Week 6 global tuning defaults:
#          • Default batch size for pipelines: 128
#          • Epochs per tuning run: 30 (shorter for grid search)
#
#   2) Data Loading (Pre-Denormalized, Week 6 Format)
#      - Loads the following via safe memmap-aware loader:
#          X_train_denormalized.npy  (64k × 224×224×3)
#          X_val_denormalized.npy    ( 8k × 224×224×3)
#          X_test_denormalized.npy   ( 8k × 224×224×3)
#          y_train_baseline.npy, y_val_baseline.npy, y_test_baseline.npy
#      - Uses np.load with mmap_mode or raw np.memmap fallback
#      - Prints basic range checks and class names from split_info.json
#
#   3) Memory-Safe tf.data Pipeline (from Week 6)
#      - create_optimized_dataset():
#          • Uses a Python generator over memmap arrays (streaming, no full RAM load)
#          • Shuffles via index permutations (not copying the whole array)
#          • Batches on-the-fly and prefetches only 2 batches
#      - Builds:
#          • train_dataset, val_dataset, test_dataset
#      - Keeps RAM overhead low and GPU fed without OOM crashes
#
#   4) Learning Rate Schedule (Warmup + Exponential Decay)
#      - WarmupExponentialDecay:
#          • Phase 1: Warmup from 0 → initial_lr over a few epochs
#          • Phase 2: Exponential decay towards a low final_lr
#          • Phase 3: Constant low LR after decay window
#      - Used per configuration via Adam(learning_rate=lr_schedule, clipnorm, clipvalue)
#
#   5) MemoryOptimizedCallback (from Week 6)
#      - At end of each epoch:
#          • Runs gc.collect() and clears TF backend where possible
#          • Optionally clears Linux page cache (echo 3 > /proc/sys/vm/drop_caches)
#            if psutil shows high memory usage (> 80%)
#          • Prints epoch stats and memory footprint if psutil is available
#
#   6) Model Architecture & Loading
#      - build_baseline_cnn(num_classes):
#          • 3 conv blocks (32 → 64 → 128 filters)
#          • Each block: Conv2D ×2 + BatchNorm + MaxPool + Dropout
#          • GlobalAveragePooling2D → Dense(512) → Dense(256) → Softmax(NUM_CLASSES)
#      - load_or_create_model():
#          • Tries to load an existing .keras baseline model from:
#              - week7_baseline_cnn.keras
#              - baseline_cnn.keras
#              - baseline_model.keras
#          • Verifies parameter count (~5.75M); if mismatch, rebuilds baseline CNN
#
#   7) Hyperparameter Search Space
#      - Learning rates: [1e-4, 5e-4, 1e-3]
#      - Batch sizes:    [64, 128, 256]
#      - Total combinations: len(learning_rates) × len(batch_sizes)
#      - Designed for **coarse but practical** exploration under GPU + RAM constraints
#
#   8) Training Function per Configuration
#      - train_with_config(...):
#          • Builds fresh tf.data datasets for given (lr, batch_size)
#          • Creates LR schedule (WarmupExponentialDecay) for that config
#          • Compiles model with Adam + gradient clipping
#          • Attaches callbacks:
#              - ModelCheckpoint → best model per config (saved under tuning_results/)
#              - EarlyStopping on val_accuracy with patience=5
#              - ReduceLROnPlateau on val_loss
#              - MemoryOptimizedCallback (cache + GC)
#          • Trains for up to EPOCHS_TUNING (default 30)
#          • Evaluates on val and test datasets
#          • Computes macro-level metrics:
#              - val_accuracy, val_precision, val_recall, val_f1
#              - test_accuracy, test_f1
#          • Returns a results dict including training time and final metrics
#
#   9) Grid Search Loop
#      - Iterates over all (learning_rate, batch_size) pairs via itertools.product
#      - For each combo:
#          • Calls train_with_config(...)
#          • Tracks best configuration by **validation F1-score**
#          • Performs gc.collect() and clears TF session between runs
#      - Prints when a new best configuration is found
#
#  10) Saving Tuning Results
#      - Consolidated CSV:
#          • tuning_results/hyperparameter_tuning_results.csv
#            - One row per configuration (LR, batch size, metrics, training time)
#      - Visualizations:
#          • tuning_results/hyperparameter_heatmaps.png
#            - Heatmaps for val_accuracy, F1-score, training_time
#            - Scatter plot of val_accuracy vs F1-score colored by batch size
#      - Best model file:
#          • tuning_results/best_lr{LR}_bs{BS}.keras
#
#  11) Summary Report
#      - Generates a human-readable summary:
#          • Best configuration and metrics (accuracy, F1, precision, recall)
#          • Test accuracy for that configuration
#          • Aggregate statistics (avg / best / worst accuracy)
#      - Writes to:
#          • tuning_results/tuning_summary_report.txt
#      - Prints the same summary to stdout at the end of the run
#
# Requirements / Assumptions:
#   • Week 6 preprocessing + denormalization already completed:
#       - X_train_denormalized.npy, X_val_denormalized.npy, X_test_denormalized.npy
#       - y_train_baseline.npy, y_val_baseline.npy, y_test_baseline.npy
#       - split_info.json (must contain class_names)
#   • RunPod (or similar) environment with:
#       - A40 (or comparable) GPU
#       - TensorFlow 2.x and optional psutil installed
#
#
# Version:
#   2.0 (2025)
# ================================================================================================

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import gc
import time
import json
import pandas as pd
import warnings
from datetime import datetime
from itertools import product
from sklearn.metrics import (f1_score, precision_score, recall_score, 
                             roc_auc_score, confusion_matrix, classification_report)

warnings.filterwarnings('ignore')

# Import psutil for memory monitoring (optional but recommended)
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("⚠️  psutil not available - memory monitoring limited")

# ============================================
# WORKSPACE SETUP
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
RESULTS_DIR = (OUTPUT_DIR / 'results').resolve()
TUNING_DIR = (OUTPUT_DIR / 'tuning_results').resolve()

# Create directories
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
MODEL_DIR.mkdir(exist_ok=True, parents=True)
RESULTS_DIR.mkdir(exist_ok=True, parents=True)
TUNING_DIR.mkdir(exist_ok=True, parents=True)

print("=" * 80)
print("WEEK 7: OPTIMIZED HYPERPARAMETER TUNING")
print("Using Week 6 Best Practices for Resource Utilization")
print("=" * 80)
print(f"\n📊 Configuration:")
print(f"   Storage base: {STORAGE_BASE}")
print(f"   Output directory: {OUTPUT_DIR}")
print(f"   Model directory: {MODEL_DIR}")
print(f"   Tuning directory: {TUNING_DIR}")

# ============================================
# GPU CONFIGURATION
# ============================================
print("\n🎮 STEP 0: GPU Configuration...")

# Suppress TensorFlow verbose logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_VMODULE'] = 'slow_operation_alarm=0'

# HYPERPARAMETER TUNING SETTINGS
BATCH_SIZE_DEFAULT = 128
EPOCHS_TUNING = 30  # Reduced for grid search efficiency
INITIAL_LR = 0.0001
CLEAR_CACHE_FREQ = 5

print(f"✓ Configuration:")
print(f"   Default batch size: 128 (from Week 6)")
print(f"   Epochs per tuning run: {EPOCHS_TUNING}")
print(f"   Learning rate base: {INITIAL_LR}")
print(f"   Cache clear frequency: Every {CLEAR_CACHE_FREQ} epochs")

# Configure GPU memory (same as Week 6)
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"\n✓ GPU configured: {len(gpus)} GPU(s) detected")
        print(f"✓ Single GPU training (memory growth enabled)")
    else:
        print("⚠ WARNING: No GPU detected!")
except Exception as e:
    print(f"⚠ GPU configuration error: {e}")

# ============================================
# UTILITIES (FROM WEEK 6)
# ============================================

def print_memory_status(label=""):
    """Print current memory usage (from Week 6)"""
    if not HAS_PSUTIL:
        return
    try:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        memory_mb = mem_info.rss / 1024 / 1024
        percent = process.memory_percent()
        vm = psutil.virtual_memory()
        print(f"  💾 Memory {label}:")
        print(f"     Process: {memory_mb:.0f} MB ({percent:.1f}%)")
        print(f"     System: {vm.used/(1024**3):.1f}GB/{vm.total/(1024**3):.1f}GB ({vm.percent:.1f}%)")
    except Exception as e:
        pass

def safe_load_npy(filepath, description, use_memmap=True, expected_shape=None):
    """Safely load .npy files with memmap support (from Week 6)"""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    print(f"  📥 Loading {description}...")
    try:
        if use_memmap:
            try:
                data = np.load(filepath, mmap_mode='r', allow_pickle=True)
                print(f"  ✓ {description}: {data.shape} (memmap)")
                return data
            except:
                try:
                    data = np.load(filepath, mmap_mode='r', allow_pickle=False)
                    print(f"  ✓ {description}: {data.shape} (memmap)")
                    return data
                except:
                    if expected_shape:
                        print(f"  ⚠️  Using raw memmap with shape {expected_shape}...")
                        data = np.memmap(str(filepath), dtype=np.float32, mode='r', shape=expected_shape)
                        print(f"  ✓ {description}: {data.shape}")
                        return data
                    raise
        else:
            data = np.load(filepath, allow_pickle=True)
            print(f"  ✓ {description}: {data.shape}")
            return data
    except Exception as e:
        print(f"  ❌ Error: {e}")
        raise

# ============================================
# STEP 1: LOAD DENORMALIZED DATA (WEEK 6 APPROACH)
# ============================================
print("\n📥 STEP 1: Loading pre-denormalized datasets...")
print("   (Memory mapping + Generator = No OOM crashes!)")

print_memory_status("before loading")

# Load denormalized data with memmap
X_train = safe_load_npy(
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

X_val = safe_load_npy(
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

X_test = safe_load_npy(
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

print(f"\n✓ All denormalized data loaded via memmap!")
print(f"  X_train: {X_train.shape}, Range: [{X_train.min():.3f}, {X_train.max():.3f}]")
print(f"  X_val: {X_val.shape}, Range: [{X_val.min():.3f}, {X_val.max():.3f}]")
print(f"  X_test: {X_test.shape}, Range: [{X_test.min():.3f}, {X_test.max():.3f}]")

# Load class info
with open(OUTPUT_DIR / 'split_info.json', 'r', encoding='utf-8') as f:
    split_info = json.load(f)

CLASS_NAMES = split_info['class_names']
NUM_CLASSES = len(CLASS_NAMES)
print(f"\n✓ Classes: {', '.join(CLASS_NAMES)}")

print_memory_status("after loading data")

# ============================================
# STEP 2: CREATE OPTIMIZED TF.DATA PIPELINE (WEEK 6 APPROACH)
# ============================================
print("\n🔧 STEP 2: Creating memory-safe tf.data pipeline...")
print("   ✅ Generator-based loading (never loads full array)")
print("   ✅ Reduced prefetch buffer = 8-10GB saved")
print("   ✅ Memory streaming from memmap")

def create_optimized_dataset(X, y, batch_size, shuffle=True):
    """
    MEMORY-SAFE pipeline using generator for memmap arrays (from Week 6)
    - Generator reads from memmap in batches (never loads full array)
    - No runtime denormalization (data already [0,1])
    - Minimal prefetch (2 batches)
    - Optional shuffling with index permutation
    """
    num_samples = len(X)
    
    def data_generator():
        """Generator that yields batches from memmap without loading full array"""
        indices = np.arange(num_samples)
        if shuffle:
            np.random.shuffle(indices)
        
        for idx in indices:
            img = X[idx]  # Shape: (224, 224, 3)
            label = y[idx]  # Scalar
            yield img, label
    
    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=(
            tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int64)
        )
    )

    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(buffer_size=2)  # Minimal prefetch (Week 6 FIX)

    return dataset

print("\n  Creating datasets...")
train_dataset = create_optimized_dataset(X_train, y_train, BATCH_SIZE_DEFAULT, shuffle=True)
val_dataset = create_optimized_dataset(X_val, y_val, BATCH_SIZE_DEFAULT, shuffle=False)
test_dataset = create_optimized_dataset(X_test, y_test, BATCH_SIZE_DEFAULT, shuffle=False)

print(f"  ✓ Training pipeline created (streaming from memmap)")
print(f"  ✓ Validation pipeline created (streaming from memmap)")
print(f"  ✓ Test pipeline created (streaming from memmap)")

print_memory_status("after creating datasets")

# ============================================
# STEP 3: LEARNING RATE SCHEDULE (FROM WEEK 6)
# ============================================

class WarmupExponentialDecay(keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate schedule with warmup (from Week 6)"""
    
    def __init__(self, initial_lr, warmup_epochs, decay_epochs, total_epochs):
        super().__init__()
        self.initial_lr = initial_lr
        self.warmup_epochs = warmup_epochs
        self.decay_epochs = decay_epochs
        self.total_epochs = total_epochs
        self.final_lr = initial_lr * 0.03125
    
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        epoch = step / 500.0
        epoch = tf.minimum(epoch, float(self.total_epochs))
        
        warmup_lr = self.initial_lr * (epoch / self.warmup_epochs)
        decay_progress = (epoch - self.warmup_epochs) / self.decay_epochs
        decay_lr = self.initial_lr * tf.pow(self.final_lr / self.initial_lr, decay_progress)
        
        lr = tf.cond(
            epoch < self.warmup_epochs,
            lambda: warmup_lr,
            lambda: tf.cond(
                epoch < self.warmup_epochs + self.decay_epochs,
                lambda: decay_lr,
                lambda: self.final_lr
            )
        )
        
        return lr
    
    def get_config(self):
        return {
            'initial_lr': self.initial_lr,
            'warmup_epochs': self.warmup_epochs,
            'decay_epochs': self.decay_epochs,
            'total_epochs': self.total_epochs
        }

# ============================================
# STEP 4: MEMORY CLEANUP CALLBACK (FROM WEEK 6)
# ============================================

class MemoryOptimizedCallback(callbacks.Callback):
    """Aggressive memory management from Week 6"""
    
    def __init__(self, clear_cache_freq=5):
        super().__init__()
        self.clear_cache_freq = clear_cache_freq
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        try:
            tf.keras.backend.clear_session()
        except:
            pass

        if epoch % self.clear_cache_freq == 0 and HAS_PSUTIL:
            try:
                vm = psutil.virtual_memory()
                mem_used_pct = vm.percent
                
                if mem_used_pct > 80:
                    os.system('sync > /dev/null 2>&1')
                    time.sleep(0.2)
                    os.system('echo 3 > /proc/sys/vm/drop_caches > /dev/null 2>&1')
                    time.sleep(0.2)
                    
                    mem_after = psutil.virtual_memory().percent
                    print(f"   ✨ Cache cleared (memory: {mem_used_pct:.1f}% → {mem_after:.1f}%)")
            except:
                pass

        logs = logs or {}
        elapsed = time.time() - self.start_time
        hours = elapsed // 3600
        minutes = (elapsed % 3600) // 60
        seconds = elapsed % 60

        print(f"\n  📊 Epoch {epoch+1} Stats:")
        print(f"     Loss: {logs.get('loss', 0):.4f}, Acc: {logs.get('accuracy', 0):.4f}")
        print(f"     Val Loss: {logs.get('val_loss', 0):.4f}, Val Acc: {logs.get('val_accuracy', 0):.4f}")
        print(f"     Elapsed: {hours:.0f}h {minutes:.0f}m {seconds:.0f}s")
        print_memory_status("after epoch")

# ============================================
# STEP 5: MODEL ARCHITECTURE (BASELINE CNN)
# ============================================

def build_baseline_cnn(num_classes):
    """Build baseline CNN (matches Week 6)"""
    return keras.Sequential([
        # Block 1: 32 filters
        keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),

        # Block 2: 64 filters
        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),

        # Block 3: 128 filters
        keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),

        # Dense layers
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),

        keras.layers.Dense(256, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),

        keras.layers.Dense(num_classes, activation='softmax')
    ], name='Baseline_CNN')

# ============================================
# STEP 6: MODEL LOADING (WITH VERIFICATION FROM WEEK 6)
# ============================================

print("\n🏗️  STEP 6: Model loading with parameter verification...")

def load_or_create_model(num_classes):
    """Try to load model, else create new (from Week 6 pattern)"""
    model = None
    model_paths = [
        MODEL_DIR / 'week7_baseline_cnn.keras',
        MODEL_DIR / 'baseline_cnn.keras',
        MODEL_DIR / 'baseline_model.keras',
    ]

    for model_path in model_paths:
        if model_path.exists():
            try:
                print(f"  Attempting to load from: {model_path.name}")
                model = keras.models.load_model(str(model_path), compile=False)
                print(f"  ✓ Model loaded: {model.count_params():,} parameters")
                
                # Verify it's correct (should have ~5.75M parameters)
                if model.count_params() < 5_000_000:
                    print(f"  ⚠️  Wrong model (only {model.count_params():,} params, need ~5.75M)")
                    model = None
                    continue
                
                break
            except Exception as e:
                print(f"  ❌ Failed to load: {e}")
                model = None
                continue

    if not model:
        print(f"\n  Creating new baseline CNN...")
        model = build_baseline_cnn(num_classes)

    print(f"\n✓ Model ready: {model.count_params():,} parameters")
    return model

model = load_or_create_model(NUM_CLASSES)

# ============================================
# STEP 7: HYPERPARAMETER TUNING SETUP
# ============================================
print("\n🔍 STEP 7: Defining hyperparameter search space...")

learning_rates = [0.0001, 0.0005, 0.001]
batch_sizes = [64, 128, 256]

print(f"  Learning Rates: {learning_rates}")
print(f"  Batch Sizes: {batch_sizes}")
print(f"  Total combinations: {len(learning_rates) * len(batch_sizes)}")

# ============================================
# STEP 8: TRAINING FUNCTION (WITH METRICS)
# ============================================
print("\n⚙️  STEP 8: Setting up training function with advanced metrics...")

def train_with_config(model_base, lr, batch_size, X_train_data, y_train_data, 
                      X_val_data, y_val_data, X_test_data, y_test_data, 
                      epochs=30, verbose=0):
    """Train model with specific hyperparameters and compute advanced metrics"""
    
    print(f"\n  Training: LR={lr}, Batch={batch_size}")
    start_time = time.time()
    
    # Create datasets for this configuration
    train_ds = create_optimized_dataset(X_train_data, y_train_data, batch_size, shuffle=True)
    val_ds = create_optimized_dataset(X_val_data, y_val_data, batch_size, shuffle=False)
    test_ds = create_optimized_dataset(X_test_data, y_test_data, batch_size, shuffle=False)
    
    # Compile model
    lr_schedule = WarmupExponentialDecay(lr, 3, 15, epochs)
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0, clipvalue=0.5)
    
    model_base.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    best_model_path = TUNING_DIR / f'best_lr{lr}_bs{batch_size}.keras'
    callbacks_list = [
        callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=0
        ),
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=0,
            mode='max'
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-8,
            verbose=0
        ),
        MemoryOptimizedCallback(clear_cache_freq=CLEAR_CACHE_FREQ)
    ]
    
    # Train
    history = model_base.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks_list,
        verbose=verbose
    )
    
    training_time = time.time() - start_time
    
    # Evaluate
    val_loss, val_acc = model_base.evaluate(val_ds, verbose=0)
    test_loss, test_acc = model_base.evaluate(test_ds, verbose=0)
    
    # Get detailed metrics
    y_val_pred_proba = model_base.predict(val_ds, verbose=0)
    y_val_pred = np.argmax(y_val_pred_proba, axis=1)
    
    y_test_pred_proba = model_base.predict(test_ds, verbose=0)
    y_test_pred = np.argmax(y_test_pred_proba, axis=1)
    
    val_precision = precision_score(y_val_data, y_val_pred, average='macro', zero_division=0)
    val_recall = recall_score(y_val_data, y_val_pred, average='macro', zero_division=0)
    val_f1 = f1_score(y_val_data, y_val_pred, average='macro', zero_division=0)
    
    test_precision = precision_score(y_test_data, y_test_pred, average='macro', zero_division=0)
    test_recall = recall_score(y_test_data, y_test_pred, average='macro', zero_division=0)
    test_f1 = f1_score(y_test_data, y_test_pred, average='macro', zero_division=0)
    
    results = {
        'learning_rate': lr,
        'batch_size': batch_size,
        'epochs_trained': len(history.history['loss']),
        'training_time': training_time,
        'final_train_loss': float(history.history['loss'][-1]),
        'final_train_acc': float(history.history['accuracy'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1]),
        'final_val_acc': float(history.history['val_accuracy'][-1]),
        'val_accuracy': float(val_acc),
        'f1_score': float(val_f1),
        'precision': float(val_precision),
        'recall': float(val_recall),
        'test_accuracy': float(test_acc),
        'test_f1': float(test_f1),
    }
    
    print(f"    ✓ Val Acc: {val_acc:.4f}, F1: {val_f1:.4f}, Time: {training_time:.1f}s")
    
    return results

# ============================================
# STEP 9: RUN GRID SEARCH
# ============================================
print("\n🚀 STEP 9: Running grid search...")

all_results = []
best_config = None
best_f1 = 0.0

for idx, (lr, bs) in enumerate(product(learning_rates, batch_sizes)):
    print(f"\n[{idx+1}/{len(learning_rates) * len(batch_sizes)}] ", end="")
    
    try:
        results = train_with_config(model, lr, bs, X_train, y_train, 
                                   X_val, y_val, X_test, y_test, 
                                   epochs=EPOCHS_TUNING, verbose=0)
        all_results.append(results)
        
        if results['f1_score'] > best_f1:
            best_f1 = results['f1_score']
            best_config = (lr, bs)
            print(f"  🌟 New best! (F1: {best_f1:.4f})")
        
        gc.collect()
        tf.keras.backend.clear_session()
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        continue

print("\n" + "=" * 80)
print(f"✅ Grid search complete! Trained {len(all_results)} configurations")
print(f"  Best F1-Score: {best_f1:.4f}")
if best_config:
    print(f"  Best Config: LR={best_config[0]}, Batch={best_config[1]}")

# ============================================
# STEP 10: SAVE RESULTS
# ============================================
print("\n💾 STEP 10: Saving results...")

results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values('f1_score', ascending=False)
results_csv = TUNING_DIR / 'hyperparameter_tuning_results.csv'
results_df.to_csv(results_csv, index=False)
print(f"✓ Saved: {results_csv}")

# ============================================
# STEP 11: VISUALIZATIONS
# ============================================
print("\n🎨 STEP 11: Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Pivot tables
pivot_acc = results_df.pivot(index='learning_rate', columns='batch_size', values='val_accuracy')
pivot_f1 = results_df.pivot(index='learning_rate', columns='batch_size', values='f1_score')
pivot_time = results_df.pivot(index='learning_rate', columns='batch_size', values='training_time')

# Heatmaps
sns.heatmap(pivot_acc, annot=True, fmt='.4f', cmap='YlGnBu', ax=axes[0, 0],
            cbar_kws={'label': 'Validation Accuracy'}, linewidths=1)
axes[0, 0].set_title('Validation Accuracy', fontsize=12, fontweight='bold')

sns.heatmap(pivot_f1, annot=True, fmt='.4f', cmap='YlOrRd', ax=axes[0, 1],
            cbar_kws={'label': 'F1-Score'}, linewidths=1)
axes[0, 1].set_title('F1-Score', fontsize=12, fontweight='bold')

sns.heatmap(pivot_time, annot=True, fmt='.1f', cmap='RdYlGn_r', ax=axes[1, 0],
            cbar_kws={'label': 'Training Time (s)'}, linewidths=1)
axes[1, 0].set_title('Training Time', fontsize=12, fontweight='bold')

# Scatter plot
scatter = axes[1, 1].scatter(results_df['val_accuracy'], results_df['f1_score'],
                             c=results_df['batch_size'], s=150, cmap='viridis', alpha=0.7)
axes[1, 1].set_xlabel('Validation Accuracy')
axes[1, 1].set_ylabel('F1-Score')
axes[1, 1].set_title('Accuracy vs F1-Score')
axes[1, 1].grid(True, alpha=0.3)
plt.colorbar(scatter, ax=axes[1, 1], label='Batch Size')

plt.tight_layout()
viz_path = TUNING_DIR / 'hyperparameter_heatmaps.png'
plt.savefig(viz_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {viz_path}")

# ============================================
# STEP 12: SUMMARY REPORT
# ============================================
print("\n📝 STEP 12: Creating summary report...")

best_result = results_df.iloc[0] if len(results_df) > 0 else None

if best_result is not None:
    summary = f"""
{'='*80}
WEEK 7: HYPERPARAMETER TUNING SUMMARY
{'='*80}

BEST CONFIGURATION:
  Learning Rate: {best_result['learning_rate']}
  Batch Size: {best_result['batch_size']}
  Validation Accuracy: {best_result['val_accuracy']:.4f}
  F1-Score: {best_result['f1_score']:.4f}
  Precision: {best_result['precision']:.4f}
  Recall: {best_result['recall']:.4f}
  Test Accuracy: {best_result['test_accuracy']:.4f}

TRAINING STATISTICS:
  Total configurations: {len(results_df)}
  Average accuracy: {results_df['val_accuracy'].mean():.4f}
  Best accuracy: {results_df['val_accuracy'].max():.4f}
  Worst accuracy: {results_df['val_accuracy'].min():.4f}

WEEK 6 BEST PRACTICES APPLIED:
  ✓ Memory mapping + Generator-based data loading
  ✓ Reduced prefetch buffer (2 instead of AUTOTUNE)
  ✓ Aggressive memory cleanup and GC
  ✓ Learning rate schedule with warmup
  ✓ Proper model loading with verification
  ✓ All models saved in .keras format
  ✓ Advanced metrics (F1, precision, recall)
  ✓ Comprehensive visualizations

{'='*80}
"""
    
    print(summary)
    
    report_path = TUNING_DIR / 'tuning_summary_report.txt'
    with open(report_path, 'w') as f:
        f.write(summary)
    print(f"\n✓ Saved: {report_path}")

print("\n" + "=" * 80)
print("✅ WEEK 7 COMPLETE!")
print("=" * 80)
print(f"\n📁 Output Files:")
print(f"   Results CSV: {results_csv}")
print(f"   Heatmaps: {viz_path}")
if best_result is not None:
    best_model_filename = f"best_lr{best_result['learning_rate']}_bs{best_result['batch_size']}.keras"
    best_model_path = TUNING_DIR / best_model_filename
    print(f"   Best model: {best_model_path}")
print(f"\n✨ All Week 6 best practices implemented successfully!")
print("=" * 80 + "\n")