"""
WEEK 9: MAXIMUM GPU UTILIZATION - A40 OPTIMIZED (FIXED)
Fixed: Data type compatibility issue in data loader
"""

import os
import gc
import json
import warnings
import time
import subprocess
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, applications, mixed_precision
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION
# ============================================

class Config:
    """GPU-optimized configuration for A40"""
    
    # Hardware
    GPU_MEMORY_LIMIT = 40 * 1024  # MB - Fixed: was 46GB causing overflow
    MAX_RAM_USAGE = 40  # GB
    
    # Data
    IMAGE_SIZE = (224, 224, 3)
    NUM_CLASSES = 8
    
    # Training Parameters - MEMORY-SAFE GPU OPTIMIZED
    BATCH_SIZE = 64  # Keep at 64 to avoid OOM (41GB data + 46GB RAM)
    GRADIENT_ACCUMULATION_STEPS = 4  # Increased to 4 for effective batch of 256
    EPOCHS_PHASE1 = 50
    EPOCHS_PHASE2 = 25
    
    # Learning rates - INCREASED for faster convergence from plateau
    LR_PHASE1 = 1e-3  # Increased from 5e-4 to escape 31% plateau
    LR_PHASE2 = 1e-4  # Increased from 3e-5 for meaningful fine-tuning updates
    
    # Data Pipeline - MEMORY-SAFE
    PREFETCH_BUFFER = 4  # Conservative prefetch (4 batches = 256 samples)
    NUM_PARALLEL_CALLS = tf.data.AUTOTUNE  # Auto-tune parallelism (CPU side)
    CACHE_DATASET = False  # NO caching - would OOM with 41GB data!
    INTERLEAVE_CYCLE = 8  # Moderate parallel loading
    
    # Models
    MODEL_CONFIG = {
        'EfficientNetB0': {'input_shape': (224, 224, 3)},
        'EfficientNetB3': {'input_shape': (224, 224, 3)},
        'ResNet50V2': {'input_shape': (224, 224, 3)},
        'DenseNet121': {'input_shape': (224, 224, 3)},
        'MobileNetV2': {'input_shape': (224, 224, 3)},
    }
    
    # Data is PRE-NORMALIZED from Week 2-6 pipeline:
    # Week 2: ImageNet norm → [-1, 1]
    # Week 6: Denormalized → [0, 1]
    # Week 9: NO preprocessing needed (already in correct range for models)
    # NOTE: We removed PREPROCESSING dict - data is ready to use as-is
    
    # Paths
    BASE_DIR = Path('/workspace') if Path('/workspace').exists() else Path.cwd()
    NETWORK_VOLUME = Path('/runpod-volume') if Path('/runpod-volume').exists() else None
    
    @classmethod
    def get_output_dir(cls):
        storage = cls.NETWORK_VOLUME if cls.NETWORK_VOLUME else cls.BASE_DIR
        return storage / 'outputs'
    
    @classmethod
    def get_transfer_dir(cls):
        return cls.get_output_dir() / 'transfer_learning_EB0'
    
    @classmethod
    def get_model_input_shape(cls, model_name):
        return cls.MODEL_CONFIG.get(model_name, {'input_shape': (224, 224, 3)})['input_shape']


# ============================================
# GPU CONFIGURATION
# ============================================

def configure_gpu():
    """Configure GPU for maximum performance"""
    
    print("🔧 Configuring GPU for MAXIMUM performance...")
    
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("❌ NO GPU DETECTED!")
        return
    
    print(f"✅ Found {len(gpus)} GPU(s)")
    
    # Mixed precision
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    print(f"✅ Mixed precision: {policy.name}")
    
    # GPU memory growth
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=Config.GPU_MEMORY_LIMIT)]
        )
        print(f"✅ GPU memory limit: {Config.GPU_MEMORY_LIMIT/1024:.0f} GB")
    except RuntimeError as e:
        print(f"⚠️  GPU config warning: {e}")
    
    # Enable optimizations
    tf.config.optimizer.set_experimental_options({
        'layout_optimizer': True,
        'constant_folding': True,
        'shape_optimization': True,
        'remapping': True,
        'arithmetic_optimization': True,
        'dependency_optimization': True,
        'loop_optimization': True,
        'function_optimization': True,
        'debug_stripper': True,
        'disable_model_pruning': False,
        'scoped_allocator_optimization': True,
        'pin_to_host_optimization': True,
        'implementation_selector': True,
        'auto_mixed_precision': True,
        'min_graph_nodes': 1
    })
    
    # Enable XLA compilation for faster execution
    tf.config.optimizer.set_jit(True)
    print("✅ Advanced GPU optimizations + XLA enabled")
    
    # Test GPU
    print("\n🧪 Testing GPU...")
    with tf.device('/GPU:0'):
        a = tf.random.normal([1000, 1000])
        b = tf.matmul(a, a)
        _ = b.numpy()
    print("✅ GPU computation successful!")


# ============================================
# GPU MONITORING
# ============================================

def print_gpu_status(label=""):
    """Print detailed GPU status"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit', 
             '--format=csv,nounits,noheader'],
            capture_output=True,
            text=True,
            timeout=3
        )
        if result.returncode == 0:
            vals = result.stdout.strip().split(',')
            gpu_pct = vals[0].strip()
            mem_pct = vals[1].strip()
            mem_used = int(vals[2].strip())
            mem_total = int(vals[3].strip())
            power_draw = float(vals[4].strip())
            power_limit = float(vals[5].strip())
            
            print(f"🎮 GPU {label}:")
            print(f"   Compute: {gpu_pct}% | Memory: {mem_pct}% ({mem_used}/{mem_total} MB)")
            print(f"   Power: {power_draw:.0f}W / {power_limit:.0f}W ({power_draw/power_limit*100:.0f}%)")
    except:
        pass


# ============================================
# FIXED DATA LOADER
# ============================================

class OptimizedDataLoader:
    """High-performance data loader with FIXED data types"""
    
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.class_names = None
    
    def load_split_info(self):
        split_info_path = self.data_dir / 'split_info.json'
        with open(split_info_path, 'r') as f:
            info = json.load(f)
        self.class_names = info['class_names']
        return info
    
    def create_optimized_dataset(self, X_path, y_path, batch_size, shuffle=True, name="dataset"):
        """
        FIXED: Proper data type handling for TensorFlow compatibility
        """
        
        print(f"\n📂 Creating OPTIMIZED {name}...")
        
        # Load with memory mapping
        X_mmap = np.load(X_path, mmap_mode='r', allow_pickle=False)
        y_full = np.load(y_path, allow_pickle=False)
        
        n_samples = X_mmap.shape[0]
        print(f"  Samples: {n_samples:,}")
        print(f"  Batch size: {batch_size}")
        print(f"  Parallel workers: {Config.NUM_PARALLEL_CALLS}")
        
        # CRITICAL FIX: Convert labels to int32 upfront
        y_full = y_full.astype(np.int32)
        print(f"  Label dtype: {y_full.dtype} (converted to int32)")
        
        # Create indices
        indices = np.arange(n_samples, dtype=np.int32)
        if shuffle:
            np.random.shuffle(indices)
        
        # FIXED: Proper data loader without .numpy() call on numpy objects
        def load_sample(idx):
            """
            Load sample with correct data types
            idx is a tf.Tensor, so we need .numpy() on it
            But X_mmap[idx] and y[idx] are already numpy, so no .numpy() call needed
            """
            # Convert TensorFlow tensor to Python int
            idx_val = int(idx)
            
            # Load image (already numpy array from memmap)
            image = X_mmap[idx_val].astype(np.float32)
            
            # Data is PRE-NORMALIZED [0, 1] from Week 6 denormalization
            # NO scaling needed - data ready for model input
            
            # Load label (already numpy int32)
            label = y_full[idx_val]  # No .numpy() here - it's already numpy!
            
            return image, label
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices(indices)
        
        # Map with proper output types
        dataset = dataset.map(
            lambda idx: tf.py_function(
                load_sample,
                [idx],
                [tf.float32, tf.int32]  # Explicitly specify output types
            ),
            num_parallel_calls=Config.NUM_PARALLEL_CALLS,
            deterministic=False
        )
        
        # Set shapes explicitly
        dataset = dataset.map(lambda x, y: (
            tf.ensure_shape(x, X_mmap.shape[1:]),
            tf.ensure_shape(y, [])
        ))
        
        # Data is PRE-NORMALIZED [0, 1] from Week 6 - ready for model input
        # NO preprocessing needed (fixes double-normalization bug)
        
        # NO CACHING - prevent OOM with 41GB data in 46GB RAM
        # We rely on memory-mapped loading instead
        
        # Batch
        dataset = dataset.batch(batch_size, drop_remainder=True)
        
        # Conservative prefetch (won't load entire dataset)
        dataset = dataset.prefetch(Config.PREFETCH_BUFFER)
        
        # Lightweight optimization options (no memory-intensive features)
        options = tf.data.Options()
        options.experimental_optimization.map_parallelization = True
        options.experimental_optimization.parallel_batch = False  # Disabled to save RAM
        options.threading.private_threadpool_size = 8  # Reduced from 16
        options.threading.max_intra_op_parallelism = 1
        dataset = dataset.with_options(options)
        
        # Cache validation dataset (safe at 8k images, ~2GB)
        if name == "Validation":
            dataset = dataset.cache()
            print(f"  💾 Validation dataset cached for stable evaluation")
        else:
            print(f"  💾 Memory-safe pipeline (no train caching, mmap-based loading)")
        
        print(f"  ✅ Optimized pipeline created")
        
        return dataset
    
    def get_dataset_for_model(self, model_name):
        """Load optimized dataset with model-specific preprocessing and augmentation"""
        
        target_size = Config.get_model_input_shape(model_name)
        print(f"\n🔄 Loading OPTIMIZED data for {model_name}")
        
        X_train_path = self.data_dir / 'X_train_denormalized.npy'
        y_train_path = self.data_dir / 'y_train.npy'
        X_val_path = self.data_dir / 'X_val_denormalized.npy'
        y_val_path = self.data_dir / 'y_val.npy'
        
        for path in [X_train_path, y_train_path, X_val_path, y_val_path]:
            if not path.exists():
                raise FileNotFoundError(f"Required: {path}")
        
        train_ds = self.create_optimized_dataset(
            X_train_path, y_train_path, Config.BATCH_SIZE, 
            shuffle=True, name="Training"
        )
        
        val_ds = self.create_optimized_dataset(
            X_val_path, y_val_path, Config.BATCH_SIZE,
            shuffle=False, name="Validation"
        )
        
        # ✅ FINAL FIX: Data is [0, 1] but EfficientNet expects [0, 255]
        # EfficientNet's preprocess_input() does NOT normalize - it expects raw pixel values!
        # Solution: Scale [0, 1] → [0, 255] to match ImageNet training
        
        def scale_to_efficientnet_range(image, label):
            """Scale [0, 1] → [0, 255] for EfficientNet pretrained weights"""
            # EfficientNet was trained on [0, 255] pixel values
            image = image * 255.0
            return image, label
        
        # Apply REDUCED augmentation BEFORE scaling (operate in [0, 1] for stability)
        def augment(image, label):
            # Augment in [0, 1] range (more stable)
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_flip_up_down(image)
            image = tf.image.rot90(image, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
            image = tf.image.random_brightness(image, 0.05)
            image = tf.image.random_contrast(image, 0.85, 1.15)
            # Clip to [0, 1] after augmentation
            image = tf.clip_by_value(image, 0.0, 1.0)
            return image, label
        
        print(f"  🔄 Applying data augmentation to training set...")
        train_ds = train_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        print(f"  🔄 Scaling to EfficientNet range [0, 255] for pretrained weights...")
        train_ds = train_ds.map(scale_to_efficientnet_range, num_parallel_calls=tf.data.AUTOTUNE)
        
        print(f"  🔄 Scaling validation set to EfficientNet range [0, 255]...")
        val_ds = val_ds.map(scale_to_efficientnet_range, num_parallel_calls=tf.data.AUTOTUNE)
        
        print(f"  ✅ Data ready: [0, 1] → augmented → [0, 255] (EfficientNet compatible)")
        
        return train_ds, val_ds, target_size


# ============================================
# MODEL BUILDER
# ============================================

def create_transfer_model(model_name, input_shape, num_classes):
    """Create transfer learning model"""
    
    base_models = {
        'EfficientNetB0': applications.EfficientNetB0,
        'EfficientNetB3': applications.EfficientNetB3,
        'ResNet50V2': applications.ResNet50V2,
        'DenseNet121': applications.DenseNet121,
        'MobileNetV2': applications.MobileNetV2,
    }
    
    print(f"\n🏗️  Building {model_name}...")
    
    base_model = base_models[model_name](
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = False
    
    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation='relu')(x)  # Increased from 256
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)  # Removed second 0.5 dropout - less aggressive
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)
    
    model = keras.Model(inputs, outputs, name=model_name)
    
    # XLA enabled globally - no per-model jit_compile needed
    
    total_params = model.count_params()
    trainable_params = sum([tf.size(var).numpy() for var in model.trainable_variables])
    print(f"  Parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    return model


# ============================================
# TRAINING CALLBACK
# ============================================

class OptimizedTrainingCallback(keras.callbacks.Callback):
    """Monitor training with detailed GPU stats"""
    
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.epoch_start = None
        self.batch_times = []
    
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()
        self.batch_times = []
    
    def on_train_batch_end(self, batch, logs=None):
        if hasattr(self, 'batch_start'):
            self.batch_times.append(time.time() - self.batch_start)
        self.batch_start = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.time() - self.epoch_start
        logs = logs or {}
        
        # Calculate throughput
        if self.batch_times:
            avg_batch_time = np.mean(self.batch_times)
            samples_per_sec = Config.BATCH_SIZE / avg_batch_time
        else:
            samples_per_sec = 0
        
        # Safely handle validation metrics (protect against NaN/missing)
        val_loss = logs.get('val_loss')
        val_acc = logs.get('val_accuracy')
        
        print(f"\n📊 Epoch {epoch+1}:")
        print(f"   Loss: {logs.get('loss', 0):.4f} | Val Loss: {val_loss if val_loss is not None else float('nan'):.4f}")
        print(f"   Acc: {logs.get('accuracy', 0):.4f} | Val Acc: {val_acc if val_acc is not None else float('nan'):.4f}")
        print(f"   LR: {float(self.model.optimizer.learning_rate):.2e}")
        print(f"   Time: {elapsed:.1f}s | Throughput: {samples_per_sec:.0f} samples/sec")
        
        # GPU status
        print_gpu_status("after epoch")
        
        gc.collect()


# ============================================
# TRAINING FUNCTION
# ============================================

def train_model_optimized(model, model_name, train_ds, val_ds, output_dir):
    """Train with all optimizations"""
    
    output_dir = Path(output_dir)
    model_dir = output_dir / model_name
    model_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\n{'='*70}")
    print(f"🎯 Training {model_name} (OPTIMIZED)")
    print(f"{'='*70}")
    print(f"  Batch size: {Config.BATCH_SIZE}")
    print(f"  Effective batch: {Config.BATCH_SIZE * Config.GRADIENT_ACCUMULATION_STEPS}")
    
    print_gpu_status("before training")
    
    # Calculate steps (CRITICAL for validation to work correctly)
    steps_per_epoch = 1000  # 64,000 / 64
    validation_steps = 125   # 8,000 / 64
    
    # ==================== PHASE 1 ====================
    print(f"\n🔒 PHASE 1: Feature extraction")
    
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=Config.LR_PHASE1,
            clipnorm=1.0  # Gradient clipping for stability
        ),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
        # Removed jit_compile - using global XLA only
        # Note: label_smoothing not supported in this TF version
    )
    
    print(f"  ✅ LR: {float(model.optimizer.learning_rate):.2e}")
    print(f"  ✅ Steps per epoch: {steps_per_epoch}, Validation steps: {validation_steps}")
    
    callbacks_p1 = [
        OptimizedTrainingCallback(f"{model_name}-Phase1"),
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,  # Increased from 8 - give model more time to improve after LR reduction
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(model_dir / 'phase1_best.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,  # Balanced - not too aggressive (was 5), not too slow
            min_lr=5e-6,  # Prevent going too small
            verbose=1
        ),
        keras.callbacks.CSVLogger(str(model_dir / 'phase1_history.csv'))
    ]
    
    # Load class weights for imbalanced dataset (passed as parameter)
    class_weight = output_dir.parent / 'class_weights.json'
    class_weights_dict = None
    if class_weight.exists():
        with open(class_weight, 'r') as f:
            class_weights_dict = {int(k): v for k, v in json.load(f).items()}
            print(f"  ✅ Using class weights for imbalanced dataset")
    
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        epochs=Config.EPOCHS_PHASE1,
        callbacks=callbacks_p1,
        class_weight=class_weights_dict,
        verbose=1
    )
    
    best_phase1 = max(history1.history['val_accuracy'])
    print(f"\n✅ Phase 1: {best_phase1:.4f}")
    
    # ==================== PHASE 2 ====================
    print(f"\n🔓 PHASE 2: Fine-tuning")
    
    # Unfreeze (more aggressive for 64k images)
    base_model = model.layers[1]
    total_layers = len(base_model.layers)
    unfreeze_from = int(total_layers * 0.5)  # Changed from 0.7 to 0.5
    
    for layer in base_model.layers[unfreeze_from:]:
        layer.trainable = True
    
    trainable_params = sum([tf.size(var).numpy() for var in model.trainable_variables])
    print(f"  Unfrozen: {total_layers - unfreeze_from}/{total_layers} layers")
    print(f"  Trainable: {trainable_params:,}")
    
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=Config.LR_PHASE2,
            clipnorm=1.0  # Gradient clipping for stability
        ),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
        # Removed jit_compile - using global XLA only
        # Note: label_smoothing not supported in this TF version
    )
    
    print(f"  ✅ LR: {float(model.optimizer.learning_rate):.2e}")
    
    callbacks_p2 = [
        OptimizedTrainingCallback(f"{model_name}-Phase2"),
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(model_dir / 'best_model.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,  # Slightly reduced from 12 for Phase 2
            min_lr=1e-5,  # Higher minimum for Phase 2 fine-tuning
            verbose=1
        ),
        keras.callbacks.CSVLogger(str(model_dir / 'phase2_history.csv'))
    ]
    
    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        epochs=Config.EPOCHS_PHASE2,
        callbacks=callbacks_p2,
        class_weight=class_weights_dict,
        verbose=1
    )
    
    # Save
    final_model_path = model_dir / 'final_model.keras'
    model.save(final_model_path)
    
    best_phase2 = max(history2.history['val_accuracy'])
    print(f"\n✅ Complete! Phase1: {best_phase1:.4f} | Phase2: {best_phase2:.4f}")
    
    return model, {
        'model_name': model_name,
        'best_val_accuracy': float(best_phase2),
        'phase1_best': float(best_phase1),
        'phase2_best': float(best_phase2),
        'history_phase1': {k: [float(v) for v in vals] for k, vals in history1.history.items()},
        'history_phase2': {k: [float(v) for v in vals] for k, vals in history2.history.items()},
        'model_path': str(final_model_path),
        'success': True
    }


# ============================================
# SEQUENTIAL TRAINING
# ============================================

def train_all_models(class_names, output_dir, models_to_train, loader):
    """Train all models"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    all_results = {}
    
    for i, model_name in enumerate(models_to_train, 1):
        print(f"\n{'#'*70}")
        print(f"# MODEL {i}/{len(models_to_train)}: {model_name}")
        print(f"{'#'*70}")
        
        try:
            train_ds, val_ds, input_shape = loader.get_dataset_for_model(model_name)
            model = create_transfer_model(model_name, input_shape, len(class_names))
            trained_model, results = train_model_optimized(
                model, model_name, train_ds, val_ds, output_dir
            )
            
            all_results[model_name] = results
            
            # Save
            with open(output_dir / f'{model_name}_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            
        except Exception as e:
            print(f"\n❌ Failed: {e}")
            import traceback
            traceback.print_exc()
            all_results[model_name] = {'success': False, 'error': str(e)}
        
        finally:
            # Cleanup
            if 'model' in locals():
                del model
            if 'trained_model' in locals():
                del trained_model
            gc.collect()
            keras.backend.clear_session()
            print("\n🧹 Cleanup complete")
    
    return all_results


# ============================================
# MAIN
# ============================================

def main():
    print("\n" + "="*70)
    print("🚀 OPTIMIZED TRANSFER LEARNING - FIXED VERSION")
    print("="*70)
    print(f"Batch size: {Config.BATCH_SIZE}")
    print(f"Effective batch: {Config.BATCH_SIZE * Config.GRADIENT_ACCUMULATION_STEPS}")
    print(f"Parallel workers: {Config.NUM_PARALLEL_CALLS}")
    print("="*70)
    
    configure_gpu()
    print_gpu_status("initial")
    
    data_dir = Config.get_output_dir()
    output_dir = Config.get_transfer_dir()
    output_dir.mkdir(exist_ok=True, parents=True)
    
    loader = OptimizedDataLoader(data_dir)
    split_info = loader.load_split_info()
    
    # Calculate class weights for imbalanced dataset (CRITICAL)
    print("\n📊 Calculating class weights for imbalanced dataset...")
    y_train_full = np.load(data_dir / 'y_train.npy', allow_pickle=False)
    unique_classes, class_counts = np.unique(y_train_full, return_counts=True)
    print(f"  Class distribution: {dict(zip(unique_classes, class_counts))}")
    
    # Compute balanced class weights
    max_count = class_counts.max()
    class_weights_dict = {int(cls): float(max_count / count) for cls, count in zip(unique_classes, class_counts)}
    print(f"  Class weights: {class_weights_dict}")
    
    # Save class weights
    class_weights_path = output_dir / 'class_weights.json'
    with open(class_weights_path, 'w') as f:
        json.dump(class_weights_dict, f, indent=2)
    
    del y_train_full
    gc.collect()
    
    models_to_train = [
        'EfficientNetB0'
    ]
    
    print(f"\n🎯 Models: {models_to_train}")
    
    all_results = train_all_models(
        loader.class_names, output_dir, models_to_train, loader
    )
    
    # Summary
    with open(output_dir / 'training_summary.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*70}")
    print("🎉 COMPLETE!")
    print(f"{'='*70}")
    
    for name, res in all_results.items():
        if res.get('success'):
            print(f"  {name}: {res['phase2_best']:.4f}")


if __name__ == "__main__":
    main()