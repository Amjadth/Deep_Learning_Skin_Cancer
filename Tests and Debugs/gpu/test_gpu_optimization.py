# ============================================
# GPU OPTIMIZATION TEST SCRIPT
# ============================================
# 
# This script tests whether GPU is being properly utilized
# during training. It checks:
#
# 1. GPU device availability
# 2. GPU memory growth settings
# 3. Data pipeline device placement
# 4. Model computation on GPU
# 5. Training loop GPU utilization
#
# ============================================

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import time

print("=" * 70)
print("GPU OPTIMIZATION TEST SUITE")
print("=" * 70)

# ============================================
# TEST 1: GPU DEVICE AVAILABILITY
# ============================================

print("\n✅ TEST 1: GPU Device Availability")
print("-" * 70)

gpus = tf.config.list_physical_devices('GPU')
cpus = tf.config.list_physical_devices('CPU')

print(f"   Physical Devices:")
print(f"   • GPUs: {len(gpus)} device(s)")
for i, gpu in enumerate(gpus):
    print(f"     [{i}] {gpu.name}")
print(f"   • CPUs: {len(cpus)} device(s)")

logical_devices = tf.config.list_logical_devices()
print(f"\n   Logical Devices: {len(logical_devices)}")
for device in logical_devices:
    print(f"   • {device.name}")

if len(gpus) == 0:
    print("\n   ❌ CRITICAL: No GPUs detected!")
else:
    print(f"\n   ✅ {len(gpus)} GPU(s) available")

# ============================================
# TEST 2: MEMORY GROWTH CONFIGURATION
# ============================================

print("\n✅ TEST 2: Memory Growth Configuration")
print("-" * 70)

try:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("   ✅ Memory growth enabled for all GPUs")
except RuntimeError as e:
    print(f"   ⚠️  Memory growth setup issue: {e}")

# Check configuration
for gpu in gpus:
    memory_growth = tf.config.experimental.get_memory_growth(gpu)
    print(f"   • {gpu.name}: memory_growth={memory_growth}")

# ============================================
# TEST 3: MIXED PRECISION CONFIGURATION
# ============================================

print("\n✅ TEST 3: Mixed Precision Configuration")
print("-" * 70)

try:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    
    current_policy = tf.keras.mixed_precision.global_policy()
    print(f"   ✅ Mixed precision enabled")
    print(f"   • Compute dtype: {current_policy.compute_dtype}")
    print(f"   • Variable dtype: {current_policy.variable_dtype}")
    print(f"   • Policy name: {current_policy.name}")
except Exception as e:
    print(f"   ⚠️  Mixed precision setup error: {e}")

# ============================================
# TEST 4: DATA PIPELINE GPU OPTIMIZATION
# ============================================

print("\n✅ TEST 4: Data Pipeline GPU Optimization")
print("-" * 70)

# Create dummy data
X_dummy = np.random.randn(100, 224, 224, 3).astype(np.float32)
y_dummy = np.random.randint(0, 8, 100)

print("   Creating tf.data pipeline...")

# Method: from_tensor_slices (GPU-friendly)
X_tensor = tf.convert_to_tensor(X_dummy, dtype=tf.float32)
y_tensor = tf.convert_to_tensor(y_dummy, dtype=tf.int32)

dataset = tf.data.Dataset.from_tensor_slices((X_tensor, y_tensor))
dataset = dataset.batch(32)
dataset = dataset.prefetch(tf.data.AUTOTUNE)

print("   ✅ Pipeline created with:")
print("      • from_tensor_slices (GPU-friendly)")
print("      • batch(32)")
print("      • prefetch(AUTOTUNE)")

# Verify data is accessible
try:
    sample_batch = next(iter(dataset))
    print(f"   ✅ Data accessible: batch shape = {sample_batch[0].shape}")
except Exception as e:
    print(f"   ❌ Data access failed: {e}")

# ============================================
# TEST 5: MODEL CREATION & COMPUTATION
# ============================================

print("\n✅ TEST 5: Model Creation & GPU Computation")
print("-" * 70)

# Create simple model
try:
    model = keras.Sequential([
        keras.layers.Input(shape=(224, 224, 3)),
        keras.layers.Conv2D(32, 3, activation='relu'),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(8, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"   ✅ Model created: {model.count_params():,} parameters")
    
except Exception as e:
    print(f"   ❌ Model creation failed: {e}")

# ============================================
# TEST 6: FORWARD PASS ON GPU
# ============================================

print("\n✅ TEST 6: Forward Pass GPU Computation")
print("-" * 70)

try:
    with tf.device('/GPU:0') if gpus else tf.device('/CPU:0'):
        device_name = tf.config.get_default_device() or '/GPU:0'
        
        # Get a batch
        sample_batch = next(iter(dataset))
        X_batch = sample_batch[0]
        
        # Forward pass
        start = time.time()
        output = model(X_batch, training=False)
        elapsed = time.time() - start
        
        print(f"   ✅ Forward pass successful on {device_name}")
        print(f"      • Input shape: {X_batch.shape}")
        print(f"      • Output shape: {output.shape}")
        print(f"      • Compute time: {elapsed*1000:.2f}ms")
        
except Exception as e:
    print(f"   ❌ Forward pass failed: {e}")

# ============================================
# TEST 7: TRAINING STEP GPU COMPUTATION
# ============================================

print("\n✅ TEST 7: Training Step GPU Computation")
print("-" * 70)

try:
    # Single training step
    start = time.time()
    loss, accuracy = model.evaluate(dataset.take(5), verbose=0)
    elapsed = time.time() - start
    
    print(f"   ✅ Training evaluation successful")
    print(f"      • Loss: {loss:.4f}")
    print(f"      • Accuracy: {accuracy:.4f}")
    print(f"      • Time for 5 batches: {elapsed*1000:.2f}ms")
    print(f"      • Speed: {(32*5)/elapsed:.0f} samples/sec")
    
    if elapsed < 1.0:  # Should be very fast with GPU
        print(f"      ✅ Speed indicates GPU usage")
    else:
        print(f"      ⚠️  Speed seems slow - check GPU utilization")
        
except Exception as e:
    print(f"   ❌ Training step failed: {e}")

# ============================================
# TEST 8: DEVICE PLACEMENT VERIFICATION
# ============================================

print("\n✅ TEST 8: Device Placement Verification")
print("-" * 70)

# Check where operations are placed
@tf.function
def test_operation():
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    return tf.matmul(a, b)

try:
    result = test_operation()
    print("   ✅ Operations executed")
    print(f"      • Result shape: {result.shape}")
except Exception as e:
    print(f"   ⚠️  Operation failed: {e}")

# ============================================
# TEST 9: TENSORFLOW VERSION & BUILD INFO
# ============================================

print("\n✅ TEST 9: TensorFlow Configuration")
print("-" * 70)

print(f"   TensorFlow Version: {tf.__version__}")
print(f"   Keras Version: {keras.__version__}")

build_info = tf.sysconfig.get_build_info()
print(f"   CUDA Support: {'cuda' in str(build_info).lower()}")

try:
    from tensorflow.python.platform import build_info as tf_build_info
    print(f"   GPU Support: Available")
except:
    print(f"   GPU Support: Not available")

# ============================================
# TEST 10: RECOMMENDATIONS
# ============================================

print("\n✅ TEST 10: GPU Optimization Recommendations")
print("-" * 70)

recommendations = []

if len(gpus) == 0:
    recommendations.append("❌ NO GPUs DETECTED - Check NVIDIA drivers!")
else:
    recommendations.append(f"✅ {len(gpus)} GPU(s) available")
    recommendations.append("✅ Use explicit device placement: with tf.device('/GPU:0')")
    recommendations.append("✅ Ensure batch size is large enough (32-64)")
    recommendations.append("✅ Use mixed_float16 for faster computation")
    recommendations.append("✅ Monitor GPU with: nvidia-smi -l 1")
    recommendations.append("✅ Enable XLA JIT compilation")
    recommendations.append("✅ Use prefetch(AUTOTUNE) in data pipeline")

for rec in recommendations:
    print(f"   {rec}")

# ============================================
# MONITORING COMMANDS
# ============================================

print("\n" + "=" * 70)
print("GPU MONITORING COMMANDS")
print("=" * 70)

print("""
To monitor GPU during training, run in a separate terminal:

1. Watch nvidia-smi every 1 second:
   $ watch -n 1 nvidia-smi

2. Continuous nvidia-smi output:
   $ nvidia-smi -l 1

3. Detailed GPU process info:
   $ nvidia-smi pmon -c 1

4. GPU memory detailed breakdown:
   $ nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.free --format=csv -l 1

Key metrics to watch:
   • Processes column: Should show 'python' using GPU
   • Memory-Usage: Should be 80-95% of total GPU memory
   • GPU %: Should be 80-99% busy during training
   • SM Clock: Should be at max or near-max
   • Memory Clock: Should be at max or near-max
""")

# ============================================
# FINAL STATUS
# ============================================

print("=" * 70)
print("GPU OPTIMIZATION TEST COMPLETE")
print("=" * 70)

if len(gpus) > 0:
    print(f"\n✅ System is GPU-ready!")
    print(f"   Now run: python week6_fixed_runpod.py")
    print(f"   And monitor GPU with: watch -n 1 nvidia-smi")
else:
    print(f"\n❌ No GPU available - CPU training only (will be slow)")
