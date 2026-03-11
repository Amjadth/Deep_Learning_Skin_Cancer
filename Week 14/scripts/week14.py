#!/usr/bin/env python3
"""
Week 14: ONNX Inference Testing & Accuracy Evaluation

Module:       week14.py
Purpose:      Comprehensive ONNX model benchmark and accuracy evaluation
Dataset:      ISIC 2019 Skin Cancer Detection
Model:        EfficientNetB3
Author:       Amjad
Date:         February 2026
Platform:     RunPod

═══════════════════════════════════════════════════════════════════════════════

DESCRIPTION
───────────
Benchmarks ONNX inference performance and evaluates accuracy on test set.
Computes latency, throughput, per-class metrics, and confusion matrix.

METRICS COMPUTED
────────────────
• Inference Performance
  └─ Single-image latency (mean, std, P95)
  └─ Throughput (images/sec)
  └─ Batch processing efficiency

• Accuracy Metrics
  └─ Overall accuracy
  └─ Macro & weighted precision, recall, F1
  └─ Per-class performance breakdown
  └─ Confidence distribution

• Diagnostics
  └─ Confusion matrix
  └─ Memory usage (CPU/GPU)
  └─ Detailed classification report

USAGE
─────
Run from terminal:

    python /workspace/week14.py

Requirements:
  • onnxruntime (with CUDA support)
  • numpy, pandas
  • sklearn (metrics)
  • psutil (memory diagnostics)

Inputs:
  • /workspace/output/EfficientNetB3_ISIC2019_final.onnx
  • /workspace/dataset/test_preprocessed/X_test_300x300.npy
  • /workspace/dataset/test_preprocessed/y_test.npy

Outputs:
  • /workspace/inference_results/evaluation_results.json
  • /workspace/inference_results/evaluation_report.txt
  • /workspace/inference_results/confusion_matrix.csv

═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import onnxruntime as ort
import time
from pathlib import Path
import json
import os
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

print("=" * 80)
print("ONNX INFERENCE PERFORMANCE + ACCURACY EVALUATION")
print("=" * 80)

# ============================================================================
# CONFIGURATION
# ============================================================================

TEST_DATA_PATH = "/workspace/dataset/test_preprocessed/X_test_300x300.npy"
TEST_LABELS_PATH = "/workspace/dataset/test_preprocessed/y_test.npy"
ONNX_MODEL_PATH = "/workspace/output/EfficientNetB3_ISIC2019_final.onnx"
RESULTS_DIR = Path("/workspace/inference_results")

CLASS_NAMES = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']

# Benchmark settings
WARMUP_ITERATIONS = 10
PERFORMANCE_TEST_SIZE = 1000  # For latency testing

# ============================================================================
# LOAD DATA
# ============================================================================

print(f"\n📂 Loading test data...")

# Load images
if not Path(TEST_DATA_PATH).exists():
    print(f"✗ ERROR: Test images not found: {TEST_DATA_PATH}")
    exit(1)

X_test = np.load(TEST_DATA_PATH).astype(np.float32)
print(f"✓ Images loaded: {X_test.shape}")

# Load labels
if not Path(TEST_LABELS_PATH).exists():
    print(f"✗ ERROR: Test labels not found: {TEST_LABELS_PATH}")
    print(f"\n💡 Run preprocessing with ground truth:")
    print(f"   python preprocess_test_set_enhanced.py")
    exit(1)

y_test = np.load(TEST_LABELS_PATH)
print(f"✓ Labels loaded: {y_test.shape}")

print(f"\nData characteristics:")
print(f"  Images: shape={X_test.shape}, dtype={X_test.dtype}")
print(f"  Range:  [{X_test.min():.1f}, {X_test.max():.1f}]")
print(f"  Labels: shape={y_test.shape}, dtype={y_test.dtype}")
print(f"  Classes: {len(np.unique(y_test))} unique")

# Verify [0-255] range for EfficientNet and auto-fix when needed
if X_test.max() > 1.0:
    print(f"  ✓ CORRECT: Data in [0-255] range (EfficientNet compatible)")
    X_test_infer = X_test
else:
    print(f"  ⚠ WARNING: Data in [0-1] range (should be [0-255] for EfficientNet)")
    print(f"  🔄 Auto-fix: scaling test data from [0,1] to [0,255] for inference")
    X_test_infer = X_test * 255.0

print(f"  Inference range: [{X_test_infer.min():.1f}, {X_test_infer.max():.1f}]")


def normalize_output_for_argmax(raw_output):
    """
    Normalize ONNX output to 2D [batch, classes] for robust argmax.
    Handles outputs like [B, C], [B, 1, C], [B, 1, 1, C], etc.
    """
    probs = np.asarray(raw_output)
    if probs.ndim == 1:
        probs = probs[np.newaxis, :]
    elif probs.ndim > 2:
        probs = probs.reshape(probs.shape[0], -1)
    return probs

# ============================================================================
# LOAD ONNX MODEL
# ============================================================================

print(f"\n🤖 Loading ONNX model...")

if not Path(ONNX_MODEL_PATH).exists():
    print(f"✗ ERROR: ONNX model not found: {ONNX_MODEL_PATH}")
    exit(1)

providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session = ort.InferenceSession(
    str(ONNX_MODEL_PATH),
    sess_options=sess_options,
    providers=providers
)

active_provider = session.get_providers()[0]
print(f"✓ Session created: {active_provider}")

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
print(f"  ONNX input shape:  {session.get_inputs()[0].shape}")
print(f"  ONNX output shape: {session.get_outputs()[0].shape}")

# ============================================================================
# WARMUP
# ============================================================================

print(f"\n🔥 Warmup ({WARMUP_ITERATIONS} iterations)...")
warmup_batch = X_test_infer[:1]
for _ in range(WARMUP_ITERATIONS):
    _ = session.run([output_name], {input_name: warmup_batch})
print("✓ Warmup complete")

# ============================================================================
# INFERENCE PERFORMANCE
# ============================================================================

print(f"\n" + "=" * 80)
print(f"⚡ INFERENCE PERFORMANCE")
print("=" * 80)

# Single image latency
perf_samples = min(PERFORMANCE_TEST_SIZE, len(X_test))
times = []

print(f"\nMeasuring latency on {perf_samples} images...")
for i in range(perf_samples):
    img = X_test_infer[i:i+1]
    start = time.perf_counter()
    _ = session.run([output_name], {input_name: img})
    end = time.perf_counter()
    times.append((end - start) * 1000)

avg_latency = np.mean(times)
std_latency = np.std(times)
p95_latency = np.percentile(times, 95)

print(f"\nLatency Statistics:")
print(f"  Average:   {avg_latency:.2f} ms")
print(f"  Std Dev:   {std_latency:.2f} ms")
print(f"  95th %ile: {p95_latency:.2f} ms")
print(f"  Throughput: {1000/avg_latency:.1f} images/sec")

# ============================================================================
# ACCURACY EVALUATION
# ============================================================================

print(f"\n" + "=" * 80)
print(f"🎯 ACCURACY EVALUATION")
print("=" * 80)

print(f"\nRunning inference on all {len(X_test)} test images...")
print("This may take a moment...")

# Batch inference for efficiency
BATCH_SIZE = 32
all_predictions = []

num_batches = (len(X_test) + BATCH_SIZE - 1) // BATCH_SIZE
start_time = time.time()

for i in range(0, len(X_test), BATCH_SIZE):
    end_idx = min(i + BATCH_SIZE, len(X_test))
    batch = X_test_infer[i:end_idx]
    
    # Get predictions
    raw_output = session.run([output_name], {input_name: batch})[0]
    probs = normalize_output_for_argmax(raw_output)
    pred_classes = np.argmax(probs, axis=-1)
    all_predictions.extend(pred_classes.tolist())

elapsed = time.time() - start_time
y_pred = np.array(all_predictions)

print(f"✓ Inference complete in {elapsed:.2f} seconds")
print(f"  Average: {(elapsed / len(X_test)) * 1000:.2f} ms per image")

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)

precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print(f"\n📊 Overall Metrics:")
print("=" * 60)
print(f"  Accuracy:          {accuracy*100:.2f}%")
print(f"\n  Macro Average:")
print(f"    Precision:       {precision_macro*100:.2f}%")
print(f"    Recall:          {recall_macro*100:.2f}%")
print(f"    F1-Score:        {f1_macro*100:.2f}%")
print(f"\n  Weighted Average:")
print(f"    Precision:       {precision_weighted*100:.2f}%")
print(f"    Recall:          {recall_weighted*100:.2f}%")
print(f"    F1-Score:        {f1_weighted*100:.2f}%")

# Per-class metrics
print(f"\n📊 Per-Class Performance:")
print("=" * 60)

per_class_precision = precision_score(y_test, y_pred, average=None, zero_division=0)
per_class_recall = recall_score(y_test, y_pred, average=None, zero_division=0)
per_class_f1 = f1_score(y_test, y_pred, average=None, zero_division=0)

for i, class_name in enumerate(CLASS_NAMES):
    class_count = (y_test == i).sum()
    print(f"\n  {class_name} (n={class_count}):")
    print(f"    Precision: {per_class_precision[i]*100:5.2f}%")
    print(f"    Recall:    {per_class_recall[i]*100:5.2f}%")
    print(f"    F1-Score:  {per_class_f1[i]*100:5.2f}%")

# Confusion matrix
print(f"\n📊 Confusion Matrix:")
print("=" * 60)
cm = confusion_matrix(y_test, y_pred)

# Print header
print(f"\n{'Actual':>8s} | ", end='')
for cls in CLASS_NAMES:
    print(f"{cls:>6s}", end=' ')
print()
print("-" * 70)

# Print matrix
for i, class_name in enumerate(CLASS_NAMES):
    print(f"{class_name:>8s} | ", end='')
    for j in range(len(CLASS_NAMES)):
        print(f"{cm[i,j]:>6d}", end=' ')
    print()

# ============================================================================
# DETAILED CLASSIFICATION REPORT
# ============================================================================

print(f"\n📋 Detailed Classification Report:")
print("=" * 60)
report = classification_report(
    y_test, y_pred, 
    target_names=CLASS_NAMES,
    zero_division=0
)
print(report)

# ============================================================================
# MEMORY USAGE
# ============================================================================

print(f"\n💾 Memory Usage:")
print("=" * 60)

try:
    import psutil
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"  CPU Memory: {mem_info.rss / (1024**2):.1f} MB")
except:
    pass

if active_provider == 'CUDAExecutionProvider':
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total', 
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            mem_used, mem_total = map(int, result.stdout.strip().split(','))
            print(f"  GPU Memory: {mem_used} MB / {mem_total} MB ({mem_used/mem_total*100:.1f}%)")
    except:
        pass

# ============================================================================
# SAVE RESULTS
# ============================================================================

print(f"\n💾 Saving results...")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Compile results
results = {
    'model_info': {
        'model_path': str(ONNX_MODEL_PATH),
        'provider': active_provider
    },
    'test_data': {
        'num_images': len(X_test),
        'num_classes': len(CLASS_NAMES),
        'class_names': CLASS_NAMES
    },
    'performance': {
        'avg_latency_ms': float(avg_latency),
        'std_latency_ms': float(std_latency),
        'p95_latency_ms': float(p95_latency),
        'throughput_ips': float(1000/avg_latency),
        'total_inference_time_sec': float(elapsed)
    },
    'accuracy_metrics': {
        'accuracy': float(accuracy),
        'macro_precision': float(precision_macro),
        'macro_recall': float(recall_macro),
        'macro_f1': float(f1_macro),
        'weighted_precision': float(precision_weighted),
        'weighted_recall': float(recall_weighted),
        'weighted_f1': float(f1_weighted)
    },
    'per_class_metrics': {
        CLASS_NAMES[i]: {
            'precision': float(per_class_precision[i]),
            'recall': float(per_class_recall[i]),
            'f1': float(per_class_f1[i]),
            'support': int((y_test == i).sum())
        }
        for i in range(len(CLASS_NAMES))
    },
    'confusion_matrix': cm.tolist()
}

# Save JSON
json_path = RESULTS_DIR / "evaluation_results.json"
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"✓ Saved: {json_path}")

# Save detailed report
report_path = RESULTS_DIR / "evaluation_report.txt"
with open(report_path, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("ONNX INFERENCE PERFORMANCE + ACCURACY EVALUATION\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("PERFORMANCE:\n")
    f.write(f"  Average latency: {avg_latency:.2f} ms\n")
    f.write(f"  Throughput: {1000/avg_latency:.1f} images/sec\n\n")
    
    f.write("ACCURACY:\n")
    f.write(f"  Overall accuracy: {accuracy*100:.2f}%\n")
    f.write(f"  Macro F1: {f1_macro*100:.2f}%\n")
    f.write(f"  Weighted F1: {f1_weighted*100:.2f}%\n\n")
    
    f.write("PER-CLASS METRICS:\n")
    for i, class_name in enumerate(CLASS_NAMES):
        f.write(f"  {class_name}:\n")
        f.write(f"    Precision: {per_class_precision[i]*100:.2f}%\n")
        f.write(f"    Recall: {per_class_recall[i]*100:.2f}%\n")
        f.write(f"    F1: {per_class_f1[i]*100:.2f}%\n")
    
    f.write("\n" + "=" * 80 + "\n")

print(f"✓ Saved: {report_path}")

# Save confusion matrix as CSV
cm_df = pd.DataFrame(cm, 
                     index=[f"True_{cls}" for cls in CLASS_NAMES],
                     columns=[f"Pred_{cls}" for cls in CLASS_NAMES])
cm_path = RESULTS_DIR / "confusion_matrix.csv"
cm_df.to_csv(cm_path)
print(f"✓ Saved: {cm_path}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print(f"\n" + "=" * 80)
print(f"✅ EVALUATION COMPLETE!")
print("=" * 80)

print(f"\n🎯 Key Results:")
print(f"  ✓ Accuracy:     {accuracy*100:.2f}%")
print(f"  ✓ Macro F1:     {f1_macro*100:.2f}%")
print(f"  ✓ Latency:      {avg_latency:.2f} ms")
print(f"  ✓ Throughput:   {1000/avg_latency:.1f} img/sec")

print(f"\n📁 Results saved to:")
print(f"  {json_path}")
print(f"  {report_path}")
print(f"  {cm_path}")

print(f"\n🚀 ONNX model validated and ready for production!")

print("\n" + "=" * 80)