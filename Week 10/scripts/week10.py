#!/usr/bin/env python3
# ============================================
# WEEK 10: CORRECTED MODEL EVALUATION - A40 GPU OPTIMIZED
# ============================================
#
# FIXES FROM ORIGINAL WEEK 10:
# ✅ Accurate model path tracking (actual saved locations)
# ✅ Correct input sizes per model (224x224 vs 300x300)
# ✅ Proper test data preprocessing matching training normalization
# ✅ Week 8 regularization models EXCLUDED (as requested)
# ✅ A40 GPU (not A6000) optimized configuration
# ✅ Verified model loading from actual training scripts
#
# MODELS TESTED:
# 1. Week 6: Baseline CNN (224×224 denormalized data)
# 2. Week 7: Tuned Baseline (224×224 denormalized data)
# 3. Week 9: EfficientNetB0 (224×224)
# 4. Week 9: EfficientNetB3 (300×300 - SPECIAL HANDLING)
#
# EXCLUDED:
# ❌ Week 8 Regularization models (as requested)
#
# TEST DATA HANDLING:
# - Week 6/7 models: Use X_test_denormalized.npy (or create if missing)
# - Week 9 B0: Use X_test_denormalized.npy (same input size)
# - Week 9 B3: Use X_test_300.npy (or create from X_test via downscaling)
#
# GPU: NVIDIA A40 (48GB VRAM)
# ============================================

import os
import gc
import json
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, mixed_precision
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import subprocess

warnings.filterwarnings('ignore')

print("=" * 80)
print("WEEK 10: CORRECTED MODEL EVALUATION - A40 GPU OPTIMIZED")
print("=" * 80)

# ============================================
# CONFIGURATION
# ============================================

class Config:
    """A40-optimized configuration"""
    
    GPU_MEMORY_LIMIT = 40 * 1024  # MB (40GB)
    BATCH_SIZE_EVAL = 64  # Conservative for A40
    
    # Paths
    BASE_DIR = Path('/workspace') if Path('/workspace').exists() else Path.cwd()
    NETWORK_VOLUME = Path('/runpod-volume') if Path('/runpod-volume').exists() else None
    
    STORAGE_BASE = NETWORK_VOLUME if NETWORK_VOLUME else BASE_DIR
    OUTPUT_DIR = (STORAGE_BASE / 'outputs').resolve()
    WEEK10_DIR = (OUTPUT_DIR / 'week10_evaluation_corrected').resolve()
    
    # Class names
    CLASS_NAMES = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']
    NUM_CLASSES = 8
    
    # Models to evaluate (WEEK 8 EXCLUDED)
    # Multiple Week 7 models from hyperparameter tuning - using best performer
    MODELS_TO_TEST = {
        'Baseline_Week6': {
            'path': OUTPUT_DIR / 'models' / 'denormalized_best_model.keras',
            'input_size': (224, 224, 3),
            'test_data': 'X_test_denormalized.npy',
            'description': 'Baseline CNN (Week 6) - 224×224 denormalized'
        },
        'Tuned_Week7_LR0.001_BS64': {
            'path': OUTPUT_DIR / 'tuning_results' / 'best_lr0.001_bs64.keras',
            'input_size': (224, 224, 3),
            'test_data': 'X_test_denormalized.npy',
            'description': 'Tuned Baseline (Week 7, LR=0.001, BS=64) - 224×224'
        },
        'Tuned_Week7_LR0.0001_BS64': {
            'path': OUTPUT_DIR / 'tuning_results' / 'best_lr0.0001_bs64.keras',
            'input_size': (224, 224, 3),
            'test_data': 'X_test_denormalized.npy',
            'description': 'Tuned Baseline (Week 7, LR=0.0001, BS=64) - 224×224'
        },
        'Tuned_Week7_LR0.001_BS128': {
            'path': OUTPUT_DIR / 'tuning_results' / 'best_lr0.001_bs128.keras',
            'input_size': (224, 224, 3),
            'test_data': 'X_test_denormalized.npy',
            'description': 'Tuned Baseline (Week 7, LR=0.001, BS=128) - 224×224'
        },
        'Tuned_Week7_LR0.0001_BS128': {
            'path': OUTPUT_DIR / 'tuning_results' / 'best_lr0.0001_bs128.keras',
            'input_size': (224, 224, 3),
            'test_data': 'X_test_denormalized.npy',
            'description': 'Tuned Baseline (Week 7, LR=0.0001, BS=128) - 224×224'
        },
        'Tuned_Week7_LR0.001_BS256': {
            'path': OUTPUT_DIR / 'tuning_results' / 'best_lr0.001_bs256.keras',
            'input_size': (224, 224, 3),
            'test_data': 'X_test_denormalized.npy',
            'description': 'Tuned Baseline (Week 7, LR=0.001, BS=256) - 224×224'
        },
        'Tuned_Week7_LR0.0001_BS256': {
            'path': OUTPUT_DIR / 'tuning_results' / 'best_lr0.0001_bs256.keras',
            'input_size': (224, 224, 3),
            'test_data': 'X_test_denormalized.npy',
            'description': 'Tuned Baseline (Week 7, LR=0.0001, BS=256) - 224×224'
        },
        'Tuned_Week7_LR0.0005_BS64': {
            'path': OUTPUT_DIR / 'tuning_results' / 'best_lr0.0005_bs64.keras',
            'input_size': (224, 224, 3),
            'test_data': 'X_test_denormalized.npy',
            'description': 'Tuned Baseline (Week 7, LR=0.0005, BS=64) - 224×224'
        },
        'Tuned_Week7_LR0.0005_BS128': {
            'path': OUTPUT_DIR / 'tuning_results' / 'best_lr0.0005_bs128.keras',
            'input_size': (224, 224, 3),
            'test_data': 'X_test_denormalized.npy',
            'description': 'Tuned Baseline (Week 7, LR=0.0005, BS=128) - 224×224'
        },
        'Tuned_Week7_LR0.0005_BS256': {
            'path': OUTPUT_DIR / 'tuning_results' / 'best_lr0.0005_bs256.keras',
            'input_size': (224, 224, 3),
            'test_data': 'X_test_denormalized.npy',
            'description': 'Tuned Baseline (Week 7, LR=0.0005, BS=256) - 224×224'
        },
        'EfficientNetB0_Week9': {
            'path': OUTPUT_DIR / 'transfer_learning_EB0' / 'EfficientNetB0' / 'best_model.keras',
            'input_size': (224, 224, 3),
            'test_data': 'X_test_denormalized.npy',
            'description': 'EfficientNetB0 (Week 9) - 224×224'
        },
        'EfficientNetB3_Week9': {
            'path': OUTPUT_DIR / 'transfer_learning_EB3' / 'EfficientNetB3' / 'best_model.keras',
            'input_size': (300, 300, 3),
            'test_data': 'X_test_300.npy',
            'description': 'EfficientNetB3 (Week 9) - 300×300'
        },
    }


# ============================================
# GPU CONFIGURATION
# ============================================

def configure_gpu():
    """Configure A40 GPU for inference"""
    
    print("\n🎮 Configuring A40 GPU for model evaluation...")
    
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("⚠️  No GPU detected!")
        return
    
    print(f"✅ Found {len(gpus)} GPU(s)")
    
    # Enable memory growth
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    # Set memory limit
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(
                memory_limit=Config.GPU_MEMORY_LIMIT
            )]
        )
        print(f"✅ GPU memory limit: {Config.GPU_MEMORY_LIMIT/1024:.0f} GB")
    except RuntimeError as e:
        print(f"⚠️  GPU config warning: {e}")


def print_gpu_status(label=""):
    """Print GPU status"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total',
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
            
            print(f"🎮 GPU {label}:")
            print(f"   Compute: {gpu_pct}% | Memory: {mem_pct}% ({mem_used}/{mem_total} MB)")
    except:
        pass


# ============================================
# DATA LOADING UTILITIES
# ============================================

def safe_load_npy_direct(filepath, expected_shape=None, description=""):
    """
    3-level fallback loading strategy for corrupted NPY files.
    FROM: Week9_EfficientNetB3.py
    
    Level 1: Standard np.load() with memmap
    Level 2: Allow pickle format
    Level 3: Direct memmap (bypasses corrupted headers)
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        print(f"   ❌ File not found: {filepath}")
        return None
    
    try:
        # Level 1: Try standard np.load first
        data = np.load(str(filepath), mmap_mode='r')
        print(f"     ✅ Loaded {description} with standard np.load()")
        return data
        
    except (ValueError, OSError) as e:
        print(f"     ⚠️  Standard load failed ({type(e).__name__}), trying allow_pickle...")
        
        try:
            # Level 2: Fallback with allow_pickle
            data = np.load(str(filepath), allow_pickle=True, mmap_mode='r')
            print(f"     ✅ Loaded {description} with allow_pickle=True")
            return data
            
        except Exception as e2:
            print(f"     ⚠️  Pickle load failed, trying direct memmap...")
            
            # Level 3: Direct memmap (last resort)
            if expected_shape:
                try:
                    data = np.memmap(
                        str(filepath),
                        dtype=np.float32,
                        mode='r',
                        shape=expected_shape
                    )
                    print(f"     ✅ Loaded {description} with direct memmap (bypassing headers)")
                    return data
                except Exception as e3:
                    print(f"     ❌ All loading methods failed: {e3}")
                    return None
            else:
                print(f"     ❌ Cannot use memmap fallback - no expected_shape provided")
                return None


def safe_load_npy(filepath, description, use_memmap=True, expected_shape=None):
    """Safely load NPY file with fallback mechanisms"""
    
    filepath = Path(filepath)
    
    if not filepath.exists():
        print(f"❌ {description} not found: {filepath}")
        return None
    
    try:
        if use_memmap:
            data = np.load(str(filepath), mmap_mode='r')
        else:
            data = np.load(str(filepath), allow_pickle=False)
        
        print(f"✅ Loaded {description}: {data.shape}")
        return data
        
    except Exception as e:
        print(f"⚠️  Standard load failed, using 3-level fallback...")
        # Use 3-level fallback for corrupted files
        return safe_load_npy_direct(filepath, expected_shape=expected_shape, description=description)


def ensure_test_data_exists(output_dir):
    """Ensure all test data files exist, create if missing"""
    
    print("\n📂 Checking test data files...")
    
    # Check X_test_denormalized
    denorm_path = output_dir / 'X_test_denormalized.npy'
    if not denorm_path.exists():
        print(f"⚠️  {denorm_path.name} not found, creating from X_test.npy...")
        
        x_test_path = output_dir / 'X_test.npy'
        if x_test_path.exists():
            # Load and denormalize
            # ImageNet normalization constants
            IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            
            print(f"   Loading {x_test_path.name}...")
            X_test = np.load(x_test_path, mmap_mode='r')
            
            print(f"   Denormalizing from ImageNet [-1,1] to [0,1]...")
            # X_test is currently in ImageNet normalized format: (x - mean) / std
            # Reverse: x = (x_norm * std) + mean
            X_test_denorm = np.zeros_like(X_test, dtype=np.float32)
            
            # Process in chunks to avoid memory issues
            chunk_size = 1000
            for i in tqdm(range(0, len(X_test), chunk_size), desc="Denormalizing"):
                end_idx = min(i + chunk_size, len(X_test))
                chunk = X_test[i:end_idx]
                
                # Denormalize
                for c in range(3):
                    X_test_denorm[i:end_idx, :, :, c] = (chunk[:, :, :, c] * IMAGENET_STD[c]) + IMAGENET_MEAN[c]
            
            # Clip to [0, 1]
            X_test_denorm = np.clip(X_test_denorm, 0, 1).astype(np.float32)
            
            print(f"   Saving denormalized test data...")
            np.save(denorm_path, X_test_denorm)
            print(f"✅ Created {denorm_path.name}")
        else:
            print(f"❌ X_test.npy not found, cannot create denormalized version")
    else:
        print(f"✅ {denorm_path.name} exists")
    
    # Check X_test_300
    test_300_path = output_dir / 'X_test_300.npy'
    if not test_300_path.exists():
        print(f"⚠️  {test_300_path.name} not found, creating from X_test_denormalized.npy...")
        
        if denorm_path.exists():
            from PIL import Image
            
            print(f"   Loading denormalized test data...")
            X_test_denorm = np.load(denorm_path, mmap_mode='r')
            
            print(f"   Downscaling to 300×300...")
            X_test_300 = np.zeros((len(X_test_denorm), 300, 300, 3), dtype=np.float32)
            
            chunk_size = 100
            for i in tqdm(range(0, len(X_test_denorm), chunk_size), desc="Downscaling"):
                end_idx = min(i + chunk_size, len(X_test_denorm))
                chunk = X_test_denorm[i:end_idx]
                
                for j in range(end_idx - i):
                    img = (chunk[j] * 255).astype(np.uint8)
                    img_pil = Image.fromarray(img)
                    img_resized = img_pil.resize((300, 300), Image.Resampling.BICUBIC)
                    X_test_300[i + j] = np.array(img_resized, dtype=np.float32) / 255.0
            
            print(f"   Saving 300×300 test data...")
            np.save(test_300_path, X_test_300)
            print(f"✅ Created {test_300_path.name}")
        else:
            print(f"❌ X_test_denormalized.npy not found, cannot create 300×300 version")
    else:
        print(f"✅ {test_300_path.name} exists")


# ============================================
# MODEL EVALUATION
# ============================================

def scale_test_data_for_model(X_test, model_name):
    """
    Scale test data from [0,1] to [0,255] for models that expect raw pixel values.
    FROM: Week9_EfficientNetB3.py & Week9_EfficientNetB0.py
    
    EfficientNet models were trained on [0,255] pixel values (ImageNet raw data)
    Other models may also expect this range for consistency.
    """
    
    # Models that need [0,255] scaling
    models_needing_255_scaling = [
        'EfficientNetB0_Week9',
        'EfficientNetB3_Week9',
        'Baseline_Week6',      # CNN trained on [0,255] range
        'Tuned_Week7'          # Tuned CNN also trained on [0,255] range
    ]
    
    if model_name in models_needing_255_scaling:
        print(f"   🔄 Scaling test data [0,1] → [0,255] for {model_name}")
        X_test_scaled = X_test * 255.0
        print(f"   ✅ Scaled to range [0,255]")
        return X_test_scaled.astype(np.float32)
    else:
        print(f"   ℹ️  No scaling needed for {model_name}")
        return X_test.astype(np.float32)


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model on test set"""
    
    print(f"\n📊 Evaluating {model_name}...")
    print(f"   Model input shape: {model.input_shape}")
    print(f"   Test data shape: {X_test.shape}")
    
    # Scale test data appropriately for model
    X_test_scaled = scale_test_data_for_model(X_test, model_name)
    
    # Create tf.data dataset
    ds_test = tf.data.Dataset.from_tensor_slices(X_test_scaled)
    ds_test = ds_test.batch(Config.BATCH_SIZE_EVAL)
    ds_test = ds_test.prefetch(2)
    
    # Get predictions
    print(f"   Running inference...")
    y_pred_proba = model.predict(ds_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Multi-class AUC
    try:
        y_test_bin = label_binarize(y_test, classes=range(Config.NUM_CLASSES))
        auc_score = roc_auc_score(y_test_bin, y_pred_proba, average='weighted', multi_class='ovr')
    except:
        auc_score = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Classification report
    class_report = classification_report(
        y_test, y_pred,
        target_names=Config.CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )
    
    results = {
        'model_name': model_name,
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'auc': float(auc_score),
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'confusion_matrix': cm,
        'classification_report': class_report
    }
    
    print(f"   ✅ Accuracy: {accuracy:.4f}")
    print(f"   ✅ Precision: {precision:.4f}")
    print(f"   ✅ Recall: {recall:.4f}")
    print(f"   ✅ F1-Score: {f1:.4f}")
    print(f"   ✅ AUC: {auc_score:.4f}")
    
    return results


# ============================================
# VISUALIZATION
# ============================================

def create_comparison_visualization(all_results, output_dir):
    """Create comparison visualization"""
    
    print("\n📊 Creating comparison visualizations...")
    
    # Extract metrics
    models = [r['model_name'] for r in all_results]
    accuracies = [r['accuracy'] for r in all_results]
    precisions = [r['precision'] for r in all_results]
    recalls = [r['recall'] for r in all_results]
    f1_scores = [r['f1'] for r in all_results]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy
    ax = axes[0, 0]
    ax.bar(models, accuracies, color='skyblue', edgecolor='black')
    ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_ylim([0, 1])
    for i, v in enumerate(accuracies):
        ax.text(i, v + 0.02, f'{v:.4f}', ha='center', fontsize=10)
    ax.tick_params(axis='x', rotation=45)
    
    # Precision
    ax = axes[0, 1]
    ax.bar(models, precisions, color='lightgreen', edgecolor='black')
    ax.set_title('Model Precision Comparison', fontsize=14, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_ylim([0, 1])
    for i, v in enumerate(precisions):
        ax.text(i, v + 0.02, f'{v:.4f}', ha='center', fontsize=10)
    ax.tick_params(axis='x', rotation=45)
    
    # Recall
    ax = axes[1, 0]
    ax.bar(models, recalls, color='lightcoral', edgecolor='black')
    ax.set_title('Model Recall Comparison', fontsize=14, fontweight='bold')
    ax.set_ylabel('Recall', fontsize=12)
    ax.set_ylim([0, 1])
    for i, v in enumerate(recalls):
        ax.text(i, v + 0.02, f'{v:.4f}', ha='center', fontsize=10)
    ax.tick_params(axis='x', rotation=45)
    
    # F1-Score
    ax = axes[1, 1]
    ax.bar(models, f1_scores, color='lightyellow', edgecolor='black')
    ax.set_title('Model F1-Score Comparison', fontsize=14, fontweight='bold')
    ax.set_ylabel('F1-Score', fontsize=12)
    ax.set_ylim([0, 1])
    for i, v in enumerate(f1_scores):
        ax.text(i, v + 0.02, f'{v:.4f}', ha='center', fontsize=10)
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    viz_path = output_dir / 'model_comparison.png'
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {viz_path}")
    plt.close()


def create_confusion_matrices(all_results, output_dir):
    """Create confusion matrices for all models"""
    
    print("📊 Creating confusion matrices...")
    
    n_models = len(all_results)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
    
    # Flatten axes for easier iteration
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for idx, results in enumerate(all_results):
        ax = axes[idx]
        cm = results['confusion_matrix']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=Config.CLASS_NAMES,
                    yticklabels=Config.CLASS_NAMES,
                    cbar_kws={'shrink': 0.8})
        ax.set_title(f"Confusion Matrix: {results['model_name']}", fontsize=10, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=8)
        ax.set_xlabel('Predicted Label', fontsize=8)
    
    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    cm_path = output_dir / 'confusion_matrices.png'
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {cm_path}")
    plt.close()


# ============================================
# MAIN
# ============================================

def main():
    
    # Setup
    configure_gpu()
    Config.WEEK10_DIR.mkdir(exist_ok=True, parents=True)
    
    print(f"\n📁 Week 10 Evaluation Directory: {Config.WEEK10_DIR}")
    print(f"📁 Output Directory: {Config.OUTPUT_DIR}")
    
    # Ensure test data exists
    ensure_test_data_exists(Config.OUTPUT_DIR)
    
    # Load labels
    print("\n📂 Loading test labels...")
    y_test = safe_load_npy(
        Config.OUTPUT_DIR / 'y_test.npy',
        'y_test',
        use_memmap=False
    )
    
    if y_test is None:
        print("❌ Could not load test labels")
        return
    
    # Evaluate each model
    print("\n" + "=" * 80)
    print("EVALUATING MODELS")
    print("=" * 80)
    
    all_results = []
    failed_models = []
    
    for model_key, model_config in Config.MODELS_TO_TEST.items():
        print(f"\n{'='*80}")
        print(f"Model: {model_key}")
        print(f"Description: {model_config['description']}")
        print(f"{'='*80}")
        
        # Check if model exists
        model_path = model_config['path']
        if not model_path.exists():
            print(f"❌ Model not found: {model_path}")
            failed_models.append({
                'model': model_key,
                'reason': f"Model file not found at {model_path}"
            })
            continue
        
        try:
            # Load model
            print(f"📥 Loading model from: {model_path.name}...")
            model = keras.models.load_model(str(model_path), compile=False)
            print(f"✅ Model loaded successfully")
            print(f"   Parameters: {model.count_params():,}")
            
            # Load test data
            test_data_path = Config.OUTPUT_DIR / model_config['test_data']
            print(f"📂 Loading test data: {test_data_path.name}")
            X_test = safe_load_npy(
                test_data_path,
                model_config['test_data'],
                use_memmap=True,
                expected_shape=(8000,) + model_config['input_size']
            )
            
            if X_test is None:
                raise RuntimeError(f"Could not load test data from {test_data_path}")
            
            # Verify input shape
            expected_shape = model_config['input_size']
            actual_shape = X_test.shape[1:]
            
            if actual_shape != expected_shape:
                print(f"⚠️  Input shape mismatch!")
                print(f"   Expected: {expected_shape}")
                print(f"   Actual: {actual_shape}")
                print(f"   Resizing test data...")
                
                # Use PIL to resize
                from PIL import Image
                
                X_test_resized = np.zeros((len(X_test),) + expected_shape, dtype=np.float32)
                
                for i in tqdm(range(len(X_test)), desc="Resizing"):
                    img = (X_test[i] * 255).astype(np.uint8)
                    img_pil = Image.fromarray(img)
                    img_resized = img_pil.resize(
                        (expected_shape[1], expected_shape[0]),
                        Image.Resampling.BICUBIC
                    )
                    X_test_resized[i] = np.array(img_resized, dtype=np.float32) / 255.0
                
                X_test = X_test_resized
                print(f"✅ Resized to {X_test.shape}")
            
            print_gpu_status("before inference")
            
            # Evaluate
            results = evaluate_model(model, X_test, y_test, model_key)
            all_results.append(results)
            
            print_gpu_status("after inference")
            
            # Cleanup
            del model
            del X_test
            gc.collect()
            
        except Exception as e:
            print(f"❌ Error evaluating {model_key}: {e}")
            import traceback
            traceback.print_exc()
            failed_models.append({
                'model': model_key,
                'reason': str(e)
            })
    
    # Summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    
    if all_results:
        # Create results dataframe
        results_df = pd.DataFrame([{
            'Model': r['model_name'],
            'Accuracy': f"{r['accuracy']:.4f}",
            'Precision': f"{r['precision']:.4f}",
            'Recall': f"{r['recall']:.4f}",
            'F1-Score': f"{r['f1']:.4f}",
            'AUC': f"{r['auc']:.4f}"
        } for r in all_results])
        
        print("\n📊 Results Table:")
        print(results_df.to_string(index=False))
        
        # Save overall results
        results_csv = Config.WEEK10_DIR / 'evaluation_results.csv'
        results_df.to_csv(results_csv, index=False)
        print(f"\n✅ Saved: {results_csv}")
        
        # ============================================
        # SAVE PER-MODEL DETAILED RESULTS
        # ============================================
        
        print("\n💾 Saving per-model detailed results...")
        
        for r in all_results:
            model_name_safe = r['model_name'].replace('/', '_')
            
            # 1. Per-model metrics CSV
            metrics_data = {
                'Metric': [
                    'Accuracy',
                    'Precision (weighted)',
                    'Recall (weighted)',
                    'F1-Score (weighted)',
                    'AUC (weighted)'
                ],
                'Value': [
                    f"{r['accuracy']:.6f}",
                    f"{r['precision']:.6f}",
                    f"{r['recall']:.6f}",
                    f"{r['f1']:.6f}",
                    f"{r['auc']:.6f}"
                ]
            }
            
            metrics_csv = Config.WEEK10_DIR / f'{model_name_safe}_metrics.csv'
            pd.DataFrame(metrics_data).to_csv(metrics_csv, index=False)
            print(f"   ✅ Saved: {metrics_csv}")
            
            # 2. Per-class detailed report CSV (from classification_report)
            class_report = r['classification_report']
            class_metrics = []
            
            for class_idx, class_name in enumerate(Config.CLASS_NAMES):
                if str(class_idx) in class_report:
                    class_data = class_report[str(class_idx)]
                    class_metrics.append({
                        'Class': class_name,
                        'Precision': f"{class_data.get('precision', 0):.6f}",
                        'Recall': f"{class_data.get('recall', 0):.6f}",
                        'F1-Score': f"{class_data.get('f1-score', 0):.6f}",
                        'Support': int(class_data.get('support', 0))
                    })
            
            # Add weighted average
            if 'weighted avg' in class_report:
                weighted = class_report['weighted avg']
                class_metrics.append({
                    'Class': 'WEIGHTED AVG',
                    'Precision': f"{weighted.get('precision', 0):.6f}",
                    'Recall': f"{weighted.get('recall', 0):.6f}",
                    'F1-Score': f"{weighted.get('f1-score', 0):.6f}",
                    'Support': int(weighted.get('support', 0))
                })
            
            class_report_csv = Config.WEEK10_DIR / f'{model_name_safe}_class_report.csv'
            pd.DataFrame(class_metrics).to_csv(class_report_csv, index=False)
            print(f"   ✅ Saved: {class_report_csv}")
            
            # 3. Per-model confusion matrix CSV
            cm = r['confusion_matrix']
            cm_df = pd.DataFrame(
                cm,
                index=[f'True_{name}' for name in Config.CLASS_NAMES],
                columns=[f'Pred_{name}' for name in Config.CLASS_NAMES]
            )
            
            cm_csv = Config.WEEK10_DIR / f'{model_name_safe}_confusion_matrix.csv'
            cm_df.to_csv(cm_csv)
            print(f"   ✅ Saved: {cm_csv}")
            
            # 4. Per-model predictions CSV (for regenerating visualizations)
            y_pred = r['y_pred']
            y_pred_proba = r['y_pred_proba']
            
            predictions_data = {
                'Sample_ID': range(len(y_pred)),
                'True_Label': y_test,
                'True_Label_Name': [Config.CLASS_NAMES[int(y)] for y in y_test],
                'Predicted_Label': y_pred,
                'Predicted_Label_Name': [Config.CLASS_NAMES[int(y)] for y in y_pred],
                'Prediction_Confidence': np.max(y_pred_proba, axis=1)
            }
            
            # Add per-class probabilities
            for class_idx, class_name in enumerate(Config.CLASS_NAMES):
                predictions_data[f'Prob_{class_name}'] = y_pred_proba[:, class_idx]
            
            predictions_df = pd.DataFrame(predictions_data)
            
            pred_csv = Config.WEEK10_DIR / f'{model_name_safe}_predictions.csv'
            predictions_df.to_csv(pred_csv, index=False)
            print(f"   ✅ Saved: {pred_csv}")
        
        # Save detailed JSON with all data
        results_json = Config.WEEK10_DIR / 'evaluation_results_detailed.json'
        detailed_results = []
        
        for r in all_results:
            detailed_results.append({
                'model_name': r['model_name'],
                'metrics': {
                    'accuracy': float(r['accuracy']),
                    'precision': float(r['precision']),
                    'recall': float(r['recall']),
                    'f1': float(r['f1']),
                    'auc': float(r['auc'])
                },
                'classification_report': r['classification_report'],
                'confusion_matrix': r['confusion_matrix'].tolist(),
                'num_samples': len(r['y_pred']),
                'timestamp': datetime.now().isoformat()
            })
        
        with open(results_json, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        print(f"\n✅ Saved: {results_json}")
        
        # Create visualizations
        create_comparison_visualization(all_results, Config.WEEK10_DIR)
        create_confusion_matrices(all_results, Config.WEEK10_DIR)
        
        # Find best model
        best_model = max(all_results, key=lambda x: x['accuracy'])
        print(f"\n🏆 Best Model: {best_model['model_name']} (Accuracy: {best_model['accuracy']:.4f})")
    
    # Report failed models
    if failed_models:
        print(f"\n❌ Failed Models ({len(failed_models)}):")
        for failure in failed_models:
            print(f"   - {failure['model']}: {failure['reason']}")
    
    print("\n✅ Week 10 Evaluation Complete!")


if __name__ == "__main__":
    main()
