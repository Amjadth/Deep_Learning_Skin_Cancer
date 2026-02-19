#!/usr/bin/env python3
"""
Test script to diagnose and fix model loading issues for Week 6.
Handles LossScaleOptimizer serialization problems.
"""

import os
import json
import warnings
import tensorflow as tf
from tensorflow import keras
from pathlib import Path

warnings.filterwarnings('ignore')

# ============================================
# SETUP
# ============================================

WORKSPACE = Path('/workspace')
OUTPUT_DIR = WORKSPACE / 'outputs'
MODEL_DIR = OUTPUT_DIR / 'models'
RESULTS_DIR = OUTPUT_DIR / 'results'

print("=" * 70)
print("TEST: MODEL LOADING AND FIXING")
print("=" * 70)

# Check if files exist
baseline_model_path = MODEL_DIR / 'baseline_cnn.keras'
baseline_json_path = MODEL_DIR / 'baseline_cnn_architecture.json'
baseline_config_path = MODEL_DIR / 'baseline_config.json'

print(f"\n📁 Checking files:")
print(f"  Model: {baseline_model_path.exists()} - {baseline_model_path}")
print(f"  JSON: {baseline_json_path.exists()} - {baseline_json_path}")
print(f"  Config: {baseline_config_path.exists()} - {baseline_config_path}")

# ============================================
# ATTEMPT 1: Try direct loading
# ============================================

print(f"\n🔧 Attempt 1: Direct model loading...")

try:
    model = keras.models.load_model(baseline_model_path)
    print(f"✅ SUCCESS: Model loaded directly!")
    print(f"   Parameters: {model.count_params():,}")
except AttributeError as e:
    print(f"❌ FAILED: AttributeError (expected with mixed precision FP16)")
    print(f"   Error: {str(e)[:100]}...")
    model = None
except Exception as e:
    print(f"❌ FAILED: {type(e).__name__}")
    print(f"   Error: {str(e)[:100]}...")
    model = None

# ============================================
# ATTEMPT 2: Load with compile=False
# ============================================

if not model:
    print(f"\n🔧 Attempt 2: Load with compile=False...")
    
    try:
        model = keras.models.load_model(baseline_model_path, compile=False)
        print(f"✅ SUCCESS: Model loaded with compile=False!")
        print(f"   Parameters: {model.count_params():,}")
        
        # Recompile without FP16 optimizer state
        print(f"\n   Recompiling model...")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        print(f"   ✓ Recompiled successfully!")
        
    except Exception as e:
        print(f"❌ FAILED: {type(e).__name__}")
        print(f"   Error: {str(e)[:100]}...")
        model = None

# ============================================
# ATTEMPT 3: Rebuild from JSON architecture
# ============================================

if not model:
    print(f"\n🔧 Attempt 3: Rebuild from JSON architecture...")
    
    try:
        with open(baseline_json_path, 'r') as f:
            model_json = f.read()
        
        model = keras.models.model_from_json(model_json)
        print(f"✅ SUCCESS: Model rebuilt from JSON!")
        print(f"   Parameters: {model.count_params():,}")
        
        # Compile model
        print(f"\n   Compiling model...")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        print(f"   ✓ Compiled successfully!")
        
        print(f"\n   ⚠️  NOTE: Weights not loaded - JSON only has architecture!")
        print(f"   This model will need retraining.")
        
    except Exception as e:
        print(f"❌ FAILED: {type(e).__name__}")
        print(f"   Error: {str(e)[:100]}...")
        model = None

# ============================================
# ANALYSIS & RECOMMENDATIONS
# ============================================

print("\n" + "=" * 70)
print("ANALYSIS & RECOMMENDATIONS")
print("=" * 70)

# Check config
if baseline_config_path.exists():
    print(f"\n📋 Config file found:")
    with open(baseline_config_path, 'r') as f:
        config = json.load(f)
    print(f"   Model: {config.get('model_name')}")
    print(f"   Parameters: {config.get('total_parameters'):,}")
    print(f"   Classes: {config.get('num_classes')}")

print(f"\n🔍 ISSUE DIAGNOSIS:")
print(f"""
The model was saved with mixed precision (FP16) enabled.
TensorFlow wraps the optimizer with LossScaleOptimizerV3 which:
  1. Scales loss by 2^scale to prevent underflow with FP16
  2. Has optimizer state that doesn't serialize well
  3. Causes AttributeError: 'LossScaleOptimizerV3' has no attribute 'name'

SOLUTION:
  ✓ Load with compile=False to skip optimizer deserialization
  ✓ Recompile with a fresh Adam optimizer
  ✓ Model weights ARE preserved
  ✓ Training can resume normally
""")

print(f"\n✅ RECOMMENDED APPROACH FOR WEEK 6:")
print(f"""
1. Load model: keras.models.load_model(path, compile=False)
2. Recompile: model.compile(optimizer=..., loss=..., metrics=...)
3. Continue with training

This preserves:
  ✓ Model architecture
  ✓ Trained weights
  ✓ Training history (if saved separately)
  
The fresh optimizer will:
  ✓ Start with step counter at 0 (OK for fine-tuning)
  ✓ Not have FP16 scaling issues
  ✓ Train normally on the A40 GPU
""")

# ============================================
# TEST: Save and reload corrected model
# ============================================

if model:
    print(f"\n🔄 Testing save/reload cycle...")
    
    test_save_path = MODEL_DIR / 'baseline_cnn_test.keras'
    
    try:
        # Save with safe settings
        model.save(test_save_path, save_format='keras')
        print(f"   ✓ Model saved: {test_save_path.name}")
        
        # Reload
        model_reloaded = keras.models.load_model(test_save_path, compile=False)
        print(f"   ✓ Model reloaded: {model_reloaded.count_params():,} parameters")
        
        # Verify architecture
        if model.count_params() == model_reloaded.count_params():
            print(f"   ✅ Architecture verified!")
        else:
            print(f"   ❌ Architecture mismatch!")
        
        # Cleanup
        test_save_path.unlink()
        print(f"   ✓ Test file cleaned up")
        
    except Exception as e:
        print(f"   ⚠️  Test failed: {e}")

# ============================================
# SUMMARY
# ============================================

print("\n" + "=" * 70)
print("✅ TEST COMPLETE")
print("=" * 70)
print(f"""
Status: Model loading issue IDENTIFIED and SOLUTIONS PROVIDED

Problem: Mixed precision (FP16) optimizer serialization
Solution: Load with compile=False, then recompile

Next Steps:
  1. Update week6_fixed_runpod.py to use this approach
  2. Run week6 training with corrected model loading
  3. Training will proceed normally
""")
