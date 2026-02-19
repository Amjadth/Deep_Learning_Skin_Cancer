#!/usr/bin/env python3
"""
Correct Data Loading Test - Matches Week 6/7 Approach
======================================================
Tests loading data the SAME WAY Week 6 and 7 load it.
Uses np.memmap fallback for files without proper NPY headers.
"""

import numpy as np
from pathlib import Path
import sys

def safe_load_npy(filepath, description, use_memmap=True, expected_shape=None):
    """Safely load .npy files with memmap support - EXACT COPY FROM WEEK 6"""
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

def main():
    """Main test routine."""
    print("\n" + "="*70)
    print("CORRECT DATA LOADING TEST (Week 6/7 Approach)")
    print("="*70)
    
    # Find outputs directory
    base_paths = [
        Path('/workspace/outputs'),
        Path('/Users/ahadraza/Side Projects/Freelancing/Skin Cancer V3.0/outputs'),
        Path.cwd() / 'outputs',
    ]
    
    output_dir = None
    for path in base_paths:
        if path.exists():
            output_dir = path
            break
    
    if not output_dir:
        print("\n❌ Could not find outputs directory")
        sys.exit(1)
    
    print(f"\n📁 Using output directory: {output_dir}")
    
    # Test data loading - EXACTLY AS WEEK 6 DOES IT
    print("\n" + "="*70)
    print("TESTING DATA LOADING (Week 6/7 Method)")
    print("="*70)
    
    try:
        print("\n🔄 Step 1: Loading training data...")
        X_train = safe_load_npy(
            output_dir / 'X_train_denormalized.npy',
            'X_train_denormalized (64k images)',
            use_memmap=True,
            expected_shape=(64000, 224, 224, 3)
        )
        y_train = safe_load_npy(
            output_dir / 'y_train_baseline.npy',
            'y_train (labels)',
            use_memmap=False
        )
        
        print("\n🔄 Step 2: Loading validation data...")
        X_val = safe_load_npy(
            output_dir / 'X_val_denormalized.npy',
            'X_val_denormalized (8k images)',
            use_memmap=True,
            expected_shape=(8000, 224, 224, 3)
        )
        y_val = safe_load_npy(
            output_dir / 'y_val_baseline.npy',
            'y_val (labels)',
            use_memmap=False
        )
        
        print("\n🔄 Step 3: Loading test data...")
        X_test = safe_load_npy(
            output_dir / 'X_test_denormalized.npy',
            'X_test_denormalized (8k images)',
            use_memmap=True,
            expected_shape=(8000, 224, 224, 3)
        )
        y_test = safe_load_npy(
            output_dir / 'y_test_baseline.npy',
            'y_test (labels)',
            use_memmap=False
        )
        
        # All loaded successfully!
        print("\n" + "="*70)
        print("✅ SUCCESS! All data loaded correctly")
        print("="*70)
        
        print(f"\n📊 Data Summary:")
        print(f"  X_train: {X_train.shape}, Range: [{X_train.min():.4f}, {X_train.max():.4f}]")
        print(f"  y_train: {y_train.shape}, Classes: {np.unique(y_train)}")
        print(f"  X_val:   {X_val.shape}, Range: [{X_val.min():.4f}, {X_val.max():.4f}]")
        print(f"  y_val:   {y_val.shape}, Classes: {np.unique(y_val)}")
        print(f"  X_test:  {X_test.shape}, Range: [{X_test.min():.4f}, {X_test.max():.4f}]")
        print(f"  y_test:  {y_test.shape}, Classes: {np.unique(y_test)}")
        
        # Verify data quality
        print(f"\n✅ Data Quality Checks:")
        
        # Check for NaN/Inf
        has_nan = np.isnan(X_train).any() or np.isnan(X_val).any() or np.isnan(X_test).any()
        has_inf = np.isinf(X_train).any() or np.isinf(X_val).any() or np.isinf(X_test).any()
        
        if has_nan:
            print(f"  ❌ Found NaN values!")
        else:
            print(f"  ✓ No NaN values")
        
        if has_inf:
            print(f"  ❌ Found Inf values!")
        else:
            print(f"  ✓ No Inf values")
        
        # Check data range
        all_data_min = min(X_train.min(), X_val.min(), X_test.min())
        all_data_max = max(X_train.max(), X_val.max(), X_test.max())
        
        if 0 <= all_data_min and all_data_max <= 1:
            print(f"  ✓ Data in [0, 1] range (properly denormalized)")
        else:
            print(f"  ⚠️  Data range: [{all_data_min:.4f}, {all_data_max:.4f}]")
        
        print(f"\n✅ READY TO USE IN WEEK 8!")
        print("="*70 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print(f"\nℹ️  Troubleshooting:")
        print(f"  1. Check if files exist in {output_dir}")
        print(f"  2. Files should be named:")
        print(f"     - X_train_denormalized.npy")
        print(f"     - X_val_denormalized.npy")
        print(f"     - X_test_denormalized.npy")
        print(f"     - y_train_baseline.npy")
        print(f"     - y_val_baseline.npy")
        print(f"     - y_test_baseline.npy")
        print(f"\n  3. If not found, run Week 6 to generate denormalized data first")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
