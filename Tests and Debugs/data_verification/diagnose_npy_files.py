#!/usr/bin/env python3
"""
Diagnostic script to check .npy files from Week 4.
Run this in your RunPod container to understand what's wrong with the files.
"""

import numpy as np
from pathlib import Path
import os

OUTPUT_DIR = Path('/workspace/outputs') if Path('/workspace/outputs').exists() else Path('./outputs')

print("=" * 70)
print("NPY FILE DIAGNOSTICS")
print("=" * 70)

files_to_check = [
    'X_train.npy',
    'X_val.npy', 
    'X_test.npy',
    'y_train.npy',
    'y_val.npy',
    'y_test.npy'
]

for filename in files_to_check:
    filepath = OUTPUT_DIR / filename
    print(f"\n📋 Checking: {filename}")
    print("-" * 70)
    
    if not filepath.exists():
        print(f"  ❌ File not found!")
        continue
    
    # Check file size
    size_bytes = filepath.stat().st_size
    size_gb = size_bytes / (1024**3)
    print(f"  📦 Size: {size_gb:.2f} GB ({size_bytes:,} bytes)")
    
    # Try to read header
    try:
        with open(filepath, 'rb') as f:
            magic = f.read(6)
            print(f"  🔍 Magic bytes: {magic}")
            
            if magic == b'\x93NUMPY':
                print(f"     ✓ Valid NumPy file header")
            else:
                print(f"     ❌ Invalid header! Should be '\\x93NUMPY'")
    except Exception as e:
        print(f"  ❌ Cannot read file: {e}")
        continue
    
    # Try different loading methods
    print(f"  🔧 Testing loading methods...")
    
    # Method 1: np.load without pickle
    try:
        data = np.load(filepath, mmap_mode='r', allow_pickle=False)
        print(f"     ✓ np.load (allow_pickle=False): {data.shape}, dtype={data.dtype}")
        del data
    except ValueError as e:
        print(f"     ⚠️  np.load (allow_pickle=False): FAILED - {str(e)[:60]}")
    except Exception as e:
        print(f"     ❌ np.load (allow_pickle=False): ERROR - {str(e)[:60]}")
    
    # Method 2: np.load with pickle
    try:
        data = np.load(filepath, mmap_mode='r', allow_pickle=True)
        print(f"     ✓ np.load (allow_pickle=True): {data.shape}, dtype={data.dtype}")
        del data
    except Exception as e:
        print(f"     ❌ np.load (allow_pickle=True): ERROR - {str(e)[:60]}")
    
    # Method 3: Direct memmap (for large files only)
    if size_gb > 1:  # Only for large files
        print(f"  🔧 Attempting direct memmap with expected shapes...")
        
        # Expected shapes from week4 output
        expected_shapes = {
            'X_train.npy': (64000, 600, 600, 3),
            'X_val.npy': (8000, 600, 600, 3),
            'X_test.npy': (8000, 600, 600, 3),
        }
        
        if filename in expected_shapes:
            expected_shape = expected_shapes[filename]
            expected_bytes = np.prod(expected_shape) * 4  # float32 = 4 bytes
            
            print(f"     Expected shape: {expected_shape}")
            print(f"     Expected size: {expected_bytes / (1024**3):.2f} GB")
            print(f"     Actual size: {size_gb:.2f} GB")
            
            # Check if size matches (with some tolerance for header)
            header_size_mb = 1  # NumPy header is typically < 1MB
            size_diff_gb = abs(size_gb - (expected_bytes / (1024**3)))
            
            if size_diff_gb < 0.1:  # Within 100MB tolerance
                print(f"     ✓ Size matches! Attempting direct memmap...")
                try:
                    data = np.memmap(str(filepath), dtype=np.float32, mode='r', shape=expected_shape)
                    print(f"     ✓ Direct memmap SUCCESS: {data.shape}, dtype={data.dtype}")
                    
                    # Sample check
                    print(f"     📊 Data sample: min={data[0].min():.2f}, max={data[0].max():.2f}, mean={data[0].mean():.2f}")
                    del data
                except Exception as e:
                    print(f"     ❌ Direct memmap FAILED: {str(e)[:60]}")
            else:
                print(f"     ⚠️  Size mismatch! Diff: {size_diff_gb:.2f} GB")
                print(f"     File may be corrupted or have different shape")

print("\n" + "=" * 70)
print("DIAGNOSIS COMPLETE")
print("=" * 70)
print("\n📋 Summary:")
print("  - If all files load with 'allow_pickle=False': Files are good!")
print("  - If files need 'allow_pickle=True': Files use pickle (common)")
print("  - If direct memmap works: Files are raw arrays (good!)")
print("  - If all methods fail: Files are corrupted (re-run week4)")
print("\n💡 Recommendation:")
print("  The updated week5_fixed_runpod.py now tries all these methods automatically.")
print("  Just run: python week5_fixed_runpod.py")
print("=" * 70)
