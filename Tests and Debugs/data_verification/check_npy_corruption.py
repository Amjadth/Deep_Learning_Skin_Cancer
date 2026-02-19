#!/usr/bin/env python3
"""
NPY File Corruption Checker
===========================
Diagnostic tool to check if NumPy .npy files are corrupted or readable.
Tests various loading methods and provides detailed diagnostics.
"""

import numpy as np
import os
from pathlib import Path
import sys

def check_file_basics(filepath):
    """Check basic file properties."""
    print(f"\n{'='*70}")
    print(f"📋 FILE: {filepath.name}")
    print(f"{'='*70}")
    
    if not filepath.exists():
        print(f"❌ FILE NOT FOUND: {filepath}")
        return False
    
    file_size_mb = filepath.stat().st_size / (1024**2)
    print(f"✓ File exists")
    print(f"✓ File size: {file_size_mb:.2f} MB")
    
    return True

def check_npy_header(filepath):
    """Check if file has valid NPY header."""
    print(f"\n🔍 Checking NPY header...")
    try:
        with open(filepath, 'rb') as f:
            magic = f.read(6)
            if magic == b'\x93NUMPY':
                print(f"✓ Valid NPY header found: {magic}")
                version = f.read(2)
                major, minor = version[0], version[1]
                print(f"✓ NPY version: {major}.{minor}")
                return True
            else:
                print(f"❌ Invalid header. Got: {magic}")
                return False
    except Exception as e:
        print(f"❌ Failed to read header: {e}")
        return False

def try_load_with_mmap(filepath):
    """Try loading with memory mapping."""
    print(f"\n🔄 Attempting load with mmap_mode='r'...")
    try:
        data = np.load(filepath, mmap_mode='r')
        print(f"✓ SUCCESS with mmap")
        print(f"  Shape: {data.shape}")
        print(f"  Dtype: {data.dtype}")
        return True, data
    except Exception as e:
        print(f"❌ FAILED: {type(e).__name__}: {e}")
        return False, None

def try_load_direct(filepath):
    """Try direct loading without mmap."""
    print(f"\n🔄 Attempting direct load (no mmap)...")
    try:
        data = np.load(filepath)
        print(f"✓ SUCCESS with direct load")
        print(f"  Shape: {data.shape}")
        print(f"  Dtype: {data.dtype}")
        return True, data
    except Exception as e:
        print(f"❌ FAILED: {type(e).__name__}: {e}")
        return False, None

def try_load_with_pickle(filepath):
    """Try loading with allow_pickle=True."""
    print(f"\n🔄 Attempting load with allow_pickle=True...")
    try:
        data = np.load(filepath, allow_pickle=True)
        print(f"✓ SUCCESS with allow_pickle=True")
        print(f"  Shape: {data.shape if hasattr(data, 'shape') else 'N/A'}")
        print(f"  Dtype: {data.dtype if hasattr(data, 'dtype') else type(data)}")
        return True, data
    except Exception as e:
        print(f"❌ FAILED: {type(e).__name__}: {e}")
        return False, None

def check_file_corruption(filepath):
    """Comprehensive corruption check."""
    print(f"\n{'='*70}")
    print(f"🔍 CORRUPTION CHECK FOR: {filepath.name}")
    print(f"{'='*70}")
    
    # Step 1: Basic checks
    if not check_file_basics(filepath):
        return {
            'file': filepath.name,
            'exists': False,
            'status': '❌ FAILED - File not found'
        }
    
    # Step 2: Check header
    has_valid_header = check_npy_header(filepath)
    
    # Step 3: Try different loading methods
    results = {
        'file': filepath.name,
        'exists': True,
        'valid_header': has_valid_header,
        'mmap_success': False,
        'direct_success': False,
        'pickle_success': False,
        'shape': None,
        'dtype': None,
        'status': None,
    }
    
    # Try mmap
    success, data = try_load_with_mmap(filepath)
    results['mmap_success'] = success
    if success:
        results['shape'] = data.shape
        results['dtype'] = str(data.dtype)
    
    # Try direct
    success, data = try_load_direct(filepath)
    results['direct_success'] = success
    if success and not results['shape']:
        results['shape'] = data.shape
        results['dtype'] = str(data.dtype)
    
    # Try with pickle
    success, data = try_load_with_pickle(filepath)
    results['pickle_success'] = success
    if success and not results['shape']:
        results['shape'] = data.shape if hasattr(data, 'shape') else 'N/A'
        results['dtype'] = str(data.dtype) if hasattr(data, 'dtype') else str(type(data))
    
    # Determine status
    if results['mmap_success'] or results['direct_success']:
        results['status'] = '✅ GOOD - File is readable'
    elif results['pickle_success']:
        results['status'] = '⚠️  PARTIAL - Only readable with allow_pickle=True'
    else:
        results['status'] = '❌ CORRUPTED - File cannot be loaded'
    
    return results

def main():
    """Main diagnostic routine."""
    print("\n" + "="*70)
    print("NPY FILE CORRUPTION CHECKER")
    print("="*70)
    
    # Define files to check
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
        print("Checked:")
        for path in base_paths:
            print(f"  - {path}")
        sys.exit(1)
    
    print(f"\n📁 Using output directory: {output_dir}")
    
    # Files to check
    files_to_check = [
        'X_train_baseline.npy',
        'y_train_baseline.npy',
        'X_val_baseline.npy',
        'y_val_baseline.npy',
        'X_test_baseline.npy',
        'y_test_baseline.npy',
    ]
    
    results_summary = []
    
    for filename in files_to_check:
        filepath = output_dir / filename
        result = check_file_corruption(filepath)
        results_summary.append(result)
    
    # Print summary
    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")
    
    for result in results_summary:
        status_icon = "✅" if "GOOD" in result['status'] else ("⚠️ " if "PARTIAL" in result['status'] else "❌")
        print(f"{status_icon} {result['file']:30} {result['status']}")
        if result['shape']:
            print(f"   Shape: {result['shape']}, Dtype: {result['dtype']}")
    
    # Final verdict
    print(f"\n{'='*70}")
    corrupted = [r for r in results_summary if "CORRUPTED" in r['status']]
    partial = [r for r in results_summary if "PARTIAL" in r['status']]
    good = [r for r in results_summary if "GOOD" in r['status']]
    
    print(f"📊 VERDICT:")
    print(f"   ✅ Good files: {len(good)}")
    print(f"   ⚠️  Partial files: {len(partial)}")
    print(f"   ❌ Corrupted files: {len(corrupted)}")
    
    if corrupted:
        print(f"\n⚠️  ACTION REQUIRED:")
        print(f"   The following files need to be regenerated:")
        for r in corrupted:
            print(f"   - {r['file']}")
        print(f"\n   Recommendation: Run Week 6 or earlier to regenerate these files")
    
    if partial:
        print(f"\n⚠️  PARTIAL COMPATIBILITY:")
        print(f"   The following files can only be loaded with allow_pickle=True:")
        for r in partial:
            print(f"   - {r['file']}")
    
    if good and not corrupted:
        print(f"\n✅ All checked files are valid and readable!")
    
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()
