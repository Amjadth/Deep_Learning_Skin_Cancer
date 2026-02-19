#!/usr/bin/env python3
"""
Denormalized Data Integrity Checker
====================================
Diagnostic tool to check if denormalized image data files exist and are valid.
Denormalized = raw float values (0-255 range), NOT normalized to 0-1.
"""

import numpy as np
import os
from pathlib import Path
import sys

def check_file_basics(filepath):
    """Check basic file properties."""
    if not filepath.exists():
        print(f"❌ FILE NOT FOUND: {filepath}")
        return False, 0
    
    file_size_mb = filepath.stat().st_size / (1024**2)
    print(f"✓ File exists")
    print(f"✓ File size: {file_size_mb:.2f} MB")
    
    return True, file_size_mb

def check_npy_header(filepath):
    """Check if file has valid NPY header."""
    try:
        with open(filepath, 'rb') as f:
            magic = f.read(6)
            if magic == b'\x93NUMPY':
                print(f"✓ Valid NPY header found")
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

def try_load_data(filepath):
    """Try loading the denormalized data."""
    try:
        data = np.load(filepath, mmap_mode='r')
        print(f"✓ SUCCESS loading with mmap")
        print(f"  Shape: {data.shape}")
        print(f"  Dtype: {data.dtype}")
        return True, data
    except Exception as e:
        print(f"❌ FAILED to load: {type(e).__name__}: {e}")
        return False, None

def check_data_statistics(data, filename):
    """Check data value ranges and statistics (should be denormalized 0-255)."""
    print(f"\n📊 Data Statistics:")
    
    # Get sample statistics (don't load entire array)
    try:
        # Sample first batch to check values
        sample = data[:min(1000, len(data))]
        
        min_val = sample.min()
        max_val = sample.max()
        mean_val = sample.mean()
        
        print(f"  Min value: {min_val:.2f}")
        print(f"  Max value: {max_val:.2f}")
        print(f"  Mean value: {mean_val:.2f}")
        
        # Check if denormalized (0-255 range)
        if 'X_' in filename or 'image' in filename.lower():
            if 0 <= min_val and max_val <= 256:
                print(f"  ✓ Looks like denormalized image data (0-255 range)")
                return True
            elif 0 <= min_val and max_val <= 1:
                print(f"  ⚠️  WARNING: Data appears NORMALIZED (0-1), not denormalized!")
                return False
            else:
                print(f"  ⚠️  Unexpected range: [{min_val:.2f}, {max_val:.2f}]")
                return True
        else:
            print(f"  ✓ Label data (value range OK)")
            return True
            
    except Exception as e:
        print(f"  ⚠️  Could not compute statistics: {e}")
        return False

def check_file_integrity(filepath, filename):
    """Comprehensive integrity check for denormalized data."""
    print(f"\n{'='*70}")
    print(f"🔍 CHECKING: {filename}")
    print(f"{'='*70}")
    
    # Step 1: Basic checks
    exists, file_size = check_file_basics(filepath)
    if not exists:
        return {
            'file': filename,
            'exists': False,
            'status': '❌ FAILED - File not found',
            'details': f'Expected at: {filepath}'
        }
    
    # Step 2: Check header
    has_valid_header = check_npy_header(filepath)
    
    # Step 3: Try loading
    print(f"\n🔄 Attempting to load data...")
    success, data = try_load_data(filepath)
    
    result = {
        'file': filename,
        'exists': True,
        'file_size_mb': file_size,
        'valid_header': has_valid_header,
        'loadable': success,
        'shape': None,
        'dtype': None,
        'is_denormalized': None,
        'status': None,
        'details': None,
    }
    
    if not success:
        result['status'] = '❌ CORRUPTED - Cannot load'
        result['details'] = 'File is not readable'
        return result
    
    # Step 4: Check data statistics
    result['shape'] = data.shape
    result['dtype'] = str(data.dtype)
    is_denormalized = check_data_statistics(data, filename)
    result['is_denormalized'] = is_denormalized
    
    # Determine status
    if has_valid_header and success and is_denormalized:
        result['status'] = '✅ GOOD - Valid denormalized data'
    elif has_valid_header and success and not is_denormalized:
        result['status'] = '⚠️  WARNING - Data not denormalized'
        result['details'] = 'Expected denormalized (0-255 range)'
    else:
        result['status'] = '❌ INVALID - Data integrity issue'
    
    return result

def main():
    """Main diagnostic routine."""
    print("\n" + "="*70)
    print("DENORMALIZED DATA INTEGRITY CHECKER")
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
        print("Checked:")
        for path in base_paths:
            print(f"  - {path}")
        sys.exit(1)
    
    print(f"\n📁 Using output directory: {output_dir}")
    
    # Files to check - denormalized data
    files_to_check = [
        ('X_train.npy', 'Training images (denormalized)'),
        ('y_train.npy', 'Training labels'),
        ('X_val.npy', 'Validation images (denormalized)'),
        ('y_val.npy', 'Validation labels'),
        ('X_test.npy', 'Test images (denormalized)'),
        ('y_test.npy', 'Test labels'),
    ]
    
    results_summary = []
    
    print(f"\n{'='*70}")
    print("CHECKING DENORMALIZED DATA FILES")
    print(f"{'='*70}")
    
    for filename, description in files_to_check:
        filepath = output_dir / filename
        result = check_file_integrity(filepath, filename)
        result['description'] = description
        results_summary.append(result)
    
    # Print summary
    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")
    
    for result in results_summary:
        status_icon = "✅" if "GOOD" in result['status'] else ("⚠️ " if "WARNING" in result['status'] else "❌")
        print(f"{status_icon} {result['file']:25} {result['status']}")
        if result['shape']:
            print(f"   Shape: {result['shape']}, Dtype: {result['dtype']}")
    
    # Final verdict
    print(f"\n{'='*70}")
    good = [r for r in results_summary if "GOOD" in r['status']]
    warning = [r for r in results_summary if "WARNING" in r['status']]
    corrupted = [r for r in results_summary if "CORRUPTED" in r['status'] or "INVALID" in r['status']]
    missing = [r for r in results_summary if not r['exists']]
    
    print(f"📊 VERDICT:")
    print(f"   ✅ Good files: {len(good)}")
    print(f"   ⚠️  Warning files: {len(warning)}")
    print(f"   ❌ Corrupted files: {len(corrupted)}")
    print(f"   ❓ Missing files: {len(missing)}")
    
    if missing:
        print(f"\n❌ MISSING FILES:")
        for r in missing:
            print(f"   - {r['file']}")
        print(f"\n   These files need to be generated. Run Week 6 to create denormalized data.")
    
    if corrupted:
        print(f"\n❌ CORRUPTED FILES:")
        for r in corrupted:
            print(f"   - {r['file']}")
    
    if warning:
        print(f"\n⚠️  WARNING:")
        for r in warning:
            print(f"   - {r['file']}: {r['details']}")
    
    if good and not corrupted and not missing:
        print(f"\n✅ All denormalized data files are valid and ready for use!")
    
    print(f"{'='*70}\n")
    
    # Additional info
    print(f"📝 NOTE:")
    print(f"   Week 8 requires DENORMALIZED data (raw floats 0-255)")
    print(f"   If files are missing or normalized, you need to:")
    print(f"   1. Run Week 6 to generate denormalized baseline data")
    print(f"   2. Or regenerate from original source images")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()
