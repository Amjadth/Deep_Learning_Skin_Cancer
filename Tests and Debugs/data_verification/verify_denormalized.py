"""
Verify Denormalized Arrays - Check if they are valid and usable
"""

import numpy as np
from pathlib import Path

def verify_denormalized_arrays():
    """Test if denormalized arrays are valid .npy files"""
    
    data_dir = Path('/workspace/outputs')
    
    print("=" * 70)
    print("🔍 VERIFYING DENORMALIZED ARRAYS")
    print("=" * 70)
    
    files_to_check = [
        'X_train_denormalized.npy',
        'X_val_denormalized.npy',
        'X_train_baseline.npy',
        'X_val_baseline.npy',
    ]
    
    results = {}
    
    for filename in files_to_check:
        filepath = data_dir / filename
        
        print(f"\n{'─'*70}")
        print(f"📋 Testing: {filename}")
        print(f"{'─'*70}")
        
        if not filepath.exists():
            print(f"  ❌ File not found")
            results[filename] = {'status': 'NOT_FOUND'}
            continue
        
        file_size_mb = filepath.stat().st_size / (1024**2)
        print(f"  File size: {file_size_mb:.1f} MB")
        
        # Check header
        try:
            with open(filepath, 'rb') as f:
                header = f.read(10)
                print(f"  Header (hex): {header.hex()}")
                
                if header[:6] == b'\x93NUMPY':
                    print(f"  ✅ Valid .npy format detected")
                    has_header = True
                else:
                    print(f"  ⚠️  Not a standard .npy file")
                    has_header = False
        except Exception as e:
            print(f"  ❌ Header read error: {e}")
            results[filename] = {'status': 'HEADER_ERROR', 'error': str(e)}
            continue
        
        # Try loading
        print(f"\n  🔄 Load Attempts:")
        
        load_strategies = {
            'no_pickle': {'allow_pickle': False},
            'with_pickle': {'allow_pickle': True},
            'fix_imports': {'allow_pickle': True, 'fix_imports': True, 'encoding': 'bytes'},
        }
        
        load_successful = False
        successful_method = None
        array_obj = None
        
        for method_name, kwargs in load_strategies.items():
            try:
                arr = np.load(filepath, **kwargs)
                
                print(f"    ✅ {method_name}:")
                print(f"       Shape: {arr.shape}")
                print(f"       Dtype: {arr.dtype}")
                print(f"       Size: {arr.nbytes / (1024**3):.1f} GB")
                print(f"       Range: [{arr.min():.4f}, {arr.max():.4f}]")
                print(f"       Mean: {arr.mean():.4f}, Std: {arr.std():.4f}")
                
                if not load_successful:
                    load_successful = True
                    successful_method = method_name
                    array_obj = arr
            
            except Exception as e:
                print(f"    ❌ {method_name}: {type(e).__name__}: {str(e)[:60]}")
        
        # Memmap test
        if load_successful:
            print(f"\n  📍 Memory Mapping:")
            try:
                mmap_arr = np.load(filepath, mmap_mode='r', allow_pickle=False)
                print(f"    ✅ Memmap successful")
                print(f"       Shape: {mmap_arr.shape}")
                print(f"       Dtype: {mmap_arr.dtype}")
                memmap_ok = True
            except Exception as e:
                print(f"    ⚠️  Memmap failed: {e}")
                memmap_ok = False
        else:
            memmap_ok = False
        
        results[filename] = {
            'status': 'OK' if load_successful else 'FAILED',
            'file_size_mb': file_size_mb,
            'has_npy_header': has_header,
            'load_successful': load_successful,
            'successful_method': successful_method,
            'memmap_ok': memmap_ok,
            'shape': list(array_obj.shape) if array_obj is not None else None,
            'dtype': str(array_obj.dtype) if array_obj is not None else None,
        }
    
    # Summary
    print(f"\n{'='*70}")
    print(f"📊 SUMMARY")
    print(f"{'='*70}\n")
    
    valid_files = [k for k, v in results.items() if v.get('status') == 'OK']
    invalid_files = [k for k, v in results.items() if v.get('status') != 'OK']
    
    print(f"✅ Valid files ({len(valid_files)}/{len(files_to_check)}):")
    for fname in valid_files:
        result = results[fname]
        shape = result.get('shape')
        method = result.get('successful_method')
        memmap = "✅" if result.get('memmap_ok') else "❌"
        print(f"  {fname}")
        print(f"    Shape: {shape}")
        print(f"    Loaded via: {method}")
        print(f"    Memmap: {memmap}")
    
    if invalid_files:
        print(f"\n❌ Invalid files ({len(invalid_files)}/{len(files_to_check)}):")
        for fname in invalid_files:
            status = results[fname].get('status')
            print(f"  {fname}: {status}")
    
    # Detailed recommendation
    print(f"\n{'='*70}")
    print(f"📋 RECOMMENDATION")
    print(f"{'='*70}\n")
    
    if 'X_train_denormalized.npy' in valid_files and 'X_val_denormalized.npy' in valid_files:
        print("✅ USE DENORMALIZED VERSIONS")
        print("   X_train_denormalized.npy ✓")
        print("   X_val_denormalized.npy ✓")
        print("\n   These are valid 224×224 arrays!")
        print("   Perfect for the transfer learning models.")
    
    elif 'X_train_baseline.npy' in valid_files and 'X_val_baseline.npy' in valid_files:
        print("✅ FALLBACK: USE BASELINE VERSIONS")
        print("   X_train_baseline.npy ✓")
        print("   X_val_baseline.npy ✓")
        print("\n   These are valid alternatives for 224×224.")
    
    else:
        print("❌ NO VALID 224×224 ARRAYS FOUND")
        print("   Need to recover from raw binary files.")
    
    return results


if __name__ == "__main__":
    results = verify_denormalized_arrays()
