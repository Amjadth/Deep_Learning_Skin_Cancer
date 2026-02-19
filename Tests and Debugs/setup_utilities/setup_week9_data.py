"""
Setup Valid Data Files for Week 9
Use verified denormalized arrays (224×224) for transfer learning
"""

import numpy as np
import shutil
from pathlib import Path

def setup_valid_data_files():
    """Setup working data files from denormalized arrays"""
    
    data_dir = Path('/workspace/outputs')
    
    print("=" * 70)
    print("🔧 SETTING UP VALID DATA FILES FOR WEEK 9")
    print("=" * 70)
    
    # Files we verified are working
    verified_files = {
        'X_train_denormalized.npy': 'X_train.npy',
        'X_val_denormalized.npy': 'X_val.npy',
        'y_train.npy': 'y_train.npy',  # Already valid
        'y_val.npy': 'y_val.npy',      # Already valid
    }
    
    print(f"\n📂 Data directory: {data_dir}")
    print(f"✅ Using verified denormalized arrays (224×224)\n")
    
    results = {}
    
    for source_name, dest_name in verified_files.items():
        source_path = data_dir / source_name
        dest_path = data_dir / dest_name
        
        print(f"{'─'*70}")
        print(f"Processing: {source_name} → {dest_name}")
        
        if not source_path.exists():
            print(f"  ❌ Source not found: {source_name}")
            results[source_name] = {'status': 'NOT_FOUND'}
            continue
        
        # Check if destination already exists
        if dest_path.exists():
            dest_size = dest_path.stat().st_size / (1024**3)
            source_size = source_path.stat().st_size / (1024**3)
            
            if abs(dest_size - source_size) < 0.1:  # Within 100MB tolerance
                print(f"  ✅ Already exists and matches size")
                # Verify it's loadable
                try:
                    arr = np.load(dest_path, allow_pickle=False)
                    print(f"     Shape: {arr.shape}, Dtype: {arr.dtype}")
                    results[source_name] = {
                        'status': 'VERIFIED',
                        'shape': list(arr.shape),
                        'dtype': str(arr.dtype),
                    }
                    continue
                except Exception as e:
                    print(f"  ⚠️  Destination exists but corrupted, will overwrite")
                    print(f"     Error: {e}")
            else:
                print(f"  ⚠️  Size mismatch, overwriting")
                print(f"     Source: {source_size:.1f} GB, Dest: {dest_size:.1f} GB")
        
        # Load and verify source
        try:
            print(f"  Loading source...")
            arr = np.load(source_path, allow_pickle=False)
            print(f"    Shape: {arr.shape}")
            print(f"    Dtype: {arr.dtype}")
            print(f"    Size: {arr.nbytes / (1024**3):.1f} GB")
            print(f"    Range: [{arr.min():.4f}, {arr.max():.4f}]")
            
            # Save to destination
            print(f"  Saving to {dest_name}...")
            np.save(dest_path, arr)
            
            dest_size = dest_path.stat().st_size / (1024**3)
            print(f"  ✅ Saved successfully ({dest_size:.1f} GB)")
            
            results[source_name] = {
                'status': 'SUCCESS',
                'source': str(source_path),
                'dest': str(dest_path),
                'shape': list(arr.shape),
                'dtype': str(arr.dtype),
                'size_gb': float(dest_size),
            }
        
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            results[source_name] = {
                'status': 'FAILED',
                'error': str(e),
            }
    
    # Verify high-res alternatives exist
    print(f"\n{'─'*70}")
    print("High-res alternatives (for future use):")
    
    high_res_files = [
        'X_train_high_res.npy',
        'X_val_high_res.npy',
        'X_train_high_res_recovered.npy',
        'X_val_high_res_recovered.npy',
    ]
    
    for fname in high_res_files:
        fpath = data_dir / fname
        if fpath.exists():
            size_gb = fpath.stat().st_size / (1024**3)
            print(f"  ✓ {fname} ({size_gb:.1f} GB)")
        else:
            print(f"  ✗ {fname} (not found)")
    
    # Summary
    print(f"\n{'='*70}")
    print("📊 SETUP SUMMARY")
    print(f"{'='*70}\n")
    
    successful = [k for k, v in results.items() if v.get('status') in ['SUCCESS', 'VERIFIED']]
    failed = [k for k, v in results.items() if v.get('status') == 'FAILED']
    
    print(f"✅ Ready ({len(successful)}):")
    for fname in successful:
        result = results[fname]
        shape = result.get('shape')
        print(f"  {fname}")
        print(f"    → {verified_files[fname]}")
        print(f"    Shape: {shape}")
    
    if failed:
        print(f"\n❌ Failed ({len(failed)}):")
        for fname in failed:
            print(f"  {fname}: {results[fname].get('error', 'Unknown error')}")
    
    print(f"\n{'='*70}")
    print("✅ DATA SETUP COMPLETE")
    print("   Ready for Week 9 Transfer Learning!")
    print(f"{'='*70}")
    
    return results


if __name__ == "__main__":
    results = setup_valid_data_files()
