"""
Recover Corrupted .npy Files - Reconstruct from raw binary data
The X_*.npy files are saved as raw binary, need to be reconstructed with proper shapes
"""

import numpy as np
import json
from pathlib import Path

class CorruptedNpyRecovery:
    """Recover corrupted .npy files by reconstructing from binary data"""
    
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
    
    def load_binary_as_array(self, filepath, expected_shape, dtype=np.float32):
        """
        Load raw binary file as numpy array with given shape.
        
        Args:
            filepath: Path to binary file
            expected_shape: Tuple of expected shape (e.g., (64000, 600, 600, 3))
            dtype: Data type (default: float32)
        
        Returns:
            Reconstructed numpy array
        """
        filepath = Path(filepath)
        
        # Calculate expected bytes
        num_elements = np.prod(expected_shape)
        expected_bytes = num_elements * np.dtype(dtype).itemsize
        
        actual_bytes = filepath.stat().st_size
        
        print(f"\n📋 Reconstructing {filepath.name}:")
        print(f"  Expected shape: {expected_shape}")
        print(f"  Expected elements: {num_elements:,}")
        print(f"  Expected bytes: {expected_bytes:,} ({expected_bytes / (1024**3):.1f} GB)")
        print(f"  Actual bytes: {actual_bytes:,} ({actual_bytes / (1024**3):.1f} GB)")
        
        if actual_bytes != expected_bytes:
            print(f"  ⚠️  Mismatch! ({actual_bytes - expected_bytes:+d} bytes)")
            
            # Try to infer correct shape
            print(f"\n  🔍 Attempting to infer shape from file size...")
            
            # Try common shapes
            possible_shapes = [
                # Training set (64000 samples)
                (64000, 600, 600, 3),  # High-res
                (64000, 224, 224, 3),  # Low-res
                
                # Validation set (8000 samples)
                (8000, 600, 600, 3),   # High-res
                (8000, 224, 224, 3),   # Low-res
            ]
            
            for shape in possible_shapes:
                required_bytes = np.prod(shape) * np.dtype(dtype).itemsize
                if required_bytes == actual_bytes:
                    print(f"  ✅ Found matching shape: {shape}")
                    expected_shape = shape
                    break
            else:
                print(f"  ❌ Could not infer shape. Trying anyway...")
        
        # Load binary data
        try:
            arr = np.fromfile(filepath, dtype=dtype)
            print(f"  Loaded {arr.shape[0]:,} elements as flat array")
            
            # Reshape
            arr = arr.reshape(expected_shape)
            print(f"  ✅ Reshaped to {arr.shape}")
            
            # Validate
            print(f"  Data range: [{arr.min():.6f}, {arr.max():.6f}]")
            print(f"  Mean: {arr.mean():.6f}, Std: {arr.std():.6f}")
            
            return arr
        
        except Exception as e:
            print(f"  ❌ Failed to reconstruct: {e}")
            raise
    
    def recover_x_data(self):
        """Recover all X data files - use denormalized versions for 224x224"""
        
        print("\n" + "="*70)
        print("🔧 USING DENORMALIZED X DATA FILES")
        print("="*70)
        
        # Get label files to determine shapes
        y_train = np.load(self.data_dir / 'y_train.npy')
        y_val = np.load(self.data_dir / 'y_val.npy')
        y_train_hr = np.load(self.data_dir / 'y_train_high_res.npy')
        y_val_hr = np.load(self.data_dir / 'y_val_high_res.npy')
        
        n_train = y_train.shape[0]  # 64000
        n_val = y_val.shape[0]      # 8000
        
        print(f"\nDataset sizes:")
        print(f"  Train: {n_train:,} samples")
        print(f"  Val: {n_val:,} samples")
        
        # Recovery plan - use denormalized versions for 224x224, corrupted files for 600x600
        recovery_plan = [
            {
                'input': 'X_train_denormalized.npy',
                'output': 'X_train.npy',
                'shape': (n_train, 224, 224, 3),
                'description': 'Training images (224×224, denormalized)',
                'recover': False  # Already in proper format
            },
            {
                'input': 'X_val_denormalized.npy',
                'output': 'X_val.npy',
                'shape': (n_val, 224, 224, 3),
                'description': 'Validation images (224×224, denormalized)',
                'recover': False  # Already in proper format
            },
            {
                'input': 'X_train_high_res.npy',
                'output': 'X_train_high_res_recovered.npy',
                'shape': (n_train, 600, 600, 3),
                'description': 'Training images (600×600, ImageNet normalized)',
                'recover': True  # Need recovery from raw binary
            },
            {
                'input': 'X_val_high_res.npy',
                'output': 'X_val_high_res_recovered.npy',
                'shape': (n_val, 600, 600, 3),
                'description': 'Validation images (600×600, ImageNet normalized)',
                'recover': True  # Need recovery from raw binary
            },
        ]
        
        results = {}
        
        for plan in recovery_plan:
            input_path = self.data_dir / plan['input']
            output_path = self.data_dir / plan['output']
            
            print(f"\n{'─'*70}")
            print(f"📥 {plan['input']} → {plan['output']}")
            print(f"   {plan['description']}")
            
            if not input_path.exists():
                print(f"  ⏭️  Input file not found, skipping")
                continue
            
            try:
                # Check if recovery is needed
                if not plan.get('recover', True):
                    # Just copy the file, it's already in proper .npy format
                    print(f"  ℹ️  File already in proper format, copying...")
                    arr = np.load(input_path)
                    print(f"     Shape: {arr.shape}")
                    print(f"     Dtype: {arr.dtype}")
                    print(f"     Range: [{arr.min():.6f}, {arr.max():.6f}]")
                    np.save(output_path, arr)
                else:
                    # Recover array from raw binary
                    arr = self.load_binary_as_array(input_path, plan['shape'], dtype=np.float32)
                    np.save(output_path, arr)
                
                print(f"\n  ✅ Saved: {output_path}")
                print(f"     File size: {output_path.stat().st_size / (1024**3):.1f} GB")
                
                results[plan['input']] = {
                    'status': 'SUCCESS',
                    'output': str(output_path),
                    'shape': list(arr.shape),
                    'dtype': str(arr.dtype),
                }
            
            except Exception as e:
                print(f"  ❌ Failed: {e}")
                results[plan['input']] = {
                    'status': 'FAILED',
                    'error': str(e),
                }
        
        return results
    
    def create_recovery_plan(self):
        """Create detailed recovery plan"""
        
        print("\n" + "="*70)
        print("📋 RECOVERY PLAN")
        print("="*70)
        
        plan = {
            'problem': 'X_*.npy files are corrupted or incompatible',
            'root_cause': 'X_train/val.npy are raw binary format, X_train/val_high_res.npy also corrupted',
            'solution': 'Use denormalized versions for 224×224, recover high-res from raw binary if needed',
            'strategy': {
                '224x224 (low-res)': 'Copy X_train/val_denormalized.npy (already valid .npy format)',
                '600x600 (high-res)': 'Recover from X_train/val_high_res.npy raw binary files',
            },
            'steps': [
                {
                    'step': 1,
                    'action': 'Use denormalized 224×224 files (already proper .npy)',
                    'files': ['X_train_denormalized.npy', 'X_val_denormalized.npy'],
                    'status': '✅ FAST - just copy'
                },
                {
                    'step': 2,
                    'action': 'Recover 600×600 files from raw binary',
                    'files': ['X_train_high_res.npy', 'X_val_high_res.npy'],
                    'status': '⏳ SLOW - requires reshape'
                },
            ],
            'output_files': {
                'X_train.npy': 'Copy from X_train_denormalized.npy',
                'X_val.npy': 'Copy from X_val_denormalized.npy',
                'X_train_high_res_recovered.npy': 'Recover from X_train_high_res.npy raw binary',
                'X_val_high_res_recovered.npy': 'Recover from X_val_high_res.npy raw binary',
            }
        }
        
        for step in plan['steps']:
            print(f"\nStep {step['step']}: {step['action']}")
            print(f"Status: {step['status']}")
            if 'files' in step:
                for f in step['files']:
                    print(f"  - {f}")
        
        return plan


def main():
    """Run recovery"""
    
    data_dir = Path('/workspace/outputs')
    
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return
    
    print(f"\n✅ Found data directory: {data_dir}")
    
    recovery = CorruptedNpyRecovery(data_dir)
    
    # Show plan
    plan = recovery.create_recovery_plan()
    
    # Perform recovery
    print(f"\n{'#'*70}")
    print(f"# ATTEMPTING RECOVERY")
    print(f"{'#'*70}")
    
    results = recovery.recover_x_data()
    
    # Summary
    print(f"\n{'='*70}")
    print(f"📊 RECOVERY SUMMARY")
    print(f"{'='*70}")
    
    successful = [k for k, v in results.items() if v.get('status') == 'SUCCESS']
    failed = [k for k, v in results.items() if v.get('status') == 'FAILED']
    
    print(f"\n✅ Recovered ({len(successful)}):")
    for fname in successful:
        print(f"  - {fname}")
    
    if failed:
        print(f"\n❌ Failed ({len(failed)}):")
        for fname in failed:
            print(f"  - {fname}: {results[fname].get('error', 'Unknown error')}")
    
    # Save results
    results_path = data_dir / 'recovery_results.json'
    import json
    with open(results_path, 'w') as f:
        json.dump({
            'plan': plan,
            'results': results,
        }, f, indent=2)
    
    print(f"\n💾 Recovery results saved to: {results_path}")
    
    print(f"\n{'='*70}")
    print(f"✅ Recovery complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
