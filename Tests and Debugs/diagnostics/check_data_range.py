"""
Quick script to check the ACTUAL data range in your denormalized files
"""
import numpy as np
from pathlib import Path

# Load a small sample to check range
data_dir = Path('/workspace/outputs') if Path('/workspace/outputs').exists() else Path.cwd() / '../outputs'

print("Checking ACTUAL data ranges...")

# Check training data
X_train_path = data_dir / 'X_train_denormalized.npy'
if X_train_path.exists():
    X_train = np.load(X_train_path, mmap_mode='r')
    
    # Sample 100 images to check range
    sample_indices = np.random.choice(len(X_train), min(100, len(X_train)), replace=False)
    samples = X_train[sample_indices]
    
    print(f"\n📊 X_train_denormalized.npy:")
    print(f"   Shape: {X_train.shape}")
    print(f"   Dtype: {X_train.dtype}")
    print(f"   Min value: {samples.min():.6f}")
    print(f"   Max value: {samples.max():.6f}")
    print(f"   Mean: {samples.mean():.6f}")
    print(f"   Std: {samples.std():.6f}")
    
    # Determine actual range
    if samples.min() >= -0.1 and samples.max() <= 1.1:
        print(f"   ✅ Data appears to be in [0, 1] range")
    elif samples.min() >= -3 and samples.max() <= 3:
        print(f"   ⚠️  Data appears to be ImageNet normalized [-2.5, 2.5] range")
    elif samples.min() >= -1.1 and samples.max() <= 1.1:
        print(f"   ⚠️  Data appears to be in [-1, 1] range")
    elif samples.min() >= -0.5 and samples.max() <= 255.5:
        print(f"   ⚠️  Data appears to be in [0, 255] range")
    else:
        print(f"   ❌ Data range is UNEXPECTED!")
    
    del X_train, samples
else:
    print(f"❌ File not found: {X_train_path}")

# Check validation data
X_val_path = data_dir / 'X_val_denormalized.npy'
if X_val_path.exists():
    X_val = np.load(X_val_path, mmap_mode='r')
    
    sample_indices = np.random.choice(len(X_val), min(50, len(X_val)), replace=False)
    samples = X_val[sample_indices]
    
    print(f"\n📊 X_val_denormalized.npy:")
    print(f"   Shape: {X_val.shape}")
    print(f"   Min value: {samples.min():.6f}")
    print(f"   Max value: {samples.max():.6f}")
    print(f"   Mean: {samples.mean():.6f}")
    print(f"   Std: {samples.std():.6f}")
    
    del X_val, samples
else:
    print(f"❌ File not found: {X_val_path}")

print("\n" + "="*70)
print("DIAGNOSIS:")
print("="*70)
print("If data is in [0, 1]: Scale to [-1, 1] with image*2-1")
print("If data is in [-1, 1]: Use as-is (already correct)")
print("If data is ImageNet normalized: Use as-is (already correct)")  
print("If data is [0, 255]: Divide by 127.5 then subtract 1")
print("="*70)
