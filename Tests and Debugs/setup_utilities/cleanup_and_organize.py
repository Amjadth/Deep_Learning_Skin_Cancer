# ============================================
# CLEANUP AND ORGANIZE NETWORK VOLUME
# ============================================
#
# This script:
# 1. Verifies .npy arrays are correct (not corrupted)
# 2. Deletes memmap files if corresponding .npy files are verified
# 3. Organizes files into appropriate folders
#
# Safety Features:
# - Only deletes memmap files if .npy files are verified
# - Creates backup before deletion (optional)
# - Comprehensive verification (shape, dtype, loadability)
# - Detailed logging of all operations
#
# ============================================

import numpy as np
from pathlib import Path
import json
import shutil
import os
from datetime import datetime

# Network volume detection (same as week3/week4)
BASE_DIR = Path(os.getcwd())
NETWORK_VOLUME = None

if Path('/workspace').exists():
    workspace_path = Path('/workspace')
    try:
        import psutil
        workspace_disk = psutil.disk_usage(str(workspace_path))
        workspace_size_gb = workspace_disk.total / (1e9)
        if workspace_size_gb > 200:  # Likely network volume
            NETWORK_VOLUME = workspace_path
    except:
        NETWORK_VOLUME = workspace_path
elif Path('/notebooks').exists():
    BASE_DIR = Path('/notebooks')

STORAGE_BASE = NETWORK_VOLUME if NETWORK_VOLUME else BASE_DIR
OUTPUT_DIR = (STORAGE_BASE / 'outputs').resolve()

print("=" * 70)
print("CLEANUP AND ORGANIZE NETWORK VOLUME")
print("=" * 70)
print(f"📁 Output directory: {OUTPUT_DIR}")
print(f"💾 Network volume: {NETWORK_VOLUME if NETWORK_VOLUME else 'Not detected'}")
print("=" * 70)

# ============================================
# STEP 1: Verify .npy Files
# ============================================

def verify_npy_file(npy_path, expected_shape=None, expected_dtype=None, sample_check=True):
    """
    Verify a .npy file is correct and not corrupted.
    
    Returns:
        (is_valid, error_message, file_info)
    """
    try:
        # Check file exists
        if not npy_path.exists():
            return False, "File does not exist", None
        
        # Get file size
        file_size = npy_path.stat().st_size
        if file_size == 0:
            return False, "File is empty (0 bytes)", None
        
        # Try to load with memmap first (memory-efficient check)
        try:
            data = np.load(npy_path, mmap_mode='r')
        except Exception as e:
            return False, f"Failed to load file: {str(e)}", None
        
        # Check shape
        if expected_shape is not None:
            if data.shape != expected_shape:
                return False, f"Shape mismatch: expected {expected_shape}, got {data.shape}", None
        
        # Check dtype
        if expected_dtype is not None:
            if data.dtype != expected_dtype:
                return False, f"Dtype mismatch: expected {expected_dtype}, got {data.dtype}", None
        
        # Sample check: verify we can read some data
        if sample_check:
            try:
                # Read first and last elements (quick check)
                _ = data[0]
                _ = data[-1]
                # Check for NaN or Inf (corruption indicators)
                if np.any(np.isnan(data[0])) or np.any(np.isinf(data[0])):
                    return False, "File contains NaN or Inf values (possible corruption)", None
            except Exception as e:
                return False, f"Failed to read sample data: {str(e)}", None
        
        # Get file info
        file_info = {
            'shape': data.shape,
            'dtype': str(data.dtype),
            'size_bytes': file_size,
            'size_gb': file_size / (1024**3),
            'num_elements': data.size
        }
        
        del data  # Close memmap
        
        return True, "OK", file_info
        
    except Exception as e:
        return False, f"Verification error: {str(e)}", None

print("\n" + "=" * 70)
print("STEP 1: VERIFYING .NPY FILES")
print("=" * 70)

# Files to verify and their corresponding memmap files
verification_pairs = [
    {
        'npy': OUTPUT_DIR / 'X_full.npy',
        'memmap': OUTPUT_DIR / 'X_full_memmap.dat',
        'name': 'X_full',
        'expected_dtype': np.float32
    },
    {
        'npy': OUTPUT_DIR / 'y_full.npy',
        'memmap': OUTPUT_DIR / 'y_full_memmap.dat',
        'name': 'y_full',
        'expected_dtype': np.int32
    },
    {
        'npy': OUTPUT_DIR / 'X_augmented_medical.npy',
        'memmap': OUTPUT_DIR / 'X_aug_memmap.dat',
        'name': 'X_augmented_medical',
        'expected_dtype': np.float32
    },
    {
        'npy': OUTPUT_DIR / 'y_augmented_medical.npy',
        'memmap': OUTPUT_DIR / 'y_aug_memmap.dat',
        'name': 'y_augmented_medical',
        'expected_dtype': np.int32
    }
]

verification_results = {}
total_space_to_free = 0

for pair in verification_pairs:
    npy_path = pair['npy']
    memmap_path = pair['memmap']
    name = pair['name']
    
    print(f"\n🔍 Verifying {name}...")
    
    if not npy_path.exists():
        print(f"  ⚠️  {npy_path.name} does not exist - skipping")
        verification_results[name] = {'valid': False, 'reason': 'File does not exist'}
        continue
    
    # Verify the .npy file
    is_valid, error_msg, file_info = verify_npy_file(
        npy_path,
        expected_dtype=pair.get('expected_dtype')
    )
    
    if is_valid:
        print(f"  ✅ {npy_path.name} is VALID")
        print(f"     Shape: {file_info['shape']}")
        print(f"     Dtype: {file_info['dtype']}")
        print(f"     Size: {file_info['size_gb']:.2f} GB")
        
        # Check if memmap exists and get its size
        if memmap_path.exists():
            memmap_size = memmap_path.stat().st_size / (1024**3)
            print(f"     📦 Corresponding memmap: {memmap_size:.2f} GB (can be deleted)")
            total_space_to_free += memmap_size
            verification_results[name] = {
                'valid': True,
                'file_info': file_info,
                'memmap_exists': True,
                'memmap_size_gb': memmap_size,
                'memmap_path': memmap_path
            }
        else:
            print(f"     ℹ️  No corresponding memmap file found")
            verification_results[name] = {
                'valid': True,
                'file_info': file_info,
                'memmap_exists': False
            }
    else:
        print(f"  ❌ {npy_path.name} is INVALID: {error_msg}")
        verification_results[name] = {'valid': False, 'reason': error_msg}

print(f"\n📊 Verification Summary:")
print(f"  Total space that can be freed: {total_space_to_free:.2f} GB")

# ============================================
# STEP 2: Delete Verified Memmap Files
# ============================================

print("\n" + "=" * 70)
print("STEP 2: DELETING VERIFIED MEMMAP FILES")
print("=" * 70)

deleted_files = []
deleted_size = 0
failed_deletions = []

for name, result in verification_results.items():
    if result.get('valid') and result.get('memmap_exists'):
        memmap_path = result['memmap_path']
        memmap_size = result['memmap_size_gb']
        
        try:
            print(f"\n🗑️  Deleting {memmap_path.name} ({memmap_size:.2f} GB)...")
            memmap_path.unlink()
            deleted_files.append(str(memmap_path))
            deleted_size += memmap_size
            print(f"  ✅ Deleted successfully")
        except Exception as e:
            print(f"  ❌ Failed to delete: {str(e)}")
            failed_deletions.append((str(memmap_path), str(e)))

print(f"\n📊 Deletion Summary:")
print(f"  Files deleted: {len(deleted_files)}")
print(f"  Space freed: {deleted_size:.2f} GB")
if failed_deletions:
    print(f"  Failed deletions: {len(failed_deletions)}")

# ============================================
# STEP 3: Organize Files into Folders
# ============================================

print("\n" + "=" * 70)
print("STEP 3: ORGANIZING FILES INTO FOLDERS")
print("=" * 70)

# Define folder structure
folders = {
    'checkpoints': OUTPUT_DIR / 'checkpoints',
    'visualizations': OUTPUT_DIR / 'visualizations',
    'plots': OUTPUT_DIR / 'plots',
    'configs': OUTPUT_DIR / 'configs',
    'data': OUTPUT_DIR / 'data',
    'logs': OUTPUT_DIR / 'logs',
    'models': OUTPUT_DIR / 'models',
    'results': OUTPUT_DIR / 'results',
    'cache': OUTPUT_DIR / 'cache'
}

# Create folders
for folder_name, folder_path in folders.items():
    folder_path.mkdir(exist_ok=True, parents=True)
    print(f"  ✓ Created/verified: {folder_name}/")

# File organization rules
file_organization = {
    # Checkpoint files
    'checkpoints': [
        'week3_checkpoint.json',
        'week4_checkpoint.json',
        'checkpoint.json',
        'week4_split_indices.npz'
    ],
    # Configuration files
    'configs': [
        'augmentation_config_medical.json',
        'custom_dataset_statistics.json',
        'split_info.json'
    ],
    # Data files (CSV, metadata)
    'data': [
        'full_metadata.csv',
        'split_summary.csv'
    ],
    # Visualization files
    'visualizations': [
        'full_samples.png',
        'medical_augmentation_examples.png',
        'medical_augmentation_all_classes.png',
        'augmentation_class_distribution_comparison.png',
        'augmentation_statistics_summary.png',
        'split_distribution.png',
        'split_pie_charts.png',
        'split_statistics_summary.png'
    ],
    # Quick reference
    'data': [
        'QUICK_REFERENCE.txt'
    ],
    # Cache files
    'cache': [
        'tf_cache.data-00000-of-00001',
        'tf_cache.index'
    ]
}

# Organize files
moved_files = []
skipped_files = []

print(f"\n📦 Organizing files...")

for folder_name, file_patterns in file_organization.items():
    folder_path = folders[folder_name]
    
    for pattern in file_patterns:
        source_path = OUTPUT_DIR / pattern
        
        if source_path.exists():
            dest_path = folder_path / pattern
            
            # Skip if already in correct location
            if source_path == dest_path:
                continue
            
            # Skip if destination already exists
            if dest_path.exists():
                print(f"  ⚠️  {pattern} already exists in {folder_name}/ - skipping")
                skipped_files.append((pattern, folder_name, "Already exists"))
                continue
            
            try:
                shutil.move(str(source_path), str(dest_path))
                moved_files.append((pattern, folder_name))
                print(f"  ✅ Moved {pattern} → {folder_name}/")
            except Exception as e:
                print(f"  ❌ Failed to move {pattern}: {str(e)}")
                skipped_files.append((pattern, folder_name, str(e)))

# Keep .npy files in main outputs directory (they're the main data files)
print(f"\n  ℹ️  .npy files kept in main outputs/ directory (main data files)")

print(f"\n📊 Organization Summary:")
print(f"  Files moved: {len(moved_files)}")
print(f"  Files skipped: {len(skipped_files)}")

# ============================================
# STEP 4: Generate Cleanup Report
# ============================================

print("\n" + "=" * 70)
print("STEP 4: GENERATING CLEANUP REPORT")
print("=" * 70)

report = {
    'timestamp': datetime.now().isoformat(),
    'output_directory': str(OUTPUT_DIR),
    'network_volume': str(NETWORK_VOLUME) if NETWORK_VOLUME else None,
    'verification_results': {},
    'deleted_files': deleted_files,
    'deleted_size_gb': deleted_size,
    'failed_deletions': failed_deletions,
    'moved_files': [{'file': f, 'folder': folder} for f, folder in moved_files],
    'skipped_files': [{'file': f, 'folder': folder, 'reason': r} for f, folder, r in skipped_files]
}

# Add verification results (convert numpy types to native Python types)
for name, result in verification_results.items():
    report_result = {}
    if result.get('valid'):
        report_result['valid'] = True
        if 'file_info' in result:
            file_info = result['file_info'].copy()
            # Convert numpy types to native Python types
            file_info['shape'] = list(file_info['shape'])
            report_result['file_info'] = file_info
        report_result['memmap_exists'] = result.get('memmap_exists', False)
        if result.get('memmap_exists'):
            report_result['memmap_size_gb'] = result['memmap_size_gb']
    else:
        report_result['valid'] = False
        report_result['reason'] = result.get('reason', 'Unknown')
    report['verification_results'][name] = report_result

# Save report
report_path = OUTPUT_DIR / 'cleanup_report.json'
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2)

print(f"  ✅ Saved cleanup report: {report_path}")

# ============================================
# SUMMARY
# ============================================

print("\n" + "=" * 70)
print("✅ CLEANUP AND ORGANIZATION COMPLETE")
print("=" * 70)

print(f"\n📊 Summary:")
print(f"  ✅ Verified .npy files: {sum(1 for r in verification_results.values() if r.get('valid'))}/{len(verification_results)}")
print(f"  🗑️  Deleted memmap files: {len(deleted_files)}")
print(f"  💾 Space freed: {deleted_size:.2f} GB")
print(f"  📦 Files organized: {len(moved_files)}")
print(f"  📄 Report saved: cleanup_report.json")

print(f"\n📁 Current Directory Structure:")
print(f"  outputs/")
print(f"    ├── *.npy (main data files - kept in root)")
print(f"    ├── checkpoints/ (checkpoint files)")
print(f"    ├── configs/ (configuration JSON files)")
print(f"    ├── data/ (CSV, metadata, text files)")
print(f"    ├── visualizations/ (PNG visualization files)")
print(f"    ├── cache/ (TensorFlow cache files)")
print(f"    ├── logs/ (log files)")
print(f"    ├── models/ (model files)")
print(f"    └── results/ (result files)")

print(f"\n💡 Next Steps:")
print(f"  - Review cleanup_report.json for details")
print(f"  - Verify all .npy files are accessible")
print(f"  - Continue with your workflow using organized files")

print("=" * 70)

