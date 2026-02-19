#!/usr/bin/env python3
"""
MEMORY CLEANUP SCRIPT FOR DENORMALIZED ARRAYS
==============================================

Purpose:
- Remove large denormalized arrays from memory after training
- Free up container RAM for next experiments
- Safe cleanup without corrupting data files

Usage:
    python cleanup_denormalized_arrays.py [--aggressive] [--verify]

Options:
    --aggressive    : Force cleanup even if files are being accessed
    --verify        : Verify cleanup success
    --dry-run       : Show what would be deleted without deleting
"""

import numpy as np
import gc
import os
from pathlib import Path
import psutil
import time
import argparse
import sys

def get_memory_status():
    """Get current memory usage"""
    try:
        vm = psutil.virtual_memory()
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        
        return {
            'process_mb': mem_info.rss / 1024 / 1024,
            'system_used_gb': vm.used / (1024**3),
            'system_total_gb': vm.total / (1024**3),
            'system_percent': vm.percent,
        }
    except Exception as e:
        print(f"⚠️  Could not get memory status: {e}")
        return None

def print_memory_status(label=""):
    """Print memory usage nicely"""
    mem = get_memory_status()
    if mem:
        print(f"\n💾 Memory {label}:")
        print(f"   Process: {mem['process_mb']:.0f} MB")
        print(f"   System: {mem['system_used_gb']:.1f}GB / {mem['system_total_gb']:.1f}GB ({mem['system_percent']:.1f}%)")

def clear_system_cache():
    """Clear Linux page cache"""
    try:
        os.system('sync > /dev/null 2>&1')
        time.sleep(0.2)
        os.system('echo 3 > /proc/sys/vm/drop_caches > /dev/null 2>&1')
        time.sleep(0.2)
        print("   ✅ System cache cleared")
        return True
    except Exception as e:
        print(f"   ⚠️  Could not clear cache: {e}")
        return False

def cleanup_arrays(base_dir, aggressive=False, dry_run=False, verify=False):
    """
    Remove denormalized arrays from memory and system
    """
    print("=" * 70)
    print("MEMORY CLEANUP: DENORMALIZED ARRAYS")
    print("=" * 70)
    
    base_dir = Path(base_dir)
    outputs_dir = base_dir / 'outputs'
    
    # Files to clean up
    arrays_to_cleanup = [
        'X_train_denormalized.npy',
        'X_val_denormalized.npy',
        'X_test_denormalized.npy',
        'X_train_baseline.npy',
        'X_val_baseline.npy',
        'X_test_baseline.npy',
    ]
    
    print(f"\n📁 Base directory: {base_dir}")
    print(f"📁 Output directory: {outputs_dir}")
    
    if not outputs_dir.exists():
        print(f"\n❌ Output directory not found: {outputs_dir}")
        return False
    
    print_memory_status("before cleanup")
    
    print(f"\n🔍 Scanning for array files...")
    files_to_delete = []
    total_size = 0
    
    for filename in arrays_to_cleanup:
        filepath = outputs_dir / filename
        if filepath.exists():
            size_gb = filepath.stat().st_size / (1024**3)
            files_to_delete.append((filepath, size_gb))
            total_size += size_gb
            print(f"   ✓ {filename}: {size_gb:.2f} GB")
    
    if not files_to_delete:
        print(f"   ℹ️  No array files found to clean up")
        return True
    
    print(f"\n📊 Summary:")
    print(f"   Files to remove: {len(files_to_delete)}")
    print(f"   Total size: {total_size:.2f} GB")
    
    if dry_run:
        print(f"\n🔍 DRY RUN - No files will be deleted")
        return True
    
    # Ask for confirmation
    if not aggressive:
        print(f"\n⚠️  This will delete {total_size:.2f} GB of files!")
        response = input("   Continue? (yes/no): ").strip().lower()
        if response != 'yes':
            print(f"   ❌ Cancelled")
            return False
    
    # Delete files
    print(f"\n🗑️  Deleting array files...")
    deleted_size = 0
    failed = []
    
    for filepath, size_gb in files_to_delete:
        try:
            filepath.unlink()
            deleted_size += size_gb
            print(f"   ✓ Deleted: {filepath.name} ({size_gb:.2f} GB)")
        except Exception as e:
            print(f"   ❌ Failed: {filepath.name} - {e}")
            failed.append(filepath.name)
    
    # Force garbage collection
    print(f"\n🧹 Running garbage collection...")
    gc.collect()
    time.sleep(1)
    print(f"   ✓ GC completed")
    
    # Clear system cache
    print(f"\n🔄 Clearing system cache...")
    clear_system_cache()
    
    # Print results
    print(f"\n{'='*70}")
    print(f"✅ CLEANUP COMPLETE!")
    print(f"{'='*70}")
    
    print(f"\n📊 Results:")
    print(f"   Deleted: {len(files_to_delete) - len(failed)} files")
    print(f"   Freed: {deleted_size:.2f} GB")
    
    if failed:
        print(f"   Failed: {len(failed)} files")
        for name in failed:
            print(f"     - {name}")
    
    print_memory_status("after cleanup")
    
    # Verify
    if verify:
        print(f"\n✔️  Verifying deletion...")
        remaining = 0
        for filename in arrays_to_cleanup:
            filepath = outputs_dir / filename
            if filepath.exists():
                print(f"   ⚠️  Still exists: {filename}")
                remaining += 1
        
        if remaining == 0:
            print(f"   ✅ All files successfully deleted!")
        else:
            print(f"   ⚠️  {remaining} files still remain")
            return False
    
    return True

def cleanup_memory_variables():
    """
    Clear variables from Python namespace
    (Note: This is mainly for reference - called at script end)
    """
    print(f"\n🧠 Clearing Python variables...")
    
    # Get all variable names
    current_vars = list(globals().keys())
    
    # List of variable prefixes to keep
    keep_prefixes = ['__', '_', 'argv', 'exit_code']
    
    deleted_count = 0
    for var_name in current_vars:
        should_keep = any(var_name.startswith(prefix) for prefix in keep_prefixes)
        
        if not should_keep and var_name not in ['print', 'input', 'globals']:
            try:
                del globals()[var_name]
                deleted_count += 1
            except:
                pass
    
    print(f"   ✓ Cleared {deleted_count} variables")
    
    gc.collect()
    print(f"   ✓ GC completed")

def main():
    parser = argparse.ArgumentParser(
        description='Clean up denormalized array files from memory'
    )
    parser.add_argument('--aggressive', action='store_true',
                       help='Skip confirmation prompts')
    parser.add_argument('--verify', action='store_true',
                       help='Verify cleanup success')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be deleted without deleting')
    parser.add_argument('--dir', type=str, default='/workspace',
                       help='Base directory (default: /workspace)')
    
    args = parser.parse_args()
    
    try:
        success = cleanup_arrays(
            base_dir=args.dir,
            aggressive=args.aggressive,
            dry_run=args.dry_run,
            verify=args.verify
        )
        
        exit_code = 0 if success else 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit_code = 1
    
    # Clear memory before exit
    cleanup_memory_variables()
    
    print(f"\n{'='*70}")
    if exit_code == 0:
        print(f"✅ Script completed successfully")
    else:
        print(f"❌ Script failed")
    print(f"{'='*70}\n")
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
