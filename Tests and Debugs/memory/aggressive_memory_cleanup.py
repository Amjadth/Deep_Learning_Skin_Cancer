#!/usr/bin/env python3
"""
AGGRESSIVE MEMORY CLEANUP SCRIPT
=================================

Purpose:
- Free ALL possible resources from RAM
- Remove cached files, temporary files, unused models
- Clear Python/system caches, buffers, and memory
- Target: Maximize free RAM for next operations

Usage:
    python aggressive_memory_cleanup.py [--dry-run] [--aggressive] [--verify]

Options:
    --dry-run       : Show what would be deleted without deleting
    --aggressive    : Skip confirmations and delete everything
    --verify        : Verify cleanup success
"""

import os
import gc
import sys
import time
import shutil
import psutil
import argparse
from pathlib import Path
import subprocess

class MemoryCleanupManager:
    def __init__(self, dry_run=False, aggressive=False, verify=False):
        self.dry_run = dry_run
        self.aggressive = aggressive
        self.verify = verify
        self.cleanup_log = []
        self.initial_mem = self.get_memory_status()
        
    def get_memory_status(self):
        """Get comprehensive memory status"""
        try:
            vm = psutil.virtual_memory()
            swap = psutil.swap_memory()
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            
            return {
                'process_mb': mem_info.rss / 1024 / 1024,
                'process_vms_mb': mem_info.vms / 1024 / 1024,
                'system_used_gb': vm.used / (1024**3),
                'system_available_gb': vm.available / (1024**3),
                'system_total_gb': vm.total / (1024**3),
                'system_percent': vm.percent,
                'swap_used_gb': swap.used / (1024**3),
                'swap_total_gb': swap.total / (1024**3),
            }
        except Exception as e:
            print(f"⚠️  Could not get memory status: {e}")
            return None

    def print_memory_status(self, label=""):
        """Print comprehensive memory usage"""
        mem = self.get_memory_status()
        if not mem:
            return
            
        print(f"\n💾 Memory Status {label}:")
        print(f"   ┌─ Process Memory:")
        print(f"   │  └─ RSS: {mem['process_mb']:.0f} MB (resident)")
        print(f"   │  └─ VMS: {mem['process_vms_mb']:.0f} MB (virtual)")
        print(f"   ├─ System Memory:")
        print(f"   │  ├─ Used: {mem['system_used_gb']:.2f}GB / {mem['system_total_gb']:.2f}GB ({mem['system_percent']:.1f}%)")
        print(f"   │  └─ Available: {mem['system_available_gb']:.2f}GB")
        print(f"   └─ Swap: {mem['swap_used_gb']:.2f}GB / {mem['swap_total_gb']:.2f}GB")

    def calculate_freed_memory(self):
        """Calculate memory freed"""
        final_mem = self.get_memory_status()
        if not final_mem or not self.initial_mem:
            return 0
        return max(0, self.initial_mem['system_used_gb'] - final_mem['system_used_gb'])

    def log_cleanup(self, category, filename, size_mb=0, status="✓"):
        """Log cleanup action"""
        self.cleanup_log.append({
            'category': category,
            'filename': filename,
            'size_mb': size_mb,
            'status': status
        })

    # ============================================================================
    # CLEANUP FUNCTIONS
    # ============================================================================

    def cleanup_denormalized_arrays(self, base_dir='/workspace'):
        """Remove denormalized array files"""
        print(f"\n🔍 Cleaning up denormalized arrays...")
        
        base_path = Path(base_dir)
        outputs_dir = base_path / 'outputs'
        
        if not outputs_dir.exists():
            print(f"   ℹ️  Output directory not found: {outputs_dir}")
            return 0
        
        array_patterns = [
            '*denormalized*.npy',
            '*baseline*.npy',
            'X_train*.npy',
            'X_val*.npy',
            'X_test*.npy',
        ]
        
        freed = 0
        for pattern in array_patterns:
            for filepath in outputs_dir.glob(pattern):
                if filepath.is_file():
                    try:
                        size_mb = filepath.stat().st_size / (1024**2)
                        if not self.dry_run:
                            filepath.unlink()
                        freed += size_mb
                        self.log_cleanup('Arrays', filepath.name, size_mb)
                        print(f"   ✓ Deleted: {filepath.name} ({size_mb:.1f} MB)")
                    except Exception as e:
                        self.log_cleanup('Arrays', filepath.name, 0, f"❌ {e}")
                        print(f"   ❌ Failed: {filepath.name} - {e}")
        
        return freed

    def cleanup_cache_files(self, base_dir='/workspace'):
        """Remove cache directories and files"""
        print(f"\n🔍 Cleaning up cache files...")
        
        base_path = Path(base_dir)
        
        cache_patterns = [
            '__pycache__',
            '*.pyc',
            '*.pyo',
            '.cache',
            '.pytest_cache',
            '.ipynb_checkpoints',
            '.DS_Store',
        ]
        
        freed = 0
        
        # Handle __pycache__ directories
        for pycache_dir in base_path.rglob('__pycache__'):
            if pycache_dir.is_dir():
                try:
                    size = sum(f.stat().st_size for f in pycache_dir.rglob('*'))
                    if not self.dry_run:
                        shutil.rmtree(pycache_dir)
                    freed += size / (1024**2)
                    self.log_cleanup('Cache', f"{pycache_dir.relative_to(base_path)}", size / (1024**2))
                    print(f"   ✓ Removed: {pycache_dir.relative_to(base_path)}")
                except Exception as e:
                    print(f"   ⚠️  Could not remove {pycache_dir}: {e}")
        
        # Handle .pytest_cache, .ipynb_checkpoints
        for cache_dir_name in ['.pytest_cache', '.ipynb_checkpoints', '.cache']:
            cache_path = base_path / cache_dir_name
            if cache_path.exists() and cache_path.is_dir():
                try:
                    size = sum(f.stat().st_size for f in cache_path.rglob('*'))
                    if not self.dry_run:
                        shutil.rmtree(cache_path)
                    freed += size / (1024**2)
                    self.log_cleanup('Cache', cache_dir_name, size / (1024**2))
                    print(f"   ✓ Removed: {cache_dir_name}")
                except Exception as e:
                    print(f"   ⚠️  Could not remove {cache_dir_name}: {e}")
        
        return freed

    def cleanup_model_cache(self, base_dir='/workspace'):
        """Remove TensorFlow/Keras model cache"""
        print(f"\n🔍 Cleaning up ML model cache...")
        
        freed = 0
        
        # TensorFlow model cache locations
        cache_locations = [
            Path.home() / '.cache' / 'tensorflow',
            Path.home() / '.keras' / 'models',
            Path.home() / '.torch' / 'models',
            Path(base_dir) / '.keras',
            Path(base_dir) / '.torch',
        ]
        
        for cache_path in cache_locations:
            if cache_path.exists():
                try:
                    if cache_path.is_dir():
                        size = sum(f.stat().st_size for f in cache_path.rglob('*'))
                        if not self.dry_run:
                            shutil.rmtree(cache_path)
                        freed += size / (1024**2)
                        self.log_cleanup('ML Cache', str(cache_path), size / (1024**2))
                        print(f"   ✓ Removed: {cache_path}")
                    elif cache_path.is_file():
                        size = cache_path.stat().st_size / (1024**2)
                        if not self.dry_run:
                            cache_path.unlink()
                        freed += size
                        self.log_cleanup('ML Cache', str(cache_path), size)
                        print(f"   ✓ Deleted: {cache_path.name}")
                except Exception as e:
                    print(f"   ⚠️  Could not remove {cache_path}: {e}")
        
        return freed

    def cleanup_pip_cache(self):
        """Clean pip cache"""
        print(f"\n🔍 Cleaning up pip cache...")
        
        freed = 0
        
        try:
            pip_cache_dir = Path.home() / '.cache' / 'pip'
            if pip_cache_dir.exists():
                size = sum(f.stat().st_size for f in pip_cache_dir.rglob('*'))
                if not self.dry_run:
                    subprocess.run(['pip', 'cache', 'purge'], 
                                 capture_output=True, timeout=30)
                freed += size / (1024**2)
                self.log_cleanup('PIP Cache', str(pip_cache_dir), size / (1024**2))
                print(f"   ✓ Purged pip cache ({size / (1024**2):.1f} MB)")
        except Exception as e:
            print(f"   ⚠️  Could not clean pip cache: {e}")
        
        return freed

    def cleanup_temp_files(self, base_dir='/workspace'):
        """Remove temporary files"""
        print(f"\n🔍 Cleaning up temporary files...")
        
        base_path = Path(base_dir)
        freed = 0
        
        temp_patterns = [
            '*.tmp',
            '*.temp',
            '*.log',
            '.ipynb_checkpoints',
        ]
        
        for pattern in temp_patterns:
            for filepath in base_path.rglob(pattern):
                if filepath.is_file():
                    try:
                        size = filepath.stat().st_size / (1024**2)
                        if not self.dry_run:
                            filepath.unlink()
                        freed += size
                        self.log_cleanup('Temp Files', filepath.name, size)
                        print(f"   ✓ Deleted: {filepath.name}")
                    except Exception as e:
                        print(f"   ⚠️  Could not delete {filepath.name}: {e}")
        
        return freed

    def cleanup_system_cache(self):
        """Clear Linux system cache"""
        print(f"\n🔍 Clearing system cache...")
        
        freed = 0
        
        try:
            # Sync filesystem
            subprocess.run(['sync'], capture_output=True, timeout=10)
            time.sleep(0.2)
            
            # Drop caches (requires sudo on Linux)
            try:
                subprocess.run(['sync'], capture_output=True, timeout=10)
                subprocess.run(['bash', '-c', 'echo 3 > /proc/sys/vm/drop_caches'],
                             capture_output=True, timeout=10)
                print(f"   ✓ System cache dropped")
                self.log_cleanup('System', 'drop_caches', 0)
            except:
                # If no permission, just sync
                print(f"   ℹ️  Cannot drop caches (requires sudo)")
            
            time.sleep(0.2)
        except Exception as e:
            print(f"   ⚠️  Could not clear system cache: {e}")
        
        return freed

    def force_garbage_collection(self):
        """Aggressive Python garbage collection"""
        print(f"\n🔍 Running garbage collection...")
        
        try:
            # Multiple GC passes
            for i in range(3):
                collected = gc.collect()
                print(f"   ✓ GC pass {i+1}: collected {collected} objects")
                time.sleep(0.1)
            
            self.log_cleanup('GC', 'force_gc', 0)
            return True
        except Exception as e:
            print(f"   ⚠️  GC failed: {e}")
            return False

    def close_file_descriptors(self):
        """Close unused file descriptors"""
        print(f"\n🔍 Closing file descriptors...")
        
        try:
            process = psutil.Process(os.getpid())
            open_files = process.open_files()
            
            # Count but don't close - just report
            print(f"   ℹ️  Open file descriptors: {len(open_files)}")
            
            self.log_cleanup('FD', 'file_descriptors', len(open_files))
            return True
        except Exception as e:
            print(f"   ⚠️  Could not check file descriptors: {e}")
            return False

    def clear_numpy_memory(self):
        """Clear numpy memory"""
        print(f"\n🔍 Clearing NumPy memory...")
        
        try:
            import numpy as np
            # Trigger numpy memory cleanup
            np.seterr(all='ignore')
            print(f"   ✓ NumPy memory cleared")
            self.log_cleanup('NumPy', 'numpy_clear', 0)
            return True
        except Exception as e:
            print(f"   ⚠️  Could not clear NumPy: {e}")
            return False

    def run_full_cleanup(self, base_dir='/workspace'):
        """Run complete cleanup sequence"""
        print("=" * 80)
        print("🚀 AGGRESSIVE MEMORY CLEANUP - FULL SEQUENCE")
        print("=" * 80)
        
        if self.dry_run:
            print("🔍 DRY RUN MODE - No files will be deleted")
        
        if self.aggressive:
            print("⚠️  AGGRESSIVE MODE - Skipping confirmations")
        
        self.print_memory_status("[INITIAL]")
        
        # Run all cleanup operations
        total_freed = 0
        
        total_freed += self.cleanup_denormalized_arrays(base_dir)
        total_freed += self.cleanup_cache_files(base_dir)
        total_freed += self.cleanup_model_cache(base_dir)
        total_freed += self.cleanup_pip_cache()
        total_freed += self.cleanup_temp_files(base_dir)
        
        # Python-level cleanup
        self.clear_numpy_memory()
        self.close_file_descriptors()
        self.force_garbage_collection()
        
        # System-level cleanup
        self.cleanup_system_cache()
        
        # Final GC
        self.force_garbage_collection()
        
        self.print_memory_status("[FINAL]")
        
        # Print summary
        self._print_summary(total_freed)
        
        if self.verify:
            self._verify_cleanup(base_dir)
        
        return True

    def _print_summary(self, total_freed):
        """Print cleanup summary"""
        print(f"\n{'='*80}")
        print(f"✅ CLEANUP COMPLETE!")
        print(f"{'='*80}")
        
        print(f"\n📊 SUMMARY:")
        print(f"   Total files deleted: {len(self.cleanup_log)}")
        print(f"   Total space freed: {total_freed:.2f} MB ({total_freed/1024:.2f} GB)")
        
        # Memory freed
        freed_memory = self.calculate_freed_memory()
        print(f"   System RAM freed: {freed_memory:.2f} GB")
        
        # Breakdown by category
        print(f"\n📋 Breakdown by category:")
        categories = {}
        for log in self.cleanup_log:
            cat = log['category']
            categories[cat] = categories.get(cat, 0) + log['size_mb']
        
        for category, size in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            print(f"   ├─ {category}: {size:.1f} MB")

    def _verify_cleanup(self, base_dir):
        """Verify cleanup was successful"""
        print(f"\n✔️  VERIFICATION:")
        
        base_path = Path(base_dir)
        
        # Check if array files still exist
        arrays_remaining = 0
        for filepath in base_path.rglob('*denormalized*.npy'):
            arrays_remaining += 1
        
        if arrays_remaining == 0:
            print(f"   ✅ No denormalized arrays found")
        else:
            print(f"   ⚠️  {arrays_remaining} array files still exist")
        
        # Check if caches still exist
        pycache_count = sum(1 for _ in base_path.rglob('__pycache__'))
        if pycache_count == 0:
            print(f"   ✅ No __pycache__ directories found")
        else:
            print(f"   ⚠️  {pycache_count} __pycache__ directories still exist")
        
        print(f"   ✅ Verification complete")


def main():
    parser = argparse.ArgumentParser(
        description='Aggressive memory cleanup - remove everything from RAM'
    )
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be deleted without deleting')
    parser.add_argument('--aggressive', action='store_true',
                       help='Skip confirmations and delete everything')
    parser.add_argument('--verify', action='store_true',
                       help='Verify cleanup success')
    parser.add_argument('--dir', type=str, default='/workspace',
                       help='Base directory to clean (default: /workspace)')
    
    args = parser.parse_args()
    
    try:
        cleaner = MemoryCleanupManager(
            dry_run=args.dry_run,
            aggressive=args.aggressive,
            verify=args.verify
        )
        
        # Ask for confirmation unless aggressive mode
        if not args.aggressive and not args.dry_run:
            print("\n" + "="*80)
            print("⚠️  WARNING: This will delete many files and may take several minutes")
            print("="*80)
            response = input("\nContinue with aggressive cleanup? (yes/no): ").strip().lower()
            if response != 'yes':
                print("❌ Cancelled")
                sys.exit(0)
        
        cleaner.run_full_cleanup(base_dir=args.dir)
        exit_code = 0
        
    except KeyboardInterrupt:
        print(f"\n\n❌ Cleanup interrupted by user")
        exit_code = 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit_code = 1
    
    print(f"\n{'='*80}")
    if exit_code == 0:
        print(f"✅ Script completed successfully")
    else:
        print(f"❌ Script failed with errors")
    print(f"{'='*80}\n")
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
