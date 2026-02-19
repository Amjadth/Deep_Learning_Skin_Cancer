#!/usr/bin/env python3
"""
Memory Cleanup & Diagnostic Script
===================================
Checks what's in memory, analyzes loaded arrays, and safely frees resources.
Perfect before running Week 8 on RunPod to ensure clean memory state.
"""

import gc
import sys
import psutil
import numpy as np
from pathlib import Path

def get_memory_info():
    """Get detailed memory information."""
    vm = psutil.virtual_memory()
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    
    return {
        'process_rss_mb': mem_info.rss / (1024**2),
        'process_vms_mb': mem_info.vms / (1024**2),
        'process_percent': process.memory_percent(),
        'system_used_gb': vm.used / (1024**3),
        'system_available_gb': vm.available / (1024**3),
        'system_total_gb': vm.total / (1024**3),
        'system_percent': vm.percent,
    }

def print_memory_header(title):
    """Print formatted header."""
    print(f"\n{'='*70}")
    print(f"📊 {title}")
    print(f"{'='*70}")

def analyze_objects():
    """Analyze what Python objects are in memory."""
    print_memory_header("PYTHON OBJECTS IN MEMORY")
    
    import sys
    
    # Get all objects
    all_objects = gc.get_objects()
    print(f"\n📦 Total objects: {len(all_objects):,}")
    
    # Count by type
    from collections import Counter
    type_counts = Counter(type(obj).__name__ for obj in all_objects)
    
    print(f"\n🔍 Top 15 object types by count:")
    for obj_type, count in type_counts.most_common(15):
        # Get size estimate
        try:
            total_size = sum(sys.getsizeof(obj) for obj in all_objects if type(obj).__name__ == obj_type)
            size_mb = total_size / (1024**2)
            print(f"   {obj_type:30} Count: {count:8,}  Size: {size_mb:8.2f} MB")
        except:
            print(f"   {obj_type:30} Count: {count:8,}")

def find_large_arrays():
    """Find large numpy arrays and other large objects in memory."""
    print_memory_header("LARGE ARRAYS & OBJECTS IN MEMORY")
    
    large_objects = []
    
    for obj in gc.get_objects():
        try:
            # Check numpy arrays
            if isinstance(obj, np.ndarray):
                size_mb = obj.nbytes / (1024**2)
                if size_mb > 1:  # Only report arrays > 1 MB
                    large_objects.append({
                        'type': 'ndarray',
                        'shape': obj.shape,
                        'dtype': str(obj.dtype),
                        'size_mb': size_mb,
                        'obj': obj
                    })
            
            # Check lists
            elif isinstance(obj, list) and len(obj) > 1000:
                try:
                    size_mb = sum(sys.getsizeof(item) for item in obj) / (1024**2)
                    if size_mb > 10:
                        large_objects.append({
                            'type': 'list',
                            'len': len(obj),
                            'size_mb': size_mb,
                            'obj': obj
                        })
                except:
                    pass
            
            # Check dicts
            elif isinstance(obj, dict) and len(obj) > 100:
                try:
                    size_mb = sys.getsizeof(obj) / (1024**2)
                    if size_mb > 10:
                        large_objects.append({
                            'type': 'dict',
                            'len': len(obj),
                            'size_mb': size_mb,
                            'obj': obj
                        })
                except:
                    pass
        except:
            pass
    
    if not large_objects:
        print("\n✅ No large arrays/objects found in memory")
        return large_objects
    
    # Sort by size
    large_objects.sort(key=lambda x: x['size_mb'], reverse=True)
    
    print(f"\n🔴 Found {len(large_objects)} large objects:\n")
    for i, obj_info in enumerate(large_objects, 1):
        print(f"{i}. Type: {obj_info['type']}")
        if obj_info['type'] == 'ndarray':
            print(f"   Shape: {obj_info['shape']}, Dtype: {obj_info['dtype']}")
        elif obj_info['type'] in ['list', 'dict']:
            print(f"   Length: {obj_info['len']}")
        print(f"   Size: {obj_info['size_mb']:.2f} MB")
        print()
    
    return large_objects

def delete_large_arrays(large_objects):
    """Actually DELETE large arrays from memory."""
    print_memory_header("DELETING LARGE ARRAYS FROM MEMORY")
    
    if not large_objects:
        print("\n✅ No large arrays to delete")
        return
    
    print(f"\n🗑️  Deleting {len(large_objects)} large objects...\n")
    
    deleted_size = 0
    for i, obj_info in enumerate(large_objects, 1):
        try:
            size_mb = obj_info['size_mb']
            deleted_size += size_mb
            
            # Get reference to the object
            obj = obj_info['obj']
            
            # Delete it
            if isinstance(obj, np.ndarray):
                del obj
                print(f"{i}. ✓ Deleted ndarray ({size_mb:.2f} MB)")
            elif obj_info['type'] in ['list', 'dict']:
                del obj
                print(f"{i}. ✓ Deleted {obj_info['type']} ({size_mb:.2f} MB)")
        except Exception as e:
            print(f"{i}. ❌ Failed to delete: {e}")
    
    print(f"\n✅ Total deleted: {deleted_size:.2f} MB")

def clear_linux_cache():
    """Clear Linux page cache safely."""
    import os
    try:
        print("  🧹 Clearing Linux page cache...")
        os.system('sync > /dev/null 2>&1')
        os.system('echo 3 > /proc/sys/vm/drop_caches > /dev/null 2>&1')
        print("  ✓ Cache cleared")
        return True
    except:
        print("  ⚠️  Could not clear Linux cache (may need root)")
        return False

def aggressive_cleanup():
    """Aggressively clean up memory - DELETE ARRAYS FIRST."""
    print_memory_header("AGGRESSIVE MEMORY CLEANUP")
    
    # Find large arrays FIRST
    print("\n🔍 Step 1: Finding large arrays in memory...")
    large_objects = find_large_arrays()
    
    # DELETE them
    if large_objects:
        print("\n🔍 Step 2: Deleting large arrays...")
        delete_large_arrays(large_objects)
    
    # Now do garbage collection
    print("\n🔍 Step 3: Running garbage collection...")
    for i in range(5):  # Increased from 3 to 5 passes
        collected = gc.collect()
        print(f"  Pass {i+1}: Collected {collected:,} objects")
    
    # Clear TensorFlow session if available
    try:
        import tensorflow as tf
        print("\n🔍 Step 4: Clearing TensorFlow session...")
        tf.keras.backend.clear_session()
        print("  ✓ TensorFlow session cleared")
    except ImportError:
        pass
    except Exception as e:
        print(f"  ⚠️  Could not clear TensorFlow: {e}")
    
    # Final garbage collection pass
    print("\n🔍 Step 5: Final garbage collection pass...")
    gc.collect()
    print("  ✓ Complete")
    
    # Clear Linux cache
    print("\n🔍 Step 6: Clearing system cache...")
    clear_linux_cache()

def cleanup_memory():
    """Aggressively clean up memory."""
    aggressive_cleanup()

def print_memory_status(label=""):
    """Print current memory status."""
    mem = get_memory_info()
    
    print(f"\n💾 MEMORY STATUS: {label}")
    print(f"   Process: {mem['process_rss_mb']:.0f} MB ({mem['process_percent']:.1f}%)")
    print(f"   System: {mem['system_used_gb']:.1f}GB / {mem['system_total_gb']:.1f}GB ({mem['system_percent']:.1f}%)")
    print(f"   Available: {mem['system_available_gb']:.1f}GB")

def main():
    """Main diagnostic and cleanup routine."""
    print("\n" + "="*70)
    print("🧹 MEMORY CLEANUP & DIAGNOSTIC TOOL")
    print("="*70)
    
    # Initial memory status
    print_memory_status("BEFORE ANALYSIS")
    
    # Analyze what's in memory
    analyze_objects()
    
    # Aggressive cleanup (finds and deletes large arrays)
    cleanup_memory()
    
    # Final memory status
    print_memory_status("AFTER CLEANUP")
    
    # Summary
    print(f"\n{'='*70}")
    print("✅ CLEANUP COMPLETE")
    print(f"{'='*70}")
    
    mem = get_memory_info()
    print(f"\n📊 FINAL STATUS:")
    print(f"   Process memory: {mem['process_rss_mb']:.0f} MB")
    print(f"   System memory: {mem['system_used_gb']:.1f}GB used / {mem['system_available_gb']:.1f}GB available")
    print(f"   System utilization: {mem['system_percent']:.1f}%")
    
    if mem['system_percent'] < 50:
        print(f"\n✅ Memory is clean and ready for training!")
    elif mem['system_percent'] < 75:
        print(f"\n⚠️  Memory usage is moderate. Ready for training.")
    else:
        print(f"\n⚠️  Memory usage is still high. Consider closing other applications.")
    
    print(f"\n{'='*70}\n")

if __name__ == '__main__':
    import os
    main()
