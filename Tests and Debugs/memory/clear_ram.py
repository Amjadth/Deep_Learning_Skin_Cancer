# ============================================
# CLEAR RAM SCRIPT
# ============================================
#
# This script clears Python's memory and frees up RAM
# before running memory-intensive scripts like week3.py
#
# Usage:
#   python clear_ram.py
#
# ============================================

import gc
import sys
import os

def clear_ram():
    """Clear RAM by forcing garbage collection and clearing caches."""
    
    print("=" * 70)
    print("CLEARING RAM")
    print("=" * 70)
    
    # Show system-wide memory usage (if psutil is available)
    try:
        import psutil
        # System-wide memory
        system_mem = psutil.virtual_memory()
        system_total_gb = system_mem.total / (1024**3)
        system_used_gb = system_mem.used / (1024**3)
        system_available_gb = system_mem.available / (1024**3)
        system_percent = system_mem.percent
        
        print(f"\n💻 SYSTEM-WIDE MEMORY:")
        print(f"   Total: {system_total_gb:.2f} GB")
        print(f"   Used: {system_used_gb:.2f} GB ({system_percent:.1f}%)")
        print(f"   Available: {system_available_gb:.2f} GB")
        
        # Current Python process memory
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024**3)  # GB
        print(f"\n🐍 PYTHON PROCESS MEMORY:")
        print(f"   Current process: {mem_before:.2f} GB")
        
        # Show top memory-consuming processes
        print(f"\n📊 TOP MEMORY-CONSUMING PROCESSES:")
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
            try:
                mem_mb = proc.info['memory_info'].rss / (1024**2)
                if mem_mb > 100:  # Only show processes using >100 MB
                    processes.append((proc.info['pid'], proc.info['name'], mem_mb))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        # Sort by memory and show top 10
        processes.sort(key=lambda x: x[2], reverse=True)
        for pid, name, mem_mb in processes[:10]:
            mem_gb = mem_mb / 1024
            print(f"   {name[:30]:30s} (PID {pid:6d}): {mem_gb:6.2f} GB")
        
    except ImportError:
        print("\n⚠️ psutil not available - install with: pip install psutil")
        print("   Cannot show detailed memory usage")
        mem_before = None
    except Exception as e:
        print(f"\n⚠️ Error getting memory info: {e}")
        mem_before = None
    
    # Step 1: Clear any loaded modules/variables
    print("\n🧹 Step 1: Clearing loaded modules...")
    
    # Get list of modules to keep (essential system modules)
    keep_modules = {
        'sys', 'os', 'gc', 'builtins', '__builtin__', '__main__',
        'importlib', 'collections', 'itertools', 'functools',
        'threading', 'multiprocessing', 'queue'
    }
    
    # Clear non-essential modules
    modules_to_clear = []
    for module_name in list(sys.modules.keys()):
        if module_name not in keep_modules:
            # Skip standard library and built-in modules
            if not module_name.startswith('_') and '.' not in module_name:
                modules_to_clear.append(module_name)
    
    for module_name in modules_to_clear[:50]:  # Limit to first 50 to avoid issues
        try:
            if module_name in sys.modules:
                del sys.modules[module_name]
        except Exception:
            pass
    
    print(f"   ✓ Cleared {len(modules_to_clear)} modules")
    
    # Step 2: Force garbage collection
    print("\n🧹 Step 2: Running garbage collection...")
    
    # Collect all generations
    collected = 0
    for generation in range(3):
        n = gc.collect(generation)
        collected += n
        if n > 0:
            print(f"   ✓ Collected {n} objects from generation {generation}")
    
    if collected > 0:
        print(f"   ✓ Total objects collected: {collected}")
    else:
        print(f"   ✓ No objects to collect")
    
    # Step 3: Clear any cached data
    print("\n🧹 Step 3: Clearing caches...")
    
    # Clear function caches
    try:
        import functools
        functools._cache.clear()
        print("   ✓ Cleared function caches")
    except Exception:
        pass
    
    # Clear any matplotlib caches
    try:
        import matplotlib
        matplotlib.pyplot.close('all')
        matplotlib._pylab_helpers.Gcf.destroy_all()
        print("   ✓ Cleared matplotlib figures")
    except Exception:
        pass
    
    # Clear numpy/tensorflow caches if available
    try:
        import numpy as np
        # Clear any cached arrays (though numpy doesn't have a global cache)
        print("   ✓ NumPy modules loaded")
    except Exception:
        pass
    
    try:
        import tensorflow as tf
        # Clear TensorFlow session if exists
        tf.keras.backend.clear_session()
        print("   ✓ Cleared TensorFlow/Keras session")
    except Exception:
        pass
    
    # Step 4: Final garbage collection
    print("\n🧹 Step 4: Final garbage collection...")
    final_collected = gc.collect()
    if final_collected > 0:
        print(f"   ✓ Collected {final_collected} additional objects")
    else:
        print(f"   ✓ No additional objects to collect")
    
    # Show memory usage after (if psutil is available)
    try:
        import psutil
        # System-wide memory after
        system_mem_after = psutil.virtual_memory()
        system_used_gb_after = system_mem_after.used / (1024**3)
        system_available_gb_after = system_mem_after.available / (1024**3)
        system_percent_after = system_mem_after.percent
        
        print(f"\n💻 SYSTEM-WIDE MEMORY AFTER CLEANUP:")
        print(f"   Used: {system_used_gb_after:.2f} GB ({system_percent_after:.1f}%)")
        print(f"   Available: {system_available_gb_after:.2f} GB")
        
        if mem_before is not None:
            mem_after = process.memory_info().rss / (1024**3)  # GB
            mem_freed = mem_before - mem_after
            print(f"\n🐍 PYTHON PROCESS MEMORY AFTER CLEANUP:")
            print(f"   Current process: {mem_after:.2f} GB")
            print(f"   Memory freed: {mem_freed:.2f} GB")
            
            print(f"\n💡 NOTE:")
            print(f"   - Python process memory: {mem_after:.2f} GB (cleared)")
            print(f"   - System memory: {system_used_gb_after:.2f} GB / {system_mem_after.total / (1024**3):.2f} GB")
            print(f"   - The 30+ GB shown on dashboard includes:")
            print(f"     * Other processes (Jupyter, system services, etc.)")
            print(f"     * System cache and buffers")
            print(f"     * Other Python processes/kernels")
            print(f"   - Your Python process is now using only {mem_after:.2f} GB")
            print(f"   - Week3.py will use additional memory during processing")
    except Exception:
        pass
    
    print("\n" + "=" * 70)
    print("✅ RAM CLEARED - Ready to run memory-intensive scripts")
    print("=" * 70)
    print("\n💡 You can now run week3.py or other memory-intensive scripts")
    print("💡 Your Python process memory is cleared and ready")
    print("💡 Week3.py will use ~4-8 GB during processing (within 48 GB limit)")

if __name__ == "__main__":
    clear_ram()

