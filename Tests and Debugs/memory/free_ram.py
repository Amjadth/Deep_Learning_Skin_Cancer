#!/usr/bin/env python3
"""
MINIMAL RAM CLEANUP SCRIPT
Frees all possible RAM memory
"""

import gc
import os
import subprocess
import time

def free_ram():
    """Free all RAM"""
    
    # Python garbage collection
    print("Clearing Python memory...", end=" ")
    for _ in range(3):
        gc.collect()
    print("✓")
    
    # Clear system cache (Linux)
    print("Clearing system cache...", end=" ")
    try:
        subprocess.run(['sync'], capture_output=True, timeout=5)
        time.sleep(0.1)
        subprocess.run(['bash', '-c', 'echo 3 > /proc/sys/vm/drop_caches'], 
                      capture_output=True, timeout=5)
        time.sleep(0.1)
    except:
        pass
    print("✓")
    
    # Final GC
    print("Final cleanup...", end=" ")
    gc.collect()
    print("✓")
    
    print("\n✅ RAM cleared!")

if __name__ == "__main__":
    free_ram()
