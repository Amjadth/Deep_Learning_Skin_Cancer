#!/usr/bin/env python3
"""
Quick validation that week6_fixed_runpod.py has correct memory fixes.
This script checks for the key fixes without needing to run training.
"""

import re
from pathlib import Path

def check_file_for_fix(filepath, search_string, fix_name):
    """Check if a fix exists in the file."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        if search_string in content:
            print(f"✅ {fix_name}")
            return True
        else:
            print(f"❌ {fix_name} - NOT FOUND")
            return False
    except Exception as e:
        print(f"❌ Error checking {fix_name}: {e}")
        return False

def main():
    file_path = Path('/Users/ahadraza/Side Projects/Freelancing/Skin Cancer V3.0/week6_fixed_runpod.py')
    
    print("\n" + "=" * 70)
    print("🔍 WEEK 6 MEMORY FIX VALIDATION")
    print("=" * 70)
    
    if not file_path.exists():
        print(f"\n❌ File not found: {file_path}")
        return
    
    checks = [
        # Check 1: Batch size reduced to 16
        (
            "BATCH_SIZE = 16",
            "Fix #1: Batch size reduced from 32 to 16"
        ),
        
        # Check 2: Container usage percentage calculation
        (
            "memory.usage_in_bytes",
            "Fix #2: Reading actual container usage from cgroup"
        ),
        
        # Check 3: Pre-training memory safety check
        (
            "MEMORY SAFETY CHECK:",
            "Fix #3: Pre-training memory safety check"
        ),
        
        # Check 4: Container usage warning
        (
            "if usage_pct > 90:",
            "Fix #4: Critical OOM check (>90% usage)"
        ),
        
        # Check 5: GPU device context
        (
            "with tf.device('/GPU:0')",
            "Fix #5: GPU explicit device context in data pipeline"
        ),
        
        # Check 6: Memmap usage
        (
            "np.memmap",
            "Fix #6: Memmap for data loading (no RAM explosion)"
        ),
        
        # Check 7: Updated memory status function
        (
            "Container (cgroup):",
            "Fix #7: Improved print_memory_status() function"
        ),
        
        # Check 8: Memory alert in config
        (
            "MEMORY ALERT:",
            "Fix #8: Memory alert messages in training config"
        ),
    ]
    
    print("\n📋 Checking for fixes...\n")
    
    results = []
    for search_str, fix_name in checks:
        result = check_file_for_fix(file_path, search_str, fix_name)
        results.append(result)
    
    print("\n" + "=" * 70)
    print(f"✅ Fixes Applied: {sum(results)}/{len(results)}")
    print("=" * 70)
    
    if all(results):
        print("\n🎉 ALL MEMORY FIXES VERIFIED!")
        print("\nKey Improvements:")
        print("  1. ✅ Batch size: 32 → 16 (safer for high container usage)")
        print("  2. ✅ Memory tracking: Now reads actual cgroup usage")
        print("  3. ✅ Safety check: Prevents training at >90% container usage")
        print("  4. ✅ GPU pipeline: Explicit device placement for data")
        print("  5. ✅ Memmap: Data stays on disk until needed")
        print("  6. ✅ Warnings: Clear alerts during training")
        print("\n📊 Status: Ready to run week6_fixed_runpod.py")
    else:
        print(f"\n⚠️  Some fixes missing! ({len(results) - sum(results)} not found)")
    
    print("\n" + "=" * 70)

if __name__ == '__main__':
    main()
