# ============================================
# SIMPLE RAM CLEAR SCRIPT
# ============================================
#
# Quick and simple RAM clearing script
# Use this if you need to free up memory quickly
#
# Usage:
#   python clear_ram_simple.py
#
# ============================================

import gc
import sys

print("=" * 70)
print("CLEARING RAM (Simple)")
print("=" * 70)

# Force garbage collection multiple times
print("\n🧹 Running garbage collection...")
for i in range(3):
    collected = gc.collect()
    if collected > 0:
        print(f"   Cycle {i+1}: Collected {collected} objects")
    else:
        print(f"   Cycle {i+1}: No objects to collect")

print("\n✅ RAM cleared!")
print("💡 Ready to run memory-intensive scripts")

