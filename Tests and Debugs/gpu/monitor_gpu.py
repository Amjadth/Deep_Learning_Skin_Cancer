#!/usr/bin/env python3
# ============================================
# GPU UTILIZATION VERIFICATION DURING TRAINING
# ============================================
#
# Run this in a SEPARATE terminal to monitor GPU while week6 is training
# This script displays real-time GPU metrics
#
# ============================================

import subprocess
import sys
import time
import re

print("=" * 80)
print("GPU UTILIZATION MONITORING")
print("=" * 80)
print("\nThis script monitors GPU performance while training runs")
print("Press Ctrl+C to stop\n")

print("📊 GPU Metrics to watch for:")
print("   • GPU % (under 'Processes'): Should be 80-99%")
print("   • Memory-Usage: Should be 80-95% full")
print("   • GPU-Util: Should be high")
print("   • Memory-Util: Should be high")
print("\n" + "=" * 80 + "\n")

try:
    while True:
        # Run nvidia-smi
        result = subprocess.run(
            [
                'nvidia-smi',
                '--query-gpu=timestamp,index,name,driver_version,pstate,utilization.gpu,utilization.memory,memory.used,memory.free,memory.total,temperature.gpu',
                '--format=csv,nounits,noheader'
            ],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            print(result.stdout)
            
            # Also check processes
            proc_result = subprocess.run(
                ['nvidia-smi', 'pmon', '-c', '1'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if proc_result.returncode == 0:
                lines = proc_result.stdout.strip().split('\n')
                if len(lines) > 2:  # Header + data
                    print("Active GPU Processes:")
                    for line in lines[2:6]:  # Show first few processes
                        if line.strip():
                            print(f"  {line}")
        else:
            print(f"Error running nvidia-smi: {result.stderr}")
            print("Make sure you have NVIDIA GPU drivers installed")
            break
        
        print("-" * 80)
        time.sleep(1)

except KeyboardInterrupt:
    print("\n\n✅ Monitoring stopped")
    sys.exit(0)
except FileNotFoundError:
    print("❌ nvidia-smi not found - NVIDIA drivers not installed")
    print("   Install drivers: https://www.nvidia.com/Download/driverDetails.aspx")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
