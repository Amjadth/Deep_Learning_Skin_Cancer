#!/usr/bin/env python3
"""
Comprehensive Memory Detection Script
Detects and displays ALL types of memory:
1. Host System RAM (physical machine)
2. Container RAM Limit (cgroup enforcement)
3. GPU VRAM (discrete GPU)
"""

import os
import sys
import json
import subprocess
from pathlib import Path

try:
    import psutil
except ImportError:
    print("❌ psutil not installed. Install: pip install psutil")
    sys.exit(1)

try:
    import tensorflow as tf
except ImportError:
    print("⚠️  TensorFlow not installed - GPU detection disabled")
    tf = None

# ============================================
# HOST MEMORY DETECTION
# ============================================

def get_host_memory_info():
    """Get host system RAM information (physical machine)."""
    try:
        mem = psutil.virtual_memory()
        return {
            'total_gb': mem.total / (1024**3),
            'used_gb': mem.used / (1024**3),
            'available_gb': mem.available / (1024**3),
            'percent': mem.percent,
            'active_gb': mem.active / (1024**3),
            'inactive_gb': mem.inactive / (1024**3),
            'buffers_gb': mem.buffers / (1024**3),
            'cached_gb': mem.cached / (1024**3),
            'shared_gb': mem.shared / (1024**3),
        }
    except Exception as e:
        print(f"❌ Error reading host memory: {e}")
        return None

# ============================================
# CONTAINER MEMORY DETECTION
# ============================================

def get_container_memory_limit():
    """Get container memory limit from cgroup (v2 and v1)."""
    try:
        # Try cgroup v2 first (newer systems)
        cgroup_v2_path = '/sys/fs/cgroup/memory.max'
        if os.path.exists(cgroup_v2_path):
            try:
                with open(cgroup_v2_path, 'r') as f:
                    limit_str = f.read().strip()
                    if limit_str != 'max':
                        limit_bytes = int(limit_str)
                        return {
                            'type': 'cgroup v2',
                            'limit_gb': limit_bytes / (1024**3),
                            'limit_bytes': limit_bytes,
                            'path': cgroup_v2_path
                        }
            except Exception as e:
                print(f"    ⚠️  cgroup v2 read error: {e}")
        
        # Try cgroup v1
        cgroup_v1_path = '/sys/fs/cgroup/memory/memory.limit_in_bytes'
        if os.path.exists(cgroup_v1_path):
            try:
                with open(cgroup_v1_path, 'r') as f:
                    limit_bytes = int(f.read().strip())
                    # Check if it's actually limited (not max value)
                    if limit_bytes < (1 << 62):  # Less than 2^62
                        return {
                            'type': 'cgroup v1',
                            'limit_gb': limit_bytes / (1024**3),
                            'limit_bytes': limit_bytes,
                            'path': cgroup_v1_path
                        }
            except Exception as e:
                print(f"    ⚠️  cgroup v1 read error: {e}")
        
        return {
            'type': 'unlimited',
            'limit_gb': None,
            'limit_bytes': None,
            'note': 'No container limits found (using full host memory)'
        }
    
    except Exception as e:
        print(f"❌ Error reading container limit: {e}")
        return None

def get_container_memory_usage():
    """Get current container memory usage from cgroup."""
    try:
        # Try cgroup v2 first
        cgroup_v2_path = '/sys/fs/cgroup/memory.current'
        if os.path.exists(cgroup_v2_path):
            with open(cgroup_v2_path, 'r') as f:
                usage_bytes = int(f.read().strip())
                return {
                    'type': 'cgroup v2',
                    'usage_gb': usage_bytes / (1024**3),
                    'usage_bytes': usage_bytes,
                    'path': cgroup_v2_path
                }
        
        # Try cgroup v1
        cgroup_v1_path = '/sys/fs/cgroup/memory/memory.usage_in_bytes'
        if os.path.exists(cgroup_v1_path):
            with open(cgroup_v1_path, 'r') as f:
                usage_bytes = int(f.read().strip())
                return {
                    'type': 'cgroup v1',
                    'usage_gb': usage_bytes / (1024**3),
                    'usage_bytes': usage_bytes,
                    'path': cgroup_v1_path
                }
        
        return {
            'type': 'not_found',
            'usage_gb': None,
            'note': 'Container usage not found in cgroup'
        }
    
    except Exception as e:
        print(f"⚠️  Error reading container usage: {e}")
        return None

# ============================================
# PROCESS MEMORY DETECTION
# ============================================

def get_process_memory_info():
    """Get current Python process memory usage."""
    try:
        process = psutil.Process()
        mem_info = process.memory_info()
        
        return {
            'rss_gb': mem_info.rss / (1024**3),  # Resident Set Size
            'rss_bytes': mem_info.rss,
            'vms_gb': mem_info.vms / (1024**3),  # Virtual Memory Size
            'vms_bytes': mem_info.vms,
            'percent': process.memory_percent(),
            'pid': process.pid,
        }
    except Exception as e:
        print(f"❌ Error reading process memory: {e}")
        return None

# ============================================
# GPU MEMORY DETECTION
# ============================================

def get_gpu_memory_info_tensorflow():
    """Get GPU memory info using TensorFlow."""
    if tf is None:
        return None
    
    try:
        gpus = tf.config.list_physical_devices('GPU')
        
        if not gpus:
            return {
                'num_gpus': 0,
                'gpu_devices': []
            }
        
        # Try to get GPU memory using nvidia-smi
        gpu_memory = get_gpu_memory_nvidia_smi()
        if gpu_memory:
            return gpu_memory
        
        # Fallback: TensorFlow device info
        return {
            'num_gpus': len(gpus),
            'gpu_devices': [gpu.name for gpu in gpus],
            'source': 'tensorflow',
            'note': 'Use nvidia-smi for actual memory allocation'
        }
    
    except Exception as e:
        return {
            'error': str(e),
            'source': 'tensorflow'
        }

def get_gpu_memory_nvidia_smi():
    """Get GPU memory info using nvidia-smi command."""
    try:
        # Check if nvidia-smi is available
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode != 0:
            return None
        
        gpu_data = []
        for line in result.stdout.strip().split('\n'):
            if not line.strip():
                continue
            
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 6:
                gpu_data.append({
                    'index': int(parts[0]),
                    'name': parts[1],
                    'memory_total_mb': int(float(parts[2])),
                    'memory_used_mb': int(float(parts[3])),
                    'memory_free_mb': int(float(parts[4])),
                    'utilization_percent': float(parts[5])
                })
        
        if gpu_data:
            return {
                'num_gpus': len(gpu_data),
                'gpus': gpu_data,
                'source': 'nvidia-smi',
                'note': 'Real-time GPU memory allocation'
            }
        
        return None
    
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
        return None

# ============================================
# PRINT FORMATTING
# ============================================

def print_section(title):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def print_subsection(title):
    """Print formatted subsection header."""
    print(f"\n  {title}")
    print("  " + "-" * 76)

def format_bytes(bytes_val):
    """Format bytes to human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f} PB"

# ============================================
# MAIN DETECTION AND DISPLAY
# ============================================

def main():
    print("\n" + "=" * 80)
    print("  🔍 COMPREHENSIVE MEMORY DETECTION SCRIPT")
    print("=" * 80)
    print(f"  System: {sys.platform}")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Host: {os.uname().nodename}")
    
    results = {
        'timestamp': str(Path('').resolve()),
        'system': sys.platform,
        'hostname': os.uname().nodename,
    }
    
    # ============================================
    # 1. HOST MEMORY
    # ============================================
    
    print_section("1️⃣  HOST SYSTEM RAM (Physical Machine)")
    
    host_mem = get_host_memory_info()
    if host_mem:
        print(f"\n  📊 Total RAM: {host_mem['total_gb']:.2f} GB")
        print(f"  📍 Used: {host_mem['used_gb']:.2f} GB ({host_mem['percent']:.1f}%)")
        print(f"  ✅ Available: {host_mem['available_gb']:.2f} GB")
        print(f"\n  💾 Breakdown:")
        print(f"     • Active: {host_mem['active_gb']:.2f} GB")
        print(f"     • Inactive: {host_mem['inactive_gb']:.2f} GB")
        print(f"     • Buffers: {host_mem['buffers_gb']:.2f} GB")
        print(f"     • Cached: {host_mem['cached_gb']:.2f} GB")
        print(f"     • Shared: {host_mem['shared_gb']:.2f} GB")
        
        results['host_memory'] = host_mem
    else:
        print("  ❌ Could not detect host memory")
    
    # ============================================
    # 2. CONTAINER MEMORY LIMIT
    # ============================================
    
    print_section("2️⃣  CONTAINER RAM LIMIT (cgroup)")
    
    container_limit = get_container_memory_limit()
    if container_limit:
        print(f"\n  🏗️  Container Type: {container_limit['type']}")
        
        if 'path' in container_limit:
            print(f"  📁 Path: {container_limit['path']}")
        
        if container_limit.get('limit_gb'):
            print(f"  📊 Memory Limit: {container_limit['limit_gb']:.2f} GB")
            print(f"  📊 Memory Limit (bytes): {container_limit['limit_bytes']:,}")
            
            # Calculate percentage of host
            if host_mem:
                pct_of_host = (container_limit['limit_gb'] / host_mem['total_gb']) * 100
                print(f"  📈 % of Host RAM: {pct_of_host:.1f}%")
        else:
            print(f"  ℹ️  {container_limit.get('note', 'Status unknown')}")
        
        results['container_limit'] = container_limit
    else:
        print("  ❌ Could not detect container limit")
    
    # ============================================
    # 3. CONTAINER MEMORY USAGE
    # ============================================
    
    print_section("3️⃣  CONTAINER RAM USAGE (cgroup)")
    
    container_usage = get_container_memory_usage()
    if container_usage:
        print(f"\n  🏗️  Container Type: {container_usage['type']}")
        
        if 'path' in container_usage:
            print(f"  📁 Path: {container_usage['path']}")
        
        if container_usage.get('usage_gb') is not None:
            print(f"  📊 Current Usage: {container_usage['usage_gb']:.2f} GB")
            print(f"  📊 Current Usage (bytes): {container_usage['usage_bytes']:,}")
            
            # Calculate percentage of limit
            if container_limit and container_limit.get('limit_gb'):
                pct_of_limit = (container_usage['usage_gb'] / container_limit['limit_gb']) * 100
                print(f"  📈 % of Container Limit: {pct_of_limit:.1f}%")
            
            # Calculate percentage of host
            if host_mem:
                pct_of_host = (container_usage['usage_gb'] / host_mem['total_gb']) * 100
                print(f"  📈 % of Host RAM: {pct_of_host:.1f}%")
        else:
            print(f"  ℹ️  {container_usage.get('note', 'Status unknown')}")
        
        results['container_usage'] = container_usage
    else:
        print("  ❌ Could not detect container usage")
    
    # ============================================
    # 4. PROCESS MEMORY
    # ============================================
    
    print_section("4️⃣  PYTHON PROCESS MEMORY")
    
    process_mem = get_process_memory_info()
    if process_mem:
        print(f"\n  🐍 Process ID (PID): {process_mem['pid']}")
        print(f"  📊 RSS (Resident Set): {process_mem['rss_gb']:.2f} GB ({process_mem['rss_bytes']:,} bytes)")
        print(f"  📊 VMS (Virtual Memory): {process_mem['vms_gb']:.2f} GB ({process_mem['vms_bytes']:,} bytes)")
        print(f"  📈 % of Host RAM: {process_mem['percent']:.2f}%")
        
        # Calculate % of container if available
        if container_limit and container_limit.get('limit_gb'):
            pct_of_container = (process_mem['rss_gb'] / container_limit['limit_gb']) * 100
            print(f"  📈 % of Container Limit: {pct_of_container:.1f}%")
        
        results['process_memory'] = process_mem
    else:
        print("  ❌ Could not detect process memory")
    
    # ============================================
    # 5. GPU MEMORY
    # ============================================
    
    print_section("5️⃣  GPU VRAM (Discrete GPU)")
    
    # Try nvidia-smi first
    gpu_memory = get_gpu_memory_nvidia_smi()
    
    if gpu_memory and gpu_memory.get('num_gpus', 0) > 0:
        print(f"\n  🎮 Source: {gpu_memory.get('source', 'unknown')}")
        print(f"  📊 Number of GPUs: {gpu_memory['num_gpus']}")
        
        if 'gpus' in gpu_memory:
            for gpu in gpu_memory['gpus']:
                print(f"\n  GPU {gpu['index']}: {gpu['name']}")
                print(f"     Total Memory: {gpu['memory_total_mb']} MB ({gpu['memory_total_mb']/1024:.2f} GB)")
                print(f"     Used Memory: {gpu['memory_used_mb']} MB ({gpu['memory_used_mb']/1024:.2f} GB)")
                print(f"     Free Memory: {gpu['memory_free_mb']} MB ({gpu['memory_free_mb']/1024:.2f} GB)")
                print(f"     Utilization: {gpu['utilization_percent']:.1f}%")
                print(f"     % Used: {(gpu['memory_used_mb']/gpu['memory_total_mb'])*100:.1f}%")
        
        print(f"\n  ℹ️  {gpu_memory.get('note', '')}")
        results['gpu_memory'] = gpu_memory
    else:
        # Fallback to TensorFlow
        gpu_memory = get_gpu_memory_info_tensorflow()
        if gpu_memory and gpu_memory.get('num_gpus', 0) > 0:
            print(f"\n  🎮 Source: {gpu_memory.get('source', 'tensorflow')}")
            print(f"  📊 Number of GPUs: {gpu_memory['num_gpus']}")
            print(f"  📋 GPU Devices:")
            for device in gpu_memory.get('gpu_devices', []):
                print(f"     • {device}")
            print(f"\n  ℹ️  {gpu_memory.get('note', 'GPU detected but no memory info available')}")
            results['gpu_memory'] = gpu_memory
        else:
            print("  ⚠️  No GPU detected (nvidia-smi not available or no GPU hardware)")
            if tf:
                print(f"  ℹ️  TensorFlow available but no GPUs found")
            results['gpu_memory'] = {'num_gpus': 0}
    
    # ============================================
    # SUMMARY AND COMPARISON
    # ============================================
    
    print_section("📊 MEMORY HIERARCHY SUMMARY")
    
    print("\n  🔺 Memory Stack (from largest to smallest):")
    
    if host_mem:
        print(f"     1. 🖥️  HOST RAM: {host_mem['total_gb']:.2f} GB (physical system)")
    
    if container_limit and container_limit.get('limit_gb'):
        print(f"     2. 📦 CONTAINER LIMIT: {container_limit['limit_gb']:.2f} GB (cgroup enforcement)")
        if host_mem:
            ratio = container_limit['limit_gb'] / host_mem['total_gb']
            print(f"        ({ratio*100:.1f}% of host RAM)")
    
    if gpu_memory and gpu_memory.get('gpus'):
        total_gpu_gb = sum(gpu['memory_total_mb'] for gpu in gpu_memory['gpus']) / 1024
        print(f"     3. 🎮 GPU VRAM: {total_gpu_gb:.2f} GB (discrete GPU)")
    
    if process_mem:
        print(f"     4. 🐍 PROCESS RSS: {process_mem['rss_gb']:.2f} GB (current Python process)")
    
    # ============================================
    # INTERPRETATION
    # ============================================
    
    print_section("🔍 INTERPRETATION GUIDE")
    
    print("\n  🖥️  HOST RAM ({:.2f} GB):".format(host_mem['total_gb'] if host_mem else 0))
    print("     • Physical system RAM (not container-specific)")
    print("     • Visible to all processes on the host")
    print("     • Not directly usable if container limit is set")
    
    if container_limit and container_limit.get('limit_gb'):
        print(f"\n  📦 CONTAINER LIMIT ({container_limit['limit_gb']:.2f} GB):")
        print("     • Maximum RAM this container can use")
        print("     • Enforced by kernel (cgroup)")
        print("     • Python process CANNOT exceed this")
        print("     • Data pipeline should stay within this limit")
    
    if container_usage and container_usage.get('usage_gb') is not None:
        print(f"\n  📊 CONTAINER USAGE ({container_usage['usage_gb']:.2f} GB):")
        print("     • Current RAM consumed by this container")
        print("     • Includes all processes in container")
        print("     • Plus kernel caches and buffers")
    
    if gpu_memory and gpu_memory.get('gpus'):
        print(f"\n  🎮 GPU VRAM:")
        print("     • Completely separate from system RAM")
        print("     • Data must be explicitly transferred to GPU")
        print("     • Training with GPU reduces RAM pressure on system")
        print("     • Use tf.device('/GPU:0') to place tensors on GPU")
    
    if process_mem:
        print(f"\n  🐍 PROCESS MEMORY ({process_mem['rss_gb']:.2f} GB):")
        print("     • RAM used by Python process")
        print("     • Should stay well below container limit")
        print("     • Use memmap for large datasets to avoid loading into RAM")
    
    # ============================================
    # RECOMMENDATIONS
    # ============================================
    
    print_section("💡 RECOMMENDATIONS")
    
    print("\n  ✅ For optimal training:")
    
    if container_limit and container_limit.get('limit_gb'):
        recommended_data = container_limit['limit_gb'] * 0.6
        print(f"\n     1. Keep data loading under {recommended_data:.1f} GB")
        print(f"        ({container_limit['limit_gb']:.1f} GB limit × 60% safety factor)")
    
    if gpu_memory and gpu_memory.get('gpus'):
        total_gpu_gb = sum(gpu['memory_total_mb'] for gpu in gpu_memory['gpus']) / 1024
        recommended_batch = min(32, int(total_gpu_gb / 2))
        print(f"\n     2. Use batch_size around {recommended_batch}-64")
        print(f"        (for {total_gpu_gb:.1f} GB GPU VRAM)")
    
    print(f"\n     3. Use memmap for large arrays (data loads from disk, not RAM)")
    print(f"     4. Use tf.data with prefetch for GPU pipeline efficiency")
    print(f"     5. Explicitly place tensors on GPU with tf.device('/GPU:0')")
    
    # ============================================
    # SAVE RESULTS
    # ============================================
    
    output_file = Path('memory_detection_results.json')
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n\n  💾 Results saved to: {output_file}")
    except Exception as e:
        print(f"\n\n  ⚠️  Could not save results: {e}")
    
    print("\n" + "=" * 80)
    print("  ✅ Memory detection complete!")
    print("=" * 80 + "\n")

if __name__ == '__main__':
    main()
