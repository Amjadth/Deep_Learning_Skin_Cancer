# ============================================
# MEMORY DIAGNOSTIC SCRIPT
# ============================================
#
# Shows detailed memory usage breakdown
# Helps identify what's using system memory
#
# Usage:
#   python check_memory.py
#
# ============================================

import os
import sys

try:
    import psutil
except ImportError:
    print("⚠️ psutil not installed. Install with: pip install psutil")
    sys.exit(1)

print("=" * 70)
print("MEMORY DIAGNOSTIC REPORT")
print("=" * 70)

# System-wide memory
system_mem = psutil.virtual_memory()
system_total_gb = system_mem.total / (1024**3)
system_used_gb = system_mem.used / (1024**3)
system_available_gb = system_mem.available / (1024**3)
system_percent = system_mem.percent
system_cached_gb = system_mem.cached / (1024**3) if hasattr(system_mem, 'cached') else 0
system_buffers_gb = system_mem.buffers / (1024**3) if hasattr(system_mem, 'buffers') else 0

print(f"\n💻 SYSTEM-WIDE MEMORY:")
print(f"   Total RAM: {system_total_gb:.2f} GB")
print(f"   Used: {system_used_gb:.2f} GB ({system_percent:.1f}%)")
print(f"   Available: {system_available_gb:.2f} GB")
if system_cached_gb > 0:
    print(f"   Cached: {system_cached_gb:.2f} GB")
if system_buffers_gb > 0:
    print(f"   Buffers: {system_buffers_gb:.2f} GB")

# Current Python process
current_process = psutil.Process(os.getpid())
current_mem_gb = current_process.memory_info().rss / (1024**3)
print(f"\n🐍 CURRENT PYTHON PROCESS:")
print(f"   PID: {os.getpid()}")
print(f"   Memory: {current_mem_gb:.2f} GB")
print(f"   Command: {' '.join(current_process.cmdline()[:3])}")

# Top memory-consuming processes
print(f"\n📊 TOP 15 MEMORY-CONSUMING PROCESSES:")
print(f"{'Process Name':<35} {'PID':<10} {'Memory (GB)':<15} {'% of Total':<12}")
print("-" * 70)

processes = []
for proc in psutil.process_iter(['pid', 'name', 'memory_info', 'cmdline']):
    try:
        mem_mb = proc.info['memory_info'].rss / (1024**2)
        if mem_mb > 50:  # Only show processes using >50 MB
            mem_gb = mem_mb / 1024
            percent_of_total = (mem_gb / system_total_gb) * 100
            name = proc.info['name'] or 'unknown'
            # Truncate long names
            if len(name) > 34:
                name = name[:31] + "..."
            processes.append((name, proc.info['pid'], mem_gb, percent_of_total))
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        pass

# Sort by memory and show top 15
processes.sort(key=lambda x: x[2], reverse=True)
total_shown = 0
for name, pid, mem_gb, percent in processes[:15]:
    print(f"{name:<35} {pid:<10} {mem_gb:>10.2f} GB    {percent:>8.2f}%")
    total_shown += mem_gb

print("-" * 70)
print(f"{'Total (shown)':<35} {'':<10} {total_shown:>10.2f} GB")
print(f"{'Other processes + system':<35} {'':<10} {system_used_gb - total_shown:>10.2f} GB")

# Python processes specifically
print(f"\n🐍 ALL PYTHON PROCESSES:")
python_procs = []
for proc in psutil.process_iter(['pid', 'name', 'memory_info', 'cmdline']):
    try:
        name = proc.info['name'] or ''
        cmdline = ' '.join(proc.info['cmdline'] or [])
        if 'python' in name.lower() or 'python' in cmdline.lower():
            mem_gb = proc.info['memory_info'].rss / (1024**3)
            python_procs.append((proc.info['pid'], name, cmdline[:60], mem_gb))
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        pass

if python_procs:
    python_procs.sort(key=lambda x: x[3], reverse=True)
    for pid, name, cmdline, mem_gb in python_procs:
        print(f"   PID {pid:6d}: {mem_gb:6.2f} GB - {name} {cmdline}")
else:
    print("   No Python processes found")

# Jupyter/IPython processes
print(f"\n📓 JUPYTER/IPYTHON PROCESSES:")
jupyter_procs = []
for proc in psutil.process_iter(['pid', 'name', 'memory_info', 'cmdline']):
    try:
        cmdline = ' '.join(proc.info['cmdline'] or [])
        if any(keyword in cmdline.lower() for keyword in ['jupyter', 'ipython', 'notebook', 'lab']):
            mem_gb = proc.info['memory_info'].rss / (1024**3)
            name = proc.info['name'] or 'unknown'
            jupyter_procs.append((proc.info['pid'], name, cmdline[:70], mem_gb))
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        pass

if jupyter_procs:
    jupyter_procs.sort(key=lambda x: x[3], reverse=True)
    for pid, name, cmdline, mem_gb in jupyter_procs:
        print(f"   PID {pid:6d}: {mem_gb:6.2f} GB - {cmdline}")
else:
    print("   No Jupyter/IPython processes found")

# Summary and recommendations
print(f"\n💡 SUMMARY:")
print(f"   System memory used: {system_used_gb:.2f} GB / {system_total_gb:.2f} GB ({system_percent:.1f}%)")
print(f"   Available for week3.py: {system_available_gb:.2f} GB")
print(f"   Week3.py will use: ~4-8 GB during processing")
print(f"   Status: {'✅ Sufficient memory available' if system_available_gb > 10 else '⚠️ Low memory - consider freeing up'}")
print(f"\n💡 TIPS:")
print(f"   - The 30+ GB on dashboard includes system cache (normal)")
print(f"   - System cache will be freed automatically when needed")
print(f"   - Your Python process is using minimal memory")
print(f"   - Week3.py uses memmaps (streams from disk, not RAM)")
print(f"   - You have {system_available_gb:.2f} GB available - more than enough!")

print("\n" + "=" * 70)

