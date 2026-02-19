# ============================================
# NETWORK VOLUME CONTENT CHECKER
# ============================================
#
# This script checks the contents of the network volume to verify
# that all critical data is safely persisted before terminating the pod.
#
# Purpose:
# - Verify network volume is mounted and accessible
# - List all files and directories on the network volume
# - Check for critical dataset files (Week 1-4 outputs)
# - Calculate total storage usage
# - Provide a safety report before pod termination
#
# Usage:
#   python check_network_volume.py
#
# Author: Deep Learning Engineer
# Date: 2024
# ============================================

import os
from pathlib import Path
from datetime import datetime
import sys

# ============================================
# NETWORK VOLUME DETECTION
# ============================================

def detect_network_volume():
    """Detect network volume location based on RunPod environment."""
    runpod_pod_id = os.environ.get('RUNPOD_POD_ID', None)
    is_pod = runpod_pod_id is not None
    is_serverless = os.environ.get('RUNPOD_WORKER_ID', None) is not None
    
    network_volume_paths = []
    
    if is_serverless:
        # Serverless: Network volumes mount at /runpod-volume
        if os.path.exists("/runpod-volume"):
            network_volume_paths.append(Path("/runpod-volume"))
    elif is_pod:
        # Pod: Network volume is mounted at /workspace if attached
        if os.path.exists("/workspace"):
            network_volume_paths.append(Path("/workspace"))
    else:
        # Fallback: Check common locations
        if os.path.exists("/runpod-volume"):
            network_volume_paths.append(Path("/runpod-volume"))
        if os.path.exists("/workspace"):
            network_volume_paths.append(Path("/workspace"))
    
    return network_volume_paths

# ============================================
# FILE SIZE HELPER
# ============================================

def format_size(size_bytes):
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"

def get_dir_size(path):
    """Get total size of directory in bytes."""
    total_size = 0
    try:
        for dirpath, _, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(filepath)
                except (OSError, FileNotFoundError):
                    pass
    except (OSError, PermissionError):
        pass
    return total_size

# ============================================
# CRITICAL FILES CHECK
# ============================================

# Expected outputs from Week 1-4
CRITICAL_FILES = {
    'Week 1': [
        'X_processed.npy',
        'y_processed.npy',
        'dataset_info.json',
        'class_distribution.png'
    ],
    'Week 2': [
        'X_resized.npy',
        'y_resized.npy',
        'resize_info.json',
        'resize_summary.csv'
    ],
    'Week 3': [
        'X_augmented_medical.npy',
        'y_augmented_medical.npy',
        'augmentation_info.json',
        'augmentation_summary.csv'
    ],
    'Week 4': [
        'X_train.npy',
        'y_train.npy',
        'X_val.npy',
        'y_val.npy',
        'X_test.npy',
        'y_test.npy',
        'split_info.json',
        'split_summary.csv'
    ]
}

def check_critical_files(output_dir):
    """Check for critical files in output directory."""
    results = {
        'Week 1': {'found': [], 'missing': [], 'total_size': 0},
        'Week 2': {'found': [], 'missing': [], 'total_size': 0},
        'Week 3': {'found': [], 'missing': [], 'total_size': 0},
        'Week 4': {'found': [], 'missing': [], 'total_size': 0}
    }
    
    if not output_dir.exists():
        return results
    
    for week, files in CRITICAL_FILES.items():
        for filename in files:
            filepath = output_dir / filename
            if filepath.exists():
                size = filepath.stat().st_size
                results[week]['found'].append({
                    'name': filename,
                    'path': str(filepath),
                    'size': size,
                    'size_human': format_size(size)
                })
                results[week]['total_size'] += size
            else:
                results[week]['missing'].append(filename)
    
    return results

# ============================================
# DIRECTORY STRUCTURE ANALYSIS
# ============================================

def analyze_directory(path, max_depth=3, current_depth=0):
    """Analyze directory structure and file sizes."""
    structure = {
        'path': str(path),
        'type': 'directory' if path.is_dir() else 'file',
        'size': 0,
        'files': 0,
        'subdirectories': {}
    }
    
    if not path.exists():
        return structure
    
    if path.is_file():
        try:
            structure['size'] = path.stat().st_size
            structure['files'] = 1
            structure['modified'] = datetime.fromtimestamp(path.stat().st_mtime).isoformat()
        except (OSError, FileNotFoundError):
            pass
        return structure
    
    try:
        items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name))
        for item in items:
            if item.is_dir() and current_depth < max_depth:
                structure['subdirectories'][item.name] = analyze_directory(
                    item, max_depth, current_depth + 1
                )
                structure['size'] += structure['subdirectories'][item.name]['size']
                structure['files'] += structure['subdirectories'][item.name]['files']
            elif item.is_file():
                try:
                    size = item.stat().st_size
                    structure['size'] += size
                    structure['files'] += 1
                except (OSError, FileNotFoundError):
                    pass
    except (OSError, PermissionError) as e:
        structure['error'] = str(e)
    
    return structure

# ============================================
# MAIN REPORT GENERATION
# ============================================

def generate_report(network_volumes):
    """Generate comprehensive report of network volume contents."""
    print("=" * 80)
    print("NETWORK VOLUME CONTENT CHECKER")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    if not network_volumes:
        print("❌ ERROR: No network volume detected!")
        print()
        print("Checked locations:")
        print("  - /runpod-volume (Serverless)")
        print("  - /workspace (Pod)")
        print()
        print("⚠️  WARNING: Data may not be persisted if pod is terminated!")
        return False
    
    all_safe = True
    
    for vol_path in network_volumes:
        print(f"📁 Network Volume: {vol_path}")
        print(f"   Exists: {vol_path.exists()}")
        print(f"   Writable: {os.access(vol_path, os.W_OK) if vol_path.exists() else 'N/A'}")
        print()
        
        if not vol_path.exists():
            print("❌ Volume path does not exist!")
            all_safe = False
            continue
        
        # Check outputs directory
        outputs_dir = vol_path / 'outputs'
        print(f"📂 Outputs Directory: {outputs_dir}")
        print(f"   Exists: {outputs_dir.exists()}")
        
        if outputs_dir.exists():
            outputs_size = get_dir_size(outputs_dir)
            print(f"   Total Size: {format_size(outputs_size)}")
            print()
            
            # Check critical files
            print("🔍 Checking Critical Files...")
            print("-" * 80)
            critical_results = check_critical_files(outputs_dir)
            
            total_found = 0
            total_missing = 0
            total_size = 0
            
            for week, results in critical_results.items():
                found_count = len(results['found'])
                missing_count = len(results['missing'])
                week_size = results['total_size']
                
                total_found += found_count
                total_missing += missing_count
                total_size += week_size
                
                status = "✅" if missing_count == 0 else "⚠️"
                print(f"{status} {week}:")
                print(f"   Found: {found_count}/{found_count + missing_count} files")
                print(f"   Size: {format_size(week_size)}")
                
                if results['found']:
                    print(f"   Files found:")
                    for file_info in results['found']:
                        print(f"     ✓ {file_info['name']} ({file_info['size_human']})")
                
                if results['missing']:
                    print(f"   Files missing:")
                    for filename in results['missing']:
                        print(f"     ✗ {filename}")
                    all_safe = False
                
                print()
            
            print("-" * 80)
            print(f"Summary: {total_found} files found, {total_missing} files missing")
            print(f"Total Size: {format_size(total_size)}")
            print()
            
            # Check visualizations directory
            viz_dir = outputs_dir / 'visualizations'
            if viz_dir.exists():
                viz_size = get_dir_size(viz_dir)
                viz_files = len(list(viz_dir.glob('*')))
                print(f"🎨 Visualizations Directory:")
                print(f"   Files: {viz_files}")
                print(f"   Size: {format_size(viz_size)}")
                print()
            
            # List all files in outputs (top level)
            print("📋 Outputs Directory Contents:")
            print("-" * 80)
            try:
                items = sorted(outputs_dir.iterdir(), key=lambda x: (x.is_file(), x.name))
                file_count = 0
                dir_count = 0
                
                for item in items:
                    if item.is_file():
                        file_count += 1
                        try:
                            size = item.stat().st_size
                            modified = datetime.fromtimestamp(item.stat().st_mtime)
                            print(f"   📄 {item.name}")
                            print(f"      Size: {format_size(size)}")
                            print(f"      Modified: {modified.strftime('%Y-%m-%d %H:%M:%S')}")
                        except (OSError, FileNotFoundError):
                            print(f"   📄 {item.name} (cannot read stats)")
                    elif item.is_dir():
                        dir_count += 1
                        try:
                            dir_size = get_dir_size(item)
                            print(f"   📁 {item.name}/")
                            print(f"      Size: {format_size(dir_size)}")
                        except (OSError, FileNotFoundError):
                            print(f"   📁 {item.name}/ (cannot read stats)")
                
                print()
                print(f"   Total: {file_count} files, {dir_count} directories")
            except (OSError, PermissionError) as e:
                print(f"   ❌ Error reading directory: {e}")
            
            print()
        
        else:
            print("❌ Outputs directory does not exist!")
            print("   This means no data has been saved to the network volume yet.")
            all_safe = False
            print()
        
        # Check total volume usage
        print("💾 Network Volume Usage:")
        print("-" * 80)
        try:
            volume_size = get_dir_size(vol_path)
            print(f"   Total Size: {format_size(volume_size)}")
            
            # Try to get disk usage (if available)
            try:
                import shutil
                total, used, free = shutil.disk_usage(vol_path)
                print(f"   Total Disk: {format_size(total)}")
                print(f"   Used Disk: {format_size(used)}")
                print(f"   Free Disk: {format_size(free)}")
                print(f"   Usage: {(used/total)*100:.1f}%")
            except (OSError, PermissionError, AttributeError):
                pass
        except (OSError, PermissionError) as e:
            print(f"   ⚠️  Could not calculate volume usage: {e}")
        
        print()
    
    # Final safety assessment
    print("=" * 80)
    print("SAFETY ASSESSMENT")
    print("=" * 80)
    
    if all_safe and network_volumes:
        print("✅ SAFE TO TERMINATE POD")
        print()
        print("All critical files are present on the network volume.")
        print("Your data is safely persisted and will survive pod termination.")
    else:
        print("⚠️  NOT SAFE TO TERMINATE POD")
        print()
        if not network_volumes:
            print("❌ No network volume detected!")
            print("   Data may be stored only in temporary workspace.")
            print("   Attach a network volume before running data processing scripts.")
        else:
            print("❌ Some critical files are missing from the network volume.")
            print("   Missing files will be lost when the pod is terminated.")
            print("   Please run the missing week scripts before terminating.")
    
    print()
    print("=" * 80)
    
    return all_safe

# ============================================
# MAIN EXECUTION
# ============================================

def main():
    """Main execution function."""
    print("🔍 Detecting network volume...")
    network_volumes = detect_network_volume()
    
    if network_volumes:
        print(f"✓ Found {len(network_volumes)} network volume(s)")
        for vol in network_volumes:
            print(f"  - {vol}")
    else:
        print("⚠️  No network volume detected")
    
    print()
    
    # Generate report
    is_safe = generate_report(network_volumes)
    
    # Exit with appropriate code
    sys.exit(0 if is_safe else 1)

if __name__ == "__main__":
    main()

