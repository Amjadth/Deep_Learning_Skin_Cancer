"""
Data File Inspector - Diagnose array types and formats
Helps identify how to properly load the .npy files
"""

import numpy as np
import json
from pathlib import Path
import struct

class DataInspector:
    """Comprehensive inspection of data files"""
    
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.results = {}
    
    def inspect_npy_file(self, filepath):
        """Deep inspection of a single .npy file"""
        filepath = Path(filepath)
        
        if not filepath.exists():
            return {'status': 'NOT_FOUND', 'error': f'File does not exist: {filepath}'}
        
        result = {
            'filename': filepath.name,
            'path': str(filepath),
            'size_mb': filepath.stat().st_size / (1024**2),
        }
        
        print(f"\n{'='*70}")
        print(f"📄 Inspecting: {filepath.name}")
        print(f"{'='*70}")
        print(f"Size: {result['size_mb']:.1f} MB")
        
        # ==================== FILE HEADER ====================
        print(f"\n🔍 File Header Analysis:")
        try:
            with open(filepath, 'rb') as f:
                header = f.read(20)
                result['first_20_bytes_hex'] = header.hex()
                result['first_20_bytes_repr'] = repr(header)
                
                print(f"  First 20 bytes (hex): {header.hex()}")
                print(f"  First 20 bytes (repr): {repr(header)}")
                
                # Check for numpy magic
                if header[:6] == b'\x93NUMPY':
                    result['format'] = 'VALID_NPY'
                    print(f"  ✅ Valid numpy .npy format detected")
                    
                    # Parse header version
                    version = struct.unpack('<BB', header[6:8])
                    print(f"  Version: {version[0]}.{version[1]}")
                    result['npy_version'] = f"{version[0]}.{version[1]}"
                else:
                    result['format'] = 'UNKNOWN'
                    print(f"  ⚠️  NOT a standard .npy file")
                    print(f"     Expected: b'\\x93NUMPY'")
                    print(f"     Got: {header[:6]}")
        
        except Exception as e:
            result['header_error'] = str(e)
            print(f"  ❌ Header read error: {e}")
        
        # ==================== NUMPY LOAD ATTEMPTS ====================
        print(f"\n🔄 Load Attempts:")
        
        load_strategies = {
            'no_pickle': {'allow_pickle': False},
            'with_pickle': {'allow_pickle': True},
            'fix_imports': {'allow_pickle': True, 'fix_imports': True, 'encoding': 'bytes'},
            'latin1': {'allow_pickle': True, 'encoding': 'latin1'},
        }
        
        loaded_successfully = False
        successful_method = None
        array_obj = None
        
        for method_name, kwargs in load_strategies.items():
            try:
                arr = np.load(filepath, **kwargs)
                
                result[f'load_{method_name}'] = {
                    'success': True,
                    'dtype': str(arr.dtype),
                    'shape': list(arr.shape),
                    'size_elements': int(np.prod(arr.shape)),
                    'nbytes': int(arr.nbytes),
                }
                
                print(f"\n  ✅ {method_name}:")
                print(f"     dtype: {arr.dtype}")
                print(f"     shape: {arr.shape}")
                print(f"     elements: {np.prod(arr.shape):,}")
                print(f"     memory: {arr.nbytes / (1024**2):.1f} MB")
                
                # Sample info
                if arr.size > 0:
                    print(f"     data_range: [{arr.min():.6f}, {arr.max():.6f}]")
                    print(f"     mean: {arr.mean():.6f}")
                    print(f"     std: {arr.std():.6f}")
                
                if not loaded_successfully:
                    loaded_successfully = True
                    successful_method = method_name
                    array_obj = arr
            
            except Exception as e:
                result[f'load_{method_name}'] = {
                    'success': False,
                    'error': str(e),
                }
                print(f"  ❌ {method_name}: {type(e).__name__}: {str(e)[:80]}")
        
        result['load_successful'] = loaded_successfully
        result['successful_method'] = successful_method
        
        # ==================== MEMMAP ATTEMPTS ====================
        if loaded_successfully:
            print(f"\n📍 Memory Mapping:")
            try:
                mmap_arr = np.load(filepath, mmap_mode='r', allow_pickle=False)
                result['memmap_success'] = True
                result['memmap_dtype'] = str(mmap_arr.dtype)
                result['memmap_shape'] = list(mmap_arr.shape)
                print(f"  ✅ Memory mapping successful")
                print(f"     dtype: {mmap_arr.dtype}")
                print(f"     shape: {mmap_arr.shape}")
            except Exception as e:
                result['memmap_success'] = False
                result['memmap_error'] = str(e)
                print(f"  ⚠️  Memory mapping failed: {e}")
        
        return result
    
    def inspect_all_files(self):
        """Inspect all relevant data files"""
        
        files_to_check = [
            'X_train.npy',
            'y_train.npy',
            'X_val.npy',
            'y_val.npy',
            'X_train_high_res.npy',
            'y_train_high_res.npy',
            'X_val_high_res.npy',
            'y_val_high_res.npy',
            'split_info.json',
        ]
        
        print(f"\n{'#'*70}")
        print(f"# DATA FILE INSPECTION")
        print(f"{'#'*70}")
        print(f"Data directory: {self.data_dir}")
        print(f"Exists: {self.data_dir.exists()}")
        
        # List actual files
        if self.data_dir.exists():
            print(f"\nActual files in directory:")
            actual_files = sorted([f.name for f in self.data_dir.iterdir() if f.is_file()])
            for fname in actual_files:
                fpath = self.data_dir / fname
                size_mb = fpath.stat().st_size / (1024**2)
                print(f"  - {fname} ({size_mb:.1f} MB)")
        
        # Check for each expected file
        print(f"\nInspecting expected files:")
        for filename in files_to_check:
            filepath = self.data_dir / filename
            
            if not filepath.exists():
                print(f"\n⏭️  {filename}: NOT FOUND")
                self.results[filename] = {'status': 'NOT_FOUND'}
                continue
            
            if filename.endswith('.json'):
                # Special handling for JSON
                print(f"\n📝 {filename}:")
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    print(f"  ✅ Valid JSON")
                    if 'class_names' in data:
                        print(f"  Classes: {data['class_names']}")
                    self.results[filename] = {'status': 'OK', 'type': 'json', 'data': data}
                except Exception as e:
                    print(f"  ❌ Error: {e}")
                    self.results[filename] = {'status': 'ERROR', 'error': str(e)}
            else:
                # .npy file
                self.results[filename] = self.inspect_npy_file(filepath)
        
        return self.results
    
    def print_summary(self):
        """Print comprehensive summary"""
        
        print(f"\n{'='*70}")
        print(f"📊 SUMMARY")
        print(f"{'='*70}")
        
        npy_files = {k: v for k, v in self.results.items() if k.endswith('.npy')}
        
        # Group by type
        loadable = {k: v for k, v in npy_files.items() if v.get('load_successful', False)}
        not_loadable = {k: v for k, v in npy_files.items() if not v.get('load_successful', False)}
        
        print(f"\n✅ Loadable files ({len(loadable)}/{len(npy_files)}):")
        for fname, result in loadable.items():
            method = result.get('successful_method', '?')
            shape = result.get('load_' + method, {}).get('shape', '?')
            print(f"  {fname}: {shape} [{method}]")
        
        if not_loadable:
            print(f"\n❌ Not loadable ({len(not_loadable)}/{len(npy_files)}):")
            for fname, result in not_loadable.items():
                if 'load_no_pickle' in result:
                    error = result['load_no_pickle'].get('error', '')[:60]
                    print(f"  {fname}: {error}")
        
        # Memmap compatibility
        print(f"\n📍 Memory Mapping Support:")
        memmap_success = {k: v for k, v in npy_files.items() if v.get('memmap_success', False)}
        memmap_fail = {k: v for k, v in npy_files.items() if not v.get('memmap_success', False) and v.get('load_successful', False)}
        
        if memmap_success:
            print(f"  ✅ Memmap compatible: {len(memmap_success)}")
            for fname in memmap_success:
                print(f"     - {fname}")
        
        if memmap_fail:
            print(f"  ⚠️  Loadable but not memmap-compatible: {len(memmap_fail)}")
            for fname in memmap_fail:
                print(f"     - {fname}")
    
    def export_results(self):
        """Export results to JSON"""
        output_path = self.data_dir / 'data_inspection_report.json'
        
        # Make results JSON serializable
        serializable_results = {}
        for k, v in self.results.items():
            if isinstance(v, dict):
                serializable_results[k] = v
            else:
                serializable_results[k] = str(v)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n💾 Report saved to: {output_path}")
        return output_path


def main():
    """Run inspection"""
    
    # Try different potential data directories
    potential_dirs = [
        Path('/workspace/outputs'),
        Path('/runpod-volume/outputs'),
        Path.cwd() / 'outputs',
    ]
    
    data_dir = None
    for potential in potential_dirs:
        if potential.exists() and any(f.name.endswith('.npy') for f in potential.iterdir()):
            data_dir = potential
            break
    
    if not data_dir:
        print("❌ Could not find data directory with .npy files")
        print(f"Checked: {potential_dirs}")
        return
    
    print(f"\n✅ Found data directory: {data_dir}")
    
    # Run inspection
    inspector = DataInspector(data_dir)
    results = inspector.inspect_all_files()
    
    # Print summary
    inspector.print_summary()
    
    # Export
    inspector.export_results()
    
    print(f"\n{'='*70}")
    print(f"✅ Inspection complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
