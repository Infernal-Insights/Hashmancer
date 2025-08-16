#!/usr/bin/env python3
"""
GPU Utilities for Hashmancer Docker Workers
Provides GPU detection, monitoring, and optimization utilities.
"""

import os
import sys
import json
import subprocess
from typing import Dict, List, Optional, Any

def get_gpu_info() -> Dict[str, Any]:
    """Get comprehensive GPU information."""
    try:
        # Try nvidia-smi first
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,driver_version',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, check=True)
        
        gpus = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = [p.strip() for p in line.split(',')]
                gpu_info = {
                    'index': int(parts[0]),
                    'name': parts[1],
                    'memory_total_mb': int(parts[2]),
                    'memory_used_mb': int(parts[3]),
                    'memory_free_mb': int(parts[4]),
                    'utilization_percent': int(parts[5]),
                    'temperature_c': int(parts[6]) if parts[6] != 'N/A' else None,
                    'driver_version': parts[7]
                }
                gpu_info['memory_usage_percent'] = round(
                    (gpu_info['memory_used_mb'] / gpu_info['memory_total_mb']) * 100, 1
                )
                gpus.append(gpu_info)
        
        return {
            'available': True,
            'count': len(gpus),
            'gpus': gpus,
            'cuda_version': get_cuda_version(),
        }
        
    except (subprocess.CalledProcessError, FileNotFoundError, Exception) as e:
        return {
            'available': False,
            'error': str(e),
            'count': 0,
            'gpus': []
        }

def get_cuda_version() -> Optional[str]:
    """Get CUDA version."""
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, check=True)
        for line in result.stdout.split('\n'):
            if 'release' in line.lower():
                # Extract version from line like "Cuda compilation tools, release 12.1, V12.1.105"
                parts = line.split('release')[1].split(',')[0].strip()
                return parts
    except:
        pass
    
    # Try alternative method
    try:
        import torch
        if torch.cuda.is_available():
            return torch.version.cuda
    except:
        pass
    
    return None

def test_gpu_access() -> Dict[str, Any]:
    """Test GPU access and functionality."""
    results = {
        'nvidia_smi': False,
        'cuda_available': False,
        'pytorch_cuda': False,
        'gpu_count': 0,
        'errors': []
    }
    
    # Test nvidia-smi
    try:
        subprocess.run(['nvidia-smi'], capture_output=True, check=True)
        results['nvidia_smi'] = True
    except Exception as e:
        results['errors'].append(f"nvidia-smi failed: {e}")
    
    # Test CUDA availability
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=count', '--format=csv,noheader'], 
                              capture_output=True, text=True, check=True)
        results['gpu_count'] = len(result.stdout.strip().split('\n'))
        results['cuda_available'] = True
    except Exception as e:
        results['errors'].append(f"CUDA test failed: {e}")
    
    # Test PyTorch CUDA
    try:
        import torch
        results['pytorch_cuda'] = torch.cuda.is_available()
        if results['pytorch_cuda']:
            results['pytorch_gpu_count'] = torch.cuda.device_count()
    except Exception as e:
        results['errors'].append(f"PyTorch CUDA test failed: {e}")
    
    return results

def optimize_gpu_settings() -> Dict[str, Any]:
    """Optimize GPU settings for hash cracking."""
    optimizations = []
    warnings = []
    
    # Check persistence mode
    try:
        result = subprocess.run([
            'nvidia-smi', '--query-gpu=persistence_mode', '--format=csv,noheader'
        ], capture_output=True, text=True, check=True)
        
        if 'Disabled' in result.stdout:
            try:
                subprocess.run(['nvidia-smi', '-pm', '1'], check=True)
                optimizations.append("Enabled GPU persistence mode")
            except subprocess.CalledProcessError:
                warnings.append("Could not enable persistence mode (requires root)")
    except Exception as e:
        warnings.append(f"Could not check persistence mode: {e}")
    
    # Check power limits
    try:
        result = subprocess.run([
            'nvidia-smi', '--query-gpu=power.max_limit,power.default_limit', '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, check=True)
        
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                max_power, default_power = line.split(',')
                max_power = float(max_power.strip())
                default_power = float(default_power.strip())
                
                if max_power > default_power:
                    warnings.append(f"GPU power limit could be increased from {default_power}W to {max_power}W")
    except Exception as e:
        warnings.append(f"Could not check power limits: {e}")
    
    return {
        'optimizations_applied': optimizations,
        'warnings': warnings,
        'timestamp': subprocess.run(['date'], capture_output=True, text=True).stdout.strip()
    }

def monitor_gpu_usage(duration_seconds: int = 60) -> Dict[str, Any]:
    """Monitor GPU usage over time."""
    import time
    
    samples = []
    start_time = time.time()
    
    while time.time() - start_time < duration_seconds:
        gpu_info = get_gpu_info()
        if gpu_info['available']:
            timestamp = time.time()
            for gpu in gpu_info['gpus']:
                samples.append({
                    'timestamp': timestamp,
                    'gpu_index': gpu['index'],
                    'utilization': gpu['utilization_percent'],
                    'memory_usage': gpu['memory_usage_percent'],
                    'temperature': gpu['temperature_c']
                })
        time.sleep(1)
    
    # Calculate averages
    if samples:
        total_samples = len(samples)
        avg_utilization = sum(s['utilization'] for s in samples) / total_samples
        avg_memory = sum(s['memory_usage'] for s in samples) / total_samples
        avg_temp = sum(s['temperature'] for s in samples if s['temperature']) / max(1, len([s for s in samples if s['temperature']]))
        
        return {
            'duration_seconds': duration_seconds,
            'sample_count': total_samples,
            'average_utilization_percent': round(avg_utilization, 1),
            'average_memory_usage_percent': round(avg_memory, 1),
            'average_temperature_c': round(avg_temp, 1),
            'samples': samples
        }
    else:
        return {'error': 'No samples collected', 'duration_seconds': duration_seconds}

def main():
    """Main CLI interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description='GPU Utilities for Hashmancer')
    parser.add_argument('command', choices=['info', 'test', 'optimize', 'monitor'], 
                       help='Command to run')
    parser.add_argument('--json', action='store_true', help='Output JSON format')
    parser.add_argument('--duration', type=int, default=60, 
                       help='Monitoring duration in seconds (for monitor command)')
    
    args = parser.parse_args()
    
    if args.command == 'info':
        result = get_gpu_info()
    elif args.command == 'test':
        result = test_gpu_access()
    elif args.command == 'optimize':
        result = optimize_gpu_settings()
    elif args.command == 'monitor':
        result = monitor_gpu_usage(args.duration)
    
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        if args.command == 'info':
            if result['available']:
                print(f"✅ {result['count']} GPU(s) detected")
                for gpu in result['gpus']:
                    print(f"   GPU {gpu['index']}: {gpu['name']}")
                    print(f"   Memory: {gpu['memory_used_mb']}/{gpu['memory_total_mb']} MB ({gpu['memory_usage_percent']}%)")
                    print(f"   Utilization: {gpu['utilization_percent']}%")
                    if gpu['temperature_c']:
                        print(f"   Temperature: {gpu['temperature_c']}°C")
            else:
                print(f"❌ GPU not available: {result.get('error', 'Unknown error')}")
                
        elif args.command == 'test':
            print("GPU Access Test Results:")
            print(f"   nvidia-smi: {'✅' if result['nvidia_smi'] else '❌'}")
            print(f"   CUDA: {'✅' if result['cuda_available'] else '❌'}")
            print(f"   PyTorch CUDA: {'✅' if result['pytorch_cuda'] else '❌'}")
            print(f"   GPU Count: {result['gpu_count']}")
            if result['errors']:
                print("   Errors:")
                for error in result['errors']:
                    print(f"     - {error}")
                    
        elif args.command == 'optimize':
            if result['optimizations_applied']:
                print("✅ Optimizations applied:")
                for opt in result['optimizations_applied']:
                    print(f"   - {opt}")
            if result['warnings']:
                print("⚠️  Warnings:")
                for warning in result['warnings']:
                    print(f"   - {warning}")
                    
        elif args.command == 'monitor':
            if 'error' in result:
                print(f"❌ Monitoring failed: {result['error']}")
            else:
                print(f"📊 GPU Monitoring Results ({result['duration_seconds']}s):")
                print(f"   Average Utilization: {result['average_utilization_percent']}%")
                print(f"   Average Memory Usage: {result['average_memory_usage_percent']}%")
                print(f"   Average Temperature: {result['average_temperature_c']}°C")
                print(f"   Samples Collected: {result['sample_count']}")

if __name__ == '__main__':
    main()