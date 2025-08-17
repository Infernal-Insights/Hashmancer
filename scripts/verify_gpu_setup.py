#!/usr/bin/env python3
"""
GPU Setup Verification Script
Ensures NVIDIA drivers and CUDA are working properly
"""

import subprocess
import sys
import os
import json
from datetime import datetime

def run_command(cmd, timeout=30):
    """Run a command and return output"""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, 
            text=True, timeout=timeout
        )
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as e:
        return -1, "", str(e)

def check_nvidia_driver():
    """Check NVIDIA driver installation"""
    print("🔍 Checking NVIDIA driver...")
    
    returncode, stdout, stderr = run_command("nvidia-smi --version")
    
    if returncode == 0:
        version_line = stdout.split('\n')[0]
        print(f"✅ NVIDIA driver: {version_line}")
        return True
    else:
        print(f"❌ NVIDIA driver not found: {stderr}")
        return False

def check_cuda_runtime():
    """Check CUDA runtime"""
    print("🔍 Checking CUDA runtime...")
    
    returncode, stdout, stderr = run_command("nvcc --version")
    
    if returncode == 0:
        for line in stdout.split('\n'):
            if 'release' in line.lower():
                print(f"✅ CUDA compiler: {line.strip()}")
                return True
    
    # Try alternative method
    if os.path.exists('/usr/local/cuda/version.txt'):
        try:
            with open('/usr/local/cuda/version.txt', 'r') as f:
                version = f.read().strip()
                print(f"✅ CUDA runtime: {version}")
                return True
        except:
            pass
    
    print(f"⚠️  CUDA compiler not found (may be runtime-only): {stderr}")
    return False

def check_gpu_devices():
    """Check available GPU devices"""
    print("🎮 Checking GPU devices...")
    
    returncode, stdout, stderr = run_command("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader")
    
    if returncode == 0 and stdout:
        gpus = []
        for line in stdout.split('\n'):
            if line.strip():
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 3:
                    gpu_info = {
                        'name': parts[0],
                        'memory': parts[1],
                        'driver': parts[2]
                    }
                    gpus.append(gpu_info)
                    print(f"✅ GPU: {parts[0]} ({parts[1]}) - Driver {parts[2]}")
        
        return len(gpus), gpus
    else:
        print(f"❌ No GPUs detected: {stderr}")
        return 0, []

def check_docker_gpu():
    """Check if running in Docker with GPU access"""
    print("🐳 Checking Docker GPU access...")
    
    # Check if we're in a container
    if os.path.exists('/.dockerenv'):
        print("✅ Running in Docker container")
        
        # Check GPU access from within container
        returncode, stdout, stderr = run_command("nvidia-smi -L")
        if returncode == 0 and stdout:
            gpu_count = len([line for line in stdout.split('\n') if line.strip().startswith('GPU')])
            print(f"✅ Container has access to {gpu_count} GPU(s)")
            return True
        else:
            print(f"❌ Container does not have GPU access: {stderr}")
            return False
    else:
        print("ℹ️  Not running in Docker")
        return True

def check_python_gpu():
    """Check Python GPU libraries"""
    print("🐍 Checking Python GPU support...")
    
    gpu_support = {}
    
    # Check PyTorch
    try:
        import torch
        gpu_support['pytorch'] = {
            'available': torch.cuda.is_available(),
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'version': torch.__version__
        }
        if torch.cuda.is_available():
            print(f"✅ PyTorch CUDA: {torch.cuda.device_count()} device(s)")
        else:
            print("⚠️  PyTorch CUDA not available")
    except ImportError:
        print("ℹ️  PyTorch not installed")
        gpu_support['pytorch'] = {'available': False, 'error': 'Not installed'}
    
    # Check NumPy (basic)
    try:
        import numpy
        print(f"✅ NumPy: {numpy.__version__}")
        gpu_support['numpy'] = {'version': numpy.__version__}
    except ImportError:
        print("⚠️  NumPy not available")
        gpu_support['numpy'] = {'error': 'Not installed'}
    
    return gpu_support

def generate_report():
    """Generate comprehensive GPU setup report"""
    print("=" * 60)
    print("🔓 HASHMANCER GPU SETUP VERIFICATION")
    print("=" * 60)
    
    report = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'hostname': os.environ.get('HOSTNAME', 'unknown'),
        'worker_id': os.environ.get('WORKER_ID', 'unknown'),
        'checks': {}
    }
    
    # Run all checks
    report['checks']['nvidia_driver'] = check_nvidia_driver()
    report['checks']['cuda_runtime'] = check_cuda_runtime()
    
    gpu_count, gpu_list = check_gpu_devices()
    report['checks']['gpu_devices'] = {
        'count': gpu_count,
        'devices': gpu_list
    }
    
    report['checks']['docker_gpu'] = check_docker_gpu()
    report['checks']['python_gpu'] = check_python_gpu()
    
    # Overall status
    critical_checks = [
        report['checks']['nvidia_driver'],
        report['checks']['docker_gpu'],
        gpu_count > 0
    ]
    
    overall_status = all(critical_checks)
    report['overall_status'] = 'ready' if overall_status else 'issues'
    
    print("=" * 60)
    if overall_status:
        print("🎉 GPU SETUP: READY FOR HASH CRACKING!")
        print("✅ All critical components are working")
    else:
        print("⚠️  GPU SETUP: ISSUES DETECTED")
        print("❌ Some components need attention")
    
    print(f"📊 Summary: {gpu_count} GPU(s) detected")
    print("=" * 60)
    
    # Save report
    try:
        with open('/app/logs/gpu_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        print("📄 Report saved to /app/logs/gpu_report.json")
    except:
        pass
    
    return report

if __name__ == "__main__":
    try:
        report = generate_report()
        sys.exit(0 if report['overall_status'] == 'ready' else 1)
    except KeyboardInterrupt:
        print("\n👋 Verification interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        sys.exit(1)