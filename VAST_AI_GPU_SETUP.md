# 🎮 Vast.ai GPU Setup Guide

This guide ensures your Hashmancer workers get proper NVIDIA CUDA support on Vast.ai.

## ✅ **Guaranteed CUDA-Ready Setup**

### **Step 1: Choose the Right Base Images**

Your Docker image already uses the correct CUDA base:
```dockerfile
FROM nvidia/cuda:12.2-devel-ubuntu22.04
```

This includes:
- ✅ **Ubuntu 22.04** (latest LTS)
- ✅ **CUDA 12.2** toolkit pre-installed
- ✅ **NVIDIA drivers** compatible
- ✅ **Development tools** for compilation

### **Step 2: Vast.ai Template Configuration**

Here's your **complete Vast.ai template**:

#### **🐳 Docker Repository And Environment**
```
Image Path:Tag: yourusername/hashmancer-worker:latest
Version Tag: latest
```

#### **⚙️ Docker Options**
```
Docker create/run options: --gpus all --restart unless-stopped -p 8081:8081
```
**Important**: The `--gpus all` flag gives your container access to all GPUs!

#### **🔌 Ports**
```
8081
```

#### **🌍 Environment Variables**
| Key | Value |
|-----|-------|
| `HASHMANCER_SERVER_IP` | `YOUR_PUBLIC_IP` |
| `HASHMANCER_SERVER_PORT` | `8080` |
| `WORKER_PORT` | `8081` |
| `MAX_CONCURRENT_JOBS` | `3` |
| `LOG_LEVEL` | `INFO` |
| `NVIDIA_VISIBLE_DEVICES` | `all` |
| `NVIDIA_DRIVER_CAPABILITIES` | `compute,utility` |

#### **🚀 Launch Mode**
```
Docker ENTRYPOINT
```

#### **📝 On-start Script**
```bash
echo "🔓 Hashmancer GPU Worker Starting..."
echo "Server: $HASHMANCER_SERVER_IP:$HASHMANCER_SERVER_PORT"
echo "GPU verification will run automatically..."

# Verify GPU access
nvidia-smi || echo "Warning: nvidia-smi not available"
```

#### **💾 Disk Space**
```
15 GB (minimum for CUDA tools)
```

### **Step 3: GPU Instance Selection**

When choosing instances on Vast.ai:

#### **✅ Recommended GPU Types:**
- **RTX 4090** - Best performance/price for hash cracking
- **RTX 3080/3090** - Good performance, lower cost
- **RTX A6000** - Professional cards, very fast
- **V100** - Data center cards, excellent for compute

#### **⚠️ Avoid These:**
- **GTX 16xx series** - No tensor cores
- **Very old GPUs** - May lack driver support
- **Mining-specific cards** - Often lack display outputs

#### **🔍 Instance Verification Checklist:**
- ✅ **CUDA Compute Capability** ≥ 6.0
- ✅ **Driver Version** ≥ 525.xx
- ✅ **Memory** ≥ 8GB for large hash lists
- ✅ **Verified** status on Vast.ai

### **Step 4: Advanced GPU Configuration**

#### **For Maximum Performance:**
Add these environment variables:

| Key | Value | Purpose |
|-----|-------|---------|
| `CUDA_VISIBLE_DEVICES` | `0,1,2,3` | Specify which GPUs to use |
| `NVIDIA_DRIVER_CAPABILITIES` | `compute,utility,graphics` | Full GPU capabilities |
| `CUDA_CACHE_DISABLE` | `0` | Enable CUDA kernel caching |
| `CUDA_LAUNCH_BLOCKING` | `0` | Async CUDA launches |

#### **For Memory Optimization:**
```bash
# In on-start script
echo "Setting GPU memory mode..."
nvidia-smi -pm 1  # Enable persistence mode
nvidia-smi -ac 877,1215  # Set application clocks (adjust per GPU)
```

### **Step 5: Verification Commands**

Your worker automatically runs these checks:

```bash
# Basic GPU detection
nvidia-smi

# Detailed GPU info
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv

# CUDA runtime verification
nvcc --version

# Docker GPU access test
python3 -c "
import subprocess
result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
print('GPUs detected:')
print(result.stdout)
"
```

## 🔧 **Troubleshooting GPU Issues**

### **Issue**: "nvidia-smi: command not found"

**Cause**: Container doesn't have GPU access

**Solution**: 
1. Add `--gpus all` to Docker options
2. Set `NVIDIA_VISIBLE_DEVICES=all` environment variable
3. Choose a GPU-enabled instance type

### **Issue**: "CUDA driver version is insufficient"

**Cause**: Host driver too old for your CUDA version

**Solution**:
1. Choose instances with newer drivers (≥525.xx)
2. Use older CUDA base image: `nvidia/cuda:11.8-devel-ubuntu20.04`

### **Issue**: "No GPU detected in container"

**Verification Steps**:
```bash
# SSH to your Vast.ai instance
ssh root@instance-ip

# Check host GPU
nvidia-smi

# Check container GPU access
docker exec your-container nvidia-smi

# Check Docker GPU runtime
docker info | grep -i nvidia
```

### **Issue**: "Out of memory errors"

**Solutions**:
1. Reduce `MAX_CONCURRENT_JOBS` to 1 or 2
2. Choose instances with more GPU memory
3. Add memory monitoring to your worker

## 🎯 **Optimized Instance Template**

Here's a **complete working template**:

```yaml
# Vast.ai Instance Configuration
Image: yourusername/hashmancer-worker:latest
Docker Options: --gpus all --restart unless-stopped -p 8081:8081 --shm-size=1g

Environment:
  HASHMANCER_SERVER_IP: "203.0.113.1"  # Your IP
  HASHMANCER_SERVER_PORT: "8080"
  WORKER_PORT: "8081"
  MAX_CONCURRENT_JOBS: "3"
  LOG_LEVEL: "INFO"
  NVIDIA_VISIBLE_DEVICES: "all"
  NVIDIA_DRIVER_CAPABILITIES: "compute,utility"
  CUDA_VISIBLE_DEVICES: "all"

Launch Mode: Docker ENTRYPOINT
Disk Space: 15 GB
Ports: 8081

On-start Script: |
  echo "🔓 Hashmancer GPU Worker"
  echo "GPUs: $(nvidia-smi -L | wc -l)"
  echo "Driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
  echo "Ready for hash cracking!"
```

## 🚀 **Performance Tuning**

### **GPU-Specific Optimizations:**

#### **RTX 4090:**
```bash
# Optimal settings for RTX 4090
MAX_CONCURRENT_JOBS=4
CUDA_DEVICE_ORDER=PCI_BUS_ID
```

#### **RTX 3080:**
```bash
# Optimal settings for RTX 3080
MAX_CONCURRENT_JOBS=3
NVIDIA_MIG_CONFIG_DEVICES=all
```

#### **Multi-GPU Setup:**
```bash
# For instances with multiple GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3
MAX_CONCURRENT_JOBS=6  # 1.5x GPU count
```

### **Memory Management:**
```bash
# Add to on-start script
echo "Configuring GPU memory..."
nvidia-smi -pm 1  # Persistence mode
nvidia-smi --reset-gpu-ecc=0  # Disable ECC if not needed
```

## ✨ **Expected Results**

After deployment, your worker logs should show:
```
🔓 Hashmancer GPU Worker Starting...
✅ NVIDIA GPU detected: NVIDIA GeForce RTX 4090 (24564 MiB) - Driver 535.xx
🔍 Running GPU verification...
✅ GPU verification passed
✅ Worker registered successfully
📋 Starting job polling...
Ready for hash cracking! 🚀
```

Your worker will automatically:
1. ✅ **Detect all available GPUs**
2. ✅ **Verify CUDA compatibility**
3. ✅ **Report GPU capabilities to server**
4. ✅ **Optimize for detected hardware**
5. ✅ **Start processing jobs with GPU acceleration**

This setup guarantees your workers will have full NVIDIA CUDA support for maximum hash cracking performance! 🎮⚡