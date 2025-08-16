# 🚀 Hashmancer Ultimate Docker Deployment Guide

This guide provides the **most seamless Docker deployment experience** for Hashmancer with full GPU support, NVIDIA drivers, Hashcat, and Darkling.

## ✨ Features

✅ **One-command deployment** - Single script deploys everything  
✅ **GPU acceleration** - Full NVIDIA CUDA support with drivers  
✅ **Dual hash engines** - Both Hashcat and Darkling available  
✅ **Auto-scaling** - Separate server and worker containers  
✅ **Production-ready** - Nginx reverse proxy, health checks, monitoring  
✅ **Rock-solid Redis** - Optimized Redis with our unified management system  
✅ **Zero configuration** - Intelligent defaults that just work  

## 🎯 Quick Start (30 seconds)

The absolute fastest way to get Hashmancer running:

```bash
# Clone and deploy in one command
git clone <repository> && cd hashmancer
./deploy-hashmancer.sh quick
```

That's it! 🎉

## 📋 Prerequisites

### Required
- **Docker** (20.10+)
- **Docker Compose** (2.0+)
- **4GB RAM** minimum
- **10GB disk space**

### For GPU Support
- **NVIDIA GPU** with compute capability 3.5+
- **NVIDIA drivers** (470+)
- **NVIDIA Container Toolkit** (auto-installed by script)

### Check Your System
```bash
# Check Docker
docker --version
docker-compose --version

# Check GPU (if available)
nvidia-smi
```

## 🚀 Deployment Options

### 1. Interactive Deployment (Recommended)
```bash
./deploy-hashmancer.sh
```
- Guided setup with choices
- Automatic GPU detection
- Custom service selection

### 2. Quick Full Deployment
```bash
./deploy-hashmancer.sh quick
```
- Deploys everything automatically
- Uses intelligent defaults
- GPU support auto-detected

### 3. Server Only
```bash
./deploy-hashmancer.sh server-only
```
- Just server and Redis
- Perfect for centralized setup

### 4. GPU Worker Only
```bash
./deploy-hashmancer.sh gpu-worker
```
- Connects to existing server
- Requires server running elsewhere

## 🏗️ Architecture

The deployment includes these services:

### Core Services
- **🖥️ Hashmancer Server** - Main application server
- **📊 Redis** - Optimized job queue and cache
- **🌐 Nginx** - Reverse proxy with SSL support

### Worker Services
- **🎮 GPU Worker** - CUDA-enabled with Hashcat + Darkling
- **💻 CPU Worker** - CPU-only with Hashcat + Darkling

### Optional Services
- **📈 Prometheus** - Metrics collection
- **📊 Grafana** - Performance dashboards

## 🎮 GPU Configuration

### Automatic Setup
The deployment script automatically:
1. Detects NVIDIA GPUs
2. Installs NVIDIA Container Toolkit
3. Configures Docker for GPU access
4. Sets up CUDA environment
5. Installs Hashcat with GPU support

### Manual GPU Setup (if needed)
```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## ⚙️ Configuration

### Environment Variables

Create `.env` file to customize:

```env
# Redis Configuration
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_MAX_CONNECTIONS=50

# Server Configuration
HASHMANCER_HOST=0.0.0.0
HASHMANCER_PORT=8080
LOG_LEVEL=INFO

# Worker Configuration
DEFAULT_ENGINE=darkling
WORKER_LOG_LEVEL=INFO

# GPU Configuration
GPU_MEMORY_LIMIT=0.9
CUDA_VISIBLE_DEVICES=all

# Performance
WORKERS=4
MAX_REQUESTS=1000
```

### Redis Configuration

Customize `config/redis.conf`:

```conf
# Memory settings
maxmemory 4gb
maxmemory-policy allkeys-lru

# Persistence
save 900 1
save 300 10
save 60 10000
appendonly yes

# Performance
tcp-keepalive 300
timeout 0
```

## 🎛️ Service Management

### Basic Operations
```bash
# View status
docker-compose -f docker-compose.ultimate.yml ps

# View logs (all services)
docker-compose -f docker-compose.ultimate.yml logs -f

# View logs (specific service)
docker-compose -f docker-compose.ultimate.yml logs -f server
docker-compose -f docker-compose.ultimate.yml logs -f worker-gpu

# Restart services
docker-compose -f docker-compose.ultimate.yml restart

# Stop services
docker-compose -f docker-compose.ultimate.yml down

# Stop and remove volumes
docker-compose -f docker-compose.ultimate.yml down -v
```

### Scale Workers
```bash
# Scale to 3 GPU workers
docker-compose -f docker-compose.ultimate.yml up -d --scale worker-gpu=3

# Scale to 5 CPU workers  
docker-compose -f docker-compose.ultimate.yml up -d --scale worker-cpu=5
```

## 📊 Monitoring

### Health Checks
```bash
# Overall health
docker-compose -f docker-compose.ultimate.yml ps

# Redis health
python redis_tool.py health

# GPU status
docker exec hashmancer-worker-gpu python /app/gpu-utils.py info

# Server health
curl http://localhost:8080/health
```

### Access Points
- **🌐 Web Interface**: http://localhost
- **🔧 Server Direct**: http://localhost:8080  
- **📡 API**: http://localhost:8000
- **📊 Redis**: localhost:6379
- **📈 Prometheus**: http://localhost:9090 (if enabled)
- **📊 Grafana**: http://localhost:3000 (if enabled)

### Performance Monitoring
```bash
# Real-time GPU monitoring
docker exec hashmancer-worker-gpu python /app/gpu-utils.py monitor --duration 60

# Worker performance
docker stats hashmancer-worker-gpu hashmancer-worker-cpu

# Redis performance
python redis_tool.py stats
```

## 🛠️ Troubleshooting

### Common Issues

#### 1. GPU Not Detected
```bash
# Check GPU availability
nvidia-smi

# Check Docker GPU support
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# Restart Docker if needed
sudo systemctl restart docker
```

#### 2. Redis Connection Issues
```bash
# Test Redis directly
docker exec hashmancer-redis redis-cli ping

# Check Redis logs
docker-compose -f docker-compose.ultimate.yml logs redis

# Redis health check
python redis_tool.py test
```

#### 3. Worker Not Connecting
```bash
# Check worker logs
docker-compose -f docker-compose.ultimate.yml logs worker-gpu

# Test server connectivity
docker exec hashmancer-worker-gpu curl http://server:8080/health

# Check network
docker network ls
```

#### 4. Build Failures
```bash
# Clean rebuild
docker-compose -f docker-compose.ultimate.yml build --no-cache

# Check disk space
df -h

# Clean Docker cache
docker system prune -f
```

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Verbose Docker Compose
docker-compose -f docker-compose.ultimate.yml --verbose up
```

## 🔧 Advanced Configuration

### Custom Worker Configuration
```yaml
# docker-compose.override.yml
version: '3.8'
services:
  worker-gpu:
    environment:
      - CUDA_VISIBLE_DEVICES=0,1  # Use specific GPUs
      - GPU_MEMORY_LIMIT=0.8      # Limit GPU memory usage
      - HASHCAT_OPTIONS=--force   # Custom Hashcat options
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0', '1']  # Specific GPU devices
              capabilities: [gpu]
```

### SSL Configuration
```bash
# Generate SSL certificates
mkdir -p ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout ssl/key.pem -out ssl/cert.pem

# Update nginx configuration
# Uncomment HTTPS server block in docker/nginx/conf.d/hashmancer.conf
```

### Custom Wordlists
```bash
# Add wordlists
mkdir -p wordlists
cp your-wordlist.txt wordlists/

# They'll be available in containers at /app/wordlists/
```

## 🚀 Production Deployment

### Security Hardening
```bash
# 1. Use SSL certificates
# 2. Change default passwords
# 3. Configure firewall
# 4. Use secrets management
# 5. Enable audit logging
```

### Performance Tuning
```bash
# Increase Redis memory
export REDIS_MEMORY=8gb

# More workers
export WORKERS=8

# GPU optimization
export GPU_MEMORY_LIMIT=0.95
```

### Backup Strategy
```bash
# Backup Redis data
python redis_tool.py backup --path backups/redis-$(date +%Y%m%d).pkl

# Backup configuration
tar -czf backups/config-$(date +%Y%m%d).tar.gz config/ .env

# Backup logs
tar -czf backups/logs-$(date +%Y%m%d).tar.gz logs/
```

## 📝 Examples

### Complete GPU Setup
```bash
# 1. Check prerequisites
nvidia-smi
docker --version

# 2. Clone and deploy
git clone <repository>
cd hashmancer

# 3. Quick deployment
./deploy-hashmancer.sh quick

# 4. Verify GPU access
docker exec hashmancer-worker-gpu nvidia-smi
docker exec hashmancer-worker-gpu python /app/gpu-utils.py info

# 5. Test hash cracking
# Access web interface at http://localhost
```

### Server + Multiple Workers
```bash
# 1. Deploy server
./deploy-hashmancer.sh server-only

# 2. On worker machines
export SERVER_HOST=your-server-ip
./deploy-hashmancer.sh gpu-worker

# 3. Scale workers
docker-compose -f docker-compose.ultimate.yml up -d --scale worker-gpu=4
```

### Development Setup
```bash
# 1. Development environment
export BUILD_ENV=development

# 2. Mount source code
docker-compose -f docker-compose.ultimate.yml \
    -f docker-compose.dev.yml up -d

# 3. Live reload enabled
# Code changes automatically reload
```

## 🆘 Support

### Getting Help
1. **Check logs**: `docker-compose logs -f [service]`
2. **Health checks**: `python redis_tool.py health`
3. **GPU diagnostics**: `docker exec hashmancer-worker-gpu python /app/gpu-utils.py test`
4. **System resources**: `docker stats`

### Reporting Issues
Include this information:
```bash
# System info
uname -a
docker --version
nvidia-smi || echo "No GPU"

# Service status
docker-compose -f docker-compose.ultimate.yml ps

# Recent logs
docker-compose -f docker-compose.ultimate.yml logs --tail=50
```

---

## 🎉 Success!

Your Hashmancer deployment should now be running with:

✅ **Server** accessible at http://localhost  
✅ **GPU workers** with CUDA + Hashcat + Darkling  
✅ **Redis** with optimized configuration  
✅ **Monitoring** and health checks  
✅ **Production-ready** infrastructure  

**Happy hash cracking!** 🔓