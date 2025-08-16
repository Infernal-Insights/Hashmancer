# 🚀 Hashmancer Seamless Installation

Get Hashmancer running in minutes with our automated installation system!

## One-Command Installation

```bash
curl -fsSL https://raw.githubusercontent.com/Infernal-Insights/Hashmancer/main/install-hashmancer.sh | bash
```

## What's New? ✨

### 🎯 **Completely Automated Setup**
- Interactive prompts for API keys and preferences
- Automatic system dependency installation
- Firewall configuration
- SSL/TLS setup with Let's Encrypt
- GPU passthrough auto-configuration

### 🐳 **Amazing Docker Support**
- Production-ready containers
- GPU-accelerated images
- Automatic health checks
- Resource management
- Security hardening

### ⚡ **Three Installation Modes**

#### 1. 🐳 Docker (Recommended)
```bash
./install-hashmancer.sh
# Select: Docker Installation
```
- Easy deployment and updates
- Isolated environment
- GPU passthrough support
- One-command deployment

#### 2. 🖥️ Native Installation
```bash
./install-hashmancer.sh
# Select: Native Installation
```
- Direct system installation
- Maximum performance
- Full system integration
- Systemd service management

#### 3. ☁️ Cloud/VPS Installation
```bash
./install-hashmancer.sh
# Select: Cloud Installation
```
- Automatic SSL/TLS with Let's Encrypt
- Domain name configuration
- Firewall setup
- Production-ready security

## 🎮 Quick Start Commands

### Docker Deployments
```bash
# Production
./docker-scripts/quick-deploy.sh production

# GPU-accelerated
./docker-scripts/quick-deploy.sh gpu

# Development
./docker-scripts/quick-deploy.sh development

# With monitoring
./docker-scripts/quick-deploy.sh monitoring
```

### Management
```bash
./scripts/start.sh      # Start services
./scripts/stop.sh       # Stop services
./scripts/status.sh     # Check status
./scripts/logs.sh       # View logs
./scripts/update.sh     # Update installation
```

## 🔧 What Gets Installed Automatically

### System Dependencies
- ✅ Docker & Docker Compose
- ✅ Redis server
- ✅ Node.js & npm
- ✅ Python 3.11+ & pip
- ✅ NGINX (for cloud deployments)
- ✅ Build tools & compilers

### GPU Support
- ✅ NVIDIA Container Toolkit
- ✅ CUDA runtime detection
- ✅ GPU passthrough configuration
- ✅ Memory optimization

### Security & Networking
- ✅ UFW firewall rules
- ✅ SSL/TLS certificates
- ✅ Reverse proxy setup
- ✅ Security headers
- ✅ Rate limiting

### Monitoring & Health
- ✅ Health check endpoints
- ✅ Automatic service recovery
- ✅ Log rotation
- ✅ Performance monitoring
- ✅ Resource limits

## 🎛️ Interactive Configuration

The installer prompts for:

### Core Settings
- Admin username & password
- HTTP/HTTPS ports
- Redis configuration
- Network access permissions

### API Integration
- OpenAI API key
- Anthropic API key
- Custom API endpoints

### Cloud Setup
- Domain name
- SSL email address
- DNS configuration

### GPU Configuration
- Automatic GPU detection
- Memory allocation
- Performance tuning

## 🐳 Docker Images

### Production Image
```dockerfile
FROM python:3.11-slim
# Multi-stage build
# Security hardened
# Non-root user
# Health checks
```

### GPU Image
```dockerfile
FROM nvidia/cuda:11.8-runtime-ubuntu22.04
# CUDA support
# PyTorch GPU
# GPU monitoring
# Memory management
```

### Worker Image
```dockerfile
FROM python:3.11-slim
# Distributed processing
# Darkling engine
# Optimized for cracking
```

## 📊 Deployment Options

### docker-compose.production.yml
- Production-ready
- Security hardened
- Resource limits
- Health monitoring

### docker-compose.gpu.yml
- GPU acceleration
- CUDA support
- Memory optimization
- Performance monitoring

### docker-compose.development.yml
- Hot reload
- Debug support
- Development tools
- Local testing

## 🔒 Security Features

### Automatic Security Setup
- Firewall configuration
- SSL/TLS certificates
- Security headers
- Rate limiting
- Input validation

### Container Security
- Non-root users
- Read-only filesystems
- Resource limits
- Network isolation
- Security scanning

## 📈 Monitoring Stack

### Built-in Monitoring
```bash
# Enable monitoring
./docker-scripts/quick-deploy.sh monitoring
```

Access:
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000
- **Health checks**: http://localhost:8000/health

### GPU Monitoring
- NVIDIA SMI metrics
- Memory usage tracking
- Temperature monitoring
- Performance analytics

## 🚨 Troubleshooting

### Quick Diagnostics
```bash
# Check installation
./scripts/status.sh

# View logs
./scripts/logs.sh

# Test connectivity
curl http://localhost:8000/health
```

### Common Solutions
```bash
# Restart services
./scripts/restart.sh

# Update installation
./scripts/update.sh

# Clean restart
docker-compose down -v && ./docker-scripts/quick-deploy.sh production
```

## 🎯 Next Steps

1. **Run the installer**: `./install-hashmancer.sh`
2. **Choose your deployment mode**
3. **Configure your preferences**
4. **Access**: http://localhost:8000
5. **Start cracking**! 💥

## 🆘 Need Help?

- 📖 **Full Guide**: [INSTALLATION.md](INSTALLATION.md)
- 🐛 **Issues**: [GitHub Issues](https://github.com/Infernal-Insights/Hashmancer/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/Infernal-Insights/Hashmancer/discussions)
- 📧 **Support**: Check our documentation first!

---

**Ready to crack some hashes?** 🎯

```bash
git clone https://github.com/Infernal-Insights/Hashmancer.git
cd Hashmancer
./install-hashmancer.sh
```

*Installation complete in under 5 minutes!* ⏱️