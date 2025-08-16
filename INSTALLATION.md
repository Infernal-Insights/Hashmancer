# Hashmancer Installation Guide

Welcome to Hashmancer's seamless installation system! This guide provides multiple installation methods to get you up and running quickly.

## 🚀 Quick Start

### One-Command Installation

For the fastest setup experience:

```bash
curl -fsSL https://raw.githubusercontent.com/Infernal-Insights/Hashmancer/main/install-hashmancer.sh | bash
```

Or download and run locally:

```bash
git clone https://github.com/Infernal-Insights/Hashmancer.git
cd Hashmancer
chmod +x install-hashmancer.sh
./install-hashmancer.sh
```

## 📋 Installation Methods

### 1. 🐳 Docker Installation (Recommended)

The Docker installation provides the easiest deployment with automatic dependency management.

#### Prerequisites
- Docker 20.10+ and Docker Compose 2.0+
- 4GB+ RAM, 10GB+ disk space
- For GPU: NVIDIA Container Toolkit

#### Quick Docker Deployment

```bash
# Production deployment
./docker-scripts/quick-deploy.sh production

# GPU-accelerated deployment
./docker-scripts/quick-deploy.sh gpu

# Development environment
./docker-scripts/quick-deploy.sh development

# With monitoring stack
./docker-scripts/quick-deploy.sh monitoring
```

#### Manual Docker Setup

1. **Clone and configure:**
   ```bash
   git clone https://github.com/Infernal-Insights/Hashmancer.git
   cd Hashmancer
   cp .env.example .env
   # Edit .env with your configuration
   ```

2. **Choose your deployment:**
   ```bash
   # Standard production
   docker-compose -f docker-compose.production.yml up -d
   
   # GPU-enabled
   docker-compose -f docker-compose.gpu.yml up -d
   
   # Development
   docker-compose -f docker-compose.development.yml up -d
   ```

3. **Access your installation:**
   - Portal: http://localhost:8000
   - Admin: http://localhost:8000/admin

### 2. 🖥️ Native Installation

Direct system installation for maximum performance and control.

#### Prerequisites
- Ubuntu 20.04+ or Debian 11+
- Python 3.11+, Node.js 18+, Redis 7+
- 8GB+ RAM, 20GB+ disk space
- Sudo privileges

#### Installation Steps

1. **Run the interactive installer:**
   ```bash
   ./install-hashmancer.sh
   ```
   Select "Native Installation" when prompted.

2. **Or manual installation:**
   ```bash
   # Install system dependencies
   sudo apt update
   sudo apt install -y python3 python3-pip python3-venv nodejs npm redis-server nginx
   
   # Create virtual environment
   python3 -m venv venv
   source venv/bin/activate
   
   # Install Python dependencies
   pip install -r hashmancer/server/requirements.txt
   
   # Configure and start services
   sudo systemctl enable redis-server nginx
   sudo systemctl start redis-server nginx
   ```

### 3. ☁️ Cloud/VPS Installation

Optimized for cloud deployment with automatic SSL/TLS and domain configuration.

#### Prerequisites
- Cloud instance with public IP
- Domain name pointing to your server
- Email address for SSL certificates

#### Installation Steps

1. **Run the installer:**
   ```bash
   ./install-hashmancer.sh
   ```
   Select "Cloud/VPS Installation" when prompted.

2. **Provide required information:**
   - Domain name (e.g., hashmancer.yourdomain.com)
   - Email for SSL certificate
   - Admin credentials
   - API keys (optional)

3. **The installer will automatically:**
   - Configure firewall rules
   - Set up NGINX reverse proxy
   - Obtain SSL certificates via Let's Encrypt
   - Configure automatic renewals

## ⚙️ Configuration Options

### Environment Variables

Create and customize your `.env` file:

```bash
cp .env.example .env
```

Key configuration options:

```bash
# Server Configuration
HTTP_PORT=8000
HTTPS_PORT=8443
ADMIN_USERNAME=admin
ADMIN_PASSWORD=your_secure_password

# API Keys (Optional)
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# GPU Configuration
ENABLE_GPU=true
GPU_TYPE=nvidia
GPU_MEMORY_FRACTION=0.8

# Security
ALLOW_EXTERNAL=true
DOMAIN_NAME=hashmancer.yourdomain.com
SSL_EMAIL=admin@yourdomain.com
```

### GPU Support

#### NVIDIA GPUs
1. Install NVIDIA drivers:
   ```bash
   sudo apt install nvidia-driver-535
   ```

2. Install NVIDIA Container Toolkit (for Docker):
   ```bash
   curl -fsSL https://nvidia.github.io/nvidia-docker/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-docker.gpg
   echo "deb [signed-by=/usr/share/keyrings/nvidia-docker.gpg] https://nvidia.github.io/nvidia-docker/ubuntu22.04/amd64 /" | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt update && sudo apt install -y nvidia-docker2
   sudo systemctl restart docker
   ```

3. Use GPU-enabled deployment:
   ```bash
   ./docker-scripts/quick-deploy.sh gpu
   ```

#### AMD GPUs
ROCm support is available for compatible AMD GPUs. Set `GPU_TYPE=amd` in your environment.

## 🛠️ Management Commands

### Docker Deployments

```bash
# View status
docker-compose ps

# View logs
docker-compose logs -f hashmancer

# Restart services
docker-compose restart

# Stop services
docker-compose down

# Update
git pull && docker-compose build --no-cache && docker-compose up -d
```

### Native Installations

```bash
# Control services
sudo systemctl start|stop|restart|status hashmancer

# View logs
sudo journalctl -u hashmancer -f

# Update
git pull && source venv/bin/activate && pip install -r hashmancer/server/requirements.txt
sudo systemctl restart hashmancer
```

### Generated Management Scripts

The installer creates helpful scripts in the `scripts/` directory:

```bash
./scripts/start.sh      # Start Hashmancer
./scripts/stop.sh       # Stop Hashmancer
./scripts/restart.sh    # Restart Hashmancer
./scripts/status.sh     # Check status
./scripts/logs.sh       # View logs
./scripts/update.sh     # Update installation
```

## 🔒 Security Considerations

### Firewall Configuration

The installer automatically configures UFW firewall rules:

```bash
# Allow SSH (port 22)
# Allow HTTP (port 80) - redirects to HTTPS
# Allow HTTPS (port 443)
# Allow Hashmancer (configured port)
# Block all other incoming connections
```

### SSL/TLS Certificates

For cloud deployments, SSL certificates are automatically obtained from Let's Encrypt and configured for auto-renewal.

### Security Best Practices

1. **Use strong admin passwords**
2. **Keep your system updated**
3. **Enable firewall rules**
4. **Use HTTPS in production**
5. **Regular backups**
6. **Monitor logs for suspicious activity**

## 📊 Monitoring and Logging

### Built-in Monitoring

Enable monitoring stack:

```bash
./docker-scripts/quick-deploy.sh monitoring
```

Access monitoring tools:
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000

### Log Management

Logs are automatically rotated and managed:

- **Application logs**: `/app/logs/` (Docker) or `logs/` (native)
- **System logs**: `journalctl -u hashmancer`
- **Web server logs**: `/var/log/nginx/`

## 🚨 Troubleshooting

### Common Issues

#### Installation Fails
```bash
# Check system requirements
./install-hashmancer.sh --check-only

# View detailed logs
./install-hashmancer.sh --verbose

# Clean and retry
./install-hashmancer.sh --clean
```

#### Services Won't Start
```bash
# Check Docker status
docker-compose ps
docker-compose logs

# Check native installation
sudo systemctl status hashmancer
sudo journalctl -u hashmancer -n 50
```

#### Port Conflicts
```bash
# Check what's using the port
sudo lsof -i :8000

# Change port in .env file
echo "HTTP_PORT=8080" >> .env
```

#### GPU Not Detected
```bash
# Check NVIDIA drivers
nvidia-smi

# Check Docker GPU support
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi

# Verify configuration
cat /etc/docker/daemon.json
```

### Getting Help

1. **Check logs first**: Application and system logs usually contain helpful error messages
2. **Review configuration**: Ensure all required settings are properly configured
3. **Test connectivity**: Verify network access and DNS resolution
4. **Check resources**: Ensure adequate CPU, memory, and disk space
5. **Update system**: Keep your OS and Docker up to date

### Support Resources

- **Documentation**: https://github.com/Infernal-Insights/Hashmancer/wiki
- **Issues**: https://github.com/Infernal-Insights/Hashmancer/issues
- **Discussions**: https://github.com/Infernal-Insights/Hashmancer/discussions

## 🔄 Updates and Maintenance

### Automatic Updates

Enable automatic updates in your `.env`:

```bash
ENABLE_AUTO_UPDATE=true
```

### Manual Updates

```bash
# Docker installations
./scripts/update.sh

# Or manually
git pull
docker-compose build --no-cache
docker-compose up -d

# Native installations
git pull
source venv/bin/activate
pip install -r hashmancer/server/requirements.txt
sudo systemctl restart hashmancer
```

### Backup and Restore

#### Backup
```bash
# Docker volumes
docker run --rm -v hashmancer_data:/data -v $(pwd):/backup busybox tar czf /backup/hashmancer-backup.tar.gz /data

# Configuration
tar czf config-backup.tar.gz .env config/ scripts/
```

#### Restore
```bash
# Stop services
docker-compose down

# Restore data
docker run --rm -v hashmancer_data:/data -v $(pwd):/backup busybox tar xzf /backup/hashmancer-backup.tar.gz -C /

# Restart
docker-compose up -d
```

## 🎯 Next Steps

After installation:

1. **Access the portal**: http://localhost:8000
2. **Log in with admin credentials**
3. **Configure API keys** (if not done during setup)
4. **Upload wordlists and rules**
5. **Start your first hash cracking job**
6. **Explore the admin panel** for advanced configuration

Welcome to Hashmancer! 🎉