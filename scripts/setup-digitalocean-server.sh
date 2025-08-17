#!/bin/bash
# Hashmancer DigitalOcean Server Setup Script
# Sets up a complete Hashmancer server on a fresh DigitalOcean droplet

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
info() { echo -e "${BLUE}[INFO]${NC} $1"; }

# Configuration
HASHMANCER_USER="hashmancer"
HASHMANCER_DIR="/opt/hashmancer"
SERVER_PORT="8080"
LOG_FILE="/var/log/hashmancer-setup.log"

# Create log file
touch "$LOG_FILE"

log "🔓 Hashmancer DigitalOcean Server Setup"
log "======================================="

# Function to check if running as root
check_root() {
    if [ "$EUID" -ne 0 ]; then
        error "This script must be run as root"
        error "Run: sudo $0"
        exit 1
    fi
}

# Function to get droplet info
get_droplet_info() {
    log "📊 Getting droplet information..."
    
    # Get public IP
    PUBLIC_IP=$(curl -s http://ifconfig.me || curl -s http://ipinfo.io/ip || echo "unknown")
    PRIVATE_IP=$(hostname -I | awk '{print $1}')
    DROPLET_ID=$(curl -s http://169.254.169.254/metadata/v1/id || echo "unknown")
    REGION=$(curl -s http://169.254.169.254/metadata/v1/region || echo "unknown")
    
    info "Public IP: $PUBLIC_IP"
    info "Private IP: $PRIVATE_IP"
    info "Droplet ID: $DROPLET_ID"
    info "Region: $REGION"
    
    # Save info for later use
    cat > /etc/hashmancer/droplet-info.json << EOF
{
    "public_ip": "$PUBLIC_IP",
    "private_ip": "$PRIVATE_IP", 
    "droplet_id": "$DROPLET_ID",
    "region": "$REGION",
    "setup_date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
}

# Function to update system
update_system() {
    log "🔄 Updating system packages..."
    
    apt-get update -qq
    apt-get upgrade -y
    apt-get install -y \
        curl \
        wget \
        git \
        python3 \
        python3-pip \
        python3-venv \
        nginx \
        redis-server \
        ufw \
        htop \
        unzip \
        build-essential \
        software-properties-common \
        supervisor
    
    log "✅ System updated successfully"
}

# Function to configure firewall
configure_firewall() {
    log "🔥 Configuring firewall..."
    
    # Reset firewall
    ufw --force reset
    
    # Default policies
    ufw default deny incoming
    ufw default allow outgoing
    
    # Allow SSH (important!)
    ufw allow ssh
    ufw allow 22
    
    # Allow Hashmancer server port
    ufw allow $SERVER_PORT
    
    # Allow HTTP/HTTPS for potential web interface
    ufw allow 80
    ufw allow 443
    
    # Allow ping
    ufw allow from any to any port 22 proto tcp
    
    # Enable firewall
    ufw --force enable
    
    log "✅ Firewall configured"
    ufw status
}

# Function to create hashmancer user
create_user() {
    log "👤 Creating hashmancer user..."
    
    if id "$HASHMANCER_USER" &>/dev/null; then
        warn "User $HASHMANCER_USER already exists"
    else
        useradd -m -s /bin/bash "$HASHMANCER_USER"
        usermod -aG sudo "$HASHMANCER_USER"
        log "✅ User $HASHMANCER_USER created"
    fi
    
    # Create hashmancer directories
    mkdir -p /etc/hashmancer
    mkdir -p /var/log/hashmancer
    mkdir -p /var/lib/hashmancer
    mkdir -p "$HASHMANCER_DIR"
    
    chown -R $HASHMANCER_USER:$HASHMANCER_USER /var/log/hashmancer
    chown -R $HASHMANCER_USER:$HASHMANCER_USER /var/lib/hashmancer
    chown -R $HASHMANCER_USER:$HASHMANCER_USER "$HASHMANCER_DIR"
}

# Function to install Docker (for potential worker testing)
install_docker() {
    log "🐳 Installing Docker..."
    
    if command -v docker &> /dev/null; then
        warn "Docker already installed"
        return
    fi
    
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    
    # Add hashmancer user to docker group
    usermod -aG docker $HASHMANCER_USER
    
    # Start and enable Docker
    systemctl start docker
    systemctl enable docker
    
    rm get-docker.sh
    log "✅ Docker installed"
}

# Function to configure Redis
configure_redis() {
    log "📊 Configuring Redis..."
    
    # Backup original config
    cp /etc/redis/redis.conf /etc/redis/redis.conf.backup
    
    # Configure Redis for Hashmancer
    cat > /etc/redis/redis.conf << 'EOF'
# Redis configuration for Hashmancer
bind 127.0.0.1
port 6379
timeout 0
tcp-keepalive 60
daemonize yes
supervised systemd
pidfile /var/run/redis/redis-server.pid
loglevel notice
logfile /var/log/redis/redis-server.log
databases 16
save 900 1
save 300 10
save 60 10000
stop-writes-on-bgsave-error yes
rdbcompression yes
rdbchecksum yes
dbfilename dump.rdb
dir /var/lib/redis
maxmemory 256mb
maxmemory-policy allkeys-lru
EOF
    
    # Start and enable Redis
    systemctl restart redis-server
    systemctl enable redis-server
    
    log "✅ Redis configured"
}

# Function to setup Python environment
setup_python() {
    log "🐍 Setting up Python environment..."
    
    # Create virtual environment
    sudo -u $HASHMANCER_USER python3 -m venv "$HASHMANCER_DIR/venv"
    
    # Install Python packages
    sudo -u $HASHMANCER_USER "$HASHMANCER_DIR/venv/bin/pip" install --upgrade pip
    sudo -u $HASHMANCER_USER "$HASHMANCER_DIR/venv/bin/pip" install \
        fastapi \
        uvicorn \
        redis \
        requests \
        click \
        tabulate \
        psutil \
        aiohttp \
        asyncio
    
    log "✅ Python environment ready"
}

# Function to install Hashmancer
install_hashmancer() {
    log "🔓 Installing Hashmancer..."
    
    # Clone or copy Hashmancer
    if [ -d "/tmp/hashmancer" ]; then
        log "Using local Hashmancer files..."
        sudo -u $HASHMANCER_USER cp -r /tmp/hashmancer/* "$HASHMANCER_DIR/"
    else
        log "Downloading Hashmancer..."
        # For now, create minimal server structure
        create_minimal_server
    fi
    
    # Make CLI executable
    if [ -f "$HASHMANCER_DIR/hashmancer-cli" ]; then
        chmod +x "$HASHMANCER_DIR/hashmancer-cli"
        ln -sf "$HASHMANCER_DIR/hashmancer-cli" /usr/local/bin/hashmancer-cli
    fi
    
    log "✅ Hashmancer installed"
}

# Function to create minimal server if source not available
create_minimal_server() {
    log "📦 Creating minimal Hashmancer server..."
    
    # Create directory structure
    sudo -u $HASHMANCER_USER mkdir -p "$HASHMANCER_DIR"/{hashmancer,scripts,logs}
    sudo -u $HASHMANCER_USER mkdir -p "$HASHMANCER_DIR/hashmancer"/{server,worker,cli}
    
    # Create minimal server
    cat > "$HASHMANCER_DIR/hashmancer/server/main.py" << 'EOF'
#!/usr/bin/env python3
"""
Minimal Hashmancer Server for DigitalOcean
"""

import asyncio
import json
import time
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hashmancer-server")

app = FastAPI(title="Hashmancer Server", version="1.0.0")

# In-memory storage (use Redis in production)
workers = {}
jobs = {}

@app.get("/")
async def root():
    return {"message": "Hashmancer Server", "status": "running", "version": "1.0.0"}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "service": "Hashmancer Server",
        "workers": len(workers),
        "jobs": len(jobs)
    }

@app.post("/worker/register")
async def register_worker(worker_data: dict):
    worker_id = worker_data.get("worker_id")
    if not worker_id:
        raise HTTPException(status_code=400, detail="worker_id required")
    
    workers[worker_id] = {
        **worker_data,
        "registered_at": datetime.utcnow().isoformat() + "Z",
        "last_seen": datetime.utcnow().isoformat() + "Z"
    }
    
    logger.info(f"Worker registered: {worker_id}")
    return {"status": "success", "worker_id": worker_id}

@app.get("/workers")
async def list_workers():
    return list(workers.values())

@app.get("/worker/{worker_id}/jobs")
async def get_worker_jobs(worker_id: str):
    # Return available jobs for worker
    return []

@app.post("/worker/{worker_id}/heartbeat")
async def worker_heartbeat(worker_id: str, heartbeat_data: dict):
    if worker_id in workers:
        workers[worker_id]["last_seen"] = datetime.utcnow().isoformat() + "Z"
        workers[worker_id]["status"] = "active"
        return {"status": "success"}
    else:
        raise HTTPException(status_code=404, detail="Worker not found")

@app.get("/jobs")
async def list_jobs():
    return list(jobs.values())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
EOF

    chown -R $HASHMANCER_USER:$HASHMANCER_USER "$HASHMANCER_DIR"
}

# Function to create systemd service
create_systemd_service() {
    log "⚙️  Creating systemd service..."
    
    cat > /etc/systemd/system/hashmancer.service << EOF
[Unit]
Description=Hashmancer Server
After=network.target redis.service
Wants=redis.service

[Service]
Type=simple
User=$HASHMANCER_USER
Group=$HASHMANCER_USER
WorkingDirectory=$HASHMANCER_DIR
Environment=PATH=$HASHMANCER_DIR/venv/bin
ExecStart=$HASHMANCER_DIR/venv/bin/python hashmancer/server/main.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=hashmancer

[Install]
WantedBy=multi-user.target
EOF
    
    # Reload systemd and enable service
    systemctl daemon-reload
    systemctl enable hashmancer
    
    log "✅ Systemd service created"
}

# Function to configure nginx (optional reverse proxy)
configure_nginx() {
    log "🌐 Configuring Nginx..."
    
    cat > /etc/nginx/sites-available/hashmancer << EOF
server {
    listen 80;
    server_name $PUBLIC_IP;
    
    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
    
    location /health {
        proxy_pass http://127.0.0.1:8080/health;
        access_log off;
    }
}
EOF
    
    # Enable site
    ln -sf /etc/nginx/sites-available/hashmancer /etc/nginx/sites-enabled/
    rm -f /etc/nginx/sites-enabled/default
    
    # Test and reload nginx
    nginx -t && systemctl reload nginx
    
    log "✅ Nginx configured"
}

# Function to start services
start_services() {
    log "🚀 Starting services..."
    
    # Start Redis
    systemctl start redis-server
    
    # Start Hashmancer
    systemctl start hashmancer
    
    # Start Nginx
    systemctl start nginx
    
    # Check service status
    sleep 5
    
    if systemctl is-active --quiet hashmancer; then
        log "✅ Hashmancer service is running"
    else
        error "❌ Hashmancer service failed to start"
        systemctl status hashmancer
    fi
    
    if systemctl is-active --quiet redis-server; then
        log "✅ Redis service is running"
    else
        warn "⚠️  Redis service not running"
    fi
    
    if systemctl is-active --quiet nginx; then
        log "✅ Nginx service is running"
    else
        warn "⚠️  Nginx service not running"
    fi
}

# Function to test server
test_server() {
    log "🧪 Testing server..."
    
    # Wait a moment for services to fully start
    sleep 10
    
    # Test local health endpoint
    if curl -s http://localhost:8080/health > /dev/null; then
        log "✅ Local health check passed"
    else
        error "❌ Local health check failed"
    fi
    
    # Test public health endpoint
    if curl -s "http://$PUBLIC_IP:8080/health" > /dev/null; then
        log "✅ Public health check passed"
    else
        warn "⚠️  Public health check failed (firewall/network issue?)"
    fi
    
    # Test Nginx proxy
    if curl -s "http://$PUBLIC_IP/health" > /dev/null; then
        log "✅ Nginx proxy working"
    else
        warn "⚠️  Nginx proxy not working"
    fi
}

# Function to show setup summary
show_summary() {
    log "📋 Setup Summary"
    log "================"
    
    info "🌐 Server Details:"
    info "   Public IP: $PUBLIC_IP"
    info "   API Endpoint: http://$PUBLIC_IP:8080"
    info "   Health Check: http://$PUBLIC_IP:8080/health"
    info "   Web Interface: http://$PUBLIC_IP (via Nginx)"
    
    info "🔑 Access Information:"
    info "   SSH: ssh root@$PUBLIC_IP"
    info "   User: $HASHMANCER_USER"
    info "   Directory: $HASHMANCER_DIR"
    
    info "⚙️  Service Commands:"
    info "   Status: systemctl status hashmancer"
    info "   Logs: journalctl -u hashmancer -f"
    info "   Restart: systemctl restart hashmancer"
    
    info "🎯 For Vast.ai Workers:"
    info "   Set HASHMANCER_SERVER_IP=$PUBLIC_IP"
    info "   Workers will connect to port 8080"
    
    warn "🔒 Security Notes:"
    warn "   • Change default passwords"
    warn "   • Configure SSL/TLS for production"
    warn "   • Monitor server logs regularly"
    warn "   • Keep system updated"
}

# Main execution
main() {
    check_root
    
    mkdir -p /etc/hashmancer
    get_droplet_info
    update_system
    configure_firewall
    create_user
    install_docker
    configure_redis
    setup_python
    install_hashmancer
    create_systemd_service
    configure_nginx
    start_services
    test_server
    show_summary
    
    log "🎉 Hashmancer server setup complete!"
    log "Your server is ready for Vast.ai workers at: http://$PUBLIC_IP:8080"
}

# Run main function
main "$@" 2>&1 | tee -a "$LOG_FILE"