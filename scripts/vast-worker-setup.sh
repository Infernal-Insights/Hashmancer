#!/bin/bash
# Hashmancer Vast.ai Worker Auto-Setup Script
# This script automatically configures a Vast.ai instance as a Hashmancer worker

set -e

# Configuration
HASHMANCER_DOCKER_IMAGE="hashmancer/worker:simple"
HASHMANCER_REPO="https://github.com/hashmancer/hashmancer.git"
HASHMANCER_SERVER_PORT="8080"
WORKER_PORT="8081"
LOG_FILE="/var/log/hashmancer-setup.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "$LOG_FILE"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

# Create log file
mkdir -p /var/log
touch "$LOG_FILE"

log "🚀 Starting Hashmancer Vast.ai Worker Setup"

# Function to detect server IP
detect_server_ip() {
    log "🔍 Detecting Hashmancer server IP..."
    
    # Method 1: Check environment variable
    if [ -n "$HASHMANCER_SERVER_IP" ]; then
        log "Using server IP from environment: $HASHMANCER_SERVER_IP"
        echo "$HASHMANCER_SERVER_IP"
        return 0
    fi
    
    # Method 2: Check for common private network ranges
    local server_ip=""
    for network in "192.168.1" "192.168.0" "10.0.0" "172.16.0"; do
        for i in {1..254}; do
            test_ip="${network}.${i}"
            if timeout 2 curl -s "http://${test_ip}:${HASHMANCER_SERVER_PORT}/health" > /dev/null 2>&1; then
                log "✅ Found Hashmancer server at: $test_ip"
                echo "$test_ip"
                return 0
            fi
        done
    done
    
    # Method 3: Broadcast discovery
    log "📡 Broadcasting for server discovery..."
    
    # Install nmap if needed
    if ! command -v nmap &> /dev/null; then
        apt-get update -qq && apt-get install -y nmap
    fi
    
    # Scan for servers
    local gateways=$(ip route | grep default | awk '{print $3}')
    for gateway in $gateways; do
        local network=$(echo $gateway | cut -d. -f1-3)
        local scan_result=$(nmap -p $HASHMANCER_SERVER_PORT --open -T4 "${network}.0/24" 2>/dev/null | grep -B 4 "open" | grep "Nmap scan report" | awk '{print $5}')
        
        for potential_ip in $scan_result; do
            if timeout 2 curl -s "http://${potential_ip}:${HASHMANCER_SERVER_PORT}/health" > /dev/null 2>&1; then
                log "✅ Found Hashmancer server at: $potential_ip"
                echo "$potential_ip"
                return 0
            fi
        done
    done
    
    error "❌ Could not detect Hashmancer server IP"
    return 1
}

# Function to install Docker
install_docker() {
    if command -v docker &> /dev/null; then
        log "✅ Docker already installed"
        return 0
    fi
    
    log "📦 Installing Docker..."
    
    # Update package index
    apt-get update -qq
    
    # Install dependencies
    apt-get install -y \
        apt-transport-https \
        ca-certificates \
        curl \
        gnupg \
        lsb-release
    
    # Add Docker's official GPG key
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    
    # Set up stable repository
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
    
    # Install Docker
    apt-get update -qq
    apt-get install -y docker-ce docker-ce-cli containerd.io
    
    # Start and enable Docker
    systemctl start docker
    systemctl enable docker
    
    log "✅ Docker installed successfully"
}

# Function to install NVIDIA Docker support
install_nvidia_docker() {
    log "🔧 Installing NVIDIA Docker support..."
    
    # Install NVIDIA Container Toolkit
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
    curl -s -L "https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list" | tee /etc/apt/sources.list.d/nvidia-docker.list
    
    apt-get update -qq && apt-get install -y nvidia-docker2
    
    # Restart Docker
    systemctl restart docker
    
    log "✅ NVIDIA Docker support installed"
}

# Function to download and start worker
start_worker() {
    local server_ip="$1"
    
    log "🐳 Starting Hashmancer worker container..."
    
    # Generate unique worker ID
    local worker_id="vast-$(hostname)-$(date +%s)"
    
    # Create worker configuration
    cat > /tmp/worker-config.json << EOF
{
    "worker_id": "$worker_id",
    "server_host": "$server_ip",
    "server_port": $HASHMANCER_SERVER_PORT,
    "worker_port": $WORKER_PORT,
    "max_concurrent_jobs": 3,
    "auto_register": true,
    "capabilities": {
        "gpu_count": $(nvidia-smi -L 2>/dev/null | wc -l || echo 0),
        "cpu_cores": $(nproc),
        "memory_gb": $(free -g | awk '/^Mem:/{print $2}'),
        "algorithms": ["MD5", "SHA1", "SHA256", "NTLM", "bcrypt"]
    }
}
EOF
    
    # Stop any existing worker containers
    docker stop hashmancer-worker 2>/dev/null || true
    docker rm hashmancer-worker 2>/dev/null || true
    
    # Determine if we need GPU support
    local docker_args=""
    if nvidia-smi &> /dev/null; then
        docker_args="--gpus all"
        log "🎮 GPU detected, enabling GPU support"
    fi
    
    # Start worker container
    docker run -d \
        --name hashmancer-worker \
        --restart unless-stopped \
        -p $WORKER_PORT:$WORKER_PORT \
        -v /tmp/worker-config.json:/app/config.json:ro \
        -v /var/log:/app/logs \
        $docker_args \
        -e HASHMANCER_SERVER="$server_ip:$HASHMANCER_SERVER_PORT" \
        -e WORKER_ID="$worker_id" \
        -e LOG_LEVEL="INFO" \
        "$HASHMANCER_DOCKER_IMAGE"
    
    log "✅ Worker container started with ID: $worker_id"
    
    # Wait for container to be ready
    sleep 10
    
    # Check if worker is running
    if docker ps | grep hashmancer-worker > /dev/null; then
        log "✅ Worker container is running successfully"
        
        # Register with server
        register_worker "$server_ip" "$worker_id"
    else
        error "❌ Worker container failed to start"
        docker logs hashmancer-worker
        return 1
    fi
}

# Function to register worker with server
register_worker() {
    local server_ip="$1"
    local worker_id="$2"
    
    log "📝 Registering worker with server..."
    
    # Prepare registration data
    local registration_data=$(cat << EOF
{
    "worker_id": "$worker_id",
    "host": "$(curl -s http://ifconfig.me || echo 'unknown')",
    "port": $WORKER_PORT,
    "capabilities": {
        "gpu_count": $(nvidia-smi -L 2>/dev/null | wc -l || echo 0),
        "gpu_models": $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | jq -R . | jq -s . || echo '[]'),
        "cpu_cores": $(nproc),
        "memory_gb": $(free -g | awk '/^Mem:/{print $2}'),
        "algorithms": ["MD5", "SHA1", "SHA256", "NTLM", "bcrypt"]
    },
    "status": "ready",
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
)
    
    # Register with server
    local max_attempts=5
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        log "Attempt $attempt/$max_attempts to register with server..."
        
        if curl -s -X POST \
            -H "Content-Type: application/json" \
            -d "$registration_data" \
            "http://$server_ip:$HASHMANCER_SERVER_PORT/worker/register" > /dev/null; then
            
            log "✅ Worker registered successfully with server"
            return 0
        else
            warn "Registration attempt $attempt failed, retrying in 10 seconds..."
            sleep 10
            attempt=$((attempt + 1))
        fi
    done
    
    error "❌ Failed to register worker after $max_attempts attempts"
    return 1
}

# Function to setup monitoring
setup_monitoring() {
    log "📊 Setting up worker monitoring..."
    
    # Create monitoring script
    cat > /usr/local/bin/hashmancer-monitor.sh << 'EOF'
#!/bin/bash
# Hashmancer Worker Monitor

LOG_FILE="/var/log/hashmancer-monitor.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

# Check if worker container is running
if ! docker ps | grep hashmancer-worker > /dev/null; then
    log "ERROR: Worker container not running, attempting restart..."
    docker start hashmancer-worker || {
        log "ERROR: Failed to restart worker container"
        exit 1
    }
fi

# Check worker health
WORKER_IP=$(docker inspect hashmancer-worker | jq -r '.[0].NetworkSettings.IPAddress')
if ! timeout 5 curl -s "http://localhost:8081/health" > /dev/null; then
    log "WARNING: Worker health check failed"
fi

log "Worker monitoring check completed"
EOF
    
    chmod +x /usr/local/bin/hashmancer-monitor.sh
    
    # Create cron job for monitoring
    echo "*/5 * * * * /usr/local/bin/hashmancer-monitor.sh" | crontab -
    
    log "✅ Worker monitoring setup complete"
}

# Function to create worker Docker image if it doesn't exist
create_worker_image() {
    log "🔨 Checking for Hashmancer worker Docker image..."
    
    if docker image inspect "$HASHMANCER_DOCKER_IMAGE" > /dev/null 2>&1; then
        log "✅ Worker image already exists"
        return 0
    fi
    
    log "🔨 Building Hashmancer worker Docker image..."
    
    # Install git for cloning
    apt-get update -qq && apt-get install -y git
    
    # Clone or download Hashmancer
    local build_dir="/tmp/hashmancer-build"
    if [ -d "$build_dir" ]; then
        rm -rf "$build_dir"
    fi
    
    # Create minimal worker structure
    mkdir -p "$build_dir/hashmancer/worker"
    
    # Create Dockerfile
    cat > "$build_dir/Dockerfile" << 'EOF'
FROM nvidia/cuda:11.8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 python3-pip curl wget && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip3 install --no-cache-dir requests psutil

COPY hashmancer/worker/simple_worker.py /app/worker.py
RUN mkdir -p /tmp/hashmancer-jobs /app/logs

EXPOSE 8081

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8081/health || exit 1

CMD ["python3", "/app/worker.py"]
EOF
    
    # Download worker script from server or create simple one
    if curl -s "http://localhost:8888/worker.py" -o "$build_dir/hashmancer/worker/simple_worker.py" 2>/dev/null; then
        log "✅ Downloaded worker script from server"
    else
        log "📦 Creating embedded worker script..."
        # Create a minimal worker script
        cat > "$build_dir/hashmancer/worker/simple_worker.py" << 'WORKER_EOF'
#!/usr/bin/env python3
import os, sys, time, json, requests, threading, signal
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler

class Worker:
    def __init__(self):
        self.worker_id = os.environ.get('WORKER_ID', f'vast-{int(time.time())}')
        self.server_host = os.environ.get('HASHMANCER_SERVER_IP', 'localhost')
        self.server_port = int(os.environ.get('HASHMANCER_SERVER_PORT', '8080'))
        self.worker_port = 8081
        self.running = True
        signal.signal(signal.SIGTERM, lambda s,f: setattr(self, 'running', False))
        
    def register(self):
        try:
            data = {'worker_id': self.worker_id, 'status': 'ready', 'capabilities': {'gpu_count': 0}}
            resp = requests.post(f'http://{self.server_host}:{self.server_port}/worker/register', json=data, timeout=10)
            return resp.status_code == 200
        except: return False
    
    def health_server(self):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self): 
                self.send_response(200); self.send_header('Content-type', 'application/json'); self.end_headers()
                self.wfile.write(b'{"status":"healthy"}')
            def log_message(self, *args): pass
        server = HTTPServer(('0.0.0.0', self.worker_port), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
    
    def run(self):
        print(f"Worker {self.worker_id} starting...")
        self.health_server()
        while not self.register(): time.sleep(10)
        print("Registered successfully")
        while self.running: time.sleep(10)

if __name__ == '__main__': Worker().run()
WORKER_EOF
    fi
    
    # Build Docker image
    cd "$build_dir"
    docker build -t "$HASHMANCER_DOCKER_IMAGE" .
    
    # Clean up build directory
    cd /
    rm -rf "$build_dir"
    
    log "✅ Worker Docker image built successfully"
}

# Main execution
main() {
    log "🔓 Hashmancer Vast.ai Worker Setup Starting..."
    
    # Install required packages
    log "📦 Installing required packages..."
    apt-get update -qq
    apt-get install -y curl wget jq
    
    # Install Docker
    install_docker
    
    # Install NVIDIA Docker if GPU is available
    if nvidia-smi &> /dev/null; then
        install_nvidia_docker
    fi
    
    # Create worker Docker image
    create_worker_image
    
    # Detect server IP
    SERVER_IP=$(detect_server_ip)
    if [ $? -ne 0 ]; then
        error "Failed to detect server IP. Please set HASHMANCER_SERVER_IP environment variable."
        exit 1
    fi
    
    # Start worker
    start_worker "$SERVER_IP"
    
    # Setup monitoring
    setup_monitoring
    
    log "🎉 Hashmancer worker setup completed successfully!"
    log "📊 Worker ID: $(docker exec hashmancer-worker cat /app/config.json | jq -r .worker_id)"
    log "🌐 Server: $SERVER_IP:$HASHMANCER_SERVER_PORT"
    log "📋 Monitor logs: docker logs -f hashmancer-worker"
    
    # Keep script running to show logs
    log "📋 Following worker logs..."
    docker logs -f hashmancer-worker
}

# Run main function
main "$@"