#!/bin/bash

# Hashmancer Seamless Server Installation Script
# This script provides a one-command installation experience

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
HASHMANCER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$HASHMANCER_DIR/.hashmancer_config"
TEMP_CONFIG="/tmp/hashmancer_install_config"

# Default ports
DEFAULT_HTTP_PORT=8000
DEFAULT_HTTPS_PORT=8443
DEFAULT_REDIS_PORT=6379

# Installation modes
INSTALL_MODE=""
USE_DOCKER=false
ENABLE_GPU=false
SETUP_DOMAIN=false

# Helper functions
print_header() {
    clear
    echo -e "${PURPLE}"
    cat << 'EOF'
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║    ██╗  ██╗ █████╗ ███████╗██╗  ██╗███╗   ███╗ █████╗       ║
    ║    ██║  ██║██╔══██╗██╔════╝██║  ██║████╗ ████║██╔══██╗      ║
    ║    ███████║███████║███████╗███████║██╔████╔██║███████║      ║
    ║    ██╔══██║██╔══██║╚════██║██╔══██║██║╚██╔╝██║██╔══██║      ║
    ║    ██║  ██║██║  ██║███████║██║  ██║██║ ╚═╝ ██║██║  ██║      ║
    ║    ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝      ║
    ║                                                              ║
    ║               SEAMLESS SERVER INSTALLATION                   ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_info() {
    echo -e "${CYAN}[INFO]${NC} $1"
}

print_question() {
    echo -e "${PURPLE}[QUESTION]${NC} $1"
}

# Check system requirements
check_system() {
    print_step "Checking system requirements..."
    
    # Check OS
    if [[ ! -f /etc/os-release ]]; then
        print_error "Cannot determine OS. This script supports Ubuntu/Debian systems."
        exit 1
    fi
    
    source /etc/os-release
    if [[ "$ID" != "ubuntu" && "$ID" != "debian" ]]; then
        print_warning "This script is optimized for Ubuntu/Debian. Other systems may work but are not officially supported."
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 0
        fi
    fi
    
    # Check if running as root
    if [[ $EUID -eq 0 ]]; then
        print_warning "Running as root. This is not recommended for security reasons."
        read -p "Continue as root? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_info "Please run this script as a regular user with sudo privileges."
            exit 0
        fi
    fi
    
    # Check for sudo privileges
    if ! sudo -n true 2>/dev/null; then
        print_info "This script requires sudo privileges for system configuration."
        sudo -v
    fi
    
    # Check available disk space (need at least 5GB)
    available_space=$(df "$HASHMANCER_DIR" | tail -1 | awk '{print $4}')
    required_space=$((5 * 1024 * 1024)) # 5GB in KB
    
    if [[ $available_space -lt $required_space ]]; then
        print_error "Insufficient disk space. Need at least 5GB available."
        exit 1
    fi
    
    # Detect GPU
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        print_success "NVIDIA GPU detected"
        ENABLE_GPU=true
    elif command -v rocm-smi >/dev/null 2>&1; then
        print_success "AMD GPU detected"
        ENABLE_GPU=true
    else
        print_info "No GPU detected or GPU drivers not installed"
    fi
    
    print_success "System requirements check completed"
}

# Installation mode selection
select_installation_mode() {
    print_header
    print_question "Select installation mode:"
    echo ""
    echo "1) 🐳 Docker Installation (Recommended)"
    echo "   • Easy deployment and updates"
    echo "   • Isolated environment"
    echo "   • GPU passthrough support"
    echo ""
    echo "2) 🖥️  Native Installation"
    echo "   • Direct system installation"
    echo "   • Better performance"
    echo "   • More configuration options"
    echo ""
    echo "3) ☁️  Cloud/VPS Installation"
    echo "   • Optimized for cloud deployment"
    echo "   • Automatic SSL/TLS setup"
    echo "   • Domain configuration"
    echo ""
    
    while true; do
        read -p "Choose installation mode (1/2/3): " choice
        case $choice in
            1)
                INSTALL_MODE="docker"
                USE_DOCKER=true
                print_success "Docker installation selected"
                break
                ;;
            2)
                INSTALL_MODE="native"
                print_success "Native installation selected"
                break
                ;;
            3)
                INSTALL_MODE="cloud"
                SETUP_DOMAIN=true
                print_success "Cloud installation selected"
                break
                ;;
            *)
                print_error "Invalid choice. Please select 1, 2, or 3."
                ;;
        esac
    done
}

# Collect configuration
collect_configuration() {
    print_header
    print_step "Configuration Setup"
    echo ""
    
    # Create temporary config file
    cat > "$TEMP_CONFIG" << EOF
# Hashmancer Installation Configuration
INSTALL_MODE="$INSTALL_MODE"
USE_DOCKER=$USE_DOCKER
ENABLE_GPU=$ENABLE_GPU
SETUP_DOMAIN=$SETUP_DOMAIN
EOF
    
    # API Keys
    print_question "API Configuration (Optional - can be configured later):"
    echo ""
    
    read -p "OpenAI API Key (optional): " openai_key
    if [[ -n "$openai_key" ]]; then
        echo "OPENAI_API_KEY=\"$openai_key\"" >> "$TEMP_CONFIG"
    fi
    
    read -p "Anthropic API Key (optional): " anthropic_key
    if [[ -n "$anthropic_key" ]]; then
        echo "ANTHROPIC_API_KEY=\"$anthropic_key\"" >> "$TEMP_CONFIG"
    fi
    
    # Server Configuration
    echo ""
    print_question "Server Configuration:"
    
    read -p "HTTP Port [$DEFAULT_HTTP_PORT]: " http_port
    http_port=${http_port:-$DEFAULT_HTTP_PORT}
    echo "HTTP_PORT=\"$http_port\"" >> "$TEMP_CONFIG"
    
    if [[ "$SETUP_DOMAIN" == "true" ]]; then
        read -p "HTTPS Port [$DEFAULT_HTTPS_PORT]: " https_port
        https_port=${https_port:-$DEFAULT_HTTPS_PORT}
        echo "HTTPS_PORT=\"$https_port\"" >> "$TEMP_CONFIG"
        
        read -p "Domain name (e.g., hashmancer.example.com): " domain_name
        if [[ -n "$domain_name" ]]; then
            echo "DOMAIN_NAME=\"$domain_name\"" >> "$TEMP_CONFIG"
        fi
        
        read -p "Email for SSL certificate: " ssl_email
        if [[ -n "$ssl_email" ]]; then
            echo "SSL_EMAIL=\"$ssl_email\"" >> "$TEMP_CONFIG"
        fi
    fi
    
    # Redis Configuration
    read -p "Redis Port [$DEFAULT_REDIS_PORT]: " redis_port
    redis_port=${redis_port:-$DEFAULT_REDIS_PORT}
    echo "REDIS_PORT=\"$redis_port\"" >> "$TEMP_CONFIG"
    
    # Admin Configuration
    echo ""
    print_question "Admin Account Setup:"
    
    while true; do
        read -p "Admin username: " admin_username
        if [[ -n "$admin_username" ]]; then
            echo "ADMIN_USERNAME=\"$admin_username\"" >> "$TEMP_CONFIG"
            break
        else
            print_error "Admin username cannot be empty"
        fi
    done
    
    while true; do
        read -s -p "Admin password: " admin_password
        echo
        read -s -p "Confirm admin password: " admin_password_confirm
        echo
        if [[ "$admin_password" == "$admin_password_confirm" && -n "$admin_password" ]]; then
            echo "ADMIN_PASSWORD=\"$admin_password\"" >> "$TEMP_CONFIG"
            break
        else
            print_error "Passwords don't match or are empty. Please try again."
        fi
    done
    
    # GPU Configuration
    if [[ "$ENABLE_GPU" == "true" ]]; then
        echo ""
        print_question "GPU Configuration:"
        
        read -p "Enable GPU acceleration? (Y/n): " enable_gpu_confirm
        enable_gpu_confirm=${enable_gpu_confirm:-Y}
        
        if [[ $enable_gpu_confirm =~ ^[Yy]$ ]]; then
            echo "GPU_ENABLED=true" >> "$TEMP_CONFIG"
            
            if command -v nvidia-smi >/dev/null 2>&1; then
                echo "GPU_TYPE=\"nvidia\"" >> "$TEMP_CONFIG"
            elif command -v rocm-smi >/dev/null 2>&1; then
                echo "GPU_TYPE=\"amd\"" >> "$TEMP_CONFIG"
            fi
        else
            echo "GPU_ENABLED=false" >> "$TEMP_CONFIG"
            ENABLE_GPU=false
        fi
    fi
    
    # Network Configuration
    echo ""
    print_question "Network Configuration:"
    
    read -p "Allow external connections? (Y/n): " allow_external
    allow_external=${allow_external:-Y}
    
    if [[ $allow_external =~ ^[Yy]$ ]]; then
        echo "ALLOW_EXTERNAL=true" >> "$TEMP_CONFIG"
    else
        echo "ALLOW_EXTERNAL=false" >> "$TEMP_CONFIG"
    fi
    
    print_success "Configuration collected"
}

# Install system dependencies
install_system_dependencies() {
    print_step "Installing system dependencies..."
    
    # Update package list
    sudo apt-get update -qq
    
    # Base packages
    local base_packages=(
        "curl"
        "wget"
        "git"
        "unzip"
        "software-properties-common"
        "apt-transport-https"
        "ca-certificates"
        "gnupg"
        "lsb-release"
        "htop"
        "nano"
        "vim"
        "systemd"
        "logrotate"
        "ufw"
    )
    
    # Install base packages
    for package in "${base_packages[@]}"; do
        if ! dpkg -l | grep -q "^ii  $package "; then
            print_info "Installing $package..."
            sudo apt-get install -y "$package"
        fi
    done
    
    if [[ "$USE_DOCKER" == "true" ]]; then
        install_docker
    else
        install_native_dependencies
    fi
    
    print_success "System dependencies installed"
}

# Install Docker
install_docker() {
    print_step "Installing Docker..."
    
    # Remove old Docker installations
    sudo apt-get remove -y docker docker-engine docker.io containerd runc 2>/dev/null || true
    
    # Add Docker repository
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    
    # Install Docker
    sudo apt-get update -qq
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
    
    # Add user to docker group
    sudo usermod -aG docker "$USER"
    
    # Install Docker Compose
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    
    # Enable and start Docker
    sudo systemctl enable docker
    sudo systemctl start docker
    
    # Install NVIDIA Container Toolkit if GPU is enabled
    if [[ "$ENABLE_GPU" == "true" && -x "$(command -v nvidia-smi)" ]]; then
        install_nvidia_docker
    fi
    
    print_success "Docker installed"
}

# Install NVIDIA Docker support
install_nvidia_docker() {
    print_step "Installing NVIDIA Container Toolkit..."
    
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    
    sudo apt-get update -qq
    sudo apt-get install -y nvidia-docker2
    
    sudo systemctl restart docker
    
    print_success "NVIDIA Container Toolkit installed"
}

# Install native dependencies
install_native_dependencies() {
    print_step "Installing native dependencies..."
    
    # Python
    sudo apt-get install -y python3 python3-pip python3-venv python3-dev
    
    # Node.js
    curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
    sudo apt-get install -y nodejs
    
    # Redis
    sudo apt-get install -y redis-server
    sudo systemctl enable redis-server
    sudo systemctl start redis-server
    
    # Build tools
    sudo apt-get install -y build-essential cmake pkg-config libssl-dev
    
    # NGINX (if cloud installation)
    if [[ "$SETUP_DOMAIN" == "true" ]]; then
        sudo apt-get install -y nginx certbot python3-certbot-nginx
        sudo systemctl enable nginx
    fi
    
    print_success "Native dependencies installed"
}

# Configure firewall
configure_firewall() {
    print_step "Configuring firewall..."
    
    # Load config
    source "$TEMP_CONFIG"
    
    # Reset UFW
    sudo ufw --force reset
    
    # Default policies
    sudo ufw default deny incoming
    sudo ufw default allow outgoing
    
    # Allow SSH
    sudo ufw allow ssh
    
    # Allow HTTP/HTTPS
    if [[ "$ALLOW_EXTERNAL" == "true" ]]; then
        sudo ufw allow "$HTTP_PORT"/tcp comment 'Hashmancer HTTP'
        
        if [[ "$SETUP_DOMAIN" == "true" ]]; then
            sudo ufw allow "$HTTPS_PORT"/tcp comment 'Hashmancer HTTPS'
            sudo ufw allow 80/tcp comment 'HTTP for Let\'s Encrypt'
        fi
    else
        # Only allow local connections
        sudo ufw allow from 127.0.0.1 to any port "$HTTP_PORT"
        sudo ufw allow from ::1 to any port "$HTTP_PORT"
    fi
    
    # Allow Redis (only locally)
    sudo ufw allow from 127.0.0.1 to any port "$REDIS_PORT"
    
    # Enable firewall
    sudo ufw --force enable
    
    print_success "Firewall configured"
}

# Setup SSL/TLS with Let's Encrypt
setup_ssl() {
    if [[ "$SETUP_DOMAIN" != "true" ]]; then
        return
    fi
    
    print_step "Setting up SSL/TLS certificates..."
    
    source "$TEMP_CONFIG"
    
    if [[ -z "${DOMAIN_NAME:-}" || -z "${SSL_EMAIL:-}" ]]; then
        print_warning "Domain name or email not provided, skipping SSL setup"
        return
    fi
    
    # Configure NGINX
    sudo tee "/etc/nginx/sites-available/hashmancer" > /dev/null << EOF
server {
    listen 80;
    server_name $DOMAIN_NAME;
    
    location /.well-known/acme-challenge/ {
        root /var/www/html;
    }
    
    location / {
        return 301 https://\$server_name\$request_uri;
    }
}

server {
    listen 443 ssl http2;
    server_name $DOMAIN_NAME;
    
    client_max_body_size 100M;
    
    location / {
        proxy_pass http://127.0.0.1:$HTTP_PORT;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_set_header X-Forwarded-Host \$host;
        proxy_set_header X-Forwarded-Port \$server_port;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}
EOF
    
    # Enable site
    sudo ln -sf /etc/nginx/sites-available/hashmancer /etc/nginx/sites-enabled/
    sudo rm -f /etc/nginx/sites-enabled/default
    
    # Test NGINX config
    sudo nginx -t
    
    # Restart NGINX
    sudo systemctl restart nginx
    
    # Get SSL certificate
    print_info "Obtaining SSL certificate for $DOMAIN_NAME..."
    sudo certbot --nginx -d "$DOMAIN_NAME" --email "$SSL_EMAIL" --agree-tos --non-interactive
    
    # Setup auto-renewal
    sudo systemctl enable certbot.timer
    
    print_success "SSL/TLS certificates configured"
}

# Create Docker configuration
create_docker_config() {
    if [[ "$USE_DOCKER" != "true" ]]; then
        return
    fi
    
    print_step "Creating Docker configuration..."
    
    source "$TEMP_CONFIG"
    
    # Create enhanced Dockerfile
    cat > "$HASHMANCER_DIR/Dockerfile.server" << EOF
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    gcc \\
    libffi-dev \\
    libssl-dev \\
    git \\
    curl \\
    redis-tools \\
    htop \\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY hashmancer/server/requirements*.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . /app

# Create non-root user
RUN useradd -m -u 1000 hashmancer && \\
    chown -R hashmancer:hashmancer /app
USER hashmancer

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

WORKDIR /app/hashmancer/server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

    # Create GPU-enabled Dockerfile if needed
    if [[ "$ENABLE_GPU" == "true" ]]; then
        cat > "$HASHMANCER_DIR/Dockerfile.gpu" << EOF
FROM nvidia/cuda:11.8-devel-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y --no-install-recommends \\
    python3 \\
    python3-pip \\
    python3-dev \\
    build-essential \\
    gcc \\
    libffi-dev \\
    libssl-dev \\
    git \\
    curl \\
    redis-tools \\
    htop \\
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Copy requirements and install Python dependencies
COPY hashmancer/server/requirements*.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Copy application
COPY . /app

# Create non-root user
RUN useradd -m -u 1000 hashmancer && \\
    chown -R hashmancer:hashmancer /app
USER hashmancer

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

WORKDIR /app/hashmancer/server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF
    fi
    
    # Create enhanced docker-compose.yml
    cat > "$HASHMANCER_DIR/docker-compose.production.yml" << EOF
version: "3.8"

services:
  redis:
    image: redis:7-alpine
    restart: unless-stopped
    ports:
      - "127.0.0.1:$REDIS_PORT:6379"
    volumes:
      - redis-data:/data
      - ./redis.conf:/usr/local/etc/redis/redis.conf:ro
    command: redis-server /usr/local/etc/redis/redis.conf
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  hashmancer:
    build:
      context: .
      dockerfile: $([ "$ENABLE_GPU" == "true" ] && echo "Dockerfile.gpu" || echo "Dockerfile.server")
    restart: unless-stopped
    ports:
      - "$HTTP_PORT:8000"
    depends_on:
      redis:
        condition: service_healthy
    environment:
      - REDIS_URL=redis://redis:6379
      - ADMIN_USERNAME=$ADMIN_USERNAME
      - ADMIN_PASSWORD=$ADMIN_PASSWORD$([ -n "${OPENAI_API_KEY:-}" ] && echo "
      - OPENAI_API_KEY=$OPENAI_API_KEY")$([ -n "${ANTHROPIC_API_KEY:-}" ] && echo "
      - ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY")$([ "$ENABLE_GPU" == "true" ] && echo "
      - CUDA_VISIBLE_DEVICES=all")
    volumes:
      - hashmancer-data:/app/data
      - hashmancer-logs:/app/logs
      - hashmancer-config:/app/config$([ "$ENABLE_GPU" == "true" ] && echo "
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]")
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

volumes:
  redis-data:
    driver: local
  hashmancer-data:
    driver: local
  hashmancer-logs:
    driver: local
  hashmancer-config:
    driver: local

networks:
  default:
    driver: bridge
EOF

    # Create Redis configuration
    cat > "$HASHMANCER_DIR/redis.conf" << EOF
# Redis configuration for Hashmancer
bind 127.0.0.1
port 6379
timeout 0
tcp-keepalive 300
save 900 1
save 300 10
save 60 10000
stop-writes-on-bgsave-error yes
rdbcompression yes
rdbchecksum yes
dbfilename dump.rdb
dir /data
maxmemory 256mb
maxmemory-policy allkeys-lru
appendonly yes
appendfilename "appendonly.aof"
appendfsync everysec
EOF
    
    print_success "Docker configuration created"
}

# Install and configure Hashmancer
install_hashmancer() {
    print_step "Installing Hashmancer..."
    
    source "$TEMP_CONFIG"
    
    if [[ "$USE_DOCKER" == "true" ]]; then
        install_hashmancer_docker
    else
        install_hashmancer_native
    fi
    
    print_success "Hashmancer installed"
}

# Install Hashmancer with Docker
install_hashmancer_docker() {
    print_step "Installing Hashmancer with Docker..."
    
    # Build and start services
    cd "$HASHMANCER_DIR"
    
    print_info "Building Docker images..."
    docker-compose -f docker-compose.production.yml build
    
    print_info "Starting services..."
    docker-compose -f docker-compose.production.yml up -d
    
    # Wait for services to be ready
    print_info "Waiting for services to start..."
    sleep 30
    
    # Check if services are healthy
    if docker-compose -f docker-compose.production.yml ps | grep -q "healthy"; then
        print_success "Docker services started successfully"
    else
        print_warning "Services may still be starting. Check logs if needed: docker-compose -f docker-compose.production.yml logs"
    fi
}

# Install Hashmancer natively
install_hashmancer_native() {
    print_step "Installing Hashmancer natively..."
    
    source "$TEMP_CONFIG"
    
    # Create Python virtual environment
    python3 -m venv "$HASHMANCER_DIR/venv"
    source "$HASHMANCER_DIR/venv/bin/activate"
    
    # Install Python dependencies
    pip install --upgrade pip
    pip install -r "$HASHMANCER_DIR/hashmancer/server/requirements.txt"
    
    # Create systemd service
    sudo tee "/etc/systemd/system/hashmancer.service" > /dev/null << EOF
[Unit]
Description=Hashmancer Server
After=network.target redis.service
Wants=redis.service

[Service]
Type=simple
User=$USER
Group=$USER
WorkingDirectory=$HASHMANCER_DIR/hashmancer/server
Environment=PATH=$HASHMANCER_DIR/venv/bin
Environment=PYTHONPATH=$HASHMANCER_DIR
Environment=REDIS_URL=redis://localhost:$REDIS_PORT
Environment=ADMIN_USERNAME=$ADMIN_USERNAME
Environment=ADMIN_PASSWORD=$ADMIN_PASSWORD$([ -n "${OPENAI_API_KEY:-}" ] && echo "
Environment=OPENAI_API_KEY=$OPENAI_API_KEY")$([ -n "${ANTHROPIC_API_KEY:-}" ] && echo "
Environment=ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY")
ExecStart=$HASHMANCER_DIR/venv/bin/uvicorn main:app --host 0.0.0.0 --port $HTTP_PORT
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
    
    # Enable and start service
    sudo systemctl daemon-reload
    sudo systemctl enable hashmancer
    sudo systemctl start hashmancer
}

# Create management scripts
create_management_scripts() {
    print_step "Creating management scripts..."
    
    mkdir -p "$HASHMANCER_DIR/scripts"
    
    if [[ "$USE_DOCKER" == "true" ]]; then
        create_docker_management_scripts
    else
        create_native_management_scripts
    fi
    
    print_success "Management scripts created"
}

# Create Docker management scripts
create_docker_management_scripts() {
    # Start script
    cat > "$HASHMANCER_DIR/scripts/start.sh" << 'EOF'
#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")/.."
docker-compose -f docker-compose.production.yml up -d
echo "Hashmancer started. Check status with: ./scripts/status.sh"
EOF
    
    # Stop script
    cat > "$HASHMANCER_DIR/scripts/stop.sh" << 'EOF'
#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")/.."
docker-compose -f docker-compose.production.yml down
echo "Hashmancer stopped"
EOF
    
    # Restart script
    cat > "$HASHMANCER_DIR/scripts/restart.sh" << 'EOF'
#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")/.."
docker-compose -f docker-compose.production.yml restart
echo "Hashmancer restarted"
EOF
    
    # Status script
    cat > "$HASHMANCER_DIR/scripts/status.sh" << 'EOF'
#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")/.."
echo "=== Docker Services Status ==="
docker-compose -f docker-compose.production.yml ps
echo ""
echo "=== Portal Accessibility ==="
if curl -f -s --max-time 5 http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Portal is accessible at http://localhost:8000"
else
    echo "❌ Portal is not accessible"
fi
echo ""
echo "=== Resource Usage ==="
docker stats --no-stream
EOF
    
    # Logs script
    cat > "$HASHMANCER_DIR/scripts/logs.sh" << 'EOF'
#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")/.."
docker-compose -f docker-compose.production.yml logs -f "${1:-hashmancer}"
EOF
    
    # Update script
    cat > "$HASHMANCER_DIR/scripts/update.sh" << 'EOF'
#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")/.."
echo "Pulling latest changes..."
git pull
echo "Rebuilding and restarting services..."
docker-compose -f docker-compose.production.yml down
docker-compose -f docker-compose.production.yml build --no-cache
docker-compose -f docker-compose.production.yml up -d
echo "Update complete"
EOF
    
    chmod +x "$HASHMANCER_DIR/scripts"/*.sh
}

# Create native management scripts
create_native_management_scripts() {
    # Start script
    cat > "$HASHMANCER_DIR/scripts/start.sh" << 'EOF'
#!/bin/bash
sudo systemctl start hashmancer
echo "Hashmancer started. Check status with: ./scripts/status.sh"
EOF
    
    # Stop script
    cat > "$HASHMANCER_DIR/scripts/stop.sh" << 'EOF'
#!/bin/bash
sudo systemctl stop hashmancer
echo "Hashmancer stopped"
EOF
    
    # Restart script
    cat > "$HASHMANCER_DIR/scripts/restart.sh" << 'EOF'
#!/bin/bash
sudo systemctl restart hashmancer
echo "Hashmancer restarted"
EOF
    
    # Status script
    cat > "$HASHMANCER_DIR/scripts/status.sh" << 'EOF'
#!/bin/bash
echo "=== Service Status ==="
sudo systemctl status hashmancer --no-pager
echo ""
echo "=== Portal Accessibility ==="
if curl -f -s --max-time 5 http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Portal is accessible at http://localhost:8000"
else
    echo "❌ Portal is not accessible"
fi
echo ""
echo "=== Recent Logs ==="
sudo journalctl -u hashmancer --no-pager -n 10
EOF
    
    # Logs script
    cat > "$HASHMANCER_DIR/scripts/logs.sh" << 'EOF'
#!/bin/bash
sudo journalctl -u hashmancer -f
EOF
    
    # Update script
    cat > "$HASHMANCER_DIR/scripts/update.sh" << 'EOF'
#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")/.."
echo "Pulling latest changes..."
git pull
echo "Installing dependencies..."
source venv/bin/activate
pip install -r hashmancer/server/requirements.txt
echo "Restarting service..."
sudo systemctl restart hashmancer
echo "Update complete"
EOF
    
    chmod +x "$HASHMANCER_DIR/scripts"/*.sh
}

# Setup health monitoring
setup_monitoring() {
    print_step "Setting up health monitoring..."
    
    if [[ "$USE_DOCKER" == "true" ]]; then
        # Docker health checks are built into the compose file
        print_info "Docker health checks configured"
    else
        # Create health check script for native installation
        cat > "$HASHMANCER_DIR/scripts/health-check.sh" << 'EOF'
#!/bin/bash
HASHMANCER_URL="http://localhost:8000"
MAX_FAILURES=3
FAILURE_COUNT_FILE="/tmp/hashmancer_failures"

get_failure_count() {
    if [[ -f "$FAILURE_COUNT_FILE" ]]; then
        cat "$FAILURE_COUNT_FILE"
    else
        echo 0
    fi
}

set_failure_count() {
    echo "$1" > "$FAILURE_COUNT_FILE"
}

reset_failure_count() {
    rm -f "$FAILURE_COUNT_FILE"
}

# Check if service is running
if ! systemctl is-active --quiet hashmancer; then
    echo "Service is not running, attempting to start..."
    systemctl start hashmancer
    exit 0
fi

# Check if portal is responding
if curl -f -s --max-time 10 "$HASHMANCER_URL/health" > /dev/null 2>&1; then
    reset_failure_count
    exit 0
else
    current_failures=$(get_failure_count)
    new_failures=$((current_failures + 1))
    set_failure_count "$new_failures"
    
    if [[ $new_failures -ge $MAX_FAILURES ]]; then
        echo "Max failures reached, restarting service..."
        systemctl restart hashmancer
        reset_failure_count
    fi
    
    exit 1
fi
EOF
        
        chmod +x "$HASHMANCER_DIR/scripts/health-check.sh"
        
        # Add cron job
        (crontab -l 2>/dev/null; echo "*/2 * * * * $HASHMANCER_DIR/scripts/health-check.sh") | crontab -
    fi
    
    print_success "Health monitoring configured"
}

# Validate installation
validate_installation() {
    print_step "Validating installation..."
    
    source "$TEMP_CONFIG"
    
    # Wait for services to be ready
    local max_attempts=30
    local attempt=1
    
    print_info "Waiting for Hashmancer to start..."
    while [[ $attempt -le $max_attempts ]]; do
        if curl -f -s --max-time 5 "http://localhost:$HTTP_PORT/health" > /dev/null 2>&1; then
            print_success "Hashmancer is running and accessible"
            break
        fi
        
        print_info "Attempt $attempt/$max_attempts - waiting..."
        sleep 5
        ((attempt++))
    done
    
    if [[ $attempt -gt $max_attempts ]]; then
        print_error "Hashmancer failed to start within timeout period"
        print_info "Check logs for more information:"
        if [[ "$USE_DOCKER" == "true" ]]; then
            print_info "  docker-compose -f docker-compose.production.yml logs"
        else
            print_info "  sudo journalctl -u hashmancer"
        fi
        return 1
    fi
    
    # Test admin access
    print_info "Testing admin access..."
    local admin_response
    admin_response=$(curl -s -w "%{http_code}" "http://localhost:$HTTP_PORT/admin" -o /dev/null)
    
    if [[ "$admin_response" == "200" || "$admin_response" == "302" ]]; then
        print_success "Admin panel is accessible"
    else
        print_warning "Admin panel may not be fully configured"
    fi
    
    # Test Redis connection
    print_info "Testing Redis connection..."
    if [[ "$USE_DOCKER" == "true" ]]; then
        if docker-compose -f docker-compose.production.yml exec -T redis redis-cli ping > /dev/null 2>&1; then
            print_success "Redis is accessible"
        else
            print_warning "Redis connection test failed"
        fi
    else
        if redis-cli -p "$REDIS_PORT" ping > /dev/null 2>&1; then
            print_success "Redis is accessible"
        else
            print_warning "Redis connection test failed"
        fi
    fi
    
    print_success "Installation validation completed"
}

# Save configuration
save_configuration() {
    print_step "Saving configuration..."
    
    # Move temp config to permanent location
    mv "$TEMP_CONFIG" "$CONFIG_FILE"
    chmod 600 "$CONFIG_FILE"
    
    # Create environment file for easy sourcing
    cat > "$HASHMANCER_DIR/.env" << EOF
# Hashmancer Environment Variables
# Source this file to load configuration: source .env

$(cat "$CONFIG_FILE" | grep -E '^[A-Z_]+=' | sed 's/^/export /')
EOF
    
    chmod 600 "$HASHMANCER_DIR/.env"
    
    print_success "Configuration saved"
}

# Create desktop integration
create_desktop_integration() {
    if [[ -z "${DISPLAY:-}" ]]; then
        return
    fi
    
    print_step "Creating desktop integration..."
    
    source "$CONFIG_FILE"
    
    # Create desktop file
    local desktop_dir="$HOME/.local/share/applications"
    mkdir -p "$desktop_dir"
    
    cat > "$desktop_dir/hashmancer.desktop" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Hashmancer
Comment=Hash Cracking and Analysis Platform
Icon=security-high
Exec=xdg-open http://localhost:$HTTP_PORT
Terminal=false
Categories=Network;Security;Development;
Keywords=hash;cracking;security;analysis;
EOF
    
    # Update desktop database
    if command -v update-desktop-database >/dev/null 2>&1; then
        update-desktop-database "$desktop_dir"
    fi
    
    print_success "Desktop integration created"
}

# Print final instructions
print_final_instructions() {
    clear
    print_header
    print_success "🎉 HASHMANCER INSTALLATION COMPLETE! 🎉"
    echo ""
    
    source "$CONFIG_FILE"
    
    # Access information
    print_info "🌐 ACCESS INFORMATION:"
    echo "   Portal URL: http://localhost:$HTTP_PORT"
    echo "   Admin Panel: http://localhost:$HTTP_PORT/admin"
    
    if [[ "$SETUP_DOMAIN" == "true" && -n "${DOMAIN_NAME:-}" ]]; then
        echo "   Public URL: https://$DOMAIN_NAME"
    fi
    
    echo "   Admin Username: $ADMIN_USERNAME"
    echo "   Admin Password: [configured during setup]"
    echo ""
    
    # Management commands
    print_info "🛠️  MANAGEMENT COMMANDS:"
    echo "   Start:    ./scripts/start.sh"
    echo "   Stop:     ./scripts/stop.sh"
    echo "   Restart:  ./scripts/restart.sh"
    echo "   Status:   ./scripts/status.sh"
    echo "   Logs:     ./scripts/logs.sh"
    echo "   Update:   ./scripts/update.sh"
    echo ""
    
    # Additional information
    print_info "📋 ADDITIONAL INFORMATION:"
    echo "   Configuration: $CONFIG_FILE"
    echo "   Environment:   .env"
    echo "   Installation:  $([ "$USE_DOCKER" == "true" ] && echo "Docker" || echo "Native")"
    
    if [[ "$ENABLE_GPU" == "true" ]]; then
        echo "   GPU Support:   Enabled"
    fi
    
    if [[ "$SETUP_DOMAIN" == "true" ]]; then
        echo "   SSL/TLS:       Configured"
    fi
    echo ""
    
    # Next steps
    print_info "🚀 NEXT STEPS:"
    echo "   1. Open your browser and navigate to the portal URL"
    echo "   2. Log in with your admin credentials"
    echo "   3. Configure API keys in the admin panel (if not done during setup)"
    echo "   4. Start exploring Hashmancer's features!"
    echo ""
    
    # Important notes
    print_warning "⚠️  IMPORTANT NOTES:"
    echo "   • Keep your admin credentials secure"
    echo "   • Regular backups are recommended"
    echo "   • Monitor resource usage for optimal performance"
    
    if [[ "$ALLOW_EXTERNAL" == "true" ]]; then
        echo "   • External access is enabled - ensure proper security measures"
    fi
    
    if [[ "$USE_DOCKER" != "true" ]]; then
        echo "   • Native installation requires manual dependency management"
    fi
    
    echo ""
    print_info "📚 For documentation and support, visit: https://github.com/Infernal-Insights/Hashmancer"
    echo ""
}

# Cleanup on exit
cleanup() {
    if [[ -f "$TEMP_CONFIG" ]]; then
        rm -f "$TEMP_CONFIG"
    fi
}

# Error handler
error_handler() {
    local exit_code=$?
    print_error "Installation failed with exit code $exit_code"
    print_info "Check the logs above for more information"
    cleanup
    exit $exit_code
}

# Main installation function
main() {
    # Set up error handling
    trap error_handler ERR
    trap cleanup EXIT
    
    # Welcome screen
    print_header
    print_info "Welcome to the Hashmancer Seamless Installation!"
    print_info "This script will automatically install and configure everything you need."
    echo ""
    
    read -p "Press Enter to continue or Ctrl+C to cancel..."
    
    # Installation steps
    check_system
    select_installation_mode
    collect_configuration
    install_system_dependencies
    configure_firewall
    setup_ssl
    create_docker_config
    install_hashmancer
    create_management_scripts
    setup_monitoring
    save_configuration
    validate_installation
    create_desktop_integration
    
    # Show final instructions
    print_final_instructions
}

# Check if script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi