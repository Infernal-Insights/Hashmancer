#!/bin/bash
set -e

# Hashmancer Ultimate Deployment Script
# This script provides the most seamless Docker deployment experience

echo "🚀 Hashmancer Ultimate Deployment Script"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if running as root (needed for some GPU operations)
check_permissions() {
    if [[ $EUID -eq 0 ]]; then
        log_warning "Running as root - this is okay for initial setup"
    else
        log_info "Running as non-root user"
    fi
}

# Check system requirements
check_requirements() {
    log_info "Checking system requirements..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    log_success "Docker is installed"
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    log_success "Docker Compose is available"
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running. Please start Docker first."
        exit 1
    fi
    log_success "Docker daemon is running"
    
    # Check for NVIDIA Docker support
    if command -v nvidia-smi &> /dev/null; then
        log_success "NVIDIA GPU detected"
        
        # Check NVIDIA Docker runtime
        if docker info 2>/dev/null | grep -q nvidia; then
            log_success "NVIDIA Docker runtime is available"
            export GPU_AVAILABLE=true
        else
            log_warning "NVIDIA Docker runtime not detected. GPU workers will use CPU fallback."
            export GPU_AVAILABLE=false
        fi
    else
        log_info "No NVIDIA GPU detected. Using CPU-only configuration."
        export GPU_AVAILABLE=false
    fi
}

# Setup NVIDIA Docker if available
setup_nvidia_docker() {
    if [[ "$GPU_AVAILABLE" == "true" ]]; then
        log_info "Setting up NVIDIA Docker support..."
        
        # Check if nvidia-container-toolkit is installed
        if ! command -v nvidia-container-runtime &> /dev/null; then
            log_warning "NVIDIA Container Toolkit not found. Attempting to install..."
            
            # Install nvidia-container-toolkit
            if command -v apt-get &> /dev/null; then
                curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
                curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
                    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
                    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
                sudo apt-get update
                sudo apt-get install -y nvidia-container-toolkit
                sudo nvidia-ctk runtime configure --runtime=docker
                sudo systemctl restart docker
                log_success "NVIDIA Container Toolkit installed"
            else
                log_warning "Could not install NVIDIA Container Toolkit automatically. Please install manually."
            fi
        else
            log_success "NVIDIA Container Toolkit is available"
        fi
    fi
}

# Create necessary directories and files
setup_directories() {
    log_info "Setting up directories and configuration..."
    
    # Create directories
    mkdir -p logs config ssl wordlists data
    
    # Create basic Redis config if it doesn't exist
    if [[ ! -f config/redis.conf ]]; then
        cat > config/redis.conf << 'EOF'
# Redis configuration for Hashmancer
bind 0.0.0.0
port 6379
timeout 0
tcp-keepalive 300
daemonize no
supervised no
pidfile /var/run/redis_6379.pid
loglevel notice
logfile ""
databases 16
save 900 1
save 300 10
save 60 10000
stop-writes-on-bgsave-error yes
rdbcompression yes
rdbchecksum yes
dbfilename dump.rdb
dir ./
maxmemory 2gb
maxmemory-policy allkeys-lru
appendonly yes
appendfilename "appendonly.aof"
appendfsync everysec
no-appendfsync-on-rewrite no
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb
aof-load-truncated yes
lua-time-limit 5000
slowlog-log-slower-than 10000
slowlog-max-len 128
EOF
        log_success "Created Redis configuration"
    fi
    
    # Create environment file
    if [[ ! -f .env ]]; then
        cat > .env << 'EOF'
# Hashmancer Environment Configuration

# Redis Configuration
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_MAX_CONNECTIONS=50
REDIS_CONNECTION_TIMEOUT=10

# Server Configuration
HASHMANCER_HOST=0.0.0.0
HASHMANCER_PORT=8080
API_PORT=8000
LOG_LEVEL=INFO

# Worker Configuration
DEFAULT_ENGINE=darkling
WORKER_LOG_LEVEL=INFO

# Security
SECURE_SESSION=true
SESSION_TIMEOUT=3600

# Performance
WORKERS=4
MAX_REQUESTS=1000
EOF
        log_success "Created environment configuration"
    fi
    
    log_success "Directory setup complete"
}

# Choose deployment mode
choose_deployment_mode() {
    echo ""
    log_info "Choose deployment mode:"
    echo "1) Full deployment (Server + GPU Worker + CPU Worker + Redis + Nginx)"
    echo "2) Server only (Server + Redis)"
    echo "3) GPU Worker only"
    echo "4) CPU Worker only"
    echo "5) Custom deployment"
    echo ""
    
    while true; do
        read -p "Enter your choice (1-5): " choice
        case $choice in
            1)
                export DEPLOYMENT_MODE="full"
                export COMPOSE_FILE="docker-compose.ultimate.yml"
                break
                ;;
            2)
                export DEPLOYMENT_MODE="server"
                export COMPOSE_FILE="docker-compose.ultimate.yml"
                export COMPOSE_PROFILES=""
                break
                ;;
            3)
                export DEPLOYMENT_MODE="gpu-worker"
                export COMPOSE_FILE="docker-compose.ultimate.yml"
                break
                ;;
            4)
                export DEPLOYMENT_MODE="cpu-worker"
                export COMPOSE_FILE="docker-compose.ultimate.yml"
                break
                ;;
            5)
                export DEPLOYMENT_MODE="custom"
                export COMPOSE_FILE="docker-compose.ultimate.yml"
                break
                ;;
            *)
                log_warning "Invalid choice. Please enter 1-5."
                ;;
        esac
    done
    
    log_success "Selected deployment mode: $DEPLOYMENT_MODE"
}

# Build and deploy
deploy_services() {
    log_info "Building and deploying Hashmancer services..."
    
    # Determine which services to deploy
    case $DEPLOYMENT_MODE in
        "full")
            services="redis server nginx"
            if [[ "$GPU_AVAILABLE" == "true" ]]; then
                services="$services worker-gpu"
            else
                services="$services worker-cpu"
            fi
            ;;
        "server")
            services="redis server nginx"
            ;;
        "gpu-worker")
            services="redis worker-gpu"
            ;;
        "cpu-worker")
            services="redis worker-cpu"
            ;;
        "custom")
            log_info "Available services: redis, server, worker-gpu, worker-cpu, nginx, prometheus, grafana"
            read -p "Enter services to deploy (space-separated): " services
            ;;
    esac
    
    log_info "Deploying services: $services"
    
    # Pull base images first
    log_info "Pulling base images..."
    docker-compose -f $COMPOSE_FILE pull redis nginx 2>/dev/null || true
    
    # Build custom images
    log_info "Building custom images..."
    docker-compose -f $COMPOSE_FILE build $services
    
    # Deploy services
    log_info "Starting services..."
    docker-compose -f $COMPOSE_FILE up -d $services
    
    log_success "Services deployed successfully!"
}

# Wait for services to be ready
wait_for_services() {
    log_info "Waiting for services to be ready..."
    
    # Wait for Redis
    log_info "Waiting for Redis..."
    timeout=60
    while ! docker exec hashmancer-redis redis-cli ping &> /dev/null; do
        sleep 2
        timeout=$((timeout - 2))
        if [[ $timeout -le 0 ]]; then
            log_error "Redis failed to start within 60 seconds"
            return 1
        fi
    done
    log_success "Redis is ready"
    
    # Wait for server if deployed
    if [[ $services == *"server"* ]]; then
        log_info "Waiting for server..."
        timeout=120
        while ! curl -f -s http://localhost:8080/health &> /dev/null; do
            sleep 5
            timeout=$((timeout - 5))
            if [[ $timeout -le 0 ]]; then
                log_error "Server failed to start within 120 seconds"
                return 1
            fi
        done
        log_success "Server is ready"
    fi
    
    # Test Redis health
    log_info "Testing Redis health..."
    if python3 redis_tool.py test &> /dev/null; then
        log_success "Redis health check passed"
    else
        log_warning "Redis health check failed, but services are running"
    fi
}

# Show deployment summary
show_summary() {
    echo ""
    echo "🎉 Deployment Complete!"
    echo "======================"
    
    # Show running services
    log_info "Running services:"
    docker-compose -f $COMPOSE_FILE ps
    
    echo ""
    log_info "Access information:"
    
    if [[ $services == *"server"* ]]; then
        echo "  🌐 Web Interface: http://localhost"
        echo "  🔧 Direct Server: http://localhost:8080"
        echo "  📡 API Endpoint: http://localhost:8000"
    fi
    
    if [[ $services == *"redis"* ]]; then
        echo "  📊 Redis: localhost:6379"
    fi
    
    if [[ $services == *"prometheus"* ]]; then
        echo "  📈 Prometheus: http://localhost:9090"
    fi
    
    if [[ $services == *"grafana"* ]]; then
        echo "  📊 Grafana: http://localhost:3000 (admin/hashmancer123)"
    fi
    
    echo ""
    log_info "Useful commands:"
    echo "  📋 View logs: docker-compose -f $COMPOSE_FILE logs -f [service]"
    echo "  🔍 Check status: docker-compose -f $COMPOSE_FILE ps"
    echo "  🛑 Stop services: docker-compose -f $COMPOSE_FILE down"
    echo "  🔄 Restart: docker-compose -f $COMPOSE_FILE restart [service]"
    echo "  🩺 Redis health: python3 redis_tool.py health"
    
    if [[ "$GPU_AVAILABLE" == "true" && $services == *"worker-gpu"* ]]; then
        echo "  🎮 GPU status: docker exec hashmancer-worker-gpu python /app/gpu-utils.py info"
    fi
    
    echo ""
    log_success "Hashmancer is ready to use!"
}

# Handle cleanup on exit
cleanup() {
    if [[ $? -ne 0 ]]; then
        log_error "Deployment failed. Check the logs above for details."
        log_info "To clean up partial deployment, run: docker-compose -f $COMPOSE_FILE down"
    fi
}

trap cleanup EXIT

# Main deployment flow
main() {
    echo ""
    check_permissions
    echo ""
    check_requirements
    echo ""
    setup_nvidia_docker
    echo ""
    setup_directories
    echo ""
    choose_deployment_mode
    echo ""
    deploy_services
    echo ""
    wait_for_services
    echo ""
    show_summary
}

# Run with options
case "${1:-}" in
    "quick")
        log_info "Quick deployment mode - using defaults"
        export DEPLOYMENT_MODE="full"
        export COMPOSE_FILE="docker-compose.ultimate.yml"
        check_requirements
        setup_directories
        deploy_services
        wait_for_services
        show_summary
        ;;
    "server-only")
        log_info "Server-only deployment mode"
        export DEPLOYMENT_MODE="server"
        export COMPOSE_FILE="docker-compose.ultimate.yml"
        check_requirements
        setup_directories
        deploy_services
        wait_for_services
        show_summary
        ;;
    "gpu-worker")
        log_info "GPU worker deployment mode"
        export DEPLOYMENT_MODE="gpu-worker"
        export COMPOSE_FILE="docker-compose.ultimate.yml"
        check_requirements
        setup_nvidia_docker
        setup_directories
        deploy_services
        wait_for_services
        show_summary
        ;;
    "help"|"-h"|"--help")
        echo "Hashmancer Deployment Script"
        echo ""
        echo "Usage: $0 [mode]"
        echo ""
        echo "Modes:"
        echo "  (no args)    Interactive deployment with options"
        echo "  quick        Quick full deployment with defaults"
        echo "  server-only  Deploy only server and Redis"
        echo "  gpu-worker   Deploy GPU worker only"
        echo "  help         Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0                # Interactive deployment"
        echo "  $0 quick          # Quick full deployment"
        echo "  $0 server-only    # Server and Redis only"
        echo ""
        exit 0
        ;;
    *)
        main
        ;;
esac