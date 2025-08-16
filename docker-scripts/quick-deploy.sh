#!/bin/bash

# Hashmancer Quick Docker Deployment Script
# This script provides various quick deployment options

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
PURPLE='\033[0;35m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE=""
DEPLOYMENT_TYPE=""

print_header() {
    echo -e "${PURPLE}"
    echo "=================================================="
    echo "     HASHMANCER QUICK DOCKER DEPLOYMENT"
    echo "=================================================="
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

show_usage() {
    echo "Usage: $0 [OPTIONS] DEPLOYMENT_TYPE"
    echo ""
    echo "Deployment Types:"
    echo "  production    - Production deployment with security hardening"
    echo "  gpu           - GPU-accelerated deployment"
    echo "  development   - Development environment with hot reload"
    echo "  monitoring    - Production with monitoring stack"
    echo ""
    echo "Options:"
    echo "  -h, --help    - Show this help message"
    echo "  -v, --verbose - Verbose output"
    echo "  --pull        - Pull latest images before starting"
    echo "  --build       - Force rebuild of images"
    echo "  --clean       - Clean volumes and containers before start"
    echo ""
    echo "Examples:"
    echo "  $0 production"
    echo "  $0 gpu --build"
    echo "  $0 development --clean"
    echo "  $0 monitoring --pull"
}

check_requirements() {
    print_step "Checking requirements..."
    
    # Check Docker
    if ! command -v docker >/dev/null 2>&1; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose >/dev/null 2>&1 && ! docker compose version >/dev/null 2>&1; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if Docker is running
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi
    
    # Check for GPU (if GPU deployment)
    if [[ "$DEPLOYMENT_TYPE" == "gpu" ]]; then
        if ! command -v nvidia-smi >/dev/null 2>&1; then
            print_warning "NVIDIA GPU drivers not detected. GPU deployment may not work."
        else
            print_info "GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
        fi
        
        # Check for nvidia-docker
        if ! docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
            print_error "NVIDIA Container Toolkit not properly configured."
            print_info "Please install and configure nvidia-docker2."
            exit 1
        fi
    fi
    
    print_success "Requirements check passed"
}

setup_environment() {
    print_step "Setting up environment..."
    
    cd "$PROJECT_DIR"
    
    # Create .env file if it doesn't exist
    if [[ ! -f ".env" ]]; then
        print_info "Creating .env file from example..."
        cp .env.example .env
        
        # Generate random secrets
        if command -v openssl >/dev/null 2>&1; then
            SECRET_KEY=$(openssl rand -hex 32)
            sed -i "s/^SECRET_KEY=.*/SECRET_KEY=$SECRET_KEY/" .env
        fi
        
        print_warning "Please edit .env file to configure your installation"
    fi
    
    # Create necessary directories
    mkdir -p {data,logs,config,uploads,wordlists,rules}
    
    # Set deployment-specific environment
    case "$DEPLOYMENT_TYPE" in
        production)
            export COMPOSE_FILE="docker-compose.production.yml"
            ;;
        gpu)
            export COMPOSE_FILE="docker-compose.gpu.yml"
            ;;
        development)
            export COMPOSE_FILE="docker-compose.development.yml"
            ;;
        monitoring)
            export COMPOSE_FILE="docker-compose.production.yml"
            export COMPOSE_PROFILES="monitoring"
            ;;
    esac
    
    print_success "Environment setup complete"
}

pull_images() {
    if [[ "${PULL_IMAGES:-false}" == "true" ]]; then
        print_step "Pulling latest images..."
        docker-compose -f "$COMPOSE_FILE" pull
        print_success "Images pulled"
    fi
}

build_images() {
    if [[ "${BUILD_IMAGES:-false}" == "true" ]]; then
        print_step "Building images..."
        docker-compose -f "$COMPOSE_FILE" build --no-cache
        print_success "Images built"
    fi
}

clean_deployment() {
    if [[ "${CLEAN_DEPLOYMENT:-false}" == "true" ]]; then
        print_step "Cleaning previous deployment..."
        docker-compose -f "$COMPOSE_FILE" down -v --remove-orphans
        print_warning "All volumes and containers removed"
    fi
}

deploy_services() {
    print_step "Deploying services..."
    
    # Start services
    if [[ -n "${COMPOSE_PROFILES:-}" ]]; then
        docker-compose -f "$COMPOSE_FILE" --profile "$COMPOSE_PROFILES" up -d
    else
        docker-compose -f "$COMPOSE_FILE" up -d
    fi
    
    print_success "Services deployed"
}

wait_for_services() {
    print_step "Waiting for services to be ready..."
    
    local max_attempts=30
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -f -s --max-time 5 http://localhost:8000/health >/dev/null 2>&1; then
            print_success "Services are ready"
            return 0
        fi
        
        print_info "Attempt $attempt/$max_attempts - waiting for services..."
        sleep 10
        ((attempt++))
    done
    
    print_error "Services failed to start within timeout"
    return 1
}

show_deployment_info() {
    print_step "Deployment Information"
    echo ""
    
    # Load environment
    source .env 2>/dev/null || true
    
    print_info "🌐 ACCESS INFORMATION:"
    echo "   Portal URL: http://localhost:${HTTP_PORT:-8000}"
    echo "   Admin Panel: http://localhost:${HTTP_PORT:-8000}/admin"
    
    if [[ "$DEPLOYMENT_TYPE" == "development" ]]; then
        echo "   Development Mode: Hot reload enabled"
        echo "   Debug Port: 8001"
    fi
    
    if [[ "$DEPLOYMENT_TYPE" == "monitoring" || "${COMPOSE_PROFILES:-}" == "monitoring" ]]; then
        echo "   Prometheus: http://localhost:9090"
        echo "   Grafana: http://localhost:3000"
    fi
    
    echo ""
    print_info "🛠️  MANAGEMENT COMMANDS:"
    echo "   View status: docker-compose -f $COMPOSE_FILE ps"
    echo "   View logs: docker-compose -f $COMPOSE_FILE logs -f"
    echo "   Stop services: docker-compose -f $COMPOSE_FILE down"
    echo "   Restart: docker-compose -f $COMPOSE_FILE restart"
    echo ""
    
    if [[ "$DEPLOYMENT_TYPE" == "gpu" ]]; then
        print_info "🚀 GPU INFORMATION:"
        if command -v nvidia-smi >/dev/null 2>&1; then
            nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
        fi
        echo ""
    fi
    
    print_info "📊 CONTAINER STATUS:"
    docker-compose -f "$COMPOSE_FILE" ps
}

main() {
    # Parse arguments
    PULL_IMAGES=false
    BUILD_IMAGES=false
    CLEAN_DEPLOYMENT=false
    VERBOSE=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                exit 0
                ;;
            -v|--verbose)
                VERBOSE=true
                set -x
                shift
                ;;
            --pull)
                PULL_IMAGES=true
                shift
                ;;
            --build)
                BUILD_IMAGES=true
                shift
                ;;
            --clean)
                CLEAN_DEPLOYMENT=true
                shift
                ;;
            production|gpu|development|monitoring)
                DEPLOYMENT_TYPE="$1"
                shift
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Validate deployment type
    if [[ -z "$DEPLOYMENT_TYPE" ]]; then
        print_error "Deployment type is required"
        show_usage
        exit 1
    fi
    
    # Main execution
    print_header
    print_info "Starting $DEPLOYMENT_TYPE deployment..."
    echo ""
    
    check_requirements
    setup_environment
    clean_deployment
    pull_images
    build_images
    deploy_services
    wait_for_services
    show_deployment_info
    
    echo ""
    print_success "🎉 $DEPLOYMENT_TYPE deployment completed successfully!"
    print_info "Access your Hashmancer installation at: http://localhost:${HTTP_PORT:-8000}"
}

# Execute main function
main "$@"