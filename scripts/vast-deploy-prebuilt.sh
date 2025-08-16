#!/bin/bash
# Deploy Hashmancer Workers on Vast.ai using Pre-built Docker Image

set -e

# Configuration
DOCKER_IMAGE="hashmancer/worker:latest"
DEFAULT_MAX_PRICE="1.00"
DEFAULT_GPU_TYPE="3080"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
info() { echo -e "${BLUE}[INFO]${NC} $1"; }

# Usage function
usage() {
    echo "🔓 Hashmancer Vast.ai Deployment Tool"
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -g, --gpu-type GPU_TYPE    GPU type to search for (default: $DEFAULT_GPU_TYPE)"
    echo "  -c, --count COUNT          Number of workers to deploy (default: 1)"
    echo "  -p, --max-price PRICE      Maximum price per hour (default: $DEFAULT_MAX_PRICE)"
    echo "  -s, --server-ip IP         Your Hashmancer server IP (required)"
    echo "  -k, --api-key KEY          Vast.ai API key (or set VAST_API_KEY env var)"
    echo "  -i, --image IMAGE          Docker image to use (default: $DOCKER_IMAGE)"
    echo "  -h, --help                 Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 -s 203.0.113.1 -g 3080 -c 2 -p 0.75"
    echo "  $0 --server-ip=203.0.113.1 --gpu-type=4090 --count=1"
    echo ""
    echo "Environment Variables:"
    echo "  VAST_API_KEY              Your Vast.ai API key"
    echo "  HASHMANCER_SERVER_IP      Your server IP (can be overridden with -s)"
}

# Parse command line arguments
GPU_TYPE="$DEFAULT_GPU_TYPE"
COUNT=1
MAX_PRICE="$DEFAULT_MAX_PRICE"
SERVER_IP="${HASHMANCER_SERVER_IP:-}"
VAST_API_KEY="${VAST_API_KEY:-}"
IMAGE="$DOCKER_IMAGE"

while [[ $# -gt 0 ]]; do
    case $1 in
        -g|--gpu-type)
            GPU_TYPE="$2"
            shift 2
            ;;
        -c|--count)
            COUNT="$2"
            shift 2
            ;;
        -p|--max-price)
            MAX_PRICE="$2"
            shift 2
            ;;
        -s|--server-ip)
            SERVER_IP="$2"
            shift 2
            ;;
        -k|--api-key)
            VAST_API_KEY="$2"
            shift 2
            ;;
        -i|--image)
            IMAGE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate required parameters
if [ -z "$SERVER_IP" ]; then
    error "Server IP is required. Use -s option or set HASHMANCER_SERVER_IP environment variable"
    usage
    exit 1
fi

if [ -z "$VAST_API_KEY" ]; then
    error "Vast.ai API key is required. Use -k option or set VAST_API_KEY environment variable"
    usage
    exit 1
fi

# Validate server accessibility
log "🔍 Validating server accessibility..."
if timeout 10 curl -s "http://$SERVER_IP:8080/health" > /dev/null; then
    log "✅ Server is accessible at $SERVER_IP:8080"
else
    warn "⚠️  Cannot reach server at $SERVER_IP:8080"
    warn "   Workers may not be able to connect"
    read -p "Continue anyway? (y/N): " continue_choice
    if [[ ! $continue_choice =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Display configuration
log "🚀 Hashmancer Vast.ai Deployment"
info "GPU Type: $GPU_TYPE"
info "Count: $COUNT"
info "Max Price: \$$MAX_PRICE/hour"
info "Server IP: $SERVER_IP"
info "Docker Image: $IMAGE"

# Search for available instances
log "🔍 Searching for available $GPU_TYPE instances..."

search_query=$(cat << EOF
{
    "query": {
        "rentable": true,
        "gpu_name": {"contains": "$GPU_TYPE"},
        "dph_total": {"lte": $MAX_PRICE}
    },
    "sort": [{"field": "dph_total", "direction": "asc"}],
    "limit": $((COUNT * 2))
}
EOF
)

# Make API request to search instances
search_response=$(curl -s -X POST \
    -H "Authorization: Bearer $VAST_API_KEY" \
    -H "Content-Type: application/json" \
    -d "$search_query" \
    "https://console.vast.ai/api/v0/bundles/" || {
        error "Failed to search Vast.ai instances"
        exit 1
    })

# Parse search results
available_instances=$(echo "$search_response" | jq -r '.offers[]? | select(.gpu_name | contains("'$GPU_TYPE'")) | "\(.id)|\(.gpu_name)|\(.dph_total)|\(.cpu_cores)|\(.ram)"')

if [ -z "$available_instances" ]; then
    error "No suitable $GPU_TYPE instances found under \$$MAX_PRICE/hour"
    exit 1
fi

# Display available instances
log "💰 Found suitable instances:"
echo "$available_instances" | head -$COUNT | while IFS='|' read -r id gpu_name price cpu ram; do
    info "  Instance $id: $gpu_name - \$$price/hr - ${cpu}CPU ${ram}GB RAM"
done

# Confirm deployment
echo ""
read -p "🤔 Deploy $COUNT workers? (y/N): " deploy_choice
if [[ ! $deploy_choice =~ ^[Yy]$ ]]; then
    log "Deployment cancelled"
    exit 0
fi

# Deploy instances
log "🚀 Deploying workers..."
deployed_count=0
failed_count=0

echo "$available_instances" | head -$COUNT | while IFS='|' read -r instance_id gpu_name price cpu ram; do
    log "Deploying worker $((deployed_count + 1))/$COUNT on instance $instance_id..."
    
    # Create deployment payload
    deployment_payload=$(cat << EOF
{
    "client_id": "hashmancer",
    "image": "$IMAGE",
    "env": {
        "HASHMANCER_SERVER_IP": "$SERVER_IP",
        "HASHMANCER_SERVER_PORT": "8080",
        "WORKER_PORT": "8081",
        "MAX_CONCURRENT_JOBS": "3",
        "LOG_LEVEL": "INFO"
    },
    "onstart": "echo 'Hashmancer worker starting with pre-built image...'",
    "runtype": "ssh"
}
EOF
    )
    
    # Deploy instance
    deploy_response=$(curl -s -X PUT \
        -H "Authorization: Bearer $VAST_API_KEY" \
        -H "Content-Type: application/json" \
        -d "$deployment_payload" \
        "https://console.vast.ai/api/v0/asks/$instance_id/" || {
            error "Failed to deploy instance $instance_id"
            failed_count=$((failed_count + 1))
            continue
        })
    
    # Check deployment result
    contract_id=$(echo "$deploy_response" | jq -r '.new_contract // empty')
    
    if [ -n "$contract_id" ] && [ "$contract_id" != "null" ]; then
        log "✅ Deployed worker on instance $instance_id (contract: $contract_id)"
        deployed_count=$((deployed_count + 1))
    else
        error_msg=$(echo "$deploy_response" | jq -r '.error // "Unknown error"')
        error "❌ Failed to deploy instance $instance_id: $error_msg"
        failed_count=$((failed_count + 1))
    fi
    
    # Small delay between deployments
    sleep 2
done

# Summary
echo ""
log "📊 Deployment Summary:"
info "  Successfully deployed: $deployed_count workers"
if [ $failed_count -gt 0 ]; then
    warn "  Failed deployments: $failed_count"
fi

if [ $deployed_count -gt 0 ]; then
    log "🎉 Workers are starting up..."
    info "Workers will automatically:"
    info "  • Download the pre-built Docker image"
    info "  • Connect to your server at $SERVER_IP:8080"
    info "  • Register themselves"
    info "  • Start processing jobs"
    echo ""
    info "Monitor workers with:"
    info "  ./hashmancer-cli worker status --all"
    info "  ./hashmancer-cli server discover"
else
    error "No workers were deployed successfully"
    exit 1
fi