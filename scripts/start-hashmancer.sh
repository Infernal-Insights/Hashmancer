#!/bin/bash

# Hashmancer Startup Script
# This script ensures the Hashmancer server starts properly and stays running

set -euo pipefail

# Configuration
# Auto-detect the Hashmancer directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HASHMANCER_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$HASHMANCER_DIR/venv"
LOG_DIR="$HASHMANCER_DIR/logs"
PID_FILE="$HASHMANCER_DIR/hashmancer.pid"
LOG_FILE="$LOG_DIR/startup.log"
MAX_RETRIES=5
RETRY_DELAY=5

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${timestamp} [$level] $message" | tee -a "$LOG_FILE"
}

log_info() { log "INFO" "$@"; }
log_warn() { log "WARN" "$@"; }
log_error() { log "ERROR" "$@"; }
log_success() { log "SUCCESS" "$@"; }

# Cleanup function
cleanup() {
    log_info "Cleaning up..."
    if [[ -f "$PID_FILE" ]]; then
        rm -f "$PID_FILE"
    fi
}

# Signal handlers
trap cleanup EXIT
trap 'log_error "Script interrupted"; exit 1' INT TERM

# Create necessary directories
setup_directories() {
    log_info "Setting up directories..."
    mkdir -p "$LOG_DIR"
    mkdir -p "$HASHMANCER_DIR/data"
    mkdir -p "$HASHMANCER_DIR/wordlists"
    mkdir -p "$HASHMANCER_DIR/masks"
    mkdir -p "$HASHMANCER_DIR/restores"
    chmod 755 "$LOG_DIR" "$HASHMANCER_DIR/data" "$HASHMANCER_DIR/wordlists" "$HASHMANCER_DIR/masks" "$HASHMANCER_DIR/restores"
}

# Check and setup Python virtual environment
setup_python_env() {
    log_info "Setting up Python environment..."
    
    cd "$HASHMANCER_DIR"
    
    # Check if virtual environment exists
    if [[ ! -d "$VENV_DIR" ]]; then
        log_info "Creating Python virtual environment..."
        python3 -m venv "$VENV_DIR"
    fi
    
    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install requirements if they exist
    if [[ -f "requirements.txt" ]]; then
        log_info "Installing Python requirements..."
        pip install -r requirements.txt
    fi
    
    # Install package in development mode
    if [[ -f "setup.py" ]] || [[ -f "pyproject.toml" ]]; then
        log_info "Installing Hashmancer package..."
        pip install -e .
    fi
}

# Check system dependencies
check_dependencies() {
    log_info "Checking system dependencies..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed"
        return 1
    fi
    
    # Check Redis
    if ! command -v redis-server &> /dev/null; then
        log_warn "Redis not found, attempting to start with system package..."
        if command -v systemctl &> /dev/null; then
            systemctl start redis-server 2>/dev/null || true
        fi
    fi
    
    # Check if Redis is running
    if ! pgrep redis-server &> /dev/null; then
        log_info "Starting Redis server..."
        redis-server --daemonize yes --port 6379 --logfile "$LOG_DIR/redis.log" || true
        sleep 2
    fi
    
    log_success "Dependencies check completed"
}

# Test server connectivity
test_server() {
    local max_attempts=10
    local attempt=1
    
    log_info "Testing server connectivity..."
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -f -s http://localhost:8000/server_status &> /dev/null; then
            log_success "Server is responding at http://localhost:8000"
            return 0
        fi
        
        log_info "Attempt $attempt/$max_attempts - waiting for server..."
        sleep 3
        ((attempt++))
    done
    
    log_error "Server failed to respond after $max_attempts attempts"
    return 1
}

# Main startup function
start_hashmancer() {
    local retry_count=0
    
    while [[ $retry_count -lt $MAX_RETRIES ]]; do
        log_info "Starting Hashmancer (attempt $((retry_count + 1))/$MAX_RETRIES)..."
        
        cd "$HASHMANCER_DIR"
        
        # Activate virtual environment
        source "$VENV_DIR/bin/activate"
        
        # Set environment variables
        export PYTHONPATH="$HASHMANCER_DIR"
        export PYTHONUNBUFFERED=1
        
        # Kill any existing processes
        pkill -f "python.*hashmancer" || true
        sleep 2
        
        # Start the server
        if [[ -f "main.py" ]]; then
            log_info "Starting with main.py..."
            python main.py &
        elif [[ -f "hashmancer/server/main.py" ]]; then
            log_info "Starting with hashmancer/server/main.py..."
            python -m hashmancer.server.main &
        elif [[ -f "run.py" ]]; then
            log_info "Starting with run.py..."
            python run.py &
        else
            log_info "Starting with uvicorn..."
            uvicorn hashmancer.server.main:app --host 0.0.0.0 --port 8000 --workers 1 &
        fi
        
        local server_pid=$!
        echo "$server_pid" > "$PID_FILE"
        
        log_info "Server started with PID: $server_pid"
        
        # Wait a moment for server to initialize
        sleep 5
        
        # Test if server is working
        if test_server; then
            log_success "Hashmancer server started successfully!"
            log_info "Portal available at: http://localhost:8000"
            log_info "Admin panel available at: http://localhost:8000/admin"
            
            # Wait for the server process
            wait "$server_pid"
            return $?
        else
            log_error "Server startup failed on attempt $((retry_count + 1))"
            kill "$server_pid" 2>/dev/null || true
            ((retry_count++))
            
            if [[ $retry_count -lt $MAX_RETRIES ]]; then
                log_info "Retrying in $RETRY_DELAY seconds..."
                sleep "$RETRY_DELAY"
            fi
        fi
    done
    
    log_error "Failed to start Hashmancer after $MAX_RETRIES attempts"
    return 1
}

# Main execution
main() {
    log_info "=== Hashmancer Startup Script ==="
    log_info "Starting Hashmancer server..."
    
    setup_directories
    check_dependencies
    setup_python_env
    start_hashmancer
}

# Execute main function
main "$@"