#!/bin/bash

# Hashmancer Restart Script
# This script safely restarts the Hashmancer service

set -euo pipefail

# Auto-detect the Hashmancer directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HASHMANCER_DIR="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${BLUE}[RESTART]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

restart_service() {
    log "Restarting Hashmancer service..."
    
    # Stop backup processes first
    if [[ -f "$HASHMANCER_DIR/scripts/backup-startup.sh" ]]; then
        "$HASHMANCER_DIR/scripts/backup-startup.sh" stop 2>/dev/null || true
    fi
    
    # Restart systemd service
    if systemctl is-active --quiet hashmancer; then
        log "Stopping systemd service..."
        systemctl stop hashmancer
        sleep 2
    fi
    
    log "Starting systemd service..."
    systemctl start hashmancer
    
    # Wait for service to start
    local attempts=0
    local max_attempts=30
    
    while [[ $attempts -lt $max_attempts ]]; do
        if systemctl is-active --quiet hashmancer; then
            log_success "Service started successfully"
            break
        fi
        
        sleep 1
        ((attempts++))
    done
    
    if [[ $attempts -eq $max_attempts ]]; then
        log_error "Service failed to start within 30 seconds"
        return 1
    fi
}

test_portal() {
    log "Testing portal connectivity..."
    
    local attempts=0
    local max_attempts=60
    
    while [[ $attempts -lt $max_attempts ]]; do
        if curl -f -s --max-time 3 http://localhost:8000/server_status > /dev/null 2>&1; then
            log_success "Portal is responding at http://localhost:8000"
            return 0
        fi
        
        sleep 2
        ((attempts++))
    done
    
    log_warn "Portal is not responding after 2 minutes"
    log_warn "Check logs with: journalctl -u hashmancer -f"
    return 1
}

main() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}      HASHMANCER RESTART UTILITY${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    
    # Check if running as root for systemd operations
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root (use sudo)"
        exit 1
    fi
    
    if restart_service; then
        if test_portal; then
            echo ""
            log_success "🎉 Restart completed successfully!"
            log_success "Portal: http://localhost:8000"
        else
            echo ""
            log_warn "⚠️  Service restarted but portal not responding"
            log_warn "Check status with: ./scripts/status-hashmancer.sh"
        fi
    else
        echo ""
        log_error "❌ Restart failed"
        log_error "Check logs with: journalctl -u hashmancer -n 20"
        exit 1
    fi
}

main "$@"