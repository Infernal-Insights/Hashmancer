#!/bin/bash

# Hashmancer Stop Script
# This script safely stops all Hashmancer processes

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
    echo -e "${BLUE}[STOP]${NC} $*"
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

stop_systemd_service() {
    log "Stopping systemd service..."
    
    if systemctl is-active --quiet hashmancer; then
        systemctl stop hashmancer
        
        # Wait for service to stop
        local attempts=0
        local max_attempts=30
        
        while [[ $attempts -lt $max_attempts ]]; do
            if ! systemctl is-active --quiet hashmancer; then
                log_success "Systemd service stopped"
                return 0
            fi
            
            sleep 1
            ((attempts++))
        done
        
        log_warn "Service did not stop gracefully, forcing stop..."
        systemctl kill hashmancer
        sleep 2
        
        if ! systemctl is-active --quiet hashmancer; then
            log_success "Service forcefully stopped"
        else
            log_error "Failed to stop service"
            return 1
        fi
    else
        log "Systemd service is already stopped"
    fi
}

stop_backup_processes() {
    log "Stopping backup processes..."
    
    if [[ -f "$HASHMANCER_DIR/scripts/backup-startup.sh" ]]; then
        "$HASHMANCER_DIR/scripts/backup-startup.sh" stop 2>/dev/null || true
        log_success "Backup processes stopped"
    else
        log "No backup startup script found"
    fi
}

stop_manual_processes() {
    log "Stopping any remaining Hashmancer processes..."
    
    # Find and stop any remaining processes
    local pids=$(pgrep -f "hashmancer\|python.*main" 2>/dev/null || true)
    
    if [[ -n "$pids" ]]; then
        log "Found remaining processes: $pids"
        
        # Try graceful shutdown first
        echo "$pids" | xargs kill -TERM 2>/dev/null || true
        sleep 5
        
        # Check if processes are still running
        local remaining_pids=$(pgrep -f "hashmancer\|python.*main" 2>/dev/null || true)
        
        if [[ -n "$remaining_pids" ]]; then
            log_warn "Some processes did not stop gracefully, forcing..."
            echo "$remaining_pids" | xargs kill -KILL 2>/dev/null || true
            sleep 2
        fi
        
        # Final check
        local final_pids=$(pgrep -f "hashmancer\|python.*main" 2>/dev/null || true)
        
        if [[ -z "$final_pids" ]]; then
            log_success "All processes stopped"
        else
            log_error "Some processes could not be stopped: $final_pids"
            return 1
        fi
    else
        log "No Hashmancer processes found"
    fi
}

cleanup_files() {
    log "Cleaning up process files..."
    
    # Remove PID files
    rm -f "$HASHMANCER_DIR/hashmancer.pid" 2>/dev/null || true
    rm -f "$HASHMANCER_DIR/backup.pid" 2>/dev/null || true
    
    log_success "Cleanup completed"
}

verify_stopped() {
    log "Verifying all processes are stopped..."
    
    if systemctl is-active --quiet hashmancer; then
        log_error "Systemd service is still running"
        return 1
    fi
    
    local remaining_pids=$(pgrep -f "hashmancer\|python.*main" 2>/dev/null || true)
    if [[ -n "$remaining_pids" ]]; then
        log_error "Some processes are still running: $remaining_pids"
        return 1
    fi
    
    if curl -f -s --max-time 2 http://localhost:8000/server_status > /dev/null 2>&1; then
        log_error "Portal is still responding"
        return 1
    fi
    
    log_success "All Hashmancer components are stopped"
    return 0
}

main() {
    local force="${1:-}"
    
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}       HASHMANCER STOP UTILITY${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    
    # Check if running as root for systemd operations
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root (use sudo)"
        exit 1
    fi
    
    # Confirm unless force flag is used
    if [[ "$force" != "--force" ]]; then
        echo -e "${YELLOW}This will stop all Hashmancer services and processes.${NC}"
        read -p "Are you sure? (y/N): " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log "Operation cancelled"
            exit 0
        fi
    fi
    
    echo ""
    
    # Stop all components
    stop_backup_processes
    stop_systemd_service
    stop_manual_processes
    cleanup_files
    
    echo ""
    
    if verify_stopped; then
        echo ""
        log_success "🛑 Hashmancer has been completely stopped"
        log_success "To start again: sudo systemctl start hashmancer"
    else
        echo ""
        log_error "❌ Some components may still be running"
        log_error "Check status with: ./scripts/status-hashmancer.sh"
        exit 1
    fi
}

main "$@"