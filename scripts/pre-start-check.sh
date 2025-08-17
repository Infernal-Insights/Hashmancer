#!/bin/bash

# Hashmancer Pre-Start Check Script
# This script performs health checks before starting the main service

set -euo pipefail

# Auto-detect the Hashmancer directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HASHMANCER_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$HASHMANCER_DIR/logs"
LOG_FILE="$LOG_DIR/pre-start.log"

# Logging function
log() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "$timestamp [PRE-START] $*" | tee -a "$LOG_FILE"
}

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

log "Starting pre-start checks..."

# Check if we're running as root or have sudo access
if [[ $EUID -ne 0 ]] && ! sudo -n true 2>/dev/null; then
    log "WARNING: Not running as root and no sudo access"
fi

# Check disk space
available_space=$(df "$HASHMANCER_DIR" | awk 'NR==2 {print $4}')
if [[ $available_space -lt 1048576 ]]; then  # Less than 1GB
    log "WARNING: Low disk space available: ${available_space}KB"
else
    log "Disk space check passed: ${available_space}KB available"
fi

# Check memory
available_memory=$(free | awk 'NR==2 {print $7}')
if [[ $available_memory -lt 512000 ]]; then  # Less than 512MB
    log "WARNING: Low memory available: ${available_memory}KB"
else
    log "Memory check passed: ${available_memory}KB available"
fi

# Check if required directories exist
for dir in "$HASHMANCER_DIR" "$HASHMANCER_DIR/hashmancer" "$HASHMANCER_DIR/hashmancer/server"; do
    if [[ ! -d "$dir" ]]; then
        log "ERROR: Required directory missing: $dir"
        exit 1
    fi
done

# Check if main server file exists
if [[ ! -f "$HASHMANCER_DIR/hashmancer/server/main.py" ]] && [[ ! -f "$HASHMANCER_DIR/main.py" ]]; then
    log "ERROR: Main server file not found"
    exit 1
fi

# Check Python installation
if ! command -v python3 &> /dev/null; then
    log "ERROR: Python 3 not found"
    exit 1
fi

# Kill any zombie processes
pkill -f "hashmancer.*zombie" || true

# Clean up old lock files
find "$HASHMANCER_DIR" -name "*.lock" -mtime +1 -delete 2>/dev/null || true

# Clean up old log files (keep last 30 days)
find "$LOG_DIR" -name "*.log" -mtime +30 -delete 2>/dev/null || true

log "Pre-start checks completed successfully"
exit 0