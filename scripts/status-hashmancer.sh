#!/bin/bash

# Hashmancer Status Check Script
# This script checks the current status of Hashmancer

set -euo pipefail

# Auto-detect the Hashmancer directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HASHMANCER_DIR="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

print_header() {
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}       HASHMANCER STATUS CHECK${NC}"
    echo -e "${CYAN}========================================${NC}"
}

check_systemd_service() {
    echo -e "${BLUE}[SERVICE]${NC} Checking systemd service status..."
    
    if systemctl is-active --quiet hashmancer; then
        echo -e "${GREEN}✓ Service is running${NC}"
    elif systemctl is-failed --quiet hashmancer; then
        echo -e "${RED}✗ Service has failed${NC}"
        systemctl status hashmancer --no-pager -l | head -10
    else
        echo -e "${YELLOW}! Service is stopped${NC}"
    fi
    
    if systemctl is-enabled --quiet hashmancer; then
        echo -e "${GREEN}✓ Service is enabled for auto-start${NC}"
    else
        echo -e "${YELLOW}! Service is not enabled${NC}"
    fi
}

check_portal() {
    echo -e "${BLUE}[PORTAL]${NC} Checking portal connectivity..."
    
    if curl -f -s --max-time 5 http://localhost:8000/server_status > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Portal is accessible at http://localhost:8000${NC}"
    else
        echo -e "${RED}✗ Portal is not responding${NC}"
    fi
}

check_processes() {
    echo -e "${BLUE}[PROCESSES]${NC} Checking running processes..."
    
    local hashmancer_procs=$(pgrep -f "hashmancer\|python.*main" | wc -l)
    if [[ $hashmancer_procs -gt 0 ]]; then
        echo -e "${GREEN}✓ Found $hashmancer_procs Hashmancer process(es)${NC}"
        pgrep -f "hashmancer\|python.*main" | xargs ps -p 2>/dev/null | tail -n +2
    else
        echo -e "${YELLOW}! No Hashmancer processes found${NC}"
    fi
}

check_backup_processes() {
    echo -e "${BLUE}[BACKUP]${NC} Checking backup startup methods..."
    
    if [[ -f "$HASHMANCER_DIR/backup.pid" ]]; then
        local pid_info=$(cat "$HASHMANCER_DIR/backup.pid")
        echo -e "${GREEN}✓ Backup process active: $pid_info${NC}"
    else
        echo -e "${YELLOW}! No backup processes running${NC}"
    fi
}

check_logs() {
    echo -e "${BLUE}[LOGS]${NC} Recent log activity..."
    
    if [[ -f "$HASHMANCER_DIR/logs/startup.log" ]]; then
        echo -e "${CYAN}Last 5 lines of startup log:${NC}"
        tail -n 5 "$HASHMANCER_DIR/logs/startup.log" 2>/dev/null || echo "No recent logs"
    else
        echo -e "${YELLOW}! No startup log found${NC}"
    fi
}

print_summary() {
    echo ""
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}              SUMMARY${NC}"
    echo -e "${CYAN}========================================${NC}"
    
    # Quick status
    if systemctl is-active --quiet hashmancer && curl -f -s --max-time 3 http://localhost:8000/server_status > /dev/null 2>&1; then
        echo -e "${GREEN}🎉 HASHMANCER IS RUNNING AND ACCESSIBLE${NC}"
        echo -e "${GREEN}Portal: http://localhost:8000${NC}"
    elif systemctl is-active --quiet hashmancer; then
        echo -e "${YELLOW}⚠️  HASHMANCER SERVICE IS RUNNING BUT PORTAL NOT RESPONDING${NC}"
        echo -e "${YELLOW}May be starting up or experiencing issues${NC}"
    else
        echo -e "${RED}❌ HASHMANCER IS NOT RUNNING${NC}"
        echo -e "${YELLOW}Try: sudo systemctl start hashmancer${NC}"
    fi
}

main() {
    print_header
    echo ""
    
    check_systemd_service
    echo ""
    
    check_portal
    echo ""
    
    check_processes
    echo ""
    
    check_backup_processes
    echo ""
    
    check_logs
    
    print_summary
}

main "$@"