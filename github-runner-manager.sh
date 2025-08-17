#!/bin/bash
set -e

# GitHub Actions Runner Management Script
# Comprehensive management tool for self-hosted GitHub Actions runners

echo "🔧 GitHub Actions Runner Manager"
echo "==============================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
RUNNER_DIR="$HOME/github-runner"
LOG_DIR="/var/log/github-runner"
SERVICE_PREFIX="actions.runner"

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

log_header() {
    echo -e "${PURPLE}🔹 $1${NC}"
}

# Get runner service name
get_service_name() {
    systemctl list-units --type=service | grep "$SERVICE_PREFIX" | head -1 | awk '{print $1}' || echo ""
}

# Show runner status
show_status() {
    log_header "Runner Status"
    
    local service_name=$(get_service_name)
    
    if [[ -n "$service_name" ]]; then
        echo "Service: $service_name"
        
        # Service status
        if systemctl is-active "$service_name" > /dev/null 2>&1; then
            log_success "Service is active"
        else
            log_error "Service is not active"
        fi
        
        if systemctl is-enabled "$service_name" > /dev/null 2>&1; then
            log_success "Service is enabled (auto-start)"
        else
            log_warning "Service is not enabled"
        fi
        
        echo ""
        echo "Detailed Status:"
        sudo systemctl status "$service_name" --no-pager -l | head -20
    else
        log_error "No GitHub runner service found"
        return 1
    fi
    
    echo ""
    log_header "System Information"
    echo "Hostname: $(hostname)"
    echo "OS: $(cat /etc/os-release | grep PRETTY_NAME | cut -d= -f2 | tr -d '"')"
    echo "CPU: $(nproc) cores"
    echo "Memory: $(free -h | grep Mem | awk '{print $2 " total, " $3 " used, " $7 " available"}')"
    echo "Disk: $(df -h "$RUNNER_DIR" 2>/dev/null | tail -1 | awk '{print $2 " total, " $3 " used, " $4 " available"}' || echo "N/A")"
    
    # GPU info
    if command -v nvidia-smi > /dev/null 2>&1; then
        echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
        echo "GPU Memory: $(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | head -1 | awk -F', ' '{print $2"/"$1" MB"}')"
    else
        echo "GPU: Not available"
    fi
    
    # Docker info
    if command -v docker > /dev/null 2>&1; then
        echo "Docker: $(docker --version | cut -d' ' -f3 | tr -d ',')"
        echo "Docker Status: $(docker info > /dev/null 2>&1 && echo "Running" || echo "Not running")"
    else
        echo "Docker: Not installed"
    fi
}

# Show detailed logs
show_logs() {
    local lines="${1:-50}"
    log_header "Runner Logs (last $lines lines)"
    
    local service_name=$(get_service_name)
    
    if [[ -n "$service_name" ]]; then
        echo "Service Logs:"
        sudo journalctl -u "$service_name" -n "$lines" --no-pager
    fi
    
    echo ""
    if [[ -f "$LOG_DIR/monitor.log" ]]; then
        echo "Monitor Logs:"
        tail -n "$lines" "$LOG_DIR/monitor.log"
    else
        log_warning "Monitor logs not found"
    fi
}

# Control runner service
control_service() {
    local action="$1"
    local service_name=$(get_service_name)
    
    if [[ -z "$service_name" ]]; then
        log_error "No GitHub runner service found"
        return 1
    fi
    
    case "$action" in
        "start")
            log_info "Starting runner service..."
            sudo systemctl start "$service_name"
            sleep 3
            if systemctl is-active "$service_name" > /dev/null 2>&1; then
                log_success "Runner service started"
            else
                log_error "Failed to start runner service"
                return 1
            fi
            ;;
        "stop")
            log_info "Stopping runner service..."
            sudo systemctl stop "$service_name"
            sleep 3
            if ! systemctl is-active "$service_name" > /dev/null 2>&1; then
                log_success "Runner service stopped"
            else
                log_error "Failed to stop runner service"
                return 1
            fi
            ;;
        "restart")
            log_info "Restarting runner service..."
            sudo systemctl restart "$service_name"
            sleep 5
            if systemctl is-active "$service_name" > /dev/null 2>&1; then
                log_success "Runner service restarted"
            else
                log_error "Failed to restart runner service"
                return 1
            fi
            ;;
        "enable")
            log_info "Enabling runner service for auto-start..."
            sudo systemctl enable "$service_name"
            log_success "Runner service enabled"
            ;;
        "disable")
            log_info "Disabling runner service auto-start..."
            sudo systemctl disable "$service_name"
            log_success "Runner service disabled"
            ;;
        *)
            log_error "Invalid action: $action"
            echo "Valid actions: start, stop, restart, enable, disable"
            return 1
            ;;
    esac
}

# Health check
health_check() {
    log_header "Health Check"
    
    local issues=0
    
    # Check service status
    local service_name=$(get_service_name)
    if [[ -n "$service_name" ]] && systemctl is-active "$service_name" > /dev/null 2>&1; then
        log_success "Runner service is active"
    else
        log_error "Runner service is not active"
        issues=$((issues + 1))
    fi
    
    # Check runner directory
    if [[ -d "$RUNNER_DIR" ]]; then
        log_success "Runner directory exists"
    else
        log_error "Runner directory not found: $RUNNER_DIR"
        issues=$((issues + 1))
    fi
    
    # Check runner configuration
    if [[ -f "$RUNNER_DIR/.runner" ]]; then
        log_success "Runner is configured"
    else
        log_error "Runner configuration not found"
        issues=$((issues + 1))
    fi
    
    # Check Docker access
    if command -v docker > /dev/null 2>&1; then
        if docker ps > /dev/null 2>&1; then
            log_success "Docker access verified"
        else
            log_warning "Docker access failed - check permissions"
            issues=$((issues + 1))
        fi
    else
        log_warning "Docker not installed"
    fi
    
    # Check GPU access (if NVIDIA GPU available)
    if command -v nvidia-smi > /dev/null 2>&1; then
        if nvidia-smi > /dev/null 2>&1; then
            log_success "GPU access verified"
        else
            log_warning "GPU access failed"
            issues=$((issues + 1))
        fi
    fi
    
    # Check disk space
    local disk_usage=$(df "$RUNNER_DIR" 2>/dev/null | tail -1 | awk '{print $5}' | sed 's/%//' || echo "0")
    if [[ "$disk_usage" -lt 90 ]]; then
        log_success "Disk space OK (${disk_usage}% used)"
    else
        log_error "Low disk space (${disk_usage}% used)"
        issues=$((issues + 1))
    fi
    
    # Check memory usage
    local mem_usage=$(free | grep Mem | awk '{printf "%.0f", $3/$2*100}')
    if [[ "$mem_usage" -lt 90 ]]; then
        log_success "Memory usage OK (${mem_usage}% used)"
    else
        log_warning "High memory usage (${mem_usage}% used)"
    fi
    
    echo ""
    if [[ $issues -eq 0 ]]; then
        log_success "All health checks passed! 🎉"
    else
        log_error "$issues issues found"
    fi
    
    return $issues
}

# Performance monitoring
performance_monitor() {
    local duration="${1:-60}"
    log_header "Performance Monitor (${duration}s)"
    
    log_info "Monitoring system performance for ${duration} seconds..."
    
    # Create monitoring script
    local monitor_script=$(mktemp)
    cat << 'EOF' > "$monitor_script"
#!/bin/bash
duration=$1
interval=5
iterations=$((duration / interval))

echo "Timestamp,CPU%,Memory%,Disk%,GPU_Util%,GPU_Mem_Used,GPU_Temp"

for ((i=1; i<=iterations; i++)); do
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # CPU usage
    cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//')
    
    # Memory usage
    mem_usage=$(free | grep Mem | awk '{printf "%.1f", $3/$2*100}')
    
    # Disk usage
    disk_usage=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
    
    # GPU stats (if available)
    if command -v nvidia-smi > /dev/null 2>&1; then
        gpu_stats=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader,nounits | head -1)
        gpu_util=$(echo "$gpu_stats" | cut -d',' -f1 | xargs)
        gpu_mem=$(echo "$gpu_stats" | cut -d',' -f2 | xargs)
        gpu_temp=$(echo "$gpu_stats" | cut -d',' -f3 | xargs)
    else
        gpu_util="N/A"
        gpu_mem="N/A"
        gpu_temp="N/A"
    fi
    
    echo "$timestamp,$cpu_usage,$mem_usage,$disk_usage,$gpu_util,$gpu_mem,$gpu_temp"
    
    if [[ $i -lt $iterations ]]; then
        sleep $interval
    fi
done
EOF
    
    chmod +x "$monitor_script"
    bash "$monitor_script" "$duration"
    rm -f "$monitor_script"
    
    echo ""
    log_success "Performance monitoring complete"
}

# Update runner
update_runner() {
    log_header "Updating GitHub Actions Runner"
    
    local service_name=$(get_service_name)
    
    if [[ -z "$service_name" ]]; then
        log_error "No GitHub runner service found"
        return 1
    fi
    
    # Stop service
    log_info "Stopping runner service..."
    sudo systemctl stop "$service_name"
    
    cd "$RUNNER_DIR"
    
    # Backup current installation
    log_info "Creating backup..."
    local backup_dir="backup-$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$backup_dir"
    cp -r ./* "$backup_dir/" 2>/dev/null || true
    
    # Get latest version
    log_info "Getting latest runner version..."
    local latest_version=$(curl -s https://api.github.com/repos/actions/runner/releases/latest | grep '"tag_name"' | sed -E 's/.*"v([^"]+)".*/\1/')
    
    if [[ -z "$latest_version" ]]; then
        log_error "Could not determine latest runner version"
        sudo systemctl start "$service_name"
        return 1
    fi
    
    log_info "Latest version: $latest_version"
    
    # Download new version
    local runner_package="actions-runner-linux-x64-${latest_version}.tar.gz"
    log_info "Downloading new version..."
    curl -o "$runner_package" -L "https://github.com/actions/runner/releases/download/v${latest_version}/${runner_package}"
    
    # Extract new version
    log_info "Extracting new version..."
    tar xzf "$runner_package" --overwrite
    
    # Install dependencies
    if [[ -f "./bin/installdependencies.sh" ]]; then
        log_info "Installing dependencies..."
        sudo ./bin/installdependencies.sh
    fi
    
    # Start service
    log_info "Starting runner service..."
    sudo systemctl start "$service_name"
    
    sleep 5
    
    if systemctl is-active "$service_name" > /dev/null 2>&1; then
        log_success "Runner updated successfully to version $latest_version"
        
        # Cleanup
        rm -f "$runner_package"
    else
        log_error "Failed to start runner after update"
        log_info "Attempting to restore from backup..."
        
        # Restore backup
        cp -r "$backup_dir"/* ./ 2>/dev/null || true
        sudo systemctl start "$service_name"
        
        if systemctl is-active "$service_name" > /dev/null 2>&1; then
            log_success "Runner restored from backup"
        else
            log_error "Failed to restore runner - manual intervention required"
        fi
        return 1
    fi
}

# Clean up runner data
cleanup() {
    log_header "Runner Cleanup"
    
    local service_name=$(get_service_name)
    
    if [[ -n "$service_name" ]]; then
        log_info "Stopping runner service..."
        sudo systemctl stop "$service_name"
    fi
    
    cd "$RUNNER_DIR"
    
    # Clean work directory
    if [[ -d "_work" ]]; then
        log_info "Cleaning work directory..."
        rm -rf _work/*
        log_success "Work directory cleaned"
    fi
    
    # Clean old logs
    if [[ -d "$LOG_DIR" ]]; then
        log_info "Cleaning old logs..."
        find "$LOG_DIR" -name "*.log" -mtime +7 -delete 2>/dev/null || true
        log_success "Old logs cleaned"
    fi
    
    # Clean Docker (optional)
    read -p "Clean Docker containers and images? (y/N): " clean_docker
    if [[ "$clean_docker" =~ ^[Yy]$ ]]; then
        log_info "Cleaning Docker..."
        docker system prune -f || true
        log_success "Docker cleaned"
    fi
    
    if [[ -n "$service_name" ]]; then
        log_info "Starting runner service..."
        sudo systemctl start "$service_name"
    fi
    
    log_success "Cleanup complete"
}

# Interactive dashboard
dashboard() {
    while true; do
        clear
        echo -e "${CYAN}🎛️  GitHub Actions Runner Dashboard${NC}"
        echo "======================================"
        echo ""
        
        # Quick status
        local service_name=$(get_service_name)
        if [[ -n "$service_name" ]] && systemctl is-active "$service_name" > /dev/null 2>&1; then
            echo -e "Status: ${GREEN}🟢 ONLINE${NC}"
        else
            echo -e "Status: ${RED}🔴 OFFLINE${NC}"
        fi
        
        echo "Runner: $(basename "$RUNNER_DIR")"
        echo "Uptime: $(systemctl show "$service_name" --property=ActiveEnterTimestamp --value 2>/dev/null | xargs -I {} date -d {} '+%Y-%m-%d %H:%M' 2>/dev/null || echo "N/A")"
        echo ""
        
        echo "📋 Available Commands:"
        echo "  1. Show Status"
        echo "  2. Show Logs"
        echo "  3. Health Check"
        echo "  4. Performance Monitor"
        echo "  5. Service Control"
        echo "  6. Update Runner"
        echo "  7. Cleanup"
        echo "  8. Exit"
        echo ""
        
        read -p "Choose option (1-8): " choice
        echo ""
        
        case $choice in
            1) show_status; read -p "Press Enter to continue..."; ;;
            2) show_logs; read -p "Press Enter to continue..."; ;;
            3) health_check; read -p "Press Enter to continue..."; ;;
            4) 
                read -p "Monitor duration in seconds (default 60): " duration
                performance_monitor "${duration:-60}"
                read -p "Press Enter to continue..."
                ;;
            5)
                echo "Service Control:"
                echo "  a. Start    b. Stop    c. Restart"
                echo "  d. Enable   e. Disable"
                read -p "Choose action: " action
                case $action in
                    a) control_service start ;;
                    b) control_service stop ;;
                    c) control_service restart ;;
                    d) control_service enable ;;
                    e) control_service disable ;;
                    *) log_error "Invalid action" ;;
                esac
                read -p "Press Enter to continue..."
                ;;
            6) 
                echo "⚠️  This will update the runner to the latest version."
                read -p "Continue? (y/N): " confirm
                if [[ "$confirm" =~ ^[Yy]$ ]]; then
                    update_runner
                fi
                read -p "Press Enter to continue..."
                ;;
            7)
                echo "⚠️  This will clean work directory and logs."
                read -p "Continue? (y/N): " confirm
                if [[ "$confirm" =~ ^[Yy]$ ]]; then
                    cleanup
                fi
                read -p "Press Enter to continue..."
                ;;
            8) echo "Goodbye!"; exit 0 ;;
            *) log_error "Invalid option"; sleep 1 ;;
        esac
    done
}

# Help function
show_help() {
    echo "GitHub Actions Runner Manager"
    echo ""
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  status                 Show runner status"
    echo "  logs [lines]          Show logs (default: 50 lines)"
    echo "  health                Perform health check"
    echo "  monitor [seconds]     Monitor performance (default: 60s)"
    echo "  start                 Start runner service"
    echo "  stop                  Stop runner service"
    echo "  restart               Restart runner service"
    echo "  enable                Enable auto-start"
    echo "  disable               Disable auto-start"
    echo "  update                Update runner to latest version"
    echo "  cleanup               Clean work directory and logs"
    echo "  dashboard             Interactive dashboard"
    echo "  help                  Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 status             # Show current status"
    echo "  $0 logs 100           # Show last 100 log lines"
    echo "  $0 monitor 300        # Monitor for 5 minutes"
    echo "  $0 dashboard          # Launch interactive dashboard"
}

# Main command handling
main() {
    case "${1:-dashboard}" in
        "status")
            show_status
            ;;
        "logs")
            show_logs "${2:-50}"
            ;;
        "health")
            health_check
            ;;
        "monitor")
            performance_monitor "${2:-60}"
            ;;
        "start"|"stop"|"restart"|"enable"|"disable")
            control_service "$1"
            ;;
        "update")
            update_runner
            ;;
        "cleanup")
            cleanup
            ;;
        "dashboard")
            dashboard
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            log_error "Unknown command: $1"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Handle interrupts gracefully
trap 'echo ""; log_warning "Operation interrupted"; exit 130' INT

main "$@"