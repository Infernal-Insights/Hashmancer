#!/bin/bash

# Hashmancer Advanced Monitoring Script
# This script provides comprehensive monitoring and automatic recovery

set -euo pipefail

# Configuration
# Auto-detect the Hashmancer directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HASHMANCER_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$HASHMANCER_DIR/logs"
MONITOR_LOG="$LOG_DIR/monitor.log"
STATUS_FILE="/tmp/hashmancer_status"
ALERT_EMAIL=""  # Set email for alerts if desired
WEBHOOK_URL=""  # Set webhook URL for alerts if desired

# Monitoring thresholds
MAX_CPU_PERCENT=90
MAX_MEMORY_PERCENT=85
MAX_DISK_PERCENT=90
MIN_FREE_DISK_GB=5
MAX_RESPONSE_TIME=10

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging function
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${timestamp} [$level] $message" | tee -a "$MONITOR_LOG"
}

log_info() { log "INFO" "$@"; }
log_warn() { log "WARN" "$@"; }
log_error() { log "ERROR" "$@"; }
log_success() { log "SUCCESS" "$@"; }

# Send alert function
send_alert() {
    local subject="$1"
    local message="$2"
    
    # Email alert
    if [[ -n "$ALERT_EMAIL" ]] && command -v mail &> /dev/null; then
        echo "$message" | mail -s "Hashmancer Alert: $subject" "$ALERT_EMAIL"
    fi
    
    # Webhook alert
    if [[ -n "$WEBHOOK_URL" ]] && command -v curl &> /dev/null; then
        curl -X POST "$WEBHOOK_URL" \
             -H "Content-Type: application/json" \
             -d "{\"text\":\"Hashmancer Alert: $subject\\n$message\"}" \
             2>/dev/null || true
    fi
    
    # System notification
    if command -v notify-send &> /dev/null; then
        notify-send "Hashmancer Alert" "$subject: $message" 2>/dev/null || true
    fi
    
    log_warn "ALERT: $subject - $message"
}

# Check service status
check_service_status() {
    if systemctl is-active --quiet hashmancer; then
        echo "running"
    elif systemctl is-failed --quiet hashmancer; then
        echo "failed"
    else
        echo "stopped"
    fi
}

# Check portal connectivity
check_portal_connectivity() {
    local start_time=$(date +%s%N)
    
    if curl -f -s --max-time "$MAX_RESPONSE_TIME" http://localhost:8000/server_status > /dev/null 2>&1; then
        local end_time=$(date +%s%N)
        local response_time=$(( (end_time - start_time) / 1000000 ))  # Convert to milliseconds
        echo "ok:$response_time"
    else
        echo "error"
    fi
}

# Check system resources
check_system_resources() {
    local issues=()
    
    # CPU usage
    local cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//')
    cpu_usage=${cpu_usage%.*}  # Remove decimal part
    if [[ $cpu_usage -gt $MAX_CPU_PERCENT ]]; then
        issues+=("High CPU usage: ${cpu_usage}%")
    fi
    
    # Memory usage
    local memory_info=$(free | awk 'NR==2{printf "%.0f", $3*100/$2}')
    if [[ $memory_info -gt $MAX_MEMORY_PERCENT ]]; then
        issues+=("High memory usage: ${memory_info}%")
    fi
    
    # Disk usage
    local disk_usage=$(df "$HASHMANCER_DIR" | awk 'NR==2{print $5}' | sed 's/%//')
    local free_space_kb=$(df "$HASHMANCER_DIR" | awk 'NR==2{print $4}')
    local free_space_gb=$((free_space_kb / 1024 / 1024))
    
    if [[ $disk_usage -gt $MAX_DISK_PERCENT ]] || [[ $free_space_gb -lt $MIN_FREE_DISK_GB ]]; then
        issues+=("Low disk space: ${disk_usage}% used, ${free_space_gb}GB free")
    fi
    
    # Check for zombie processes
    local zombie_count=$(ps aux | awk '$8 ~ /^Z/ { count++ } END { print count+0 }')
    if [[ $zombie_count -gt 0 ]]; then
        issues+=("Zombie processes detected: $zombie_count")
    fi
    
    # Return issues as comma-separated string
    if [[ ${#issues[@]} -gt 0 ]]; then
        printf '%s\n' "${issues[@]}" | paste -sd '|'
    else
        echo "ok"
    fi
}

# Check log errors
check_log_errors() {
    local error_count=0
    local recent_logs="$LOG_DIR/startup.log"
    
    if [[ -f "$recent_logs" ]]; then
        # Count errors in last 10 minutes
        error_count=$(grep -c "ERROR" "$recent_logs" 2>/dev/null | tail -n 100 | grep "$(date -d '10 minutes ago' '+%Y-%m-%d %H:')" | wc -l || echo 0)
    fi
    
    echo "$error_count"
}

# Perform health check
perform_health_check() {
    local service_status
    local portal_status
    local resource_status
    local error_count
    
    log_info "Performing health check..."
    
    # Check service
    service_status=$(check_service_status)
    log_info "Service status: $service_status"
    
    # Check portal
    portal_status=$(check_portal_connectivity)
    log_info "Portal status: $portal_status"
    
    # Check resources
    resource_status=$(check_system_resources)
    log_info "Resource status: $resource_status"
    
    # Check for errors
    error_count=$(check_log_errors)
    log_info "Recent errors: $error_count"
    
    # Store status
    cat > "$STATUS_FILE" << EOF
service_status=$service_status
portal_status=$portal_status
resource_status=$resource_status
error_count=$error_count
last_check=$(date '+%Y-%m-%d %H:%M:%S')
EOF
    
    # Evaluate health and take action
    local action_taken=false
    
    # Service not running
    if [[ "$service_status" != "running" ]]; then
        log_error "Service is not running, attempting to start..."
        systemctl start hashmancer
        send_alert "Service Down" "Hashmancer service was not running and has been restarted"
        action_taken=true
    fi
    
    # Portal not responding
    if [[ "$portal_status" == "error" ]] && [[ "$service_status" == "running" ]]; then
        log_error "Portal not responding, restarting service..."
        systemctl restart hashmancer
        send_alert "Portal Unresponsive" "Hashmancer portal was not responding and service has been restarted"
        action_taken=true
    fi
    
    # High response time
    if [[ "$portal_status" =~ ^ok:([0-9]+)$ ]]; then
        local response_time="${BASH_REMATCH[1]}"
        if [[ $response_time -gt $((MAX_RESPONSE_TIME * 1000)) ]]; then
            log_warn "High response time: ${response_time}ms"
            send_alert "High Response Time" "Portal response time is ${response_time}ms (threshold: $((MAX_RESPONSE_TIME * 1000))ms)"
        fi
    fi
    
    # Resource issues
    if [[ "$resource_status" != "ok" ]]; then
        log_warn "Resource issues detected: $resource_status"
        send_alert "Resource Issues" "$resource_status"
    fi
    
    # High error rate
    if [[ $error_count -gt 5 ]]; then
        log_warn "High error rate: $error_count errors in last 10 minutes"
        send_alert "High Error Rate" "$error_count errors detected in the last 10 minutes"
    fi
    
    if [[ "$action_taken" == false ]]; then
        log_success "Health check passed"
    fi
}

# Cleanup old logs
cleanup_logs() {
    local days_to_keep=30
    
    log_info "Cleaning up old logs (keeping last $days_to_keep days)..."
    
    find "$LOG_DIR" -name "*.log" -mtime +$days_to_keep -delete 2>/dev/null || true
    
    # Truncate large log files
    for log_file in "$LOG_DIR"/*.log; do
        if [[ -f "$log_file" ]] && [[ $(stat -c%s "$log_file") -gt 104857600 ]]; then  # 100MB
            tail -n 1000 "$log_file" > "${log_file}.tmp"
            mv "${log_file}.tmp" "$log_file"
            log_info "Truncated large log file: $log_file"
        fi
    done
}

# Generate health report
generate_health_report() {
    echo "=== Hashmancer Health Report ==="
    echo "Generated: $(date)"
    echo ""
    
    if [[ -f "$STATUS_FILE" ]]; then
        source "$STATUS_FILE"
        echo "Last Check: $last_check"
        echo "Service Status: $service_status"
        echo "Portal Status: $portal_status"
        echo "Resource Status: $resource_status"
        echo "Recent Errors: $error_count"
    else
        echo "No status file found"
    fi
    
    echo ""
    echo "=== System Information ==="
    echo "Uptime: $(uptime)"
    echo "Load Average: $(uptime | awk -F'load average:' '{print $2}')"
    echo "Memory Usage: $(free -h | awk 'NR==2{printf "Used: %s/%s (%.2f%%)", $3, $2, $3*100/$2}')"
    echo "Disk Usage: $(df -h "$HASHMANCER_DIR" | awk 'NR==2{printf "%s used, %s available (%s full)", $3, $4, $5}')"
    echo ""
    
    echo "=== Service Information ==="
    systemctl status hashmancer --no-pager -l || true
    echo ""
    
    echo "=== Recent Logs ==="
    journalctl -u hashmancer --no-pager -n 10 || true
}

# Main execution
main() {
    local mode="${1:-check}"
    
    case "$mode" in
        "check")
            perform_health_check
            ;;
        "report")
            generate_health_report
            ;;
        "cleanup")
            cleanup_logs
            ;;
        "status")
            if [[ -f "$STATUS_FILE" ]]; then
                cat "$STATUS_FILE"
            else
                echo "No status available"
            fi
            ;;
        *)
            echo "Usage: $0 {check|report|cleanup|status}"
            echo "  check   - Perform health check and auto-recovery"
            echo "  report  - Generate detailed health report"
            echo "  cleanup - Clean up old logs"
            echo "  status  - Show current status"
            exit 1
            ;;
    esac
}

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Execute main function
main "$@"