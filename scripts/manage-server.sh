#!/bin/bash
# Hashmancer Server Management Script for DigitalOcean

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }
info() { echo -e "${BLUE}[INFO]${NC} $1"; }

# Configuration
HASHMANCER_DIR="/opt/hashmancer"
SERVICE_NAME="hashmancer"

# Function to show usage
usage() {
    echo "🔓 Hashmancer Server Management"
    echo "Usage: $0 {start|stop|restart|status|logs|update|backup|workers|health}"
    echo ""
    echo "Commands:"
    echo "  start     - Start the Hashmancer server"
    echo "  stop      - Stop the Hashmancer server"
    echo "  restart   - Restart the Hashmancer server"
    echo "  status    - Show server status"
    echo "  logs      - Show server logs (follow with -f)"
    echo "  update    - Update server software"
    echo "  backup    - Create backup of server data"
    echo "  workers   - Show connected workers"
    echo "  health    - Check server health"
    echo "  monitor   - Monitor server in real-time"
}

# Function to check if running as root/sudo
check_permissions() {
    if [ "$EUID" -ne 0 ]; then
        error "This script requires sudo privileges"
        error "Run: sudo $0 $1"
        exit 1
    fi
}

# Function to start server
start_server() {
    log "🚀 Starting Hashmancer server..."
    
    systemctl start $SERVICE_NAME
    systemctl start redis-server
    systemctl start nginx
    
    sleep 3
    
    if systemctl is-active --quiet $SERVICE_NAME; then
        log "✅ Hashmancer server started successfully"
        show_access_info
    else
        error "❌ Failed to start Hashmancer server"
        systemctl status $SERVICE_NAME
        exit 1
    fi
}

# Function to stop server
stop_server() {
    log "🛑 Stopping Hashmancer server..."
    
    systemctl stop $SERVICE_NAME
    
    if systemctl is-active --quiet $SERVICE_NAME; then
        error "❌ Failed to stop Hashmancer server"
        exit 1
    else
        log "✅ Hashmancer server stopped"
    fi
}

# Function to restart server
restart_server() {
    log "🔄 Restarting Hashmancer server..."
    
    systemctl restart $SERVICE_NAME
    systemctl restart redis-server
    
    sleep 3
    
    if systemctl is-active --quiet $SERVICE_NAME; then
        log "✅ Hashmancer server restarted successfully"
        show_access_info
    else
        error "❌ Failed to restart Hashmancer server"
        systemctl status $SERVICE_NAME
        exit 1
    fi
}

# Function to show status
show_status() {
    log "📊 Hashmancer Server Status"
    log "=========================="
    
    # Service status
    if systemctl is-active --quiet $SERVICE_NAME; then
        info "🟢 Hashmancer: Running"
    else
        warn "🔴 Hashmancer: Stopped"
    fi
    
    if systemctl is-active --quiet redis-server; then
        info "🟢 Redis: Running"
    else
        warn "🔴 Redis: Stopped"
    fi
    
    if systemctl is-active --quiet nginx; then
        info "🟢 Nginx: Running"
    else
        warn "🔴 Nginx: Stopped"
    fi
    
    # System resources
    echo ""
    info "💻 System Resources:"
    info "   CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | awk -F'%' '{print $1}')% used"
    info "   Memory: $(free | grep Mem | awk '{printf("%.1f%% used", $3/$2 * 100.0)}')"
    info "   Disk: $(df -h / | awk 'NR==2{printf "%s used", $5}')"
    
    # Network info
    PUBLIC_IP=$(curl -s http://ifconfig.me 2>/dev/null || echo "unknown")
    echo ""
    info "🌐 Network:"
    info "   Public IP: $PUBLIC_IP"
    info "   Server Port: 8080"
    
    # Check connectivity
    if curl -s http://localhost:8080/health > /dev/null 2>&1; then
        info "✅ Health check: Passed"
    else
        warn "❌ Health check: Failed"
    fi
}

# Function to show logs
show_logs() {
    if [ "$2" = "-f" ]; then
        log "📋 Following Hashmancer logs (Ctrl+C to stop)..."
        journalctl -u $SERVICE_NAME -f
    else
        log "📋 Recent Hashmancer logs:"
        journalctl -u $SERVICE_NAME -n 50 --no-pager
    fi
}

# Function to update server
update_server() {
    log "🔄 Updating Hashmancer server..."
    
    # Update system packages
    apt-get update -qq
    apt-get upgrade -y
    
    # Restart services
    systemctl restart $SERVICE_NAME
    
    log "✅ Server updated successfully"
}

# Function to backup server
backup_server() {
    log "💾 Creating server backup..."
    
    BACKUP_DIR="/var/backups/hashmancer"
    BACKUP_FILE="hashmancer-backup-$(date +%Y%m%d-%H%M%S).tar.gz"
    
    mkdir -p "$BACKUP_DIR"
    
    # Create backup
    tar -czf "$BACKUP_DIR/$BACKUP_FILE" \
        -C /opt hashmancer \
        -C /etc systemd/system/hashmancer.service \
        -C /etc hashmancer \
        -C /var/log hashmancer
    
    log "✅ Backup created: $BACKUP_DIR/$BACKUP_FILE"
    
    # Keep only last 5 backups
    cd "$BACKUP_DIR"
    ls -t hashmancer-backup-*.tar.gz | tail -n +6 | xargs -r rm
    
    info "📊 Available backups:"
    ls -lah hashmancer-backup-*.tar.gz 2>/dev/null || echo "   No backups found"
}

# Function to show workers
show_workers() {
    log "👷 Connected Workers"
    log "=================="
    
    # Get workers from API
    if curl -s http://localhost:8080/workers > /dev/null 2>&1; then
        WORKERS=$(curl -s http://localhost:8080/workers)
        
        if [ "$WORKERS" = "[]" ]; then
            warn "No workers currently connected"
        else
            echo "$WORKERS" | python3 -m json.tool
        fi
    else
        error "Unable to connect to server API"
    fi
}

# Function to check health
check_health() {
    log "🏥 Health Check"
    log "=============="
    
    # Check local health
    if curl -s http://localhost:8080/health > /dev/null; then
        HEALTH=$(curl -s http://localhost:8080/health)
        echo "$HEALTH" | python3 -m json.tool
        log "✅ Local health check passed"
    else
        error "❌ Local health check failed"
    fi
    
    # Check public health
    PUBLIC_IP=$(curl -s http://ifconfig.me 2>/dev/null || echo "unknown")
    if [ "$PUBLIC_IP" != "unknown" ]; then
        if curl -s "http://$PUBLIC_IP:8080/health" > /dev/null; then
            log "✅ Public health check passed"
        else
            warn "⚠️  Public health check failed"
        fi
    fi
}

# Function to monitor server
monitor_server() {
    log "📊 Real-time Server Monitor (Ctrl+C to stop)"
    log "==========================================="
    
    while true; do
        clear
        echo "🔓 Hashmancer Server Monitor - $(date)"
        echo "======================================"
        
        # Service status
        if systemctl is-active --quiet $SERVICE_NAME; then
            echo "🟢 Hashmancer: Running"
        else
            echo "🔴 Hashmancer: Stopped"
        fi
        
        # System stats
        echo ""
        echo "💻 System Resources:"
        echo "   CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | awk -F'%' '{print $1}')%"
        echo "   Memory: $(free | grep Mem | awk '{printf("%.1f%%", $3/$2 * 100.0)}')"
        echo "   Load: $(uptime | awk -F'load average:' '{print $2}')"
        
        # Worker count
        if curl -s http://localhost:8080/workers > /dev/null 2>&1; then
            WORKER_COUNT=$(curl -s http://localhost:8080/workers | python3 -c "import sys, json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo "0")
            echo "   Workers: $WORKER_COUNT connected"
        fi
        
        # Recent log entries
        echo ""
        echo "📋 Recent Logs:"
        journalctl -u $SERVICE_NAME -n 5 --no-pager --output=short | tail -n 5
        
        sleep 5
    done
}

# Function to show access info
show_access_info() {
    PUBLIC_IP=$(curl -s http://ifconfig.me 2>/dev/null || echo "unknown")
    
    info "🌐 Server Access Information:"
    info "   API Endpoint: http://$PUBLIC_IP:8080"
    info "   Health Check: http://$PUBLIC_IP:8080/health"
    info "   Workers List: http://$PUBLIC_IP:8080/workers"
    info ""
    info "🎯 For Vast.ai Workers:"
    info "   Set HASHMANCER_SERVER_IP=$PUBLIC_IP"
}

# Main script logic
case "$1" in
    start)
        check_permissions
        start_server
        ;;
    stop)
        check_permissions
        stop_server
        ;;
    restart)
        check_permissions
        restart_server
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs "$@"
        ;;
    update)
        check_permissions
        update_server
        ;;
    backup)
        check_permissions
        backup_server
        ;;
    workers)
        show_workers
        ;;
    health)
        check_health
        ;;
    monitor)
        monitor_server
        ;;
    *)
        usage
        exit 1
        ;;
esac