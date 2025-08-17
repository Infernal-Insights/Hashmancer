#!/bin/bash

# Hashmancer Auto-Start Installation Script
# Run this script once to set up automatic startup of Hashmancer

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration - Auto-detect the Hashmancer directory
HASHMANCER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_NAME="hashmancer"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
SCRIPT_DIR="$HASHMANCER_DIR/scripts"

# Helper functions
print_header() {
    echo -e "${CYAN}"
    echo "=================================="
    echo "  HASHMANCER AUTO-START SETUP"
    echo "=================================="
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

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        print_error "This script must be run as root (use sudo)"
        print_info "Example: sudo ./install-service.sh"
        exit 1
    fi
}

# Install system dependencies
install_dependencies() {
    print_step "Installing system dependencies..."
    
    # Update package list
    apt-get update -qq
    
    # Install essential packages
    local packages=(
        "python3"
        "python3-pip" 
        "python3-venv"
        "redis-server"
        "curl"
        "wget"
        "htop"
        "systemd"
        "logrotate"
    )
    
    for package in "${packages[@]}"; do
        if ! dpkg -l | grep -q "^ii  $package "; then
            print_info "Installing $package..."
            apt-get install -y "$package"
        else
            print_info "$package is already installed"
        fi
    done
    
    # Ensure Redis is enabled and started
    systemctl enable redis-server
    systemctl start redis-server
    
    print_success "System dependencies installed"
}

# Make scripts executable
setup_scripts() {
    print_step "Setting up scripts..."
    
    chmod +x "$SCRIPT_DIR/start-hashmancer.sh"
    chmod +x "$SCRIPT_DIR/pre-start-check.sh"
    
    # Create logs directory
    mkdir -p "$HASHMANCER_DIR/logs"
    chmod 755 "$HASHMANCER_DIR/logs"
    
    print_success "Scripts configured"
}

# Install systemd service
install_systemd_service() {
    print_step "Installing systemd service..."
    
    # Replace placeholder with actual path and copy service file
    sed "s|HASHMANCER_DIR_PLACEHOLDER|$HASHMANCER_DIR|g" "$HASHMANCER_DIR/${SERVICE_NAME}.service" > "$SERVICE_FILE"
    
    # Set proper permissions
    chmod 644 "$SERVICE_FILE"
    
    # Reload systemd daemon
    systemctl daemon-reload
    
    # Enable the service
    systemctl enable "$SERVICE_NAME"
    
    print_success "Systemd service installed and enabled"
}

# Setup log rotation
setup_log_rotation() {
    print_step "Setting up log rotation..."
    
    cat > "/etc/logrotate.d/hashmancer" << EOF
$HASHMANCER_DIR/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 root root
    postrotate
        systemctl reload hashmancer || true
    endscript
}
EOF
    
    print_success "Log rotation configured"
}

# Setup firewall rules (if UFW is installed)
setup_firewall() {
    if command -v ufw &> /dev/null; then
        print_step "Configuring firewall..."
        
        # Allow SSH (important!)
        ufw allow ssh
        
        # Allow Hashmancer port
        ufw allow 8000/tcp comment 'Hashmancer Portal'
        
        # Enable firewall if not already enabled
        if ! ufw status | grep -q "Status: active"; then
            print_warning "UFW firewall will be enabled. Make sure SSH access is properly configured!"
            read -p "Continue? (y/N): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                ufw --force enable
            fi
        fi
        
        print_success "Firewall configured"
    else
        print_info "UFW not installed, skipping firewall configuration"
    fi
}

# Create health check script
create_health_check() {
    print_step "Creating health check script..."
    
    cat > "$SCRIPT_DIR/health-check.sh" << 'EOF'
#!/bin/bash

# Hashmancer Health Check Script
# This script monitors the Hashmancer service and restarts it if needed

HASHMANCER_URL="http://localhost:8000"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HASHMANCER_DIR="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$HASHMANCER_DIR/logs/health-check.log"
MAX_FAILURES=3
FAILURE_COUNT_FILE="/tmp/hashmancer_failures"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [HEALTH] $*" | tee -a "$LOG_FILE"
}

# Get current failure count
get_failure_count() {
    if [[ -f "$FAILURE_COUNT_FILE" ]]; then
        cat "$FAILURE_COUNT_FILE"
    else
        echo 0
    fi
}

# Set failure count
set_failure_count() {
    echo "$1" > "$FAILURE_COUNT_FILE"
}

# Reset failure count
reset_failure_count() {
    rm -f "$FAILURE_COUNT_FILE"
}

# Check if service is running
if ! systemctl is-active --quiet hashmancer; then
    log "Service is not running, attempting to start..."
    systemctl start hashmancer
    exit 0
fi

# Check if portal is responding
if curl -f -s --max-time 10 "$HASHMANCER_URL/server_status" > /dev/null 2>&1; then
    log "Health check passed"
    reset_failure_count
    exit 0
else
    # Health check failed
    current_failures=$(get_failure_count)
    new_failures=$((current_failures + 1))
    set_failure_count "$new_failures"
    
    log "Health check failed (attempt $new_failures/$MAX_FAILURES)"
    
    if [[ $new_failures -ge $MAX_FAILURES ]]; then
        log "Max failures reached, restarting service..."
        systemctl restart hashmancer
        reset_failure_count
    fi
    
    exit 1
fi
EOF
    
    chmod +x "$SCRIPT_DIR/health-check.sh"
    
    # Add cron job for health checks
    (crontab -l 2>/dev/null; echo "*/2 * * * * $SCRIPT_DIR/health-check.sh --quiet") | crontab -
    
    print_success "Health check configured"
}

# Create maintenance scripts
create_maintenance_scripts() {
    print_step "Creating maintenance scripts..."
    
    # Stop script
    cat > "$SCRIPT_DIR/stop-hashmancer.sh" << 'EOF'
#!/bin/bash
echo "Stopping Hashmancer service..."
systemctl stop hashmancer
echo "Hashmancer stopped"
EOF
    
    # Restart script
    cat > "$SCRIPT_DIR/restart-hashmancer.sh" << 'EOF'
#!/bin/bash
echo "Restarting Hashmancer service..."
systemctl restart hashmancer
echo "Hashmancer restarted"
EOF
    
    # Status script
    cat > "$SCRIPT_DIR/status-hashmancer.sh" << 'EOF'
#!/bin/bash
echo "=== Hashmancer Service Status ==="
systemctl status hashmancer --no-pager
echo ""
echo "=== Portal Accessibility ==="
if curl -f -s --max-time 5 http://localhost:8000/server_status > /dev/null; then
    echo "✅ Portal is accessible at http://localhost:8000"
else
    echo "❌ Portal is not accessible"
fi
echo ""
echo "=== Recent Logs ==="
journalctl -u hashmancer --no-pager -n 10
EOF
    
    chmod +x "$SCRIPT_DIR"/*.sh
    
    print_success "Maintenance scripts created"
}

# Setup boot-time network wait
setup_network_wait() {
    print_step "Setting up network dependency..."
    
    # Ensure systemd-networkd-wait-online is enabled
    systemctl enable systemd-networkd-wait-online.service 2>/dev/null || true
    
    print_success "Network dependency configured"
}

# Test the installation
test_installation() {
    print_step "Testing installation..."
    
    # Start the service
    systemctl start "$SERVICE_NAME"
    
    # Wait a moment
    sleep 10
    
    # Check service status
    if systemctl is-active --quiet "$SERVICE_NAME"; then
        print_success "Service is running"
    else
        print_error "Service failed to start"
        print_info "Check logs with: journalctl -u $SERVICE_NAME"
        return 1
    fi
    
    # Test portal accessibility
    local max_attempts=20
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -f -s --max-time 5 http://localhost:8000/server_status > /dev/null 2>&1; then
            print_success "Portal is accessible at http://localhost:8000"
            return 0
        fi
        
        print_info "Waiting for portal... (attempt $attempt/$max_attempts)"
        sleep 3
        ((attempt++))
    done
    
    print_warning "Portal may not be fully ready yet. Check logs if needed."
    return 0
}

# Create desktop shortcut (if GUI is available)
create_desktop_shortcut() {
    if [[ -n "${DISPLAY:-}" ]] && command -v xdg-user-dir &> /dev/null; then
        print_step "Creating desktop shortcut..."
        
        local desktop_dir
        desktop_dir=$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")
        
        if [[ -d "$desktop_dir" ]]; then
            cat > "$desktop_dir/Hashmancer Portal.desktop" << 'EOF'
[Desktop Entry]
Version=1.0
Type=Application
Name=Hashmancer Portal
Comment=Open Hashmancer Hash Cracking Portal
Exec=xdg-open http://localhost:8000
Icon=applications-internet
Terminal=false
Categories=Network;Security;
EOF
            
            chmod +x "$desktop_dir/Hashmancer Portal.desktop"
            print_success "Desktop shortcut created"
        fi
    fi
}

# Main installation function
main() {
    print_header
    
    print_info "This script will set up Hashmancer to start automatically on boot"
    print_info "and ensure it stays running with health monitoring."
    echo ""
    
    # Confirmation
    read -p "Continue with installation? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Installation cancelled"
        exit 0
    fi
    
    check_root
    install_dependencies
    setup_scripts
    install_systemd_service
    setup_log_rotation
    setup_firewall
    create_health_check
    create_maintenance_scripts
    setup_network_wait
    test_installation
    create_desktop_shortcut
    
    echo ""
    print_header
    print_success "🎉 INSTALLATION COMPLETE! 🎉"
    echo ""
    print_info "Hashmancer is now configured to:"
    echo "  ✅ Start automatically on boot"
    echo "  ✅ Restart automatically if it crashes"
    echo "  ✅ Monitor health every 2 minutes"
    echo "  ✅ Rotate logs automatically"
    echo "  ✅ Handle network dependencies"
    echo ""
    print_info "Portal will be available at: http://localhost:8000"
    print_info "Admin panel available at: http://localhost:8000/admin"
    echo ""
    print_info "Useful commands:"
    echo "  • Check status: $SCRIPT_DIR/status-hashmancer.sh"
    echo "  • Restart service: $SCRIPT_DIR/restart-hashmancer.sh"
    echo "  • Stop service: $SCRIPT_DIR/stop-hashmancer.sh"
    echo "  • View logs: journalctl -u hashmancer -f"
    echo ""
    print_warning "The portal should be accessible after the next reboot, or you can"
    print_warning "start it now by running: $SCRIPT_DIR/restart-hashmancer.sh"
    echo ""
}

# Execute main function
main "$@"