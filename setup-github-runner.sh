#!/bin/bash
set -e

# GitHub Self-Hosted Runner Setup for Hashmancer
# This script sets up a self-hosted GitHub Actions runner with GPU support

echo "🚀 Setting up GitHub Self-Hosted Runner for Hashmancer"
echo "====================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Configuration
RUNNER_DIR="$HOME/github-runner"
SERVICE_NAME="github-runner"
RUNNER_USER=$(whoami)

# Collect GitHub information
collect_github_info() {
    echo ""
    log_info "GitHub Repository Information"
    echo "You'll need to get this information from:"
    echo "GitHub Repository → Settings → Actions → Runners → New self-hosted runner"
    echo ""
    
    if [[ -z "$GITHUB_REPO_URL" ]]; then
        read -p "Enter your GitHub repository URL (e.g., https://github.com/username/hashmancer): " GITHUB_REPO_URL
    fi
    
    if [[ -z "$GITHUB_TOKEN" ]]; then
        echo ""
        log_warning "You need to get the registration token from GitHub:"
        echo "1. Go to: $GITHUB_REPO_URL/settings/actions/runners"
        echo "2. Click 'New self-hosted runner'"
        echo "3. Select 'Linux' and copy the token from the './config.sh' command"
        echo ""
        read -p "Enter the GitHub registration token: " GITHUB_TOKEN
    fi
    
    if [[ -z "$RUNNER_NAME" ]]; then
        RUNNER_NAME="hashmancer-$(hostname)"
        log_info "Using runner name: $RUNNER_NAME"
    fi
    
    # Default labels for our GPU-equipped system
    if [[ -z "$RUNNER_LABELS" ]]; then
        RUNNER_LABELS="self-hosted,linux,x64"
        
        # Add GPU label if NVIDIA GPU is detected
        if command -v nvidia-smi &> /dev/null; then
            RUNNER_LABELS="$RUNNER_LABELS,gpu,cuda"
            log_success "NVIDIA GPU detected - adding GPU labels"
        fi
        
        # Add Docker label
        if command -v docker &> /dev/null; then
            RUNNER_LABELS="$RUNNER_LABELS,docker"
            log_success "Docker detected - adding Docker label"
        fi
        
        log_info "Using labels: $RUNNER_LABELS"
    fi
}

# Check system requirements
check_requirements() {
    log_info "Checking system requirements..."
    
    # Check for required tools
    local missing_tools=()
    
    if ! command -v curl &> /dev/null; then
        missing_tools+=("curl")
    fi
    
    if ! command -v tar &> /dev/null; then
        missing_tools+=("tar")
    fi
    
    if ! command -v docker &> /dev/null; then
        log_warning "Docker not found - some tests may not work"
    fi
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log_info "Installing missing tools..."
        
        if command -v apt-get &> /dev/null; then
            sudo apt-get update
            sudo apt-get install -y "${missing_tools[@]}"
        elif command -v yum &> /dev/null; then
            sudo yum install -y "${missing_tools[@]}"
        else
            log_error "Cannot install missing tools automatically. Please install: ${missing_tools[*]}"
            exit 1
        fi
    fi
    
    log_success "System requirements met"
}

# Download and setup GitHub runner
setup_runner() {
    log_info "Setting up GitHub Actions runner..."
    
    # Create runner directory
    mkdir -p "$RUNNER_DIR"
    cd "$RUNNER_DIR"
    
    # Get the latest runner version
    log_info "Getting latest GitHub Actions runner version..."
    local latest_version=$(curl -s https://api.github.com/repos/actions/runner/releases/latest | grep '"tag_name"' | sed -E 's/.*"v([^"]+)".*/\1/')
    
    if [[ -z "$latest_version" ]]; then
        log_error "Could not determine latest runner version"
        exit 1
    fi
    
    log_info "Latest runner version: $latest_version"
    
    # Download runner if not already present
    local runner_package="actions-runner-linux-x64-${latest_version}.tar.gz"
    
    if [[ ! -f "$runner_package" ]]; then
        log_info "Downloading GitHub Actions runner..."
        curl -o "$runner_package" -L "https://github.com/actions/runner/releases/download/v${latest_version}/${runner_package}"
    else
        log_info "Runner package already exists"
    fi
    
    # Extract runner if not already extracted
    if [[ ! -f "./config.sh" ]]; then
        log_info "Extracting runner..."
        tar xzf "$runner_package"
    else
        log_info "Runner already extracted"
    fi
    
    # Install dependencies
    if [[ -f "./bin/installdependencies.sh" ]]; then
        log_info "Installing runner dependencies..."
        sudo ./bin/installdependencies.sh
    fi
    
    log_success "Runner setup complete"
}

# Configure the runner
configure_runner() {
    log_info "Configuring GitHub Actions runner..."
    
    cd "$RUNNER_DIR"
    
    # Check if already configured
    if [[ -f ".runner" ]]; then
        log_warning "Runner appears to be already configured"
        read -p "Do you want to reconfigure? (y/N): " reconfigure
        if [[ "$reconfigure" =~ ^[Yy]$ ]]; then
            log_info "Removing existing configuration..."
            ./config.sh remove --token "$GITHUB_TOKEN" || true
        else
            log_info "Skipping configuration"
            return 0
        fi
    fi
    
    # Configure runner
    log_info "Configuring runner with GitHub..."
    ./config.sh \
        --url "$GITHUB_REPO_URL" \
        --token "$GITHUB_TOKEN" \
        --name "$RUNNER_NAME" \
        --labels "$RUNNER_LABELS" \
        --work "_work" \
        --unattended \
        --replace
    
    log_success "Runner configured successfully"
}

# Install runner as a system service
install_service() {
    log_info "Installing runner as a system service..."
    
    cd "$RUNNER_DIR"
    
    # Install service
    sudo ./svc.sh install "$RUNNER_USER"
    
    # Create systemd service override for better integration
    local service_override_dir="/etc/systemd/system/actions.runner.$(basename "$GITHUB_REPO_URL").${RUNNER_NAME}.service.d"
    sudo mkdir -p "$service_override_dir"
    
    cat << EOF | sudo tee "$service_override_dir/override.conf" > /dev/null
[Unit]
Description=GitHub Actions Runner for Hashmancer ($RUNNER_NAME)
After=network.target docker.service
Wants=docker.service

[Service]
Environment=PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/cuda/bin
Environment=LD_LIBRARY_PATH=/usr/local/cuda/lib64
Environment=CUDA_HOME=/usr/local/cuda
Environment=NVIDIA_VISIBLE_DEVICES=all
Environment=NVIDIA_DRIVER_CAPABILITIES=compute,utility
Environment=DOCKER_HOST=unix:///var/run/docker.sock
Restart=always
RestartSec=10
KillMode=process

[Install]
WantedBy=multi-user.target
EOF
    
    # Reload systemd and start service
    sudo systemctl daemon-reload
    sudo ./svc.sh start
    
    # Enable service to start on boot
    sudo systemctl enable "actions.runner.$(basename "$GITHUB_REPO_URL").${RUNNER_NAME}.service"
    
    log_success "Runner service installed and started"
}

# Test runner functionality
test_runner() {
    log_info "Testing runner functionality..."
    
    # Test basic runner status
    cd "$RUNNER_DIR"
    
    # Check service status
    local service_name="actions.runner.$(basename "$GITHUB_REPO_URL").${RUNNER_NAME}.service"
    if sudo systemctl is-active "$service_name" &> /dev/null; then
        log_success "Runner service is active"
    else
        log_error "Runner service is not active"
        return 1
    fi
    
    # Test Docker access (if available)
    if command -v docker &> /dev/null; then
        if docker ps &> /dev/null; then
            log_success "Docker access verified"
        else
            log_warning "Docker access failed - may need to add user to docker group"
        fi
    fi
    
    # Test GPU access (if available)
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            log_success "GPU access verified"
        else
            log_warning "GPU access failed"
        fi
    fi
    
    # Test our deployment tools
    if [[ -f "$(dirname "$RUNNER_DIR")/hashmancer/redis_tool.py" ]]; then
        if python3 "$(dirname "$RUNNER_DIR")/hashmancer/redis_tool.py" --help &> /dev/null; then
            log_success "Hashmancer tools accessible"
        else
            log_warning "Hashmancer tools not accessible from runner"
        fi
    fi
    
    log_success "Runner functionality tests completed"
}

# Setup monitoring and logging
setup_monitoring() {
    log_info "Setting up runner monitoring..."
    
    # Create log directory
    sudo mkdir -p /var/log/github-runner
    sudo chown "$RUNNER_USER:$RUNNER_USER" /var/log/github-runner
    
    # Create monitoring script
    cat << 'EOF' > "$RUNNER_DIR/monitor-runner.sh"
#!/bin/bash
# GitHub Runner monitoring script

RUNNER_DIR="$HOME/github-runner"
LOG_FILE="/var/log/github-runner/monitor.log"

log_with_timestamp() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

# Check runner service status
SERVICE_NAME=$(systemctl list-units --type=service | grep "actions.runner" | head -1 | awk '{print $1}')

if [[ -n "$SERVICE_NAME" ]]; then
    if systemctl is-active "$SERVICE_NAME" > /dev/null 2>&1; then
        log_with_timestamp "INFO: Runner service $SERVICE_NAME is active"
    else
        log_with_timestamp "ERROR: Runner service $SERVICE_NAME is not active"
        # Attempt to restart
        log_with_timestamp "INFO: Attempting to restart runner service"
        sudo systemctl restart "$SERVICE_NAME"
    fi
else
    log_with_timestamp "ERROR: No GitHub runner service found"
fi

# Check disk space
DISK_USAGE=$(df "$RUNNER_DIR" | tail -1 | awk '{print $5}' | sed 's/%//')
if [[ "$DISK_USAGE" -gt 90 ]]; then
    log_with_timestamp "WARNING: Disk usage is ${DISK_USAGE}% in runner directory"
fi

# Check GPU status (if available)
if command -v nvidia-smi > /dev/null 2>&1; then
    if ! nvidia-smi > /dev/null 2>&1; then
        log_with_timestamp "WARNING: GPU access check failed"
    fi
fi

# Check Docker status (if available)
if command -v docker > /dev/null 2>&1; then
    if ! docker ps > /dev/null 2>&1; then
        log_with_timestamp "WARNING: Docker access check failed"
    fi
fi
EOF
    
    chmod +x "$RUNNER_DIR/monitor-runner.sh"
    
    # Setup cron job for monitoring
    (crontab -l 2>/dev/null; echo "*/5 * * * * $RUNNER_DIR/monitor-runner.sh") | crontab -
    
    log_success "Runner monitoring setup complete"
}

# Create management scripts
create_management_scripts() {
    log_info "Creating runner management scripts..."
    
    # Runner status script
    cat << EOF > "$RUNNER_DIR/runner-status.sh"
#!/bin/bash
# GitHub Runner status script

SERVICE_NAME=\$(systemctl list-units --type=service | grep "actions.runner" | head -1 | awk '{print \$1}')

echo "🔍 GitHub Runner Status"
echo "======================"
echo "Runner Name: $RUNNER_NAME"
echo "Labels: $RUNNER_LABELS"
echo "Directory: $RUNNER_DIR"
echo ""

if [[ -n "\$SERVICE_NAME" ]]; then
    echo "Service Status:"
    sudo systemctl status "\$SERVICE_NAME" --no-pager -l
    echo ""
fi

echo "System Information:"
echo "  OS: \$(cat /etc/os-release | grep PRETTY_NAME | cut -d= -f2 | tr -d '\"')"
echo "  CPU: \$(nproc) cores"
echo "  Memory: \$(free -h | grep Mem | awk '{print \$2}')"

if command -v nvidia-smi > /dev/null 2>&1; then
    echo "  GPU: \$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
fi

if command -v docker > /dev/null 2>&1; then
    echo "  Docker: \$(docker --version | cut -d' ' -f3 | tr -d ',')"
fi

echo ""
echo "Recent Logs:"
if [[ -f "/var/log/github-runner/monitor.log" ]]; then
    tail -10 "/var/log/github-runner/monitor.log"
fi
EOF
    
    chmod +x "$RUNNER_DIR/runner-status.sh"
    
    # Runner restart script
    cat << EOF > "$RUNNER_DIR/restart-runner.sh"
#!/bin/bash
# GitHub Runner restart script

SERVICE_NAME=\$(systemctl list-units --type=service | grep "actions.runner" | head -1 | awk '{print \$1}')

if [[ -n "\$SERVICE_NAME" ]]; then
    echo "🔄 Restarting GitHub Runner service..."
    sudo systemctl restart "\$SERVICE_NAME"
    sleep 5
    
    if sudo systemctl is-active "\$SERVICE_NAME" > /dev/null 2>&1; then
        echo "✅ Runner service restarted successfully"
    else
        echo "❌ Failed to restart runner service"
        exit 1
    fi
else
    echo "❌ No GitHub runner service found"
    exit 1
fi
EOF
    
    chmod +x "$RUNNER_DIR/restart-runner.sh"
    
    # Runner logs script
    cat << 'EOF' > "$RUNNER_DIR/runner-logs.sh"
#!/bin/bash
# GitHub Runner logs script

SERVICE_NAME=$(systemctl list-units --type=service | grep "actions.runner" | head -1 | awk '{print $1}')

echo "📋 GitHub Runner Logs"
echo "===================="

if [[ -n "$SERVICE_NAME" ]]; then
    echo "Service Logs (last 50 lines):"
    sudo journalctl -u "$SERVICE_NAME" -n 50 --no-pager
    echo ""
fi

if [[ -f "/var/log/github-runner/monitor.log" ]]; then
    echo "Monitor Logs (last 20 lines):"
    tail -20 "/var/log/github-runner/monitor.log"
fi
EOF
    
    chmod +x "$RUNNER_DIR/runner-logs.sh"
    
    log_success "Management scripts created in $RUNNER_DIR"
}

# Show setup summary
show_summary() {
    echo ""
    echo "🎉 GitHub Self-Hosted Runner Setup Complete!"
    echo "============================================"
    echo ""
    echo "📊 Runner Information:"
    echo "  Name: $RUNNER_NAME"
    echo "  Labels: $RUNNER_LABELS"
    echo "  Repository: $GITHUB_REPO_URL"
    echo "  Directory: $RUNNER_DIR"
    echo ""
    echo "🛠️ Management Commands:"
    echo "  Status: $RUNNER_DIR/runner-status.sh"
    echo "  Restart: $RUNNER_DIR/restart-runner.sh"
    echo "  Logs: $RUNNER_DIR/runner-logs.sh"
    echo ""
    echo "📝 Service Management:"
    local service_name="actions.runner.$(basename "$GITHUB_REPO_URL").${RUNNER_NAME}.service"
    echo "  Start: sudo systemctl start $service_name"
    echo "  Stop: sudo systemctl stop $service_name"
    echo "  Status: sudo systemctl status $service_name"
    echo "  Logs: sudo journalctl -u $service_name -f"
    echo ""
    echo "🔍 Monitoring:"
    echo "  Monitor logs: tail -f /var/log/github-runner/monitor.log"
    echo "  Cron job installed for automatic monitoring every 5 minutes"
    echo ""
    echo "📋 Next Steps:"
    echo "1. Verify the runner appears in your GitHub repository:"
    echo "   $GITHUB_REPO_URL/settings/actions/runners"
    echo ""
    echo "2. Update your workflow files to use the self-hosted runner:"
    echo "   runs-on: [self-hosted, linux, gpu] # or your specific labels"
    echo ""
    echo "3. Test the runner with a simple workflow push"
    echo ""
    log_success "Runner is ready for GitHub Actions!"
}

# Handle cleanup on exit
cleanup() {
    if [[ $? -ne 0 ]]; then
        log_error "Setup failed. Check the output above for details."
        echo ""
        echo "🔧 Troubleshooting:"
        echo "  - Verify GitHub token is valid and has correct permissions"
        echo "  - Check network connectivity to GitHub"
        echo "  - Ensure user has sudo privileges"
        echo "  - Check disk space and permissions"
    fi
}

trap cleanup EXIT

# Main setup flow
main() {
    echo ""
    collect_github_info
    echo ""
    check_requirements
    echo ""
    setup_runner
    echo ""
    configure_runner
    echo ""
    install_service
    echo ""
    test_runner
    echo ""
    setup_monitoring
    echo ""
    create_management_scripts
    echo ""
    show_summary
}

# Handle command line arguments
case "${1:-}" in
    "status")
        if [[ -f "$RUNNER_DIR/runner-status.sh" ]]; then
            "$RUNNER_DIR/runner-status.sh"
        else
            log_error "Runner not set up yet. Run without arguments to set up."
        fi
        ;;
    "restart")
        if [[ -f "$RUNNER_DIR/restart-runner.sh" ]]; then
            "$RUNNER_DIR/restart-runner.sh"
        else
            log_error "Runner not set up yet. Run without arguments to set up."
        fi
        ;;
    "logs")
        if [[ -f "$RUNNER_DIR/runner-logs.sh" ]]; then
            "$RUNNER_DIR/runner-logs.sh"
        else
            log_error "Runner not set up yet. Run without arguments to set up."
        fi
        ;;
    "help"|"-h"|"--help")
        echo "GitHub Self-Hosted Runner Setup for Hashmancer"
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  (no args)    Setup new runner"
        echo "  status       Show runner status"
        echo "  restart      Restart runner service"
        echo "  logs         Show runner logs"
        echo "  help         Show this help"
        echo ""
        echo "Environment Variables:"
        echo "  GITHUB_REPO_URL    GitHub repository URL"
        echo "  GITHUB_TOKEN       GitHub registration token"
        echo "  RUNNER_NAME        Custom runner name"
        echo "  RUNNER_LABELS      Custom runner labels"
        echo ""
        exit 0
        ;;
    *)
        main
        ;;
esac