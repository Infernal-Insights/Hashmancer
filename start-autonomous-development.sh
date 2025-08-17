#!/bin/bash
set -e

# Hashmancer Autonomous Development Startup Script
# ================================================
# Initializes and starts the autonomous development system for continuous improvement

echo "🤖 Starting Hashmancer Autonomous Development System"
echo "===================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
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

log_header() {
    echo -e "${PURPLE}🔹 $1${NC}"
}

# Configuration
PYTHON_ENV="${PYTHON_ENV:-python3}"
CONFIG_FILE="autonomous-dev-config.yaml"
LOG_FILE="/tmp/autonomous_dev_startup.log"
PID_FILE="/tmp/autonomous_dev.pid"

# Pre-flight checks
pre_flight_checks() {
    log_header "Pre-flight System Checks"
    
    local checks_passed=0
    local checks_total=0
    
    # Check Python environment
    checks_total=$((checks_total + 1))
    if command -v python3 > /dev/null 2>&1; then
        log_success "Python 3 is available"
        checks_passed=$((checks_passed + 1))
    else
        log_error "Python 3 not found"
    fi
    
    # Check required Python packages
    checks_total=$((checks_total + 1))
    if python3 -c "import asyncio, aiohttp, redis, psutil, yaml" 2>/dev/null; then
        log_success "Required Python packages available"
        checks_passed=$((checks_passed + 1))
    else
        log_warning "Some Python packages missing - installing..."
        pip3 install asyncio aiohttp redis psutil pyyaml || {
            log_error "Failed to install Python packages"
        }
    fi
    
    # Check Docker
    checks_total=$((checks_total + 1))
    if command -v docker > /dev/null 2>&1 && docker ps > /dev/null 2>&1; then
        log_success "Docker is available and running"
        checks_passed=$((checks_passed + 1))
    else
        log_error "Docker not available or not running"
    fi
    
    # Check NVIDIA GPUs
    checks_total=$((checks_total + 1))
    if command -v nvidia-smi > /dev/null 2>&1; then
        local gpu_count=$(nvidia-smi --list-gpus | wc -l)
        if [[ $gpu_count -eq 2 ]]; then
            log_success "Dual GPU setup detected ($gpu_count GPUs)"
            checks_passed=$((checks_passed + 1))
            
            # Show GPU info
            echo "  GPU Information:"
            nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | while IFS=, read -r index name memory; do
                echo "    GPU $index: $name ($memory)"
            done
        else
            log_warning "Expected 2 GPUs, found $gpu_count"
        fi
    else
        log_error "NVIDIA drivers not available"
    fi
    
    # Check Redis
    checks_total=$((checks_total + 1))
    if command -v redis-cli > /dev/null 2>&1 && redis-cli ping > /dev/null 2>&1; then
        log_success "Redis is available and responding"
        checks_passed=$((checks_passed + 1))
    else
        log_warning "Redis not available - will start with Docker"
    fi
    
    # Check disk space
    checks_total=$((checks_total + 1))
    local disk_usage=$(df . | tail -1 | awk '{print $5}' | sed 's/%//')
    if [[ $disk_usage -lt 90 ]]; then
        log_success "Disk space OK (${disk_usage}% used)"
        checks_passed=$((checks_passed + 1))
    else
        log_error "Low disk space (${disk_usage}% used)"
    fi
    
    # Check memory
    checks_total=$((checks_total + 1))
    local mem_usage=$(free | grep Mem | awk '{printf "%.0f", $3/$2*100}')
    if [[ $mem_usage -lt 85 ]]; then
        log_success "Memory OK (${mem_usage}% used)"
        checks_passed=$((checks_passed + 1))
    else
        log_warning "High memory usage (${mem_usage}% used)"
    fi
    
    # Check API keys
    checks_total=$((checks_total + 1))
    if [[ -n "$ANTHROPIC_API_KEY" || -n "$CLAUDE_API_KEY" ]]; then
        log_success "Claude API key configured"
        checks_passed=$((checks_passed + 1))
    else
        log_warning "Claude API key not configured - limited functionality"
    fi
    
    echo ""
    log_info "Pre-flight checks: $checks_passed/$checks_total passed"
    
    if [[ $checks_passed -lt $((checks_total - 2)) ]]; then
        log_error "Too many critical checks failed. Please resolve issues before starting."
        exit 1
    fi
}

# Initialize environment
initialize_environment() {
    log_header "Initializing Development Environment"
    
    # Create necessary directories
    mkdir -p /tmp/autonomous_dev_logs
    mkdir -p /tmp/autonomous_dev_data
    
    # Set up configuration file if it doesn't exist
    if [[ ! -f "$CONFIG_FILE" ]]; then
        log_info "Creating default configuration file"
        cp autonomous-dev-config.yaml.template "$CONFIG_FILE" 2>/dev/null || {
            log_warning "No config template found - using defaults"
        }
    fi
    
    # Set environment variables
    export PYTHONPATH="$PWD:$PYTHONPATH"
    export CUDA_VISIBLE_DEVICES=0,1  # Ensure both GPUs are visible
    
    # GPU optimization setup
    log_info "Optimizing GPU settings for RTX 2080 Ti"
    
    # Enable persistence mode for stability
    sudo nvidia-smi -pm 1 2>/dev/null || log_warning "Could not enable GPU persistence mode"
    
    # Set optimal power limits (250W per GPU)
    for gpu in 0 1; do
        sudo nvidia-smi -i $gpu -pl 250 2>/dev/null || log_warning "Could not set power limit for GPU $gpu"
    done
    
    log_success "Environment initialized"
}

# Deploy Hashmancer environment
deploy_hashmancer() {
    log_header "Deploying Hashmancer Environment"
    
    # Check if already running
    if docker ps | grep -q hashmancer; then
        log_info "Hashmancer containers already running"
        
        # Check health
        if curl -s http://localhost:8080/health > /dev/null 2>&1; then
            log_success "Hashmancer is healthy"
            return 0
        else
            log_warning "Hashmancer containers unhealthy - redeploying"
            docker-compose -f docker-compose.ultimate.yml down -v
        fi
    fi
    
    # Deploy fresh environment
    log_info "Starting Hashmancer deployment..."
    
    if ./deploy-hashmancer.sh quick > "$LOG_FILE" 2>&1; then
        log_success "Hashmancer deployed successfully"
        
        # Wait for services to be ready
        log_info "Waiting for services to be ready..."
        local timeout=120
        while ! curl -s http://localhost:8080/health > /dev/null 2>&1; do
            sleep 5
            timeout=$((timeout - 5))
            if [[ $timeout -le 0 ]]; then
                log_error "Services failed to start within timeout"
                return 1
            fi
        done
        
        log_success "All services are ready"
        
        # Show deployment status
        echo "  Service Status:"
        docker-compose -f docker-compose.ultimate.yml ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"
        
    else
        log_error "Hashmancer deployment failed"
        echo "Check logs: tail -f $LOG_FILE"
        return 1
    fi
}

# Start monitoring services
start_monitoring() {
    log_header "Starting System Monitoring"
    
    # Start GPU monitoring
    log_info "Starting GPU monitoring..."
    nohup python3 gpu-optimization-system.py > /tmp/gpu_monitoring.log 2>&1 &
    echo $! > /tmp/gpu_monitoring.pid
    
    # Start performance monitoring
    log_info "Starting performance monitoring..."
    nohup python3 -c "
import asyncio
import time
import subprocess
import json

async def monitor_performance():
    while True:
        try:
            # GPU stats
            result = subprocess.run(['nvidia-smi', '--query-gpu=index,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw', '--format=csv,noheader,nounits'], capture_output=True, text=True)
            
            timestamp = int(time.time())
            with open('/tmp/performance_data.log', 'a') as f:
                f.write(f'{timestamp},{result.stdout.strip()}\n')
            
            await asyncio.sleep(30)  # Monitor every 30 seconds
        except Exception as e:
            print(f'Monitoring error: {e}')
            await asyncio.sleep(60)

asyncio.run(monitor_performance())
" > /tmp/performance_monitoring.log 2>&1 &
    echo $! > /tmp/performance_monitoring.pid
    
    log_success "Monitoring services started"
}

# Start autonomous development
start_autonomous_development() {
    log_header "Starting Autonomous Development System"
    
    # Check if already running
    if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        log_warning "Autonomous development system already running (PID: $(cat "$PID_FILE"))"
        return 0
    fi
    
    # Start the autonomous development framework
    log_info "Launching autonomous development framework..."
    
    nohup python3 autonomous-dev-framework.py > /tmp/autonomous_dev_main.log 2>&1 &
    local main_pid=$!
    echo $main_pid > "$PID_FILE"
    
    # Wait a moment to check if it started successfully
    sleep 5
    
    if kill -0 $main_pid 2>/dev/null; then
        log_success "Autonomous development system started successfully (PID: $main_pid)"
        
        # Show initial status
        echo "  System Configuration:"
        echo "    - Development cycles: 6 per day (every 4 hours)"
        echo "    - Maximum Opus API calls: 15 per day"
        echo "    - GPU temperature threshold: 80°C"
        echo "    - Performance monitoring: 30-second intervals"
        echo "    - Log analysis: 5-minute intervals"
        
    else
        log_error "Failed to start autonomous development system"
        rm -f "$PID_FILE"
        return 1
    fi
}

# Show system status
show_status() {
    log_header "Autonomous Development System Status"
    
    # Main system status
    if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        log_success "Main system: Running (PID: $(cat "$PID_FILE"))"
    else
        log_error "Main system: Not running"
    fi
    
    # GPU status
    if command -v nvidia-smi > /dev/null 2>&1; then
        echo ""
        echo "🎮 GPU Status:"
        nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader | while IFS=, read -r index name temp util mem_used mem_total power; do
            echo "  GPU $index ($name): ${temp}°C, ${util}% util, ${mem_used}/${mem_total} MB, ${power}W"
        done
    fi
    
    # Docker services status
    echo ""
    echo "🐳 Docker Services:"
    if docker ps --format "table {{.Names}}\t{{.Status}}" | grep hashmancer > /dev/null; then
        docker ps --format "table {{.Names}}\t{{.Status}}" | grep hashmancer | while read name status; do
            echo "  $name: $status"
        done
    else
        echo "  No Hashmancer containers running"
    fi
    
    # Performance data
    if [[ -f "/tmp/performance_data.log" ]]; then
        echo ""
        echo "📊 Recent Performance:"
        tail -1 /tmp/performance_data.log 2>/dev/null | while IFS=, read timestamp gpu0_data gpu1_data; do
            echo "  Last update: $(date -d @$timestamp)"
            echo "  GPU performance data available"
        done
    fi
    
    # Logs location
    echo ""
    echo "📋 Log Files:"
    echo "  Main system: tail -f /tmp/autonomous_dev_main.log"
    echo "  GPU monitoring: tail -f /tmp/gpu_monitoring.log"
    echo "  Performance data: tail -f /tmp/performance_data.log"
    echo "  Startup log: tail -f $LOG_FILE"
}

# Stop autonomous development
stop_autonomous_development() {
    log_header "Stopping Autonomous Development System"
    
    # Stop main system
    if [[ -f "$PID_FILE" ]]; then
        local main_pid=$(cat "$PID_FILE")
        if kill -0 "$main_pid" 2>/dev/null; then
            log_info "Stopping main system (PID: $main_pid)"
            kill "$main_pid"
            sleep 5
            if kill -0 "$main_pid" 2>/dev/null; then
                log_warning "Force stopping main system"
                kill -9 "$main_pid"
            fi
        fi
        rm -f "$PID_FILE"
    fi
    
    # Stop monitoring services
    for pid_file in /tmp/gpu_monitoring.pid /tmp/performance_monitoring.pid; do
        if [[ -f "$pid_file" ]]; then
            local pid=$(cat "$pid_file")
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid" 2>/dev/null
            fi
            rm -f "$pid_file"
        fi
    done
    
    log_success "Autonomous development system stopped"
}

# Main menu
show_menu() {
    echo ""
    echo "🤖 Hashmancer Autonomous Development Control"
    echo "============================================"
    echo "1. Start autonomous development"
    echo "2. Show system status"
    echo "3. Stop autonomous development"
    echo "4. View logs"
    echo "5. Run single development cycle (test)"
    echo "6. Exit"
    echo ""
}

# Interactive mode
interactive_mode() {
    while true; do
        show_menu
        read -p "Choose option (1-6): " choice
        
        case $choice in
            1)
                pre_flight_checks
                initialize_environment
                deploy_hashmancer
                start_monitoring
                start_autonomous_development
                ;;
            2)
                show_status
                ;;
            3)
                stop_autonomous_development
                ;;
            4)
                echo "Available logs:"
                echo "  1. Main system log"
                echo "  2. GPU monitoring log"
                echo "  3. Performance data"
                echo "  4. Startup log"
                read -p "Choose log (1-4): " log_choice
                
                case $log_choice in
                    1) tail -f /tmp/autonomous_dev_main.log ;;
                    2) tail -f /tmp/gpu_monitoring.log ;;
                    3) tail -f /tmp/performance_data.log ;;
                    4) tail -f "$LOG_FILE" ;;
                    *) echo "Invalid choice" ;;
                esac
                ;;
            5)
                log_info "Running single development cycle..."
                python3 -c "
import asyncio
from autonomous_dev_framework import AutonomousDevFramework

async def test_cycle():
    framework = AutonomousDevFramework()
    await framework.initialize()
    await framework.run_development_cycle()
    await framework.cleanup()

asyncio.run(test_cycle())
"
                ;;
            6)
                echo "Goodbye!"
                exit 0
                ;;
            *)
                log_error "Invalid option"
                ;;
        esac
        
        echo ""
        read -p "Press Enter to continue..."
    done
}

# Handle command line arguments
case "${1:-interactive}" in
    "start")
        pre_flight_checks
        initialize_environment
        deploy_hashmancer
        start_monitoring
        start_autonomous_development
        show_status
        ;;
    "stop")
        stop_autonomous_development
        ;;
    "status")
        show_status
        ;;
    "restart")
        stop_autonomous_development
        sleep 5
        pre_flight_checks
        initialize_environment
        deploy_hashmancer
        start_monitoring
        start_autonomous_development
        ;;
    "interactive"|"")
        interactive_mode
        ;;
    "help"|"-h"|"--help")
        echo "Hashmancer Autonomous Development Control"
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  start       Start autonomous development"
        echo "  stop        Stop autonomous development"
        echo "  status      Show system status"
        echo "  restart     Restart autonomous development"
        echo "  interactive Interactive mode (default)"
        echo "  help        Show this help"
        echo ""
        ;;
    *)
        log_error "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac