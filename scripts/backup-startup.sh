#!/bin/bash

# Hashmancer Backup Startup Methods
# This script provides alternative ways to start Hashmancer if systemd fails

set -euo pipefail

# Auto-detect the Hashmancer directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HASHMANCER_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$HASHMANCER_DIR/logs"
BACKUP_LOG="$LOG_DIR/backup-startup.log"

# Logging function
log() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "$timestamp [BACKUP] $*" | tee -a "$BACKUP_LOG"
}

# Method 1: Direct Python execution
start_method_direct() {
    log "Attempting direct Python startup..."
    
    cd "$HASHMANCER_DIR"
    
    # Try different Python execution methods
    local methods=(
        "python3 -m hashmancer.server.main"
        "python3 hashmancer/server/main.py"
        "python3 main.py"
    )
    
    for method in "${methods[@]}"; do
        if $method > "$LOG_DIR/direct-startup.log" 2>&1 & then
            local pid=$!
            sleep 5
            
            if kill -0 $pid 2>/dev/null; then
                log "Direct startup successful with: $method (PID: $pid)"
                echo $pid > "$HASHMANCER_DIR/backup.pid"
                return 0
            else
                log "Direct startup failed with: $method"
            fi
        fi
    done
    
    return 1
}

# Method 2: Screen session startup
start_method_screen() {
    log "Attempting screen session startup..."
    
    if ! command -v screen &> /dev/null; then
        log "Screen not installed, skipping..."
        return 1
    fi
    
    cd "$HASHMANCER_DIR"
    
    # Start in detached screen session
    screen -dmS hashmancer bash -c "
        source venv/bin/activate 2>/dev/null || true
        export PYTHONPATH='$HASHMANCER_DIR'
        python3 -m hashmancer.server.main 2>&1 | tee '$LOG_DIR/screen-startup.log'
    "
    
    sleep 5
    
    if screen -list | grep -q hashmancer; then
        log "Screen startup successful"
        echo "screen:hashmancer" > "$HASHMANCER_DIR/backup.pid"
        return 0
    else
        log "Screen startup failed"
        return 1
    fi
}

# Method 3: Tmux session startup
start_method_tmux() {
    log "Attempting tmux session startup..."
    
    if ! command -v tmux &> /dev/null; then
        log "Tmux not installed, skipping..."
        return 1
    fi
    
    cd "$HASHMANCER_DIR"
    
    # Start in detached tmux session
    tmux new-session -d -s hashmancer "
        source venv/bin/activate 2>/dev/null || true
        export PYTHONPATH='$HASHMANCER_DIR'
        python3 -m hashmancer.server.main 2>&1 | tee '$LOG_DIR/tmux-startup.log'
    "
    
    sleep 5
    
    if tmux list-sessions | grep -q hashmancer; then
        log "Tmux startup successful"
        echo "tmux:hashmancer" > "$HASHMANCER_DIR/backup.pid"
        return 0
    else
        log "Tmux startup failed"
        return 1
    fi
}

# Method 4: Nohup startup
start_method_nohup() {
    log "Attempting nohup startup..."
    
    cd "$HASHMANCER_DIR"
    
    # Activate virtual environment if it exists
    if [[ -f "venv/bin/activate" ]]; then
        source venv/bin/activate
    fi
    
    export PYTHONPATH="$HASHMANCER_DIR"
    
    nohup python3 -m hashmancer.server.main > "$LOG_DIR/nohup-startup.log" 2>&1 &
    local pid=$!
    
    sleep 5
    
    if kill -0 $pid 2>/dev/null; then
        log "Nohup startup successful (PID: $pid)"
        echo $pid > "$HASHMANCER_DIR/backup.pid"
        return 0
    else
        log "Nohup startup failed"
        return 1
    fi
}

# Method 5: Docker startup (if available)
start_method_docker() {
    log "Attempting Docker startup..."
    
    if ! command -v docker &> /dev/null; then
        log "Docker not installed, skipping..."
        return 1
    fi
    
    if [[ ! -f "$HASHMANCER_DIR/Dockerfile" ]]; then
        log "Dockerfile not found, skipping..."
        return 1
    fi
    
    cd "$HASHMANCER_DIR"
    
    # Build image if it doesn't exist
    if ! docker image inspect hashmancer:latest &> /dev/null; then
        log "Building Docker image..."
        docker build -t hashmancer:latest . > "$LOG_DIR/docker-build.log" 2>&1
    fi
    
    # Run container
    docker run -d --name hashmancer-backup -p 8000:8000 hashmancer:latest > "$LOG_DIR/docker-startup.log" 2>&1
    
    sleep 10
    
    if docker ps | grep -q hashmancer-backup; then
        log "Docker startup successful"
        echo "docker:hashmancer-backup" > "$HASHMANCER_DIR/backup.pid"
        return 0
    else
        log "Docker startup failed"
        return 1
    fi
}

# Test portal connectivity
test_portal() {
    local max_attempts=30
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -f -s --max-time 5 http://localhost:8000/server_status &> /dev/null; then
            log "Portal is responding"
            return 0
        fi
        
        sleep 2
        ((attempt++))
    done
    
    log "Portal failed to respond after $max_attempts attempts"
    return 1
}

# Stop backup processes
stop_backup() {
    log "Stopping backup processes..."
    
    if [[ -f "$HASHMANCER_DIR/backup.pid" ]]; then
        local pid_info=$(cat "$HASHMANCER_DIR/backup.pid")
        
        case "$pid_info" in
            screen:*)
                local session_name="${pid_info#screen:}"
                screen -S "$session_name" -X quit 2>/dev/null || true
                log "Stopped screen session: $session_name"
                ;;
            tmux:*)
                local session_name="${pid_info#tmux:}"
                tmux kill-session -t "$session_name" 2>/dev/null || true
                log "Stopped tmux session: $session_name"
                ;;
            docker:*)
                local container_name="${pid_info#docker:}"
                docker stop "$container_name" 2>/dev/null || true
                docker rm "$container_name" 2>/dev/null || true
                log "Stopped Docker container: $container_name"
                ;;
            *)
                # Regular PID
                if [[ "$pid_info" =~ ^[0-9]+$ ]]; then
                    kill "$pid_info" 2>/dev/null || true
                    log "Stopped process: $pid_info"
                fi
                ;;
        esac
        
        rm -f "$HASHMANCER_DIR/backup.pid"
    fi
    
    # Kill any remaining processes
    pkill -f "hashmancer" || true
    pkill -f "python.*main" || true
}

# Main startup function
start_backup() {
    log "Starting backup startup sequence..."
    
    # Ensure log directory exists
    mkdir -p "$LOG_DIR"
    
    # Stop any existing backup processes
    stop_backup
    
    # Try startup methods in order of preference
    local methods=(
        "start_method_direct"
        "start_method_nohup"
        "start_method_screen"
        "start_method_tmux"
        "start_method_docker"
    )
    
    for method in "${methods[@]}"; do
        log "Trying startup method: $method"
        
        if $method; then
            if test_portal; then
                log "Backup startup successful with method: $method"
                log "Portal available at: http://localhost:8000"
                return 0
            else
                log "Portal test failed for method: $method"
                stop_backup
            fi
        fi
    done
    
    log "All backup startup methods failed"
    return 1
}

# Status check
check_backup_status() {
    if [[ -f "$HASHMANCER_DIR/backup.pid" ]]; then
        local pid_info=$(cat "$HASHMANCER_DIR/backup.pid")
        log "Backup process info: $pid_info"
        
        case "$pid_info" in
            screen:*)
                local session_name="${pid_info#screen:}"
                if screen -list | grep -q "$session_name"; then
                    log "Screen session is running: $session_name"
                    return 0
                fi
                ;;
            tmux:*)
                local session_name="${pid_info#tmux:}"
                if tmux list-sessions 2>/dev/null | grep -q "$session_name"; then
                    log "Tmux session is running: $session_name"
                    return 0
                fi
                ;;
            docker:*)
                local container_name="${pid_info#docker:}"
                if docker ps | grep -q "$container_name"; then
                    log "Docker container is running: $container_name"
                    return 0
                fi
                ;;
            *)
                if [[ "$pid_info" =~ ^[0-9]+$ ]] && kill -0 "$pid_info" 2>/dev/null; then
                    log "Process is running: $pid_info"
                    return 0
                fi
                ;;
        esac
    fi
    
    log "No backup process is running"
    return 1
}

# Main execution
main() {
    local action="${1:-start}"
    
    case "$action" in
        "start")
            start_backup
            ;;
        "stop")
            stop_backup
            ;;
        "status")
            check_backup_status
            ;;
        "restart")
            stop_backup
            sleep 2
            start_backup
            ;;
        *)
            echo "Usage: $0 {start|stop|status|restart}"
            echo "  start   - Start Hashmancer using backup methods"
            echo "  stop    - Stop backup processes"
            echo "  status  - Check backup process status"
            echo "  restart - Restart backup processes"
            exit 1
            ;;
    esac
}

# Execute main function
main "$@"