#!/bin/bash

# Hashmancer Setup Validation Script
# This script validates that all auto-start components are properly configured

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

# Counters
passed=0
failed=0
warnings=0

print_header() {
    echo -e "${CYAN}"
    echo "========================================"
    echo "  HASHMANCER SETUP VALIDATION"
    echo "========================================"
    echo -e "${NC}"
}

print_test() {
    echo -e "${BLUE}[TEST]${NC} $1"
}

print_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((passed++))
}

print_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((failed++))
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
    ((warnings++))
}

print_info() {
    echo -e "${CYAN}[INFO]${NC} $1"
}

# Test functions
test_files_exist() {
    print_test "Checking required files exist..."
    
    local required_files=(
        "$HASHMANCER_DIR/hashmancer.service"
        "$HASHMANCER_DIR/install-service.sh"
        "$HASHMANCER_DIR/scripts/start-hashmancer.sh"
        "$HASHMANCER_DIR/scripts/pre-start-check.sh"
        "$HASHMANCER_DIR/scripts/backup-startup.sh"
        "$HASHMANCER_DIR/scripts/monitor-hashmancer.sh"
        "$HASHMANCER_DIR/AUTO-START-README.md"
    )
    
    local all_exist=true
    for file in "${required_files[@]}"; do
        if [[ -f "$file" ]]; then
            print_info "✓ Found: $(basename "$file")"
        else
            print_fail "✗ Missing: $file"
            all_exist=false
        fi
    done
    
    if [[ "$all_exist" == true ]]; then
        print_pass "All required files exist"
    else
        print_fail "Some required files are missing"
    fi
}

test_scripts_executable() {
    print_test "Checking script permissions..."
    
    local scripts=(
        "$HASHMANCER_DIR/install-service.sh"
        "$HASHMANCER_DIR/scripts/start-hashmancer.sh"
        "$HASHMANCER_DIR/scripts/pre-start-check.sh"
        "$HASHMANCER_DIR/scripts/backup-startup.sh"
        "$HASHMANCER_DIR/scripts/monitor-hashmancer.sh"
    )
    
    local all_executable=true
    for script in "${scripts[@]}"; do
        if [[ -x "$script" ]]; then
            print_info "✓ Executable: $(basename "$script")"
        else
            print_fail "✗ Not executable: $script"
            all_executable=false
        fi
    done
    
    if [[ "$all_executable" == true ]]; then
        print_pass "All scripts are executable"
    else
        print_fail "Some scripts are not executable"
    fi
}

test_systemd_service() {
    print_test "Checking systemd service configuration..."
    
    if [[ -f "/etc/systemd/system/hashmancer.service" ]]; then
        print_pass "Systemd service file installed"
        
        if systemctl is-enabled hashmancer &>/dev/null; then
            print_pass "Service is enabled for auto-start"
        else
            print_warn "Service is not enabled (run: sudo systemctl enable hashmancer)"
        fi
        
        if systemctl is-active hashmancer &>/dev/null; then
            print_pass "Service is currently running"
        else
            print_warn "Service is not currently running"
        fi
    else
        print_fail "Systemd service not installed"
    fi
}

test_dependencies() {
    print_test "Checking system dependencies..."
    
    local dependencies=(
        "python3"
        "systemctl"
        "curl"
    )
    
    local all_available=true
    for dep in "${dependencies[@]}"; do
        if command -v "$dep" &>/dev/null; then
            print_info "✓ Available: $dep"
        else
            print_fail "✗ Missing: $dep"
            all_available=false
        fi
    done
    
    if [[ "$all_available" == true ]]; then
        print_pass "All dependencies available"
    else
        print_fail "Some dependencies are missing"
    fi
}

test_directories() {
    print_test "Checking directory structure..."
    
    local directories=(
        "$HASHMANCER_DIR"
        "$HASHMANCER_DIR/scripts"
        "$HASHMANCER_DIR/logs"
        "$HASHMANCER_DIR/hashmancer"
        "$HASHMANCER_DIR/hashmancer/server"
    )
    
    local all_exist=true
    for dir in "${directories[@]}"; do
        if [[ -d "$dir" ]]; then
            print_info "✓ Directory: $(basename "$dir")"
        else
            print_fail "✗ Missing directory: $dir"
            all_exist=false
        fi
    done
    
    if [[ "$all_exist" == true ]]; then
        print_pass "Directory structure is correct"
    else
        print_fail "Directory structure has issues"
    fi
}

test_python_environment() {
    print_test "Checking Python environment..."
    
    if [[ -d "$HASHMANCER_DIR/venv" ]]; then
        print_pass "Virtual environment exists"
    else
        print_warn "Virtual environment not found (will be created on first run)"
    fi
    
    if [[ -f "$HASHMANCER_DIR/hashmancer/server/main.py" ]]; then
        print_pass "Main server file exists"
    else
        print_fail "Main server file not found"
    fi
}

test_log_rotation() {
    print_test "Checking log rotation configuration..."
    
    if [[ -f "/etc/logrotate.d/hashmancer" ]]; then
        print_pass "Log rotation configured"
    else
        print_warn "Log rotation not configured (run install script)"
    fi
}

test_cron_jobs() {
    print_test "Checking cron jobs..."
    
    if crontab -l 2>/dev/null | grep -q "health-check.sh"; then
        print_pass "Health check cron job configured"
    else
        print_warn "Health check cron job not configured (run install script)"
    fi
}

test_firewall() {
    print_test "Checking firewall configuration..."
    
    if command -v ufw &>/dev/null; then
        if ufw status | grep -q "8000"; then
            print_pass "Firewall allows port 8000"
        else
            print_warn "Firewall may not allow port 8000"
        fi
    else
        print_info "UFW not installed, skipping firewall check"
    fi
}

test_portal_accessibility() {
    print_test "Testing portal accessibility..."
    
    if systemctl is-active hashmancer &>/dev/null; then
        if curl -f -s --max-time 10 http://localhost:8000/server_status &>/dev/null; then
            print_pass "Portal is accessible at http://localhost:8000"
        else
            print_warn "Portal is not responding (may be starting up)"
        fi
    else
        print_info "Service not running, skipping portal test"
    fi
}

# Summary function
print_summary() {
    echo ""
    echo -e "${CYAN}========================================"
    echo "  VALIDATION SUMMARY"
    echo -e "========================================${NC}"
    echo -e "${GREEN}Passed: $passed${NC}"
    echo -e "${YELLOW}Warnings: $warnings${NC}"
    echo -e "${RED}Failed: $failed${NC}"
    echo ""
    
    if [[ $failed -eq 0 ]]; then
        if [[ $warnings -eq 0 ]]; then
            echo -e "${GREEN}🎉 PERFECT! Setup is completely ready!${NC}"
            echo -e "${GREEN}Hashmancer will start automatically on boot.${NC}"
        else
            echo -e "${YELLOW}✅ GOOD! Setup is mostly ready with minor warnings.${NC}"
            echo -e "${YELLOW}Hashmancer should work but consider addressing warnings.${NC}"
        fi
    else
        echo -e "${RED}❌ ISSUES FOUND! Setup needs attention.${NC}"
        echo -e "${RED}Please address the failed tests before proceeding.${NC}"
    fi
    
    echo ""
    echo -e "${CYAN}Next steps:${NC}"
    if [[ ! -f "/etc/systemd/system/hashmancer.service" ]]; then
        echo "  1. Run: sudo ./install-service.sh"
    else
        echo "  1. Service is installed ✓"
    fi
    echo "  2. Reboot to test auto-start: sudo reboot"
    echo "  3. Check portal: http://localhost:8000"
    echo "  4. Monitor logs: journalctl -u hashmancer -f"
}

# Main execution
main() {
    print_header
    
    test_files_exist
    test_scripts_executable
    test_directories
    test_python_environment
    test_dependencies
    test_systemd_service
    test_log_rotation
    test_cron_jobs
    test_firewall
    test_portal_accessibility
    
    print_summary
    
    # Exit with appropriate code
    if [[ $failed -gt 0 ]]; then
        exit 1
    else
        exit 0
    fi
}

# Execute main function
main "$@"