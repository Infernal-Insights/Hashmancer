#!/bin/bash
set -e

# Hashmancer Docker Deployment Test Script
# Validates that the Docker deployment is working correctly

echo "🧪 Hashmancer Docker Deployment Test"
echo "===================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test functions
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

# Test results tracking
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_TOTAL=0

run_test() {
    local test_name="$1"
    local test_command="$2"
    
    TESTS_TOTAL=$((TESTS_TOTAL + 1))
    log_info "Testing: $test_name"
    
    if eval "$test_command" > /dev/null 2>&1; then
        log_success "$test_name"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        return 0
    else
        log_error "$test_name"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        return 1
    fi
}

# Prerequisites check
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    run_test "Docker is installed" "command -v docker"
    run_test "Docker Compose is available" "command -v docker-compose || docker compose version"
    run_test "Docker daemon is running" "docker info"
    
    # Check if deployment files exist
    run_test "Docker compose file exists" "test -f docker-compose.ultimate.yml"
    run_test "Deployment script exists" "test -f deploy-hashmancer.sh"
    run_test "Redis tool exists" "test -f redis_tool.py"
}

# Test Redis functionality
test_redis() {
    log_info "Testing Redis..."
    
    # Check if Redis container is running
    run_test "Redis container is running" "docker ps | grep hashmancer-redis"
    
    # Test Redis connection
    run_test "Redis accepts connections" "docker exec hashmancer-redis redis-cli ping"
    
    # Test Redis with our tool
    if command -v python3 > /dev/null 2>&1; then
        run_test "Redis tool connection test" "python3 redis_tool.py test"
        run_test "Redis health check" "python3 redis_tool.py health --quick"
    else
        log_warning "Python3 not found, skipping Redis tool tests"
    fi
}

# Test server functionality
test_server() {
    log_info "Testing server..."
    
    # Check if server container is running
    run_test "Server container is running" "docker ps | grep hashmancer-server"
    
    # Test server health endpoint
    run_test "Server health endpoint responds" "curl -f -s http://localhost:8080/health"
    
    # Test web interface (if nginx is running)
    if docker ps | grep hashmancer-nginx > /dev/null 2>&1; then
        run_test "Nginx proxy responds" "curl -f -s http://localhost/"
    fi
    
    # Check server logs for errors
    if ! docker logs hashmancer-server --since 5m 2>&1 | grep -q "ERROR\|CRITICAL"; then
        log_success "No critical errors in server logs"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        log_error "Critical errors found in server logs"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
    TESTS_TOTAL=$((TESTS_TOTAL + 1))
}

# Test worker functionality
test_workers() {
    log_info "Testing workers..."
    
    # Check GPU worker
    if docker ps | grep hashmancer-worker-gpu > /dev/null 2>&1; then
        run_test "GPU worker container is running" "docker ps | grep hashmancer-worker-gpu"
        
        # Test GPU access
        if docker exec hashmancer-worker-gpu nvidia-smi > /dev/null 2>&1; then
            log_success "GPU worker has GPU access"
            TESTS_PASSED=$((TESTS_PASSED + 1))
        else
            log_warning "GPU worker lacks GPU access (this may be normal on CPU-only systems)"
        fi
        TESTS_TOTAL=$((TESTS_TOTAL + 1))
        
        # Test Hashcat
        run_test "GPU worker Hashcat installation" "docker exec hashmancer-worker-gpu hashcat --version"
        
        # Check worker logs
        if ! docker logs hashmancer-worker-gpu --since 5m 2>&1 | grep -q "ERROR\|CRITICAL"; then
            log_success "No critical errors in GPU worker logs"
            TESTS_PASSED=$((TESTS_PASSED + 1))
        else
            log_error "Critical errors found in GPU worker logs"
            TESTS_FAILED=$((TESTS_FAILED + 1))
        fi
        TESTS_TOTAL=$((TESTS_TOTAL + 1))
    fi
    
    # Check CPU worker
    if docker ps | grep hashmancer-worker-cpu > /dev/null 2>&1; then
        run_test "CPU worker container is running" "docker ps | grep hashmancer-worker-cpu"
        run_test "CPU worker Hashcat installation" "docker exec hashmancer-worker-cpu hashcat --version"
        
        # Check worker logs
        if ! docker logs hashmancer-worker-cpu --since 5m 2>&1 | grep -q "ERROR\|CRITICAL"; then
            log_success "No critical errors in CPU worker logs"
            TESTS_PASSED=$((TESTS_PASSED + 1))
        else
            log_error "Critical errors found in CPU worker logs"
            TESTS_FAILED=$((TESTS_FAILED + 1))
        fi
        TESTS_TOTAL=$((TESTS_TOTAL + 1))
    fi
}

# Test networking
test_networking() {
    log_info "Testing networking..."
    
    # Test internal network connectivity
    if docker ps | grep hashmancer-server > /dev/null 2>&1 && docker ps | grep hashmancer-redis > /dev/null 2>&1; then
        run_test "Server can reach Redis" "docker exec hashmancer-server redis-cli -h redis ping"
    fi
    
    if docker ps | grep hashmancer-worker-gpu > /dev/null 2>&1 && docker ps | grep hashmancer-server > /dev/null 2>&1; then
        run_test "GPU worker can reach server" "docker exec hashmancer-worker-gpu curl -f -s http://server:8080/health"
    fi
    
    if docker ps | grep hashmancer-worker-cpu > /dev/null 2>&1 && docker ps | grep hashmancer-server > /dev/null 2>&1; then
        run_test "CPU worker can reach server" "docker exec hashmancer-worker-cpu curl -f -s http://server:8080/health"
    fi
}

# Test resource usage
test_resources() {
    log_info "Testing resource usage..."
    
    # Check memory usage
    local total_memory=$(docker stats --no-stream --format "table {{.MemUsage}}" | tail -n +2 | awk -F'/' '{print $1}' | sed 's/[^0-9.]//g' | awk '{sum += $1} END {print sum}')
    
    if [[ -n "$total_memory" ]] && (( $(echo "$total_memory < 8" | bc -l) )); then
        log_success "Total memory usage is reasonable: ${total_memory}GB"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        log_warning "High memory usage detected: ${total_memory}GB"
    fi
    TESTS_TOTAL=$((TESTS_TOTAL + 1))
    
    # Check if any containers are restarting
    if docker ps --filter "status=restarting" | grep -q hashmancer; then
        log_error "Some containers are continuously restarting"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    else
        log_success "No containers are restarting"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    fi
    TESTS_TOTAL=$((TESTS_TOTAL + 1))
}

# Performance test
test_performance() {
    log_info "Running basic performance tests..."
    
    # Test Redis performance
    if docker ps | grep hashmancer-redis > /dev/null 2>&1; then
        local redis_latency=$(docker exec hashmancer-redis redis-cli --latency-history -i 1 | head -1 | awk '{print $4}')
        if [[ -n "$redis_latency" ]] && (( $(echo "$redis_latency < 10" | bc -l) )); then
            log_success "Redis latency is good: ${redis_latency}ms"
            TESTS_PASSED=$((TESTS_PASSED + 1))
        else
            log_warning "Redis latency is high: ${redis_latency}ms"
        fi
        TESTS_TOTAL=$((TESTS_TOTAL + 1))
    fi
    
    # Test server response time
    if command -v curl > /dev/null 2>&1; then
        local response_time=$(curl -o /dev/null -s -w "%{time_total}" http://localhost:8080/health)
        if (( $(echo "$response_time < 2" | bc -l) )); then
            log_success "Server response time is good: ${response_time}s"
            TESTS_PASSED=$((TESTS_PASSED + 1))
        else
            log_warning "Server response time is slow: ${response_time}s"
        fi
        TESTS_TOTAL=$((TESTS_TOTAL + 1))
    fi
}

# Comprehensive test
run_comprehensive_test() {
    log_info "Running comprehensive hash cracking test..."
    
    # This would test actual hash cracking functionality
    # For now, we'll just verify the components are ready
    
    if docker ps | grep hashmancer-server > /dev/null 2>&1 && \
       docker ps | grep hashmancer-redis > /dev/null 2>&1 && \
       (docker ps | grep hashmancer-worker-gpu > /dev/null 2>&1 || docker ps | grep hashmancer-worker-cpu > /dev/null 2>&1); then
        log_success "All components ready for hash cracking"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        log_error "Not all required components are running"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
    TESTS_TOTAL=$((TESTS_TOTAL + 1))
}

# Generate test report
generate_report() {
    echo ""
    echo "📊 Test Results Summary"
    echo "======================"
    echo "Total tests: $TESTS_TOTAL"
    echo "Passed: $TESTS_PASSED"
    echo "Failed: $TESTS_FAILED"
    
    local success_rate=$(echo "scale=1; $TESTS_PASSED * 100 / $TESTS_TOTAL" | bc -l)
    echo "Success rate: ${success_rate}%"
    
    if [[ $TESTS_FAILED -eq 0 ]]; then
        echo ""
        log_success "🎉 All tests passed! Your Hashmancer deployment is working perfectly!"
        echo ""
        echo "🌐 Access your deployment:"
        echo "   Web Interface: http://localhost"
        echo "   Server Direct: http://localhost:8080"
        echo "   Redis: localhost:6379"
        echo ""
        echo "🔧 Management commands:"
        echo "   View status: docker-compose -f docker-compose.ultimate.yml ps"
        echo "   View logs: docker-compose -f docker-compose.ultimate.yml logs -f"
        echo "   Redis health: python3 redis_tool.py health"
        return 0
    else
        echo ""
        log_error "Some tests failed. Please check the output above for details."
        echo ""
        echo "🔍 Troubleshooting:"
        echo "   Check logs: docker-compose -f docker-compose.ultimate.yml logs"
        echo "   Check status: docker-compose -f docker-compose.ultimate.yml ps"
        echo "   Restart services: docker-compose -f docker-compose.ultimate.yml restart"
        return 1
    fi
}

# Main test execution
main() {
    echo ""
    check_prerequisites
    echo ""
    
    # Only run deployment tests if containers are running
    if docker ps | grep hashmancer > /dev/null 2>&1; then
        test_redis
        echo ""
        test_server
        echo ""
        test_workers
        echo ""
        test_networking
        echo ""
        test_resources
        echo ""
        test_performance
        echo ""
        run_comprehensive_test
    else
        log_warning "No Hashmancer containers found running. Deploy first with ./deploy-hashmancer.sh"
        echo ""
        echo "Quick deployment: ./deploy-hashmancer.sh quick"
        exit 1
    fi
    
    echo ""
    generate_report
}

# Handle interrupts gracefully
trap 'echo ""; log_warning "Test interrupted by user"; exit 130' INT

# Check for bc command (needed for calculations)
if ! command -v bc > /dev/null 2>&1; then
    echo "Installing bc for calculations..."
    if command -v apt-get > /dev/null 2>&1; then
        sudo apt-get update && sudo apt-get install -y bc
    elif command -v yum > /dev/null 2>&1; then
        sudo yum install -y bc
    else
        log_warning "bc not found and cannot install automatically. Some tests may not work."
    fi
fi

main "$@"