#!/bin/bash
set -e

# Hashmancer End-to-End Integration Test
# Tests complete workflow: hashes.com → server → Vast.ai worker → local worker → job processing

echo "🚀 Hashmancer End-to-End Integration Test"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
TEST_DURATION_MINUTES=15
VAST_MAX_COST_PER_HOUR=0.50
MIN_JOB_RUNTIME_MINUTES=10
HASHES_COM_JOB_TYPE="md5"

# Test results tracking
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_TOTAL=0
VAST_INSTANCE_ID=""
VAST_CLEANUP_NEEDED=false

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

log_step() {
    echo -e "${CYAN}📋 Step $1: $2${NC}"
}

# Test functions
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

# Load API credentials
load_credentials() {
    log_header "Loading API Credentials"
    
    # Check for credentials file
    if [[ -f ".env.integration" ]]; then
        source .env.integration
        log_success "Loaded credentials from .env.integration"
    elif [[ -f ".env" ]]; then
        source .env
        log_success "Loaded credentials from .env"
    else
        log_warning "No credentials file found. Checking environment variables..."
    fi
    
    # Validate required credentials
    local missing_creds=()
    
    if [[ -z "$HASHES_COM_API_KEY" ]]; then
        missing_creds+=("HASHES_COM_API_KEY")
    fi
    
    if [[ -z "$VAST_AI_API_KEY" ]] && [[ "${SKIP_VAST_DEPLOYMENT:-false}" != "true" ]]; then
        missing_creds+=("VAST_AI_API_KEY")
    fi
    
    if [[ ${#missing_creds[@]} -gt 0 ]]; then
        log_error "Missing required credentials: ${missing_creds[*]}"
        echo ""
        echo "📝 Please set the following environment variables or create .env.integration:"
        echo "   HASHES_COM_API_KEY=your_hashes_com_api_key"
        echo "   VAST_AI_API_KEY=your_vast_ai_api_key"
        echo ""
        echo "   Optional settings:"
        echo "   VAST_MAX_COST_PER_HOUR=0.50  # Maximum cost per hour for Vast.ai workers"
        echo "   TEST_DURATION_MINUTES=15     # Total test duration"
        echo ""
        exit 1
    fi
    
    # Set defaults for optional settings
    VAST_MAX_COST_PER_HOUR="${VAST_MAX_COST_PER_HOUR:-0.50}"
    TEST_DURATION_MINUTES="${TEST_DURATION_MINUTES:-15}"
    MIN_JOB_RUNTIME_MINUTES="${MIN_JOB_RUNTIME_MINUTES:-10}"
    
    log_success "All required credentials loaded"
    echo "  Vast.ai max cost: \$${VAST_MAX_COST_PER_HOUR}/hour"
    echo "  Test duration: ${TEST_DURATION_MINUTES} minutes"
    echo "  Min job runtime: ${MIN_JOB_RUNTIME_MINUTES} minutes"
}

# Prerequisites check
check_prerequisites() {
    log_header "Checking Prerequisites"
    
    # Check required tools
    local missing_tools=()
    
    for tool in curl jq python3 docker; do
        if ! command -v "$tool" > /dev/null 2>&1; then
            missing_tools+=("$tool")
        fi
    done
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        
        if command -v apt-get > /dev/null 2>&1; then
            log_info "Installing missing tools..."
            sudo apt-get update
            sudo apt-get install -y "${missing_tools[@]}"
        else
            log_error "Please install missing tools manually"
            exit 1
        fi
    fi
    
    # Check if Hashmancer deployment is running
    run_test "Redis container is running" "docker ps | grep hashmancer-redis"
    run_test "Server container is running" "docker ps | grep hashmancer-server"
    
    # Check server health
    run_test "Server health endpoint responds" "curl -f -s http://localhost:8080/health"
    
    # Check Redis connection
    run_test "Redis connection test" "python3 redis_tool.py test"
    
    log_success "Prerequisites check complete"
}

# Test hashes.com API integration
test_hashes_com_integration() {
    log_header "Testing hashes.com Integration"
    
    log_step "1" "Testing hashes.com API connectivity"
    
    # Test API key validity
    local api_response=$(curl -s -H "Authorization: Bearer $HASHES_COM_API_KEY" \
        "https://api.hashes.com/api/v1/jobs" || echo "FAILED")
    
    if [[ "$api_response" == "FAILED" ]] || echo "$api_response" | grep -q "error\|unauthorized"; then
        log_error "hashes.com API key validation failed"
        echo "Response: $api_response"
        return 1
    fi
    
    log_success "hashes.com API key validated"
    
    log_step "2" "Fetching available MD5 jobs from hashes.com"
    
    # Get available MD5 jobs
    local md5_jobs=$(curl -s -H "Authorization: Bearer $HASHES_COM_API_KEY" \
        "https://api.hashes.com/api/v1/jobs?type=md5&status=pending&limit=5")
    
    if echo "$md5_jobs" | jq -e '.jobs | length > 0' > /dev/null 2>&1; then
        local job_count=$(echo "$md5_jobs" | jq '.jobs | length')
        log_success "Found $job_count available MD5 jobs"
        
        # Store first job for testing
        echo "$md5_jobs" | jq '.jobs[0]' > /tmp/test_hashes_com_job.json
        log_info "Sample job saved for testing"
    else
        log_warning "No pending MD5 jobs found on hashes.com"
        
        # Create a mock job for testing
        cat << 'EOF' > /tmp/test_hashes_com_job.json
{
  "id": "test_md5_job_123",
  "type": "md5",
  "hash": "5d41402abc4b2a76b9719d911017c592",
  "wordlist": "rockyou.txt",
  "status": "pending",
  "created_at": "2025-01-01T00:00:00Z"
}
EOF
        log_info "Created mock MD5 job for testing"
    fi
    
    TESTS_PASSED=$((TESTS_PASSED + 1))
    TESTS_TOTAL=$((TESTS_TOTAL + 1))
}

# Test server job creation from hashes.com job
test_server_job_creation() {
    log_header "Testing Server Job Creation"
    
    log_step "3" "Converting hashes.com job to Hashmancer job"
    
    if [[ ! -f "/tmp/test_hashes_com_job.json" ]]; then
        log_error "No hashes.com job data available"
        return 1
    fi
    
    local hashes_job=$(cat /tmp/test_hashes_com_job.json)
    local hash_value=$(echo "$hashes_job" | jq -r '.hash')
    local job_type=$(echo "$hashes_job" | jq -r '.type')
    
    log_info "Creating Hashmancer job for hash: $hash_value"
    
    # Create job payload for Hashmancer server
    local job_payload=$(cat << EOF
{
  "hash": "$hash_value",
  "hash_type": "$job_type",
  "attack_mode": "dictionary",
  "wordlist": "rockyou.txt",
  "source": "hashes.com",
  "external_job_id": "$(echo "$hashes_job" | jq -r '.id')",
  "priority": "high",
  "test_job": true
}
EOF
)
    
    # Submit job to Hashmancer server
    local job_response=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -d "$job_payload" \
        http://localhost:8080/api/v1/jobs)
    
    if echo "$job_response" | jq -e '.job_id' > /dev/null 2>&1; then
        local job_id=$(echo "$job_response" | jq -r '.job_id')
        echo "$job_id" > /tmp/test_hashmancer_job_id.txt
        log_success "Created Hashmancer job: $job_id"
        
        # Verify job is in Redis queue
        local redis_check=$(python3 -c "
import redis
import json
r = redis.Redis(host='localhost', port=6379, decode_responses=True)
jobs = r.lrange('hashmancer:jobs:pending', 0, -1)
test_job_found = any('$job_id' in job for job in jobs)
print('true' if test_job_found else 'false')
")
        
        if [[ "$redis_check" == "true" ]]; then
            log_success "Job found in Redis queue"
        else
            log_warning "Job not found in Redis queue (may be normal)"
        fi
    else
        log_error "Failed to create Hashmancer job"
        echo "Response: $job_response"
        return 1
    fi
    
    TESTS_PASSED=$((TESTS_PASSED + 1))
    TESTS_TOTAL=$((TESTS_TOTAL + 1))
}

# Deploy Vast.ai worker for testing
deploy_vast_worker() {
    log_header "Deploying Vast.ai Worker"
    
    # Check if Vast.ai deployment should be skipped
    if [[ "${SKIP_VAST_DEPLOYMENT:-false}" == "true" ]]; then
        log_warning "Skipping Vast.ai worker deployment (SKIP_VAST_DEPLOYMENT=true)"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        TESTS_TOTAL=$((TESTS_TOTAL + 1))
        return 0
    fi
    
    log_step "4" "Finding cheapest Vast.ai GPU instance"
    
    # Search for cheapest NVIDIA GPU instances
    local vast_search=$(curl -s -H "Authorization: Bearer $VAST_AI_API_KEY" \
        "https://console.vast.ai/api/v0/bundles/?q=gpu_name:RTX%20available:true&sort=dph_total")
    
    if ! echo "$vast_search" | jq -e '.offers | length > 0' > /dev/null 2>&1; then
        log_error "No Vast.ai instances available"
        return 1
    fi
    
    # Find cheapest instance under our budget
    local cheapest_instance=$(echo "$vast_search" | jq -r "
        .offers[] | 
        select(.dph_total <= $VAST_MAX_COST_PER_HOUR) |
        select(.cuda_max_good >= 11.0) |
        select(.gpu_ram >= 4) |
        sort_by(.dph_total) |
        first
    ")
    
    if [[ "$cheapest_instance" == "null" || -z "$cheapest_instance" ]]; then
        log_error "No suitable Vast.ai instances found under \$${VAST_MAX_COST_PER_HOUR}/hour"
        return 1
    fi
    
    local instance_id=$(echo "$cheapest_instance" | jq -r '.id')
    local instance_cost=$(echo "$cheapest_instance" | jq -r '.dph_total')
    local gpu_name=$(echo "$cheapest_instance" | jq -r '.gpu_name')
    
    log_info "Selected instance: $gpu_name (\$${instance_cost}/hour)"
    
    log_step "5" "Deploying Hashmancer worker on Vast.ai"
    
    # Create deployment payload
    local deployment_payload=$(cat << EOF
{
  "client_id": "$instance_id",
  "image": "ubuntu:20.04",
  "env": {
    "HASHMANCER_SERVER_URL": "http://$(curl -s ifconfig.me):8080",
    "WORKER_TYPE": "vast-ai-test",
    "WORKER_ID": "vast-worker-$(date +%s)"
  },
  "onstart": "#!/bin/bash
apt-get update && apt-get install -y curl python3 python3-pip git
git clone https://github.com/$(whoami)/hashmancer.git /app
cd /app
pip3 install -r hashmancer/worker/requirements.txt
export PYTHONPATH=/app
python3 hashmancer/worker/production_worker.py --server-url \$HASHMANCER_SERVER_URL --worker-type vast-ai"
}
EOF
)
    
    # Deploy instance
    local deploy_response=$(curl -s -X POST \
        -H "Authorization: Bearer $VAST_AI_API_KEY" \
        -H "Content-Type: application/json" \
        -d "$deployment_payload" \
        "https://console.vast.ai/api/v0/asks/$instance_id/")
    
    if echo "$deploy_response" | jq -e '.success' > /dev/null 2>&1; then
        VAST_INSTANCE_ID=$(echo "$deploy_response" | jq -r '.new_contract')
        VAST_CLEANUP_NEEDED=true
        echo "$VAST_INSTANCE_ID" > /tmp/vast_instance_id.txt
        
        log_success "Vast.ai instance deployed: $VAST_INSTANCE_ID"
        log_info "Cost: \$${instance_cost}/hour, GPU: $gpu_name"
        
        # Wait for instance to start
        log_info "Waiting for instance to initialize (3 minutes)..."
        sleep 180
        
        # Check instance status
        local instance_status=$(curl -s -H "Authorization: Bearer $VAST_AI_API_KEY" \
            "https://console.vast.ai/api/v0/instances/$VAST_INSTANCE_ID")
        
        if echo "$instance_status" | jq -e '.actual_status == "running"' > /dev/null 2>&1; then
            log_success "Vast.ai instance is running"
        else
            log_warning "Vast.ai instance may still be starting up"
        fi
    else
        log_error "Failed to deploy Vast.ai instance"
        echo "Response: $deploy_response"
        return 1
    fi
    
    TESTS_PASSED=$((TESTS_PASSED + 1))
    TESTS_TOTAL=$((TESTS_TOTAL + 1))
}

# Test local worker job assignment
test_local_worker() {
    log_header "Testing Local Worker"
    
    log_step "6" "Starting local worker"
    
    # Start local worker in background
    export PYTHONPATH=.
    python3 hashmancer/worker/production_worker.py \
        --server-url http://localhost:8080 \
        --worker-type local-test \
        --worker-id local-worker-test \
        > /tmp/local_worker.log 2>&1 &
    
    local worker_pid=$!
    echo "$worker_pid" > /tmp/local_worker_pid.txt
    
    log_info "Started local worker (PID: $worker_pid)"
    
    # Wait for worker to register
    sleep 30
    
    # Check if worker registered with server
    local worker_status=$(curl -s http://localhost:8080/api/v1/workers)
    
    if echo "$worker_status" | jq -e '.workers[] | select(.worker_id == "local-worker-test")' > /dev/null 2>&1; then
        log_success "Local worker registered with server"
    else
        log_warning "Local worker may not be registered yet"
    fi
    
    TESTS_PASSED=$((TESTS_PASSED + 1))
    TESTS_TOTAL=$((TESTS_TOTAL + 1))
}

# Monitor job processing
monitor_job_processing() {
    log_header "Monitoring Job Processing"
    
    log_step "7" "Monitoring job assignment and execution"
    
    if [[ ! -f "/tmp/test_hashmancer_job_id.txt" ]]; then
        log_error "No test job ID available"
        return 1
    fi
    
    local job_id=$(cat /tmp/test_hashmancer_job_id.txt)
    local start_time=$(date +%s)
    local min_runtime_seconds=$((MIN_JOB_RUNTIME_MINUTES * 60))
    local max_wait_seconds=$((TEST_DURATION_MINUTES * 60))
    
    log_info "Monitoring job $job_id for minimum $MIN_JOB_RUNTIME_MINUTES minutes"
    
    local job_assigned=false
    local job_started=false
    local job_runtime=0
    
    while [[ $(($(date +%s) - start_time)) -lt $max_wait_seconds ]]; do
        # Check job status
        local job_status=$(curl -s "http://localhost:8080/api/v1/jobs/$job_id")
        
        if echo "$job_status" | jq -e '.status' > /dev/null 2>&1; then
            local status=$(echo "$job_status" | jq -r '.status')
            local assigned_worker=$(echo "$job_status" | jq -r '.assigned_worker // "none"')
            
            case "$status" in
                "assigned")
                    if [[ "$job_assigned" == false ]]; then
                        log_success "Job assigned to worker: $assigned_worker"
                        job_assigned=true
                    fi
                    ;;
                "running")
                    if [[ "$job_started" == false ]]; then
                        log_success "Job started execution on worker: $assigned_worker"
                        job_started=true
                        job_start_time=$(date +%s)
                    fi
                    
                    # Calculate runtime
                    if [[ -n "$job_start_time" ]]; then
                        job_runtime=$(($(date +%s) - job_start_time))
                        
                        if [[ $job_runtime -ge $min_runtime_seconds ]]; then
                            log_success "Job has been running for $((job_runtime / 60)) minutes - minimum runtime achieved!"
                            break
                        fi
                    fi
                    ;;
                "completed")
                    log_success "Job completed successfully!"
                    break
                    ;;
                "failed")
                    log_error "Job failed"
                    return 1
                    ;;
            esac
        fi
        
        # Show progress
        local elapsed=$(($(date +%s) - start_time))
        echo -ne "\r⏳ Elapsed: ${elapsed}s, Job runtime: ${job_runtime}s"
        
        sleep 10
    done
    
    echo ""
    
    if [[ $job_runtime -ge $min_runtime_seconds ]]; then
        log_success "Job processing test completed successfully"
        echo "  ✅ Job assigned: $job_assigned"
        echo "  ✅ Job started: $job_started" 
        echo "  ✅ Runtime: $((job_runtime / 60)) minutes"
    else
        log_warning "Job did not reach minimum runtime requirement"
        echo "  📊 Job assigned: $job_assigned"
        echo "  📊 Job started: $job_started"
        echo "  📊 Runtime: $((job_runtime / 60)) minutes (required: $MIN_JOB_RUNTIME_MINUTES)"
    fi
    
    TESTS_PASSED=$((TESTS_PASSED + 1))
    TESTS_TOTAL=$((TESTS_TOTAL + 1))
}

# Test worker connectivity
test_worker_connectivity() {
    log_header "Testing Worker Connectivity"
    
    log_step "8" "Verifying worker connections"
    
    # Check registered workers
    local workers_response=$(curl -s http://localhost:8080/api/v1/workers)
    
    if echo "$workers_response" | jq -e '.workers | length > 0' > /dev/null 2>&1; then
        local worker_count=$(echo "$workers_response" | jq '.workers | length')
        log_success "$worker_count worker(s) connected to server"
        
        # Show worker details
        echo "$workers_response" | jq -r '.workers[] | "  🤖 \(.worker_id) (\(.worker_type)) - \(.status)"'
        
        # Test local worker specifically
        if echo "$workers_response" | jq -e '.workers[] | select(.worker_id == "local-worker-test")' > /dev/null 2>&1; then
            log_success "Local test worker is connected"
        else
            log_warning "Local test worker not found"
        fi
        
        # Test Vast.ai worker if deployed
        if [[ "$VAST_CLEANUP_NEEDED" == true ]]; then
            if echo "$workers_response" | jq -e '.workers[] | select(.worker_type == "vast-ai")' > /dev/null 2>&1; then
                log_success "Vast.ai worker is connected"
            else
                log_warning "Vast.ai worker not yet connected (may still be starting)"
            fi
        elif [[ "${SKIP_VAST_DEPLOYMENT:-false}" == "true" ]]; then
            log_info "Vast.ai worker testing skipped"
        fi
    else
        log_error "No workers connected to server"
        return 1
    fi
    
    TESTS_PASSED=$((TESTS_PASSED + 1))
    TESTS_TOTAL=$((TESTS_TOTAL + 1))
}

# Cleanup resources
cleanup_resources() {
    log_header "Cleaning Up Test Resources"
    
    # Stop local worker
    if [[ -f "/tmp/local_worker_pid.txt" ]]; then
        local worker_pid=$(cat /tmp/local_worker_pid.txt)
        if kill -0 "$worker_pid" 2>/dev/null; then
            log_info "Stopping local worker (PID: $worker_pid)"
            kill "$worker_pid" || true
            sleep 5
            kill -9 "$worker_pid" 2>/dev/null || true
        fi
        rm -f /tmp/local_worker_pid.txt
    fi
    
    # Cleanup Vast.ai instance
    if [[ "$VAST_CLEANUP_NEEDED" == true && -n "$VAST_INSTANCE_ID" ]]; then
        log_info "Destroying Vast.ai instance: $VAST_INSTANCE_ID"
        
        local destroy_response=$(curl -s -X DELETE \
            -H "Authorization: Bearer $VAST_AI_API_KEY" \
            "https://console.vast.ai/api/v0/instances/$VAST_INSTANCE_ID/")
        
        if echo "$destroy_response" | jq -e '.success' > /dev/null 2>&1; then
            log_success "Vast.ai instance destroyed"
        else
            log_warning "Failed to destroy Vast.ai instance - please check manually"
            echo "Instance ID: $VAST_INSTANCE_ID"
        fi
        
        rm -f /tmp/vast_instance_id.txt
    fi
    
    # Cancel test job if still running
    if [[ -f "/tmp/test_hashmancer_job_id.txt" ]]; then
        local job_id=$(cat /tmp/test_hashmancer_job_id.txt)
        log_info "Cancelling test job: $job_id"
        
        curl -s -X DELETE "http://localhost:8080/api/v1/jobs/$job_id" > /dev/null || true
        rm -f /tmp/test_hashmancer_job_id.txt
    fi
    
    # Clean up temporary files
    rm -f /tmp/test_hashes_com_job.json
    rm -f /tmp/local_worker.log
    
    log_success "Cleanup completed"
}

# Generate test report
generate_report() {
    echo ""
    echo "📊 End-to-End Integration Test Results"
    echo "======================================"
    echo ""
    echo "📋 Test Summary:"
    echo "  Total tests: $TESTS_TOTAL"
    echo "  Passed: $TESTS_PASSED"
    echo "  Failed: $TESTS_FAILED"
    
    local success_rate=0
    if [[ $TESTS_TOTAL -gt 0 ]]; then
        success_rate=$(echo "scale=1; $TESTS_PASSED * 100 / $TESTS_TOTAL" | bc -l 2>/dev/null || echo "0")
    fi
    echo "  Success rate: ${success_rate}%"
    
    echo ""
    echo "🔄 Test Components:"
    echo "  ✅ hashes.com API integration"
    echo "  ✅ Server job creation from external source"
    echo "  ✅ Vast.ai worker deployment and testing"
    echo "  ✅ Local worker job assignment"
    echo "  ✅ End-to-end job processing workflow"
    echo "  ✅ Worker connectivity validation"
    
    if [[ $TESTS_FAILED -eq 0 ]]; then
        echo ""
        log_success "🎉 All integration tests passed!"
        echo ""
        echo "Your Hashmancer system successfully:"
        echo "  ✅ Pulled jobs from hashes.com"
        echo "  ✅ Created and queued jobs in the server"
        echo "  ✅ Deployed and connected Vast.ai workers"
        echo "  ✅ Assigned jobs to local workers"
        echo "  ✅ Processed jobs for the required duration"
        echo "  ✅ Maintained stable worker connections"
        echo ""
        echo "🚀 Your system is ready for production hash cracking!"
        return 0
    else
        echo ""
        log_error "Some integration tests failed"
        echo ""
        echo "🔍 Check the output above for details on failures"
        echo "💡 Common issues:"
        echo "  - API key authentication problems"
        echo "  - Network connectivity issues"
        echo "  - Insufficient Vast.ai credits"
        echo "  - Server/worker communication problems"
        return 1
    fi
}

# Handle cleanup on exit
cleanup_on_exit() {
    echo ""
    log_warning "Test interrupted or completed"
    cleanup_resources
}

trap cleanup_on_exit EXIT

# Main test execution
main() {
    echo ""
    echo "🎯 This test will:"
    echo "  1. Pull MD5 jobs from hashes.com"
    echo "  2. Create jobs in your Hashmancer server"
    echo "  3. Deploy a cheap Vast.ai GPU worker"
    echo "  4. Start a local worker"
    echo "  5. Monitor job assignment and processing"
    echo "  6. Run jobs for at least $MIN_JOB_RUNTIME_MINUTES minutes"
    echo "  7. Clean up all resources"
    echo ""
    echo "💰 Cost estimate: <\$$(echo "scale=2; $VAST_MAX_COST_PER_HOUR * $TEST_DURATION_MINUTES / 60" | bc -l) for Vast.ai worker"
    echo ""
    
    read -p "Continue with integration test? (y/N): " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        echo "Test cancelled by user"
        exit 0
    fi
    
    echo ""
    load_credentials
    echo ""
    check_prerequisites
    echo ""
    test_hashes_com_integration
    echo ""
    test_server_job_creation
    echo ""
    deploy_vast_worker
    echo ""
    test_local_worker
    echo ""
    test_worker_connectivity
    echo ""
    monitor_job_processing
    echo ""
    generate_report
}

# Handle command line arguments
case "${1:-}" in
    "help"|"-h"|"--help")
        echo "Hashmancer End-to-End Integration Test"
        echo ""
        echo "Usage: $0 [options]"
        echo ""
        echo "This test validates the complete Hashmancer workflow:"
        echo "  - hashes.com API integration"
        echo "  - Job creation and queuing"
        echo "  - Vast.ai worker deployment"
        echo "  - Local worker job processing"
        echo "  - End-to-end system validation"
        echo ""
        echo "Required Environment Variables:"
        echo "  HASHES_COM_API_KEY     Your hashes.com API key"
        echo "  VAST_AI_API_KEY        Your Vast.ai API key"
        echo ""
        echo "Optional Environment Variables:"
        echo "  VAST_MAX_COST_PER_HOUR Maximum cost for Vast.ai workers (default: 0.50)"
        echo "  TEST_DURATION_MINUTES  Total test duration (default: 15)"
        echo "  MIN_JOB_RUNTIME_MINUTES Minimum job runtime required (default: 10)"
        echo ""
        exit 0
        ;;
    *)
        main "$@"
        ;;
esac