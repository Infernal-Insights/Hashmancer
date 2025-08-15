#!/bin/bash
set -e

echo "🚀 Starting Hashmancer Worker on Vast.ai"
echo "========================================"

# Print system info
echo "📊 System Information:"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
echo "GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1) MB"
echo "CUDA Version: $(nvcc --version | grep release | awk '{print $6}' | cut -c2-)"
echo "Worker ID: ${HASHMANCER_WORKER_ID:-unknown}"
echo "Job ID: ${HASHMANCER_JOB_ID:-none}"
echo "Server URL: ${HASHMANCER_SERVER_URL:-not_set}"

# Update Hashmancer to latest version
echo "🔄 Updating Hashmancer..."
cd /workspace/hashmancer
git pull origin main

# Rebuild if needed
echo "🔨 Ensuring Darkling is up to date..."
cd darkling/build
make -j$(nproc)

# Install any new Python dependencies
echo "📦 Updating Python dependencies..."
cd /workspace/hashmancer
pip install -r requirements.txt

# Set up worker configuration
echo "⚙️ Configuring worker..."
cat > worker_config.json << EOF
{
    "server_url": "${HASHMANCER_SERVER_URL}",
    "worker_id": "${HASHMANCER_WORKER_ID}",
    "job_id": "${HASHMANCER_JOB_ID}",
    "api_key": "${HASHMANCER_API_KEY}",
    "gpu_devices": "auto",
    "heartbeat_interval": 30,
    "max_job_runtime": 7200,
    "auto_assign": true,
    "performance_monitoring": true,
    "log_level": "INFO"
}
EOF

# Create logs directory
mkdir -p logs

# Start worker with proper logging
echo "🎯 Starting Hashmancer worker..."
echo "Configuration:"
cat worker_config.json | jq .

# Run the worker
if [ -f "worker/worker_main.py" ]; then
    python3 worker/worker_main.py --config worker_config.json 2>&1 | tee logs/worker.log
else
    echo "❌ Worker script not found! Running interactive mode..."
    echo "Available commands:"
    echo "  ./darkling/build/darkling --help"
    echo "  python3 -m hashmancer.cli --help"
    echo ""
    echo "To run hashcat directly:"
    echo "  hashcat -m 0 hashes.txt wordlist.txt"
    echo ""
    echo "Entering interactive shell..."
    /bin/bash
fi