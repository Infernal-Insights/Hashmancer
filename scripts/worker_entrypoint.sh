#!/bin/bash
# Hashmancer Worker Entrypoint Script

set -e

echo "🔓 Hashmancer Production Worker Starting..."
echo "============================================="

# Display configuration
echo "Worker Configuration:"
echo "  Worker ID: ${WORKER_ID:-auto-generated}"
echo "  Server: ${HASHMANCER_SERVER_IP:-not_set}:${HASHMANCER_SERVER_PORT:-8080}"
echo "  Worker Port: ${WORKER_PORT:-8081}"
echo "  Max Jobs: ${MAX_CONCURRENT_JOBS:-3}"
echo "  Log Level: ${LOG_LEVEL:-INFO}"

# Validate required environment
if [ -z "$HASHMANCER_SERVER_IP" ]; then
    echo "❌ ERROR: HASHMANCER_SERVER_IP environment variable is required"
    echo "   Set it when running the container:"
    echo "   docker run -e HASHMANCER_SERVER_IP=your.server.ip hashmancer/worker"
    exit 1
fi

# Create log directory
mkdir -p /app/logs

# Test server connectivity
echo "🔍 Testing server connectivity..."
if timeout 10 curl -s "http://${HASHMANCER_SERVER_IP}:${HASHMANCER_SERVER_PORT:-8080}/health" > /dev/null; then
    echo "✅ Server is reachable"
else
    echo "❌ WARNING: Cannot reach server at ${HASHMANCER_SERVER_IP}:${HASHMANCER_SERVER_PORT:-8080}"
    echo "   The worker will keep trying to connect..."
fi

# Check for GPU support
echo "🎮 Checking GPU support..."
if nvidia-smi > /dev/null 2>&1; then
    echo "✅ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    
    # Run comprehensive GPU verification
    echo "🔍 Running GPU verification..."
    python3 /app/verify_gpu_setup.py
    
    if [ $? -eq 0 ]; then
        echo "✅ GPU verification passed"
    else
        echo "⚠️  GPU verification found issues (continuing anyway)"
    fi
else
    echo "ℹ️  No NVIDIA GPU detected (CPU-only mode)"
fi

# Set default worker ID if not provided
if [ -z "$WORKER_ID" ]; then
    export WORKER_ID="vast-$(hostname)-$(date +%s)"
    echo "📝 Generated Worker ID: $WORKER_ID"
fi

# Start the worker
echo "🚀 Starting Hashmancer worker..."
echo "Press Ctrl+C to stop"
echo "============================================="

# Execute the Python worker
exec python3 /app/worker.py