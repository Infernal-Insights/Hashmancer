#!/bin/bash
set -e

echo "🔧 Starting Hashmancer Worker..."

# Detect worker type
if command -v nvidia-smi > /dev/null 2>&1; then
    export WORKER_TYPE="gpu"
    echo "🎮 GPU worker detected"
    
    # Show GPU information
    echo "📊 GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits
    
    # Test GPU access
    if python -c "import torch; print(f'✅ CUDA available: {torch.cuda.is_available()}'); print(f'📱 GPU count: {torch.cuda.device_count()}')" 2>/dev/null; then
        echo "✅ GPU access confirmed"
    else
        echo "⚠️  GPU access test failed, but continuing..."
    fi
else
    export WORKER_TYPE="cpu"
    echo "💻 CPU worker mode"
fi

# Wait for server to be ready
echo "⏳ Waiting for server..."
while ! curl -f -s http://${SERVER_HOST:-server}:${SERVER_PORT:-8080}/health > /dev/null 2>&1; do
    echo "   Server not ready, waiting 5 seconds..."
    sleep 5
done
echo "✅ Server is ready!"

# Wait for Redis to be ready
echo "⏳ Waiting for Redis..."
while ! python -c "
import redis
import os
try:
    r = redis.Redis(host=os.getenv('REDIS_HOST', 'redis'), port=int(os.getenv('REDIS_PORT', '6379')))
    r.ping()
    print('✅ Redis connection successful')
except Exception as e:
    print(f'❌ Redis connection failed: {e}')
    exit(1)
"; do
    echo "   Redis not ready, waiting 3 seconds..."
    sleep 3
done

# Test Hashcat installation
echo "🔍 Testing Hashcat installation..."
if hashcat --version > /dev/null 2>&1; then
    echo "✅ Hashcat is installed and working"
    hashcat --version | head -1
else
    echo "❌ Hashcat test failed"
    exit 1
fi

# Test Darkling availability
echo "🔍 Testing Darkling availability..."
if python -c "import hashmancer.darkling; print('✅ Darkling is available')" 2>/dev/null; then
    echo "✅ Darkling is available"
else
    echo "⚠️  Darkling test failed, but continuing..."
fi

# Set up logging
mkdir -p /app/logs
export LOG_FILE="/app/logs/worker-${WORKER_ID:-unknown}-$(date +%Y%m%d).log"

echo "📝 Logging to: $LOG_FILE"

# Set worker ID if not provided
if [ -z "$WORKER_ID" ]; then
    export WORKER_ID="${WORKER_TYPE}-worker-$(hostname)"
    echo "🏷️  Generated worker ID: $WORKER_ID"
fi

echo "🚀 Starting worker with configuration:"
echo "   Worker ID: $WORKER_ID"
echo "   Worker Type: $WORKER_TYPE"
echo "   Server: ${SERVER_HOST:-server}:${SERVER_PORT:-8080}"
echo "   Redis: ${REDIS_HOST:-redis}:${REDIS_PORT:-6379}"
echo "   Default Engine: ${DEFAULT_ENGINE:-darkling}"

# Start the worker
cd /app
exec python -m hashmancer.worker.production_worker \
    --worker-id "$WORKER_ID" \
    --server-host "${SERVER_HOST:-server}" \
    --server-port "${SERVER_PORT:-8080}" \
    --log-level "${LOG_LEVEL:-INFO}" \
    2>&1 | tee -a "$LOG_FILE"