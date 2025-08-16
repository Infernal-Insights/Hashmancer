#!/bin/bash

# GPU Worker health check script
set -e

# Check if worker process is running
if ! pgrep -f "hashmancer.worker" > /dev/null 2>&1; then
    echo "❌ Worker health check failed - Worker process not found"
    exit 1
fi

# Test Redis connection
if ! python -c "
import redis
import os
try:
    r = redis.Redis(host=os.getenv('REDIS_HOST', 'redis'), port=int(os.getenv('REDIS_PORT', '6379')))
    r.ping()
except Exception:
    exit(1)
" > /dev/null 2>&1; then
    echo "❌ Worker health check failed - Redis connection failed"
    exit 1
fi

# Test server connection
if ! curl -f -s "http://${SERVER_HOST:-server}:${SERVER_PORT:-8080}/health" > /dev/null 2>&1; then
    echo "❌ Worker health check failed - Server connection failed"
    exit 1
fi

# GPU-specific checks
if command -v nvidia-smi > /dev/null 2>&1; then
    # Check GPU is accessible
    if ! nvidia-smi > /dev/null 2>&1; then
        echo "❌ GPU health check failed - nvidia-smi not working"
        exit 1
    fi
    
    # Check GPU memory usage isn't too high
    GPU_MEMORY_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    GPU_MEMORY_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    
    if [ "$GPU_MEMORY_USED" -gt 0 ] && [ "$GPU_MEMORY_TOTAL" -gt 0 ]; then
        GPU_USAGE_PERCENT=$((GPU_MEMORY_USED * 100 / GPU_MEMORY_TOTAL))
        if [ "$GPU_USAGE_PERCENT" -gt 95 ]; then
            echo "⚠️  GPU memory usage very high: ${GPU_USAGE_PERCENT}%"
        fi
    fi
fi

# Test Hashcat
if ! hashcat --version > /dev/null 2>&1; then
    echo "❌ Worker health check failed - Hashcat not working"
    exit 1
fi

echo "✅ Worker health check passed"
exit 0