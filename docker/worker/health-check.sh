#!/bin/bash

# CPU Worker health check script
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

# Test Hashcat
if ! hashcat --version > /dev/null 2>&1; then
    echo "❌ Worker health check failed - Hashcat not working"
    exit 1
fi

echo "✅ Worker health check passed"
exit 0