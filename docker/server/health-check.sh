#!/bin/bash

# Health check script for Hashmancer server
set -e

# Check if server is responding
if ! curl -f -s http://localhost:8080/health > /dev/null 2>&1; then
    echo "❌ Server health check failed - HTTP endpoint not responding"
    exit 1
fi

# Check Redis connection
if ! python /app/redis_tool.py test > /dev/null 2>&1; then
    echo "❌ Server health check failed - Redis connection failed"
    exit 1
fi

# Check if main process is running
if ! pgrep -f "hashmancer.server" > /dev/null 2>&1; then
    echo "❌ Server health check failed - Main process not found"
    exit 1
fi

echo "✅ Server health check passed"
exit 0