#!/bin/bash
set -e

echo "🚀 Starting Hashmancer Server..."

# Wait for Redis to be ready
echo "⏳ Waiting for Redis..."
while ! python /app/redis_tool.py test > /dev/null 2>&1; do
    echo "   Redis not ready, waiting 2 seconds..."
    sleep 2
done
echo "✅ Redis is ready!"

# Run Redis health check and optimization
echo "🔍 Running Redis health check..."
python /app/redis_tool.py health --quick

# Cleanup any stale Redis data
echo "🧹 Cleaning up stale Redis data..."
python /app/redis_tool.py cleanup --dry-run

# Set up logging
mkdir -p /app/logs
export LOG_FILE="/app/logs/server-$(date +%Y%m%d).log"

# Start the server based on environment
if [ "$BUILD_ENV" = "development" ]; then
    echo "🔧 Starting in development mode..."
    cd /app && python -m hashmancer.server.main \
        --host 0.0.0.0 \
        --port 8080 \
        --reload \
        2>&1 | tee -a "$LOG_FILE"
else
    echo "🚀 Starting in production mode with Gunicorn..."
    
    # Start with Gunicorn for production
    exec gunicorn \
        --bind 0.0.0.0:8080 \
        --workers ${WORKERS:-4} \
        --worker-class uvicorn.workers.UvicornWorker \
        --max-requests ${MAX_REQUESTS:-1000} \
        --max-requests-jitter ${MAX_REQUESTS_JITTER:-50} \
        --timeout 120 \
        --keep-alive 5 \
        --access-logfile "$LOG_FILE" \
        --error-logfile "$LOG_FILE" \
        --log-level ${LOG_LEVEL:-info} \
        --preload \
        hashmancer.server.app.app:app
fi