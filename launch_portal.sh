#!/bin/bash
set -e

# Activate virtual environment
source venv/bin/activate

# Load environment
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Check API key
if [ -z "$VAST_API_KEY" ]; then
    echo "❌ Please set VAST_API_KEY in .env file"
    exit 1
fi

echo "🚀 Starting Hashmancer Portal..."
echo "Portal URL: http://localhost:${PORTAL_PORT:-8080}"
echo "API Key: ${VAST_API_KEY:0:10}..."

# Start the portal
cd server
python worker_portal.py