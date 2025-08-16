#!/bin/bash
# Test the Docker setup without actually deploying to Vast.ai

echo "🧪 Testing Hashmancer Docker Setup"
echo "=================================="

# Test 1: Check if Docker files exist
echo "📋 Checking Docker files..."
files=(
    "docker/Dockerfile.worker"
    "hashmancer/worker/production_worker.py"
    "hashmancer/worker/health_server.py"
    "scripts/worker_entrypoint.sh"
    "scripts/build-and-push-docker.sh"
    "scripts/vast-deploy-prebuilt.sh"
)

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file exists"
    else
        echo "❌ $file missing"
    fi
done

# Test 2: Check CLI commands
echo ""
echo "🔧 Testing CLI commands..."
if ./hashmancer-cli --help > /dev/null 2>&1; then
    echo "✅ Main CLI works"
else
    echo "❌ Main CLI not working"
fi

if ./hashmancer-cli server --help > /dev/null 2>&1; then
    echo "✅ Server CLI works"
else
    echo "❌ Server CLI not working"
fi

if ./hashmancer-cli worker --help > /dev/null 2>&1; then
    echo "✅ Worker CLI works"
else
    echo "❌ Worker CLI not working"
fi

# Test 3: Check script permissions
echo ""
echo "🔑 Checking script permissions..."
scripts=(
    "scripts/build-and-push-docker.sh"
    "scripts/vast-deploy-prebuilt.sh"
    "hashmancer-cli"
    "demo-vast-setup.sh"
)

for script in "${scripts[@]}"; do
    if [ -x "$script" ]; then
        echo "✅ $script is executable"
    else
        echo "❌ $script is not executable"
        chmod +x "$script" 2>/dev/null && echo "   Fixed permissions" || echo "   Could not fix permissions"
    fi
done

# Test 4: Show Docker command that would be used
echo ""
echo "🐳 Docker build command:"
echo "docker build -f docker/Dockerfile.worker -t hashmancer/worker:latest ."

# Test 5: Show Vast.ai setup command
echo ""
echo "🚀 Example Vast.ai deployment:"
echo "./scripts/vast-deploy-prebuilt.sh \\"
echo "  --server-ip YOUR_PUBLIC_IP \\"
echo "  --gpu-type 3080 \\"
echo "  --count 2 \\"
echo "  --max-price 0.75 \\"
echo "  --api-key YOUR_VAST_API_KEY"

# Test 6: Check environment setup
echo ""
echo "🌍 Environment check:"
if [ -n "$VAST_API_KEY" ]; then
    echo "✅ VAST_API_KEY is set"
else
    echo "⚠️  VAST_API_KEY not set (export VAST_API_KEY=your_key)"
fi

if [ -n "$HASHMANCER_SERVER_IP" ]; then
    echo "✅ HASHMANCER_SERVER_IP is set to: $HASHMANCER_SERVER_IP"
else
    echo "ℹ️  HASHMANCER_SERVER_IP not set (will auto-detect or prompt)"
fi

echo ""
echo "📖 Next steps:"
echo "1. Install Docker if you haven't already"
echo "2. Run: ./scripts/build-and-push-docker.sh"
echo "3. Make sure your server is publicly accessible"
echo "4. Deploy workers with: ./scripts/vast-deploy-prebuilt.sh"
echo ""
echo "📚 See DOCKER_SETUP_GUIDE.md for complete instructions"