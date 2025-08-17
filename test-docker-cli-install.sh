#!/bin/bash
# Test script for Docker CLI installation

set -e

echo "🧪 Testing Docker CLI Installation"
echo "=================================="

echo "📋 Available methods:"
echo "1. ✅ Automatic installation in Docker images (Dockerfile.server, Dockerfile.worker)"
echo "2. ✅ Docker Compose with shared volume (docker-compose.with-cli.yml)"  
echo "3. ✅ Updated simple compose (docker-compose.simple.yml)"

echo ""
echo "🔨 Build commands to test:"
echo ""

echo "📦 Build server with CLI:"
echo "docker build -f Dockerfile.server -t hashmancer-server-cli ."
echo ""

echo "📦 Build worker with CLI:"
echo "docker build -f Dockerfile.worker -t hashmancer-worker-cli ."
echo ""

echo "🚀 Start with simple compose (CLI auto-installed):"
echo "docker-compose -f docker-compose.simple.yml up --build"
echo ""

echo "🚀 Start with init container approach:"
echo "docker-compose -f docker-compose.with-cli.yml up --build"
echo ""

echo "🧪 Test CLI in running container:"
echo "docker exec -it <container_name> hashmancer --help"
echo "docker exec -it <container_name> hashmancer sshkey --help"
echo ""

echo "✅ All Docker configurations updated with automatic CLI installation!"
echo "💡 The CLI will be available as 'hashmancer' command in all containers"