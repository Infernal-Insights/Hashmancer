#!/bin/bash
# Build and Push Hashmancer Worker Docker Image

set -e

# Configuration - CHANGE THIS TO YOUR DOCKER HUB USERNAME
DOCKER_USERNAME="${DOCKER_USERNAME:-hashmancer}"  # Change this!
IMAGE_NAME="$DOCKER_USERNAME/hashmancer-worker"
IMAGE_TAG="latest"
FULL_IMAGE_NAME="$IMAGE_NAME:$IMAGE_TAG"

echo "🐳 Building Hashmancer Worker Docker Image"
echo "==========================================="
echo "Image: $FULL_IMAGE_NAME"

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first:"
    echo "   https://docs.docker.com/engine/install/"
    exit 1
fi

# Build the image
echo "🔨 Building Docker image..."
docker build -f docker/Dockerfile.worker -t "$FULL_IMAGE_NAME" . || {
    echo "❌ Docker build failed"
    exit 1
}

echo "✅ Docker image built successfully"

# Test the image
echo "🧪 Testing Docker image..."
docker run --rm --name hashmancer-test \
    -e HASHMANCER_SERVER_IP=test.server.com \
    -e WORKER_ID=test-worker \
    "$FULL_IMAGE_NAME" &

# Wait a moment for container to start
sleep 5

# Check if container is running
if docker ps | grep hashmancer-test > /dev/null; then
    echo "✅ Container is running"
    
    # Test health endpoint
    if docker exec hashmancer-test curl -s http://localhost:8081/health > /dev/null; then
        echo "✅ Health endpoint is working"
    else
        echo "⚠️  Health endpoint not responding (may be normal during startup)"
    fi
    
    # Stop test container
    docker stop hashmancer-test || true
else
    echo "⚠️  Test container not running (check logs)"
fi

# Show image info
echo ""
echo "📊 Image Information:"
docker images "$IMAGE_NAME" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedSince}}"

echo ""
echo "🚀 Image built successfully: $FULL_IMAGE_NAME"
echo ""
echo "📋 Next Steps:"
echo "1. Test the image locally:"
echo "   docker run -e HASHMANCER_SERVER_IP=your.server.ip $FULL_IMAGE_NAME"
echo ""
echo "2. Push to Docker Hub (optional):"
echo "   docker login"
echo "   docker push $FULL_IMAGE_NAME"
echo ""
echo "3. Use in Vast.ai:"
echo "   Image: $FULL_IMAGE_NAME"
echo "   Environment: HASHMANCER_SERVER_IP=your.server.ip"
echo ""

# Optionally push to Docker Hub
read -p "🤔 Do you want to push to Docker Hub? (y/N): " push_choice
if [[ $push_choice =~ ^[Yy]$ ]]; then
    echo "📤 Pushing to Docker Hub..."
    
    # Check if logged in
    if ! docker info | grep Username > /dev/null 2>&1; then
        echo "🔐 Please log in to Docker Hub:"
        docker login
    fi
    
    # Push the image
    docker push "$FULL_IMAGE_NAME" || {
        echo "❌ Push failed. Make sure you have permission to push to $IMAGE_NAME"
        exit 1
    }
    
    echo "✅ Image pushed to Docker Hub: $FULL_IMAGE_NAME"
    echo "🌐 Now anyone can use: docker pull $FULL_IMAGE_NAME"
else
    echo "ℹ️  Skipping Docker Hub push"
fi

echo ""
echo "🎉 Docker image is ready!"