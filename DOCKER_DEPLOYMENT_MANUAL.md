# Hashmancer Docker Deployment Manual

Complete guide for deploying Hashmancer using Docker in multiple environments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Docker Hub Setup](#docker-hub-setup)
3. [Building Docker Images](#building-docker-images)
4. [Publishing to Docker Hub](#publishing-to-docker-hub)
5. [Deployment Scenarios](#deployment-scenarios)
   - [Simple (All-in-One)](#simple-all-in-one-deployment)
   - [Server Only](#server-only-deployment)
   - [Worker Only](#worker-only-deployment)
   - [Distributed Setup](#distributed-setup)
6. [Vast.ai Integration](#vastai-integration)
7. [Environment Variables](#environment-variables)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software
- Docker (20.10+)
- Docker Compose (1.29+)
- Git

### Required Accounts
- Docker Hub account (for publishing images)
- Optional: Vast.ai account (for GPU workers)

### Installation
```bash
# Install Docker (Ubuntu/Debian)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Verify installation
docker --version
docker-compose --version
```

---

## Docker Hub Setup

### 1. Create Docker Hub Account
- Go to https://hub.docker.com
- Create account (free tier is sufficient)
- Note your username

### 2. Login to Docker Hub
```bash
docker login
# Enter your Docker Hub username and password
```

### 3. Create Repositories (Optional)
- Go to Docker Hub dashboard
- Create repositories:
  - `your-username/hashmancer-server`
  - `your-username/hashmancer-worker`

---

## Building Docker Images

### 1. Clone Repository
```bash
git clone https://github.com/your-username/hashmancer.git
cd hashmancer
```

### 2. Build Server Image
```bash
# Build server image
docker build -f Dockerfile.server -t hashmancer-server:latest .

# Tag for Docker Hub (replace 'yourusername' with your Docker Hub username)
docker tag hashmancer-server:latest yourusername/hashmancer-server:latest
```

### 3. Build Worker Image
```bash
# Build worker image
docker build -f Dockerfile.worker -t hashmancer-worker:latest .

# Tag for Docker Hub
docker tag hashmancer-worker:latest yourusername/hashmancer-worker:latest
```

### 4. Verify Images
```bash
docker images | grep hashmancer
```

---

## Publishing to Docker Hub

### 1. Push Server Image
```bash
docker push yourusername/hashmancer-server:latest
```

### 2. Push Worker Image
```bash
docker push yourusername/hashmancer-worker:latest
```

### 3. Verify Upload
- Go to Docker Hub dashboard
- Check that images appear in your repositories
- Note the pull commands for later use

---

## Deployment Scenarios

## Simple (All-in-One) Deployment

**Use Case:** Local development, testing, single-machine demos

### Setup
```bash
# Start all services (Redis, Server, Worker)
docker-compose -f docker-compose.simple.yml up --build

# Run in background
docker-compose -f docker-compose.simple.yml up --build -d

# View logs
docker-compose -f docker-compose.simple.yml logs -f

# Stop all services
docker-compose -f docker-compose.simple.yml down
```

### Access
- **Server UI:** http://localhost:8080
- **Redis:** localhost:6379

---

## Server Only Deployment

**Use Case:** Central server for distributed workers, cloud deployment

### Method 1: Docker Run
```bash
# Start Redis first
docker run -d --name redis -p 6379:6379 redis:7-alpine

# Start server
docker run -d \
  --name hashmancer-server \
  -p 8080:8080 \
  --link redis:redis \
  -e REDIS_URL=redis://redis:6379 \
  yourusername/hashmancer-server:latest
```

### Method 2: Docker Compose
Create `docker-compose.server.yml`:
```yaml
version: "3.8"

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    command: redis-server --appendonly yes
    
  server:
    image: yourusername/hashmancer-server:latest
    ports:
      - "8080:8080"
    depends_on:
      - redis
    environment:
      - REDIS_URL=redis://redis:6379
    restart: unless-stopped

volumes:
  redis-data:
```

```bash
docker-compose -f docker-compose.server.yml up -d
```

### Cloud Deployment (DigitalOcean/AWS/GCP)
```bash
# On cloud instance
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Download compose file
wget https://raw.githubusercontent.com/your-username/hashmancer/main/docker-compose.server.yml

# Start server
docker-compose -f docker-compose.server.yml up -d

# Check status
docker-compose -f docker-compose.server.yml ps
```

---

## Worker Only Deployment

**Use Case:** Additional compute nodes, GPU farms, Vast.ai instances

### Prerequisites
- Running server with accessible IP/port
- Redis accessible from worker

### Basic Worker Deployment
```bash
docker run -d \
  --name hashmancer-worker \
  -e SERVER_URL=http://YOUR_SERVER_IP:8080 \
  -e REDIS_URL=redis://YOUR_REDIS_IP:6379 \
  -v ~/.hashmancer:/root/.hashmancer \
  yourusername/hashmancer-worker:latest
```

### GPU Worker Deployment
```bash
# For NVIDIA GPUs
docker run -d \
  --gpus all \
  --name hashmancer-worker-gpu \
  -e SERVER_URL=http://YOUR_SERVER_IP:8080 \
  -e REDIS_URL=redis://YOUR_REDIS_IP:6379 \
  -v ~/.hashmancer:/root/.hashmancer \
  yourusername/hashmancer-worker:latest
```

### Multiple Workers (Same Machine)
```bash
# Worker 1
docker run -d --name worker-1 \
  -e SERVER_URL=http://YOUR_SERVER_IP:8080 \
  -e REDIS_URL=redis://YOUR_REDIS_IP:6379 \
  yourusername/hashmancer-worker:latest

# Worker 2
docker run -d --name worker-2 \
  -e SERVER_URL=http://YOUR_SERVER_IP:8080 \
  -e REDIS_URL=redis://YOUR_REDIS_IP:6379 \
  yourusername/hashmancer-worker:latest
```

---

## Distributed Setup

**Use Case:** Production deployments with dedicated server and multiple worker machines

### Architecture
```
[Server Machine]     [Worker Machine 1]     [Worker Machine 2]     [Vast.ai GPU]
├── Redis           ├── Worker              ├── Worker              ├── Worker
├── Server          └── Worker              └── Worker              └── Worker
└── Monitor                                 └── Worker              └── Worker
```

### Server Machine Setup
```bash
# 1. Deploy server (see Server Only section above)
docker-compose -f docker-compose.server.yml up -d

# 2. Get server IP
curl ifconfig.me

# 3. Configure firewall
sudo ufw allow 8080  # Server port
sudo ufw allow 6379  # Redis port (if external access needed)
```

### Worker Machine Setup
```bash
# On each worker machine
docker run -d \
  --name hashmancer-worker \
  --restart unless-stopped \
  -e SERVER_URL=http://SERVER_MACHINE_IP:8080 \
  -e REDIS_URL=redis://SERVER_MACHINE_IP:6379 \
  yourusername/hashmancer-worker:latest
```

---

## Vast.ai Integration

**Use Case:** Scalable GPU compute with automatic worker deployment

### 1. Prepare Vast.ai Template

Create `vast-worker-template.sh`:
```bash
#!/bin/bash

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
fi

# Set your server details
SERVER_IP="YOUR_SERVER_IP"
SERVER_PORT="8080"
REDIS_IP="YOUR_REDIS_IP"
REDIS_PORT="6379"

# Run worker
docker run -d \
  --gpus all \
  --name hashmancer-worker \
  --restart unless-stopped \
  -e SERVER_URL=http://${SERVER_IP}:${SERVER_PORT} \
  -e REDIS_URL=redis://${REDIS_IP}:${REDIS_PORT} \
  -e WORKER_NAME="vast-$(hostname)" \
  yourusername/hashmancer-worker:latest

# Monitor worker
docker logs -f hashmancer-worker
```

### 2. Vast.ai Instance Creation

**Via CLI:**
```bash
# Install vastai CLI
pip install vastai

# Login
vastai set api-key YOUR_API_KEY

# Search for instances
vastai search offers 'reliability > 0.95 num_gpus=1 gpu_name=RTX_3090'

# Rent instance
vastai create instance INSTANCE_ID \
  --image nvidia/cuda:11.8-devel-ubuntu20.04 \
  --onstart-cmd "curl -s https://your-server.com/vast-worker-template.sh | bash"
```

**Via Web Interface:**
1. Go to https://vast.ai
2. Search for GPU instances
3. Select instance and click "Rent"
4. In "On-Start Script" field, paste:
```bash
curl -s https://your-server.com/vast-worker-template.sh | bash
```

### 3. Auto-Scaling Script

Create `vast-autoscale.sh` for automatic worker management:
```bash
#!/bin/bash

DOCKER_IMAGE="yourusername/hashmancer-worker:latest"
SERVER_URL="http://YOUR_SERVER_IP:8080"
REDIS_URL="redis://YOUR_REDIS_IP:6379"

# Function to create worker
create_worker() {
    vastai create instance --image nvidia/cuda:11.8-devel-ubuntu20.04 \
        --onstart-cmd "docker run -d --gpus all --name worker -e SERVER_URL=${SERVER_URL} -e REDIS_URL=${REDIS_URL} ${DOCKER_IMAGE}"
}

# Monitor queue and scale
while true; do
    QUEUE_SIZE=$(curl -s ${SERVER_URL}/api/queue/size)
    ACTIVE_WORKERS=$(curl -s ${SERVER_URL}/api/workers/active)
    
    if [ "$QUEUE_SIZE" -gt 10 ] && [ "$ACTIVE_WORKERS" -lt 5 ]; then
        echo "Scaling up workers..."
        create_worker
    fi
    
    sleep 60
done
```

---

## Environment Variables

### Server Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | `redis://localhost:6379` | Redis connection string |
| `PORT` | `8080` | Server port |
| `HOST` | `0.0.0.0` | Server host |
| `LOG_LEVEL` | `info` | Logging level |
| `WORKERS` | `4` | Number of worker processes |

### Worker Environment Variables
| Variable | Required | Description |
|----------|----------|-------------|
| `SERVER_URL` | Yes | Hashmancer server URL |
| `REDIS_URL` | Yes | Redis connection string |
| `WORKER_NAME` | No | Custom worker identifier |
| `GPU_DEVICES` | No | Specific GPU devices to use |
| `MAX_JOBS` | No | Maximum concurrent jobs |

### Usage Examples
```bash
# Server with custom settings
docker run -d \
  -e REDIS_URL=redis://remote-redis:6379 \
  -e PORT=9000 \
  -e LOG_LEVEL=debug \
  -p 9000:9000 \
  yourusername/hashmancer-server:latest

# Worker with custom settings
docker run -d \
  -e SERVER_URL=http://server:9000 \
  -e REDIS_URL=redis://remote-redis:6379 \
  -e WORKER_NAME=gpu-farm-1 \
  -e MAX_JOBS=4 \
  yourusername/hashmancer-worker:latest
```

---

## Troubleshooting

### Common Issues

#### 1. Connection Refused Errors
```bash
# Check if services are running
docker ps

# Check service logs
docker logs hashmancer-server
docker logs hashmancer-worker

# Check network connectivity
docker exec hashmancer-worker ping server-ip
```

#### 2. Image Not Found
```bash
# Verify image exists on Docker Hub
docker search yourusername/hashmancer-worker

# Pull image manually
docker pull yourusername/hashmancer-worker:latest
```

#### 3. GPU Not Available
```bash
# Check GPU status
nvidia-smi

# Verify Docker GPU support
docker run --gpus all nvidia/cuda:11.0-base nvidia-smi

# Check container GPU access
docker exec hashmancer-worker nvidia-smi
```

#### 4. Permission Errors
```bash
# Fix Docker permissions
sudo usermod -aG docker $USER
newgrp docker

# Fix file permissions
sudo chown -R $USER:$USER ~/.hashmancer
```

### Health Checks

#### Server Health
```bash
# HTTP health check
curl http://localhost:8080/health

# Container health
docker exec hashmancer-server /app/health-check.sh
```

#### Worker Health
```bash
# Check worker process
docker exec hashmancer-worker pgrep -f hashmancer.worker

# Check worker logs
docker logs hashmancer-worker --tail 50
```

#### Redis Health
```bash
# Redis ping
docker exec redis redis-cli ping

# Check Redis connections
docker exec redis redis-cli client list
```

### Performance Monitoring

#### Resource Usage
```bash
# Container stats
docker stats

# Detailed container info
docker exec hashmancer-worker ps aux
docker exec hashmancer-worker free -h
docker exec hashmancer-worker df -h
```

#### Log Analysis
```bash
# Real-time logs
docker logs -f hashmancer-server
docker logs -f hashmancer-worker

# Search logs
docker logs hashmancer-worker 2>&1 | grep ERROR
```

### Cleanup Commands

#### Stop All Containers
```bash
docker stop $(docker ps -aq --filter "name=hashmancer")
```

#### Remove All Containers
```bash
docker rm $(docker ps -aq --filter "name=hashmancer")
```

#### Clean Up Images
```bash
# Remove old images
docker image prune

# Remove specific images
docker rmi yourusername/hashmancer-server:latest
docker rmi yourusername/hashmancer-worker:latest
```

#### Reset Everything
```bash
# Stop and remove everything
docker-compose -f docker-compose.simple.yml down -v

# Remove all hashmancer containers and images
docker stop $(docker ps -aq --filter "name=hashmancer")
docker rm $(docker ps -aq --filter "name=hashmancer")
docker rmi $(docker images --filter "reference=*hashmancer*" -q)
```

---

## Quick Reference

### Build and Push Workflow
```bash
# 1. Build images
docker build -f Dockerfile.server -t yourusername/hashmancer-server:latest .
docker build -f Dockerfile.worker -t yourusername/hashmancer-worker:latest .

# 2. Test locally
docker-compose -f docker-compose.simple.yml up --build

# 3. Push to Docker Hub
docker push yourusername/hashmancer-server:latest
docker push yourusername/hashmancer-worker:latest
```

### Deployment Commands
```bash
# Local development
docker-compose -f docker-compose.simple.yml up -d

# Server only
docker run -d -p 8080:8080 yourusername/hashmancer-server:latest

# Worker only
docker run -d -e SERVER_URL=http://server:8080 yourusername/hashmancer-worker:latest

# Vast.ai worker
docker run -d --gpus all -e SERVER_URL=http://YOUR_IP:8080 yourusername/hashmancer-worker:latest
```

---

## Support

For issues and questions:
- Check logs: `docker logs container-name`
- Review environment variables
- Verify network connectivity
- Check Docker Hub image availability
- Monitor resource usage

Remember to replace `yourusername` with your actual Docker Hub username throughout this guide.