# Hashmancer Windows Setup Guide

## Step 1: Verify Prerequisites

Open PowerShell as Administrator and run:

```powershell
# Check Docker
docker --version
docker-compose --version

# Check GPU drivers
nvidia-smi

# Check if WSL2 is enabled (required for Docker)
wsl --list --verbose
```

## Step 2: Download Hashmancer

```powershell
# Clone the repository
git clone https://github.com/your-repo/hashmancer.git
cd hashmancer

# Or download and extract the ZIP file to C:\hashmancer
```

## Step 3: Setup Docker Environment

Create `.env` file in the hashmancer directory:

```env
# Server Configuration
HASHMANCER_SERVER_IP=localhost
HASHMANCER_SERVER_PORT=8080
WORKER_PORT=8081

# Worker Configuration
WORKER_ID=windows-local-worker
MAX_CONCURRENT_JOBS=3

# Redis Configuration
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=hashmancer123

# Security Configuration
HASHMANCER_SECRET_KEY=your-secret-key-here-change-this

# Logging
LOG_LEVEL=INFO
LOG_DIR=./logs

# GPU Configuration
NVIDIA_VISIBLE_DEVICES=all
CUDA_VISIBLE_DEVICES=0
```

## Step 4: Create Docker Compose Configuration

Create `docker-compose.windows.yml`:

```yaml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    container_name: hashmancer-redis
    command: redis-server --requirepass hashmancer123
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    networks:
      - hashmancer-network

  server:
    build:
      context: .
      dockerfile: Dockerfile.production
      target: server
    container_name: hashmancer-server
    ports:
      - "8080:8080"
    environment:
      - HASHMANCER_SERVER_IP=0.0.0.0
      - HASHMANCER_SERVER_PORT=8080
      - REDIS_HOST=redis
      - REDIS_PORT=6379
      - REDIS_PASSWORD=hashmancer123
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
    depends_on:
      - redis
    restart: unless-stopped
    networks:
      - hashmancer-network

  worker:
    build:
      context: .
      dockerfile: Dockerfile.gpu
      target: worker
    container_name: hashmancer-worker
    environment:
      - HASHMANCER_SERVER_IP=server
      - HASHMANCER_SERVER_PORT=8080
      - WORKER_PORT=8081
      - WORKER_ID=windows-docker-worker
      - MAX_CONCURRENT_JOBS=3
    volumes:
      - ./logs:/app/logs
      - ./worker_data:/app/worker_data
    depends_on:
      - server
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    networks:
      - hashmancer-network

volumes:
  redis_data:

networks:
  hashmancer-network:
    driver: bridge
```

## Step 5: Create Dockerfiles

### Dockerfile.production (for server)
```dockerfile
FROM python:3.11-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create logs directory
RUN mkdir -p /app/logs /app/data

# Server stage
FROM base as server
EXPOSE 8080
CMD ["python", "-m", "hashmancer.server.main"]

# Worker stage  
FROM base as worker
EXPOSE 8081
CMD ["python", "-m", "hashmancer.worker.production_worker"]
```

### Dockerfile.gpu (for worker with GPU support)
```dockerfile
FROM nvidia/cuda:12.0-devel-ubuntu20.04 as base

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-pip \
    python3.11-dev \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -s /usr/bin/python3.11 /usr/bin/python
RUN ln -s /usr/bin/pip3 /usr/bin/pip

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install GPU-specific packages
RUN pip install --no-cache-dir \
    pycuda \
    pyopencl

# Copy application code
COPY . .

# Create directories
RUN mkdir -p /app/logs /app/worker_data

# Worker stage
FROM base as worker
EXPOSE 8081
CMD ["python", "-m", "hashmancer.worker.production_worker"]
```

## Step 6: Create Required Directories

```powershell
mkdir logs
mkdir data
mkdir worker_data
```

## Step 7: Build and Start Services

```powershell
# Build the containers
docker-compose -f docker-compose.windows.yml build

# Start all services
docker-compose -f docker-compose.windows.yml up -d

# Check status
docker-compose -f docker-compose.windows.yml ps
```

## Step 8: Verify Installation

```powershell
# Check server health
curl http://localhost:8080/health

# Check server status
curl http://localhost:8080

# View logs
docker-compose -f docker-compose.windows.yml logs server
docker-compose -f docker-compose.windows.yml logs worker
docker-compose -f docker-compose.windows.yml logs redis
```

## Step 9: Access the Web Interface

Open your browser and go to:
- **Main Portal**: http://localhost:8080
- **Health Check**: http://localhost:8080/health
- **API Documentation**: http://localhost:8080/docs

## Step 10: Test GPU Functionality

```powershell
# Test Nvidia GPU in worker container
docker exec hashmancer-worker nvidia-smi

# Check worker registration
curl http://localhost:8080/api/workers
```

## Troubleshooting

### Common Issues:

1. **Docker Desktop not running**:
   ```powershell
   # Start Docker Desktop
   # Ensure WSL2 integration is enabled
   ```

2. **GPU not detected**:
   ```powershell
   # Install NVIDIA Container Toolkit
   # Enable GPU support in Docker Desktop
   ```

3. **Port conflicts**:
   ```powershell
   # Check if ports are in use
   netstat -an | findstr :8080
   netstat -an | findstr :6379
   ```

4. **Permission issues**:
   ```powershell
   # Run PowerShell as Administrator
   # Check Docker Desktop permissions
   ```

## Testing Commands

```powershell
# View real-time logs
docker-compose -f docker-compose.windows.yml logs -f

# Restart specific service
docker-compose -f docker-compose.windows.yml restart server

# Stop all services
docker-compose -f docker-compose.windows.yml down

# Clean restart
docker-compose -f docker-compose.windows.yml down -v
docker-compose -f docker-compose.windows.yml up -d
```

## Performance Testing

```powershell
# Submit a test job
curl -X POST http://localhost:8080/api/jobs \
  -H "Content-Type: application/json" \
  -d '{"algorithm": "MD5", "hashes": ["5d41402abc4b2a76b9719d911017c592"], "attack_mode": "dictionary"}'

# Check job status
curl http://localhost:8080/api/jobs
```