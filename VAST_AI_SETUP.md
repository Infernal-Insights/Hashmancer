# Vast.ai Automatic Worker Setup

This document explains how to automatically deploy Hashmancer workers on Vast.ai that will connect to your server, register themselves, and start processing jobs.

## 🚀 Quick Start

### 1. Start Your Hashmancer Server

First, make sure your Hashmancer server is running and accessible:

```bash
# Start the server
./hashmancer-cli server start --host 0.0.0.0 --port 8080

# Check server status
./hashmancer-cli server status
```

### 2. Host the Worker Setup Script

The Vast.ai workers need to download a setup script. Host it on your server:

```bash
# Host the setup script (runs in background)
./hashmancer-cli server host-setup --port 8888 &
```

This will:
- Host the worker setup script at `http://YOUR_IP:8888/setup`
- Display your public IP and the command Vast.ai workers will use
- Keep running until you stop it

### 3. Deploy Workers Automatically

Use the CLI to find and deploy the cheapest workers:

```bash
# Add a single RTX 3080 worker (finds cheapest available)
./hashmancer-cli server add-worker 3080

# Add multiple RTX 4090 workers with custom price limit
./hashmancer-cli server add-worker 4090 --count 3 --max-price 0.75

# Add any available GPU under $0.50/hour
./hashmancer-cli server add-worker rtx --count 5 --max-price 0.50
```

### 4. Monitor Workers

Check your workers are connecting:

```bash
# Discover workers on your network
./hashmancer-cli server discover

# Check worker status
./hashmancer-cli worker status --all

# View worker logs
./hashmancer-cli worker logs <worker-id> --follow
```

## 🔧 How It Works

### Automatic Setup Process

When you deploy a Vast.ai worker, this happens automatically:

1. **🐳 Docker Installation**: Installs Docker and NVIDIA Docker support
2. **🔍 Server Discovery**: Automatically finds your Hashmancer server IP
3. **📦 Image Build**: Downloads and builds the Hashmancer worker Docker image  
4. **🚀 Worker Start**: Launches the worker container with GPU support
5. **📝 Registration**: Registers with your server and starts accepting jobs
6. **📊 Monitoring**: Sets up health checks and automatic restart

### Server IP Detection

The worker setup script uses multiple methods to find your server:

1. **Environment Variable**: `HASHMANCER_SERVER_IP` (if set)
2. **Network Scanning**: Scans common private networks (192.168.x.x, 10.x.x.x)
3. **Broadcast Discovery**: Uses nmap to find servers on the network
4. **Health Check**: Verifies server is running by checking `/health` endpoint

### Worker Features

Each deployed worker provides:

- **🏥 Health Monitoring**: HTTP health check endpoint on port 8081
- **🔄 Auto-Restart**: Automatically restarts if it crashes
- **📊 System Info**: Reports GPU count, memory, CPU cores to server
- **🎯 Job Processing**: Processes hash cracking jobs from your server
- **📋 Logging**: Comprehensive logging for debugging

## 🛠️ Advanced Usage

### Custom Docker Images

To use your own worker Docker image:

```bash
# Build custom image
docker build -t my-hashmancer-worker:latest -f docker/worker/Dockerfile.simple .

# Update the setup script to use your image
export HASHMANCER_DOCKER_IMAGE="my-hashmancer-worker:latest"
```

### Manual Vast.ai Deployment

You can also deploy manually through the Vast.ai interface:

1. **Image**: `nvidia/cuda:11.8-devel-ubuntu20.04`
2. **On-start script**: 
   ```bash
   export HASHMANCER_SERVER_IP=YOUR_SERVER_IP
   wget -O - http://YOUR_SERVER_IP:8888/setup | bash
   ```

### Environment Variables

Configure workers using environment variables:

- `HASHMANCER_SERVER_IP`: Your server's IP address
- `HASHMANCER_SERVER_PORT`: Server port (default: 8080)
- `WORKER_PORT`: Worker health check port (default: 8081)
- `MAX_JOBS`: Maximum concurrent jobs (default: 3)
- `WORKER_ID`: Custom worker ID (auto-generated if not set)

### Firewall Configuration

Make sure these ports are open on your server:

- **8080**: Hashmancer server (for worker communication)
- **8888**: Setup script hosting (temporary)

On the worker side (Vast.ai), port 8081 is used for health checks.

## 📋 Example Workflows

### Scenario 1: Quick GPU Scaling

You have a large hash cracking job and need to scale up quickly:

```bash
# Start hosting the setup script
./hashmancer-cli server host-setup &

# Add 5 cheap GPUs (any type under $0.60/hour)
./hashmancer-cli server add-worker rtx --count 5 --max-price 0.60

# Monitor deployment
watch "./hashmancer-cli worker status --all"

# Check job distribution
./hashmancer-cli server jobs
```

### Scenario 2: Targeted GPU Types

You need specific GPU types for optimal performance:

```bash
# Add RTX 4090s for high-end work
./hashmancer-cli server add-worker 4090 --count 2 --max-price 1.00

# Add RTX 3080s for medium work  
./hashmancer-cli server add-worker 3080 --count 3 --max-price 0.75

# Add GTX 1080s for light work
./hashmancer-cli server add-worker 1080 --count 5 --max-price 0.30
```

### Scenario 3: Automated Job Processing

Set up workers to automatically pull and process jobs from hashes.com:

```bash
# Start workers
./hashmancer-cli server add-worker 3080 --count 2

# Watch for new jobs and auto-pull them
./hashmancer-cli hashes watch --api-key YOUR_KEY --exclude-md5 --exclude-btc &

# Monitor progress
./hashmancer-cli server jobs --format table
```

## 🐛 Troubleshooting

### Workers Not Connecting

1. **Check Setup Script Hosting**:
   ```bash
   curl http://YOUR_IP:8888/setup
   ```

2. **Verify Server Accessibility**:
   ```bash
   curl http://YOUR_IP:8080/health
   ```

3. **Check Worker Logs**:
   ```bash
   # On the Vast.ai instance
   docker logs hashmancer-worker
   ```

### Server Discovery Issues

If workers can't find your server automatically:

1. **Set Server IP Explicitly**:
   ```bash
   export HASHMANCER_SERVER_IP=YOUR_PUBLIC_IP
   ```

2. **Check Firewall Rules**:
   ```bash
   # Allow worker connections
   sudo ufw allow 8080
   sudo ufw allow 8888
   ```

### Worker Registration Failures

1. **Check Server Endpoints**:
   ```bash
   curl -X POST http://YOUR_IP:8080/worker/register \
     -H "Content-Type: application/json" \
     -d '{"worker_id":"test","status":"ready"}'
   ```

2. **Verify Network Connectivity**:
   ```bash
   # From worker to server
   telnet YOUR_SERVER_IP 8080
   ```

## 🔐 Security Considerations

### Network Security

- Use VPN or private networks when possible
- Restrict server access to known IP ranges
- Use strong authentication for sensitive operations

### API Keys

- Store Vast.ai API keys securely (environment variables)
- Use read-only keys when possible
- Rotate keys regularly

### Worker Security

- Workers run in isolated Docker containers
- No persistent data storage on workers
- Automatic cleanup when jobs complete

## 💰 Cost Management

### Monitoring Costs

```bash
# Check current worker costs
./hashmancer-cli server jobs --format json | jq '.[] | select(.status=="running") | .worker_cost'

# Set cost alerts
./hashmancer-cli server add-worker 3080 --max-price 0.50 --cost-alert 10.00
```

### Auto-Scaling

Set up automatic scaling based on job queue:

```bash
# Scale up when queue is full
if [ $(./hashmancer-cli server jobs | grep pending | wc -l) -gt 5 ]; then
    ./hashmancer-cli server add-worker 3080 --count 2
fi

# Scale down when idle
./hashmancer-cli worker status --all --format json | \
  jq -r '.[] | select(.active_jobs == 0) | .id' | \
  xargs -I {} ./hashmancer-cli worker stop {}
```

This automation ensures you only pay for compute power when you need it!

## 🎉 Summary

With this setup, you can:

✅ **One-command worker deployment** - `./hashmancer-cli server add-worker 3080`
✅ **Automatic server discovery** - Workers find your server automatically  
✅ **GPU-optimized containers** - Full CUDA/OpenCL support
✅ **Health monitoring** - Workers report status and restart if needed
✅ **Cost optimization** - Find cheapest available hardware
✅ **Scalable architecture** - Add/remove workers as needed

Your Vast.ai workers will automatically download the Docker image, connect to your server, register themselves, and start processing jobs - completely hands-free! 🚀