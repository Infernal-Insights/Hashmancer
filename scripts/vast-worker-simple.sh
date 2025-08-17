#!/bin/bash
# Simplified Vast.ai Worker Setup Script
# This script is embedded directly in the Vast.ai on-start command

set -e

# Configuration from environment variables
HASHMANCER_SERVER_IP="${HASHMANCER_SERVER_IP:-}"
HASHMANCER_SERVER_PORT="${HASHMANCER_SERVER_PORT:-8080}"
WORKER_PORT="${WORKER_PORT:-8081}"
WORKER_ID="vast-$(hostname)-$(date +%s)"

echo "🔓 Hashmancer Vast.ai Worker Starting..."
echo "Worker ID: $WORKER_ID"
echo "Server: $HASHMANCER_SERVER_IP:$HASHMANCER_SERVER_PORT"

# Validate required environment
if [ -z "$HASHMANCER_SERVER_IP" ]; then
    echo "❌ HASHMANCER_SERVER_IP environment variable is required"
    exit 1
fi

# Install dependencies
apt-get update -qq
apt-get install -y python3 python3-pip curl wget

# Install Python packages
pip3 install requests psutil

# Create worker script directly
cat > /app/worker.py << 'EOF'
#!/usr/bin/env python3
import os
import sys
import time
import json
import requests
import threading
import signal
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler

class HashmancerWorker:
    def __init__(self):
        self.worker_id = os.environ.get('WORKER_ID', f'vast-{int(time.time())}')
        self.server_host = os.environ.get('HASHMANCER_SERVER_IP', 'localhost')
        self.server_port = int(os.environ.get('HASHMANCER_SERVER_PORT', '8080'))
        self.worker_port = int(os.environ.get('WORKER_PORT', '8081'))
        self.running = True
        
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self.shutdown)
        signal.signal(signal.SIGINT, self.shutdown)
        
        print(f"Worker {self.worker_id} initialized")
        print(f"Server: {self.server_host}:{self.server_port}")
    
    def shutdown(self, signum, frame):
        print("Shutting down worker...")
        self.running = False
    
    def get_capabilities(self):
        """Get system capabilities"""
        capabilities = {
            'cpu_cores': os.cpu_count() or 4,
            'memory_gb': 8,  # Default
            'gpu_count': 0,
            'gpu_models': [],
            'algorithms': ['MD5', 'SHA1', 'SHA256', 'NTLM', 'bcrypt']
        }
        
        # Try to get GPU info
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
            if result.returncode == 0:
                gpu_lines = [line for line in result.stdout.strip().split('\n') if line.strip()]
                capabilities['gpu_count'] = len(gpu_lines)
                capabilities['gpu_models'] = [line.split(': ')[1] for line in gpu_lines if ': ' in line]
        except:
            pass
        
        return capabilities
    
    def register_with_server(self):
        """Register with the Hashmancer server"""
        registration_data = {
            'worker_id': self.worker_id,
            'host': self.get_public_ip(),
            'port': self.worker_port,
            'capabilities': self.get_capabilities(),
            'status': 'ready',
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
        
        max_attempts = 10
        for attempt in range(1, max_attempts + 1):
            try:
                print(f"Registration attempt {attempt}/{max_attempts}")
                
                response = requests.post(
                    f'http://{self.server_host}:{self.server_port}/worker/register',
                    json=registration_data,
                    timeout=10
                )
                
                if response.status_code == 200:
                    print("✅ Successfully registered with server")
                    return True
                else:
                    print(f"Registration failed: {response.status_code} - {response.text}")
                    
            except Exception as e:
                print(f"Registration error: {e}")
            
            if attempt < max_attempts:
                time.sleep(30)  # Wait longer between attempts
        
        print("❌ Failed to register with server")
        return False
    
    def get_public_ip(self):
        """Get public IP address"""
        try:
            response = requests.get('http://ifconfig.me', timeout=5)
            return response.text.strip()
        except:
            return 'unknown'
    
    def health_server(self):
        """Start health check server"""
        class HealthHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/health':
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    
                    health_data = {
                        'status': 'healthy',
                        'worker_id': worker.worker_id,
                        'server': f'{worker.server_host}:{worker.server_port}',
                        'capabilities': worker.get_capabilities(),
                        'timestamp': datetime.utcnow().isoformat() + 'Z'
                    }
                    
                    self.wfile.write(json.dumps(health_data).encode())
                else:
                    self.send_response(404)
                    self.end_headers()
                    self.wfile.write(b'Not found')
            
            def log_message(self, format, *args):
                pass  # Suppress HTTP logs
        
        worker = self
        server = HTTPServer(('0.0.0.0', self.worker_port), HealthHandler)
        
        thread = threading.Thread(target=server.serve_forever)
        thread.daemon = True
        thread.start()
        
        print(f"🏥 Health server started on port {self.worker_port}")
    
    def job_polling_loop(self):
        """Poll for jobs from server"""
        print("📋 Starting job polling...")
        
        while self.running:
            try:
                # Poll for jobs
                response = requests.get(
                    f'http://{self.server_host}:{self.server_port}/worker/{self.worker_id}/jobs',
                    timeout=5
                )
                
                if response.status_code == 200:
                    jobs = response.json()
                    if jobs:
                        print(f"📋 Received {len(jobs)} jobs")
                        # Process jobs here (simplified for demo)
                        for job in jobs:
                            print(f"Processing job: {job.get('id', 'unknown')}")
                            
                elif response.status_code == 404:
                    # No jobs available
                    pass
                else:
                    print(f"Job polling returned: {response.status_code}")
                
            except Exception as e:
                print(f"Job polling error: {e}")
            
            time.sleep(10)  # Poll every 10 seconds
    
    def run(self):
        """Main worker loop"""
        print(f"🚀 Starting Hashmancer worker: {self.worker_id}")
        
        # Start health server
        self.health_server()
        
        # Register with server
        if not self.register_with_server():
            print("Failed to register with server, exiting")
            return 1
        
        # Start job polling
        self.job_polling_loop()
        
        print("👋 Worker shutdown")
        return 0

def main():
    worker = HashmancerWorker()
    return worker.run()

if __name__ == '__main__':
    sys.exit(main())
EOF

# Create app directory
mkdir -p /app

# Set environment variables for the worker
export WORKER_ID="$WORKER_ID"
export HASHMANCER_SERVER_IP="$HASHMANCER_SERVER_IP"
export HASHMANCER_SERVER_PORT="$HASHMANCER_SERVER_PORT"
export WORKER_PORT="$WORKER_PORT"

# Start the worker
echo "🚀 Starting worker..."
python3 /app/worker.py