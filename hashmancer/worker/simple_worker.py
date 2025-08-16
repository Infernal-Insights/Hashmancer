#!/usr/bin/env python3
"""
Simple Hashmancer Worker for Vast.ai instances
Connects to server, registers, and processes jobs
"""

import asyncio
import json
import os
import sys
import time
import requests
import subprocess
import signal
from datetime import datetime
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HashmancerSimpleWorker:
    def __init__(self):
        self.worker_id = os.environ.get('WORKER_ID', f'vast-worker-{int(time.time())}')
        self.server_host = os.environ.get('HASHMANCER_SERVER_IP', 'localhost')
        self.server_port = int(os.environ.get('HASHMANCER_SERVER_PORT', '8080'))
        self.worker_port = int(os.environ.get('WORKER_PORT', '8081'))
        self.max_jobs = int(os.environ.get('MAX_JOBS', '3'))
        
        self.running_jobs = {}
        self.running = True
        
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self.shutdown)
        signal.signal(signal.SIGINT, self.shutdown)
        
        logger.info(f"Worker initialized: {self.worker_id}")
        logger.info(f"Server: {self.server_host}:{self.server_port}")
    
    def shutdown(self, signum, frame):
        """Graceful shutdown"""
        logger.info("Shutting down worker...")
        self.running = False
    
    def get_system_info(self):
        """Get system capabilities"""
        capabilities = {
            'cpu_cores': os.cpu_count(),
            'algorithms': ['MD5', 'SHA1', 'SHA256', 'NTLM', 'bcrypt']
        }
        
        # Get GPU info if available
        try:
            result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
            if result.returncode == 0:
                gpu_lines = result.stdout.strip().split('\n')
                capabilities['gpu_count'] = len(gpu_lines)
                capabilities['gpu_models'] = [line.split(': ')[1] for line in gpu_lines if ': ' in line]
            else:
                capabilities['gpu_count'] = 0
                capabilities['gpu_models'] = []
        except:
            capabilities['gpu_count'] = 0
            capabilities['gpu_models'] = []
        
        # Get memory info
        try:
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()
                for line in meminfo.split('\n'):
                    if line.startswith('MemTotal:'):
                        mem_kb = int(line.split()[1])
                        capabilities['memory_gb'] = mem_kb // (1024 * 1024)
                        break
        except:
            capabilities['memory_gb'] = 4  # Default
        
        return capabilities
    
    def register_with_server(self):
        """Register worker with the Hashmancer server"""
        capabilities = self.get_system_info()
        
        registration_data = {
            'worker_id': self.worker_id,
            'host': self.get_public_ip(),
            'port': self.worker_port,
            'capabilities': capabilities,
            'status': 'ready',
            'max_jobs': self.max_jobs,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
        
        max_attempts = 5
        for attempt in range(1, max_attempts + 1):
            try:
                logger.info(f"Registration attempt {attempt}/{max_attempts}")
                
                response = requests.post(
                    f'http://{self.server_host}:{self.server_port}/worker/register',
                    json=registration_data,
                    timeout=10
                )
                
                if response.status_code == 200:
                    logger.info("✅ Successfully registered with server")
                    return True
                else:
                    logger.warning(f"Registration failed: {response.status_code}")
                    
            except Exception as e:
                logger.error(f"Registration error: {e}")
            
            if attempt < max_attempts:
                time.sleep(10)
        
        logger.error("❌ Failed to register with server")
        return False
    
    def get_public_ip(self):
        """Get public IP address"""
        try:
            response = requests.get('http://ifconfig.me', timeout=5)
            return response.text.strip()
        except:
            return 'unknown'
    
    def poll_for_jobs(self):
        """Poll server for new jobs"""
        while self.running:
            try:
                if len(self.running_jobs) >= self.max_jobs:
                    time.sleep(10)
                    continue
                
                response = requests.get(
                    f'http://{self.server_host}:{self.server_port}/worker/{self.worker_id}/jobs',
                    timeout=5
                )
                
                if response.status_code == 200:
                    jobs = response.json()
                    for job in jobs:
                        if job['id'] not in self.running_jobs:
                            self.start_job(job)
                elif response.status_code == 404:
                    # No jobs available
                    pass
                else:
                    logger.warning(f"Job polling returned: {response.status_code}")
                
            except Exception as e:
                logger.error(f"Job polling error: {e}")
            
            time.sleep(5)
    
    def start_job(self, job):
        """Start processing a job"""
        job_id = job['id']
        logger.info(f"🚀 Starting job: {job_id}")
        
        # Create job directory
        job_dir = Path(f'/tmp/hashmancer-jobs/{job_id}')
        job_dir.mkdir(parents=True, exist_ok=True)
        
        # Save job data
        with open(job_dir / 'job.json', 'w') as f:
            json.dump(job, f, indent=2)
        
        # Mark job as running
        self.running_jobs[job_id] = {
            'start_time': time.time(),
            'status': 'running',
            'job_data': job
        }
        
        # Report job started
        try:
            requests.post(
                f'http://{self.server_host}:{self.server_port}/worker/{self.worker_id}/job/{job_id}/status',
                json={'status': 'started', 'timestamp': datetime.utcnow().isoformat() + 'Z'},
                timeout=5
            )
        except Exception as e:
            logger.error(f"Failed to report job start: {e}")
        
        # Start background processing
        import threading
        thread = threading.Thread(target=self.process_job, args=(job,))
        thread.daemon = True
        thread.start()
    
    def process_job(self, job):
        """Process a job (simulate hash cracking)"""
        job_id = job['id']
        
        try:
            logger.info(f"Processing job {job_id}...")
            
            # Simulate processing based on job type
            algorithm = job.get('algorithm', 'MD5').upper()
            hash_count = job.get('hash_count', 1)
            
            # Estimate processing time (simulation)
            base_time = 10  # Base 10 seconds
            time_per_hash = 2 if algorithm == 'MD5' else 5
            processing_time = base_time + (hash_count * time_per_hash)
            
            logger.info(f"Estimated processing time: {processing_time}s")
            
            # Simulate work with progress updates
            for i in range(10):
                if not self.running or job_id not in self.running_jobs:
                    break
                
                time.sleep(processing_time / 10)
                progress = (i + 1) * 10
                
                # Send progress update
                try:
                    requests.post(
                        f'http://{self.server_host}:{self.server_port}/worker/{self.worker_id}/job/{job_id}/progress',
                        json={'progress': progress, 'timestamp': datetime.utcnow().isoformat() + 'Z'},
                        timeout=5
                    )
                except:
                    pass
            
            if self.running and job_id in self.running_jobs:
                # Job completed successfully
                self.running_jobs[job_id]['status'] = 'completed'
                
                # Generate mock results
                results = {
                    'cracked_hashes': hash_count // 2,  # Simulate 50% success rate
                    'total_hashes': hash_count,
                    'processing_time': processing_time,
                    'algorithm': algorithm
                }
                
                # Report completion
                try:
                    requests.post(
                        f'http://{self.server_host}:{self.server_port}/worker/{self.worker_id}/job/{job_id}/status',
                        json={
                            'status': 'completed',
                            'results': results,
                            'timestamp': datetime.utcnow().isoformat() + 'Z'
                        },
                        timeout=10
                    )
                    
                    logger.info(f"✅ Job {job_id} completed successfully")
                except Exception as e:
                    logger.error(f"Failed to report job completion: {e}")
                
                # Remove from running jobs
                del self.running_jobs[job_id]
            
        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}")
            
            if job_id in self.running_jobs:
                self.running_jobs[job_id]['status'] = 'failed'
                
                # Report failure
                try:
                    requests.post(
                        f'http://{self.server_host}:{self.server_port}/worker/{self.worker_id}/job/{job_id}/status',
                        json={
                            'status': 'failed',
                            'error': str(e),
                            'timestamp': datetime.utcnow().isoformat() + 'Z'
                        },
                        timeout=5
                    )
                except:
                    pass
                
                del self.running_jobs[job_id]
    
    def health_check_server(self):
        """Simple health check endpoint"""
        from http.server import HTTPServer, BaseHTTPRequestHandler
        import json
        
        class HealthHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/health':
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    
                    health_data = {
                        'status': 'healthy',
                        'worker_id': worker.worker_id,
                        'running_jobs': len(worker.running_jobs),
                        'max_jobs': worker.max_jobs,
                        'server': f'{worker.server_host}:{worker.server_port}',
                        'capabilities': worker.get_system_info()
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
        
        import threading
        thread = threading.Thread(target=server.serve_forever)
        thread.daemon = True
        thread.start()
        
        logger.info(f"🏥 Health check server started on port {self.worker_port}")
    
    def run(self):
        """Main worker loop"""
        logger.info(f"🚀 Starting Hashmancer worker: {self.worker_id}")
        
        # Start health check server
        self.health_check_server()
        
        # Register with server
        if not self.register_with_server():
            logger.error("Failed to register with server, exiting")
            return 1
        
        # Start job polling
        logger.info("📋 Starting job polling...")
        self.poll_for_jobs()
        
        logger.info("👋 Worker shutdown complete")
        return 0

def main():
    """Entry point"""
    worker = HashmancerSimpleWorker()
    return worker.run()

if __name__ == '__main__':
    sys.exit(main())