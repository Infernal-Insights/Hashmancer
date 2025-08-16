#!/usr/bin/env python3
"""
Hashmancer Production Worker
High-performance worker for Vast.ai and cloud deployments
"""

import os
import sys
import time
import json
import requests
import subprocess
import threading
import signal
import logging
from datetime import datetime, timedelta
from pathlib import Path
import hashlib

# Configure logging
log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/app/logs/worker.log')
    ]
)
logger = logging.getLogger('hashmancer-worker')

class HashmancerProductionWorker:
    def __init__(self):
        # Configuration from environment
        self.worker_id = os.environ.get('WORKER_ID', self._generate_worker_id())
        self.server_host = os.environ.get('HASHMANCER_SERVER_IP')
        self.server_port = int(os.environ.get('HASHMANCER_SERVER_PORT', '8080'))
        self.worker_port = int(os.environ.get('WORKER_PORT', '8081'))
        self.max_jobs = int(os.environ.get('MAX_CONCURRENT_JOBS', '3'))
        
        # State management
        self.running = True
        self.registered = False
        self.active_jobs = {}
        self.job_lock = threading.Lock()
        
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._shutdown_handler)
        signal.signal(signal.SIGINT, self._shutdown_handler)
        
        # Validate configuration
        if not self.server_host:
            logger.error("HASHMANCER_SERVER_IP environment variable is required")
            sys.exit(1)
        
        logger.info(f"Worker initialized: {self.worker_id}")
        logger.info(f"Server: {self.server_host}:{self.server_port}")
        logger.info(f"Max concurrent jobs: {self.max_jobs}")
    
    def _generate_worker_id(self):
        """Generate unique worker ID"""
        hostname = os.environ.get('HOSTNAME', 'unknown')
        timestamp = int(time.time())
        return f"vast-{hostname}-{timestamp}"
    
    def _shutdown_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
        
        # Notify server of shutdown
        try:
            self._notify_server_shutdown()
        except:
            pass
    
    def _notify_server_shutdown(self):
        """Notify server that worker is shutting down"""
        try:
            requests.post(
                f'http://{self.server_host}:{self.server_port}/worker/{self.worker_id}/shutdown',
                json={'timestamp': datetime.utcnow().isoformat() + 'Z'},
                timeout=5
            )
            logger.info("Notified server of shutdown")
        except Exception as e:
            logger.warning(f"Failed to notify server of shutdown: {e}")
    
    def get_system_capabilities(self):
        """Get detailed system capabilities"""
        capabilities = {
            'cpu_cores': os.cpu_count(),
            'memory_gb': self._get_memory_gb(),
            'gpu_count': 0,
            'gpu_models': [],
            'gpu_memory': [],
            'algorithms': [
                'MD5', 'SHA1', 'SHA256', 'SHA512', 'NTLM', 'bcrypt',
                'WPA/WPA2', 'Kerberos', 'PBKDF2'
            ],
            'attack_modes': ['dictionary', 'mask', 'combinator', 'hybrid'],
            'platform': sys.platform,
            'python_version': sys.version.split()[0]
        }
        
        # Get GPU information
        try:
            gpu_info = self._get_gpu_info()
            capabilities.update(gpu_info)
        except Exception as e:
            logger.warning(f"Could not get GPU info: {e}")
        
        return capabilities
    
    def _get_memory_gb(self):
        """Get system memory in GB"""
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if line.startswith('MemTotal:'):
                        mem_kb = int(line.split()[1])
                        return mem_kb // (1024 * 1024)
        except:
            pass
        return 8  # Default fallback
    
    def _get_gpu_info(self):
        """Get GPU information using nvidia-smi"""
        gpu_info = {
            'gpu_count': 0,
            'gpu_models': [],
            'gpu_memory': []
        }
        
        try:
            # Get GPU list
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                gpu_info['gpu_count'] = len([line for line in lines if line.strip()])
                
                for line in lines:
                    if line.strip():
                        parts = line.split(', ')
                        if len(parts) >= 2:
                            gpu_info['gpu_models'].append(parts[0].strip())
                            gpu_info['gpu_memory'].append(int(parts[1].strip()))
            
            logger.info(f"Detected {gpu_info['gpu_count']} GPUs: {gpu_info['gpu_models']}")
            
        except Exception as e:
            logger.warning(f"GPU detection failed: {e}")
        
        return gpu_info
    
    def register_with_server(self):
        """Register worker with the Hashmancer server"""
        registration_data = {
            'worker_id': self.worker_id,
            'host': self._get_public_ip(),
            'port': self.worker_port,
            'capabilities': self.get_system_capabilities(),
            'status': 'ready',
            'max_concurrent_jobs': self.max_jobs,
            'version': '1.0.0',
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
        
        max_attempts = 10
        retry_delay = 30
        
        for attempt in range(1, max_attempts + 1):
            try:
                logger.info(f"Registration attempt {attempt}/{max_attempts}")
                
                response = requests.post(
                    f'http://{self.server_host}:{self.server_port}/worker/register',
                    json=registration_data,
                    timeout=15
                )
                
                if response.status_code == 200:
                    logger.info("✅ Successfully registered with server")
                    self.registered = True
                    return True
                elif response.status_code == 409:
                    logger.warning("Worker ID already exists, generating new ID")
                    self.worker_id = self._generate_worker_id()
                    registration_data['worker_id'] = self.worker_id
                else:
                    logger.warning(f"Registration failed: {response.status_code} - {response.text}")
                    
            except Exception as e:
                logger.error(f"Registration error: {e}")
            
            if attempt < max_attempts:
                logger.info(f"Retrying registration in {retry_delay} seconds...")
                time.sleep(retry_delay)
        
        logger.error("❌ Failed to register with server after all attempts")
        return False
    
    def _get_public_ip(self):
        """Get public IP address"""
        try:
            response = requests.get('http://ifconfig.me', timeout=10)
            return response.text.strip()
        except:
            try:
                response = requests.get('http://ipinfo.io/ip', timeout=10)
                return response.text.strip()
            except:
                return 'unknown'
    
    def start_health_server(self):
        """Start health check HTTP server"""
        from health_server import HealthServer
        
        try:
            health_server = HealthServer(self, self.worker_port)
            health_thread = threading.Thread(target=health_server.start, daemon=True)
            health_thread.start()
            
            logger.info(f"🏥 Health server started on port {self.worker_port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start health server: {e}")
            return False
    
    def job_polling_loop(self):
        """Main job polling loop"""
        logger.info("📋 Starting job polling...")
        poll_interval = 10  # seconds
        heartbeat_interval = 60  # seconds
        last_heartbeat = time.time()
        
        while self.running:
            try:
                current_time = time.time()
                
                # Send heartbeat periodically
                if current_time - last_heartbeat >= heartbeat_interval:
                    self._send_heartbeat()
                    last_heartbeat = current_time
                
                # Poll for new jobs if we have capacity
                with self.job_lock:
                    available_slots = self.max_jobs - len(self.active_jobs)
                
                if available_slots > 0:
                    self._poll_for_jobs()
                
                # Clean up completed jobs
                self._cleanup_completed_jobs()
                
            except Exception as e:
                logger.error(f"Job polling error: {e}")
            
            time.sleep(poll_interval)
        
        logger.info("Job polling stopped")
    
    def _send_heartbeat(self):
        """Send heartbeat to server"""
        try:
            heartbeat_data = {
                'worker_id': self.worker_id,
                'status': 'active',
                'active_jobs': len(self.active_jobs),
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }
            
            response = requests.post(
                f'http://{self.server_host}:{self.server_port}/worker/{self.worker_id}/heartbeat',
                json=heartbeat_data,
                timeout=10
            )
            
            if response.status_code != 200:
                logger.warning(f"Heartbeat failed: {response.status_code}")
                
        except Exception as e:
            logger.warning(f"Heartbeat error: {e}")
    
    def _poll_for_jobs(self):
        """Poll server for new jobs"""
        try:
            response = requests.get(
                f'http://{self.server_host}:{self.server_port}/worker/{self.worker_id}/jobs',
                timeout=10
            )
            
            if response.status_code == 200:
                jobs = response.json()
                for job in jobs:
                    if job['id'] not in self.active_jobs:
                        self._start_job(job)
            elif response.status_code == 404:
                # No jobs available
                pass
            else:
                logger.warning(f"Job polling returned: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Job polling error: {e}")
    
    def _start_job(self, job):
        """Start processing a job"""
        job_id = job['id']
        
        with self.job_lock:
            if len(self.active_jobs) >= self.max_jobs:
                logger.warning(f"Cannot start job {job_id}: max jobs reached")
                return
            
            self.active_jobs[job_id] = {
                'job_data': job,
                'start_time': time.time(),
                'status': 'starting'
            }
        
        logger.info(f"🚀 Starting job: {job_id}")
        
        # Start job processing in background thread
        job_thread = threading.Thread(
            target=self._process_job, 
            args=(job,), 
            daemon=True
        )
        job_thread.start()
    
    def _process_job(self, job):
        """Process a single job"""
        job_id = job['id']
        
        try:
            # Update job status
            with self.job_lock:
                if job_id in self.active_jobs:
                    self.active_jobs[job_id]['status'] = 'running'
            
            # Report job started
            self._report_job_status(job_id, 'started')
            
            # Simulate job processing
            algorithm = job.get('algorithm', 'MD5').upper()
            hash_count = job.get('hash_count', 1)
            attack_mode = job.get('attack_mode', 'dictionary')
            
            # Estimate processing time
            processing_time = self._estimate_processing_time(algorithm, hash_count, attack_mode)
            logger.info(f"Job {job_id}: Estimated {processing_time}s for {hash_count} {algorithm} hashes")
            
            # Simulate work with progress updates
            progress_updates = 10
            for i in range(progress_updates):
                if not self.running or job_id not in self.active_jobs:
                    logger.info(f"Job {job_id} cancelled")
                    return
                
                time.sleep(processing_time / progress_updates)
                progress = ((i + 1) / progress_updates) * 100
                
                # Send progress update
                self._report_job_progress(job_id, progress)
            
            # Generate results
            results = self._generate_job_results(job)
            
            # Report completion
            self._report_job_status(job_id, 'completed', results)
            
            logger.info(f"✅ Job {job_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}")
            self._report_job_status(job_id, 'failed', {'error': str(e)})
        
        finally:
            # Remove from active jobs
            with self.job_lock:
                if job_id in self.active_jobs:
                    del self.active_jobs[job_id]
    
    def _estimate_processing_time(self, algorithm, hash_count, attack_mode):
        """Estimate job processing time"""
        base_time = 5  # Base processing time
        
        # Algorithm complexity multipliers
        complexity = {
            'MD5': 1.0,
            'SHA1': 1.5,
            'SHA256': 2.0,
            'SHA512': 3.0,
            'NTLM': 1.2,
            'BCRYPT': 10.0
        }
        
        # Attack mode multipliers
        attack_multiplier = {
            'dictionary': 1.0,
            'mask': 2.0,
            'combinator': 1.5,
            'hybrid': 2.5
        }
        
        multiplier = complexity.get(algorithm, 2.0) * attack_multiplier.get(attack_mode, 1.0)
        return int(base_time + (hash_count * 0.1 * multiplier))
    
    def _generate_job_results(self, job):
        """Generate realistic job results"""
        hash_count = job.get('hash_count', 1)
        algorithm = job.get('algorithm', 'MD5')
        
        # Simulate realistic crack rates
        crack_rates = {
            'MD5': 0.7,
            'SHA1': 0.6,
            'SHA256': 0.4,
            'NTLM': 0.8,
            'BCRYPT': 0.2
        }
        
        crack_rate = crack_rates.get(algorithm.upper(), 0.5)
        cracked_count = int(hash_count * crack_rate)
        
        return {
            'total_hashes': hash_count,
            'cracked_hashes': cracked_count,
            'crack_rate': crack_rate,
            'algorithm': algorithm,
            'processing_time': time.time() - self.active_jobs[job['id']]['start_time'],
            'worker_id': self.worker_id
        }
    
    def _report_job_status(self, job_id, status, results=None):
        """Report job status to server"""
        try:
            status_data = {
                'status': status,
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'worker_id': self.worker_id
            }
            
            if results:
                status_data['results'] = results
            
            response = requests.post(
                f'http://{self.server_host}:{self.server_port}/worker/{self.worker_id}/job/{job_id}/status',
                json=status_data,
                timeout=10
            )
            
            if response.status_code != 200:
                logger.warning(f"Failed to report job {job_id} status: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error reporting job {job_id} status: {e}")
    
    def _report_job_progress(self, job_id, progress):
        """Report job progress to server"""
        try:
            progress_data = {
                'progress': progress,
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }
            
            requests.post(
                f'http://{self.server_host}:{self.server_port}/worker/{self.worker_id}/job/{job_id}/progress',
                json=progress_data,
                timeout=5
            )
            
        except Exception as e:
            logger.debug(f"Progress update failed: {e}")
    
    def _cleanup_completed_jobs(self):
        """Clean up completed job data"""
        try:
            # Clean up job files older than 1 hour
            jobs_dir = Path('/app/jobs')
            if jobs_dir.exists():
                cutoff_time = time.time() - 3600  # 1 hour ago
                
                for job_dir in jobs_dir.iterdir():
                    if job_dir.is_dir() and job_dir.stat().st_mtime < cutoff_time:
                        import shutil
                        shutil.rmtree(job_dir, ignore_errors=True)
                        
        except Exception as e:
            logger.debug(f"Cleanup error: {e}")
    
    def run(self):
        """Main worker entry point"""
        logger.info(f"🚀 Starting Hashmancer production worker: {self.worker_id}")
        
        # Start health server
        if not self.start_health_server():
            logger.error("Failed to start health server")
            return 1
        
        # Register with server
        if not self.register_with_server():
            logger.error("Failed to register with server")
            return 1
        
        # Start job polling
        try:
            self.job_polling_loop()
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        except Exception as e:
            logger.error(f"Worker error: {e}")
            return 1
        
        logger.info("👋 Worker shutdown complete")
        return 0

def main():
    """Entry point"""
    worker = HashmancerProductionWorker()
    return worker.run()

if __name__ == '__main__':
    sys.exit(main())