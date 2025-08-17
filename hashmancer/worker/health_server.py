#!/usr/bin/env python3
"""
Health Server for Hashmancer Worker
Provides HTTP health check endpoint and basic worker info
"""

import json
import time
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
import logging

logger = logging.getLogger('health-server')

class HealthHandler(BaseHTTPRequestHandler):
    def __init__(self, worker_instance, *args, **kwargs):
        self.worker = worker_instance
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/health':
            self._handle_health_check()
        elif self.path == '/status':
            self._handle_status()
        elif self.path == '/jobs':
            self._handle_jobs()
        elif self.path == '/metrics':
            self._handle_metrics()
        else:
            self._handle_not_found()
    
    def _handle_health_check(self):
        """Basic health check endpoint"""
        health_data = {
            'status': 'healthy' if self.worker.running else 'shutting_down',
            'worker_id': self.worker.worker_id,
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'uptime': time.time() - getattr(self.worker, 'start_time', time.time())
        }
        
        self._send_json_response(health_data)
    
    def _handle_status(self):
        """Detailed worker status"""
        with self.worker.job_lock:
            active_jobs = len(self.worker.active_jobs)
            job_details = [
                {
                    'job_id': job_id,
                    'status': job_info['status'],
                    'runtime': time.time() - job_info['start_time']
                }
                for job_id, job_info in self.worker.active_jobs.items()
            ]
        
        status_data = {
            'worker_id': self.worker.worker_id,
            'status': 'active' if self.worker.running else 'shutting_down',
            'registered': self.worker.registered,
            'server': f"{self.worker.server_host}:{self.worker.server_port}",
            'capabilities': self.worker.get_system_capabilities(),
            'active_jobs': active_jobs,
            'max_jobs': self.worker.max_jobs,
            'job_details': job_details,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
        
        self._send_json_response(status_data)
    
    def _handle_jobs(self):
        """Current job information"""
        with self.worker.job_lock:
            jobs_data = []
            for job_id, job_info in self.worker.active_jobs.items():
                job_data = job_info['job_data'].copy()
                job_data.update({
                    'worker_status': job_info['status'],
                    'start_time': job_info['start_time'],
                    'runtime': time.time() - job_info['start_time']
                })
                jobs_data.append(job_data)
        
        self._send_json_response({
            'worker_id': self.worker.worker_id,
            'active_jobs': jobs_data,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        })
    
    def _handle_metrics(self):
        """Performance metrics"""
        import psutil
        
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            metrics = {
                'worker_id': self.worker.worker_id,
                'system': {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_available_gb': memory.available / (1024**3),
                    'disk_percent': disk.percent,
                    'disk_free_gb': disk.free / (1024**3)
                },
                'worker': {
                    'active_jobs': len(self.worker.active_jobs),
                    'max_jobs': self.worker.max_jobs,
                    'registered': self.worker.registered,
                    'running': self.worker.running
                },
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }
            
            # GPU metrics if available
            try:
                import subprocess
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu', 
                     '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=5
                )
                
                if result.returncode == 0:
                    gpu_metrics = []
                    for i, line in enumerate(result.stdout.strip().split('\n')):
                        if line.strip():
                            parts = line.split(', ')
                            if len(parts) >= 4:
                                gpu_metrics.append({
                                    'gpu_id': i,
                                    'utilization_percent': int(parts[0]),
                                    'memory_used_mb': int(parts[1]),
                                    'memory_total_mb': int(parts[2]),
                                    'temperature_c': int(parts[3])
                                })
                    
                    metrics['gpu'] = gpu_metrics
                    
            except:
                pass
            
            self._send_json_response(metrics)
            
        except Exception as e:
            self._send_json_response({
                'error': f'Failed to get metrics: {e}',
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }, status_code=500)
    
    def _handle_not_found(self):
        """Handle 404 responses"""
        self.send_response(404)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        error_data = {
            'error': 'Not found',
            'path': self.path,
            'available_endpoints': ['/health', '/status', '/jobs', '/metrics']
        }
        
        self.wfile.write(json.dumps(error_data).encode())
    
    def _send_json_response(self, data, status_code=200):
        """Send JSON response"""
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')  # CORS for web interfaces
        self.end_headers()
        
        response_json = json.dumps(data, indent=2)
        self.wfile.write(response_json.encode())
    
    def log_message(self, format, *args):
        """Suppress HTTP request logs (use debug level)"""
        logger.debug(f"{self.client_address[0]} - {format % args}")

class HealthServer:
    def __init__(self, worker_instance, port):
        self.worker = worker_instance
        self.port = port
        self.server = None
    
    def start(self):
        """Start the health server"""
        try:
            # Create handler class with worker instance
            def handler(*args, **kwargs):
                return HealthHandler(self.worker, *args, **kwargs)
            
            self.server = HTTPServer(('0.0.0.0', self.port), handler)
            logger.info(f"Health server listening on port {self.port}")
            
            # Set start time for uptime calculation
            self.worker.start_time = time.time()
            
            self.server.serve_forever()
            
        except Exception as e:
            logger.error(f"Health server failed: {e}")
            raise
    
    def stop(self):
        """Stop the health server"""
        if self.server:
            self.server.shutdown()
            logger.info("Health server stopped")