#!/usr/bin/env python3
"""Serve the actual enhanced portal HTML file."""

import http.server
import socketserver
import json
from pathlib import Path
from datetime import datetime

class EnhancedPortalHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/portal':
            # Serve the actual enhanced portal
            portal_file = Path("hashmancer/server/portal_enhanced.html")
            if portal_file.exists():
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(portal_file.read_bytes())
            else:
                self.send_response(404)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(b"Enhanced portal file not found")
                
        elif self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            health_data = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "service": "Hashmancer Enhanced Portal",
                "portal_ready": True
            }
            self.wfile.write(json.dumps(health_data).encode())
            
        elif self.path == '/api/server_status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            status_data = {
                "status": "running",
                "uptime": "active",
                "workers": [],
                "redis_connected": True,
                "enhanced_features": True
            }
            self.wfile.write(json.dumps(status_data).encode())
            
        elif self.path == '/api/jobs':
            # Return empty jobs list - the frontend now loads from localStorage
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            jobs_data = []  # Empty for now, frontend handles localStorage
            self.wfile.write(json.dumps(jobs_data).encode())
            
        elif self.path.startswith('/logs'):
            # Handle logs endpoint with filtering
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            # Parse query parameters
            from urllib.parse import urlparse, parse_qs
            parsed_url = urlparse(self.path)
            query_params = parse_qs(parsed_url.query)
            
            log_type = query_params.get('type', ['all'])[0]
            worker_filter = query_params.get('worker', [''])[0]
            search_term = query_params.get('search', [''])[0].lower()
            
            # Generate dynamic log entries with current timestamp
            import random
            from datetime import datetime, timedelta
            
            current_time = datetime.now()
            sample_logs = []
            
            # Generate logs from the last 2 hours for more variety
            for i in range(50):
                log_time = current_time - timedelta(minutes=random.randint(0, 120))
                worker_id = random.choice(['server', 'agent-001', 'agent-002', 'agent-003'])
                
                # Different message types based on worker
                if worker_id == 'server':
                    messages = [
                        ("Hashmancer portal server started", "info"),
                        ("Task pipeline initialized with 16 attack strategies", "info"), 
                        ("Notification sent via Discord webhook", "info"),
                        ("Job execution pipeline completed", "success"),
                        ("New hash uploaded for processing", "info"),
                        ("Template 'Corporate Audit' saved successfully", "success"),
                        ("Configuration updated: chunk size set to 600s", "info"),
                        ("User logged in successfully", "info"),
                        ("Backup created successfully", "success"),
                        ("Database connection established", "info")
                    ]
                else:
                    temp = random.randint(65, 85)
                    temp_status = 'normal' if temp < 80 else 'warning'
                    messages = [
                        ("Connected to Hashmancer server", "info"),
                        (f"Benchmark completed: {random.randint(800, 2500)/1000:.1f}M H/s", "success"),
                        (f"Started task execution for job_{log_time.strftime('%Y%m%d_%H%M%S')}", "info"),
                        (f"Hash cracked: {random.choice(['password123', 'admin2024', 'summer2023', 'welcome!', '123456'])}", "success"),
                        (f"Chunk processing completed ({random.randint(10, 95)}% keyspace)", "info"),
                        ("Task completed successfully", "success"),
                        (f"GPU temperature: {temp}°C ({temp_status})", "warning" if temp_status == "warning" else "info"),
                        (f"Memory usage: {random.randint(40, 90)}%", "info"),
                        (f"Processing mask attack: ?u?l?l?l?l?l?d?d?s", "info"),
                        ("Connection lost, attempting to reconnect...", "warning"),
                        ("Keyspace calculation completed", "info"),
                        (f"Speed: {random.randint(1000, 5000)} kH/s", "info")
                    ]
                
                message, level = random.choice(messages)
                
                log_entry = {
                    "datetime": log_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "worker_id": worker_id,
                    "message": message,
                    "level": level
                }
                
                # Apply filters
                if worker_filter and worker_id != worker_filter:
                    continue
                    
                if log_type != 'all' and level != log_type:
                    continue
                    
                if search_term and search_term not in message.lower():
                    continue
                
                sample_logs.append(log_entry)
            
            # Sort logs by datetime (newest first)
            sample_logs.sort(key=lambda x: x['datetime'], reverse=True)
            
            # Limit to 100 entries for performance
            sample_logs = sample_logs[:100]
            
            self.wfile.write(json.dumps(sample_logs).encode())
            
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        if self.path == '/api/jobs':
            # Handle job creation
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                job_data = json.loads(post_data.decode('utf-8'))
                
                # Generate a job ID
                job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(job_data.get('hash', ''))}"
                
                # Simulate job creation response
                response_data = {
                    "status": "success",
                    "job_id": job_id,
                    "message": "Job created successfully",
                    "hash": job_data.get('hash'),
                    "algorithm": job_data.get('algorithm'),
                    "attack_mode": job_data.get('attack_mode'),
                    "estimated_time": "5-15 minutes",
                    "priority": "normal"
                }
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response_data).encode())
                
            except json.JSONDecodeError:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                error_response = {"error": "Invalid JSON data"}
                self.wfile.write(json.dumps(error_response).encode())
                
        else:
            self.send_response(404)
            self.end_headers()

if __name__ == "__main__":
    PORT = 8078
    
    print("🚀 Hashmancer Enhanced Portal Server")
    print("✨ Serving the REAL enhanced portal!")
    print(f"🌐 Access: http://localhost:{PORT}")
    print("=" * 50)
    
    with socketserver.TCPServer(("", PORT), EnhancedPortalHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n👋 Enhanced portal server stopped")
            httpd.shutdown()