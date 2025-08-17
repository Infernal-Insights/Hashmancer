#!/usr/bin/env python3
"""
Hashmancer Worker CLI
Worker management, status monitoring, and job control
"""

import click
import requests
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
import sys

# Install missing dependencies if needed
try:
    from tabulate import tabulate
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tabulate"])
    from tabulate import tabulate

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

@click.group()
def worker_cli():
    """👷 Worker management and control"""
    pass

@worker_cli.command()
@click.option('--server', default='localhost:8080', help='Server address (host:port)')
@click.option('--worker-id', help='Worker ID (auto-generated if not provided)')
@click.option('--port', default=8081, help='Worker port')
@click.option('--max-jobs', default=3, help='Maximum concurrent jobs')
def start(server, worker_id, port, max_jobs):
    """Start a worker instance"""
    if not worker_id:
        import uuid
        worker_id = f"worker-{uuid.uuid4().hex[:8]}"
    
    click.echo(f"🚀 Starting worker: {worker_id}")
    click.echo(f"   Server: {server}")
    click.echo(f"   Port: {port}")
    click.echo(f"   Max jobs: {max_jobs}")
    
    try:
        # Start worker process
        cmd = [
            sys.executable, "-m", "hashmancer.worker",
            "--server", server,
            "--worker-id", worker_id,
            "--port", str(port),
            "--max-jobs", str(max_jobs)
        ]
        
        click.echo(f"Executing: {' '.join(cmd)}")
        subprocess.run(cmd, cwd=project_root)
        
    except KeyboardInterrupt:
        click.echo(f"\n👋 Worker {worker_id} stopped")
    except Exception as e:
        click.echo(f"❌ Failed to start worker: {e}")

@worker_cli.command()
@click.option('--worker-id', help='Specific worker ID to check')
@click.option('--all', 'check_all', is_flag=True, help='Check all known workers')
@click.option('--format', type=click.Choice(['table', 'json']), default='table')
def status(worker_id, check_all, format):
    """Check worker status"""
    if worker_id:
        workers = [worker_id]
    elif check_all:
        # Get list of all workers from server
        try:
            response = requests.get("http://localhost:8080/workers", timeout=5)
            if response.status_code == 200:
                workers_data = response.json()
                workers = [w.get('id') for w in workers_data]
            else:
                click.echo(f"❌ Failed to get workers list: {response.status_code}")
                return
        except requests.RequestException as e:
            click.echo(f"❌ Error getting workers: {e}")
            return
    else:
        click.echo("❌ Must specify --worker-id or --all")
        return
    
    workers_status = []
    for wid in workers:
        status_info = get_worker_status(wid)
        workers_status.append(status_info)
    
    if format == 'json':
        click.echo(json.dumps(workers_status, indent=2))
    else:
        if not workers_status:
            click.echo("No workers found")
            return
        
        from tabulate import tabulate
        headers = ['Worker ID', 'Status', 'Jobs', 'Speed', 'Uptime', 'Last Seen']
        rows = []
        
        for worker in workers_status:
            rows.append([
                worker.get('id', 'unknown')[:12],
                worker.get('status', 'unknown'),
                f"{worker.get('active_jobs', 0)}/{worker.get('max_jobs', 0)}",
                worker.get('speed', 'unknown'),
                worker.get('uptime', 'unknown'),
                worker.get('last_seen', 'unknown')[:19]
            ])
        
        click.echo(tabulate(rows, headers=headers, tablefmt='grid'))

@worker_cli.command()
@click.argument('worker_id')
@click.option('--server', default='localhost:8080', help='Server address')
def connect(worker_id, server):
    """Connect worker to server"""
    click.echo(f"🔗 Connecting worker {worker_id} to {server}")
    
    try:
        # Send connect request
        payload = {
            'worker_id': worker_id,
            'action': 'connect',
            'timestamp': datetime.now().isoformat()
        }
        
        response = requests.post(f"http://{server}/worker/connect", json=payload, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            click.echo(f"✅ Worker connected successfully")
            click.echo(f"   Session ID: {data.get('session_id')}")
            click.echo(f"   Server assignment: {data.get('assignment')}")
        else:
            click.echo(f"❌ Connection failed: {response.status_code}")
            
    except requests.RequestException as e:
        click.echo(f"❌ Connection error: {e}")

@worker_cli.command()
@click.argument('worker_id')
@click.option('--job-id', help='Specific job to skip')
@click.option('--reason', default='manual', help='Reason for skipping')
def skip(worker_id, job_id, reason):
    """Skip current job or specific job"""
    click.echo(f"⏭️  Skipping job for worker {worker_id}")
    
    try:
        payload = {
            'worker_id': worker_id,
            'action': 'skip_job',
            'job_id': job_id,
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        }
        
        response = requests.post("http://localhost:8080/worker/skip", json=payload, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            skipped_job = data.get('skipped_job', job_id or 'current')
            click.echo(f"✅ Job {skipped_job} skipped successfully")
            next_job = data.get('next_job')
            if next_job:
                click.echo(f"   Next job: {next_job}")
        else:
            click.echo(f"❌ Skip failed: {response.status_code}")
            
    except requests.RequestException as e:
        click.echo(f"❌ Skip error: {e}")

@worker_cli.command()
@click.argument('worker_id')
@click.option('--force', is_flag=True, help='Force stop without confirmation')
def stop(worker_id, force):
    """Stop a worker"""
    if not force and not click.confirm(f'Stop worker {worker_id}?'):
        click.echo("Cancelled")
        return
    
    click.echo(f"🛑 Stopping worker {worker_id}")
    
    try:
        payload = {
            'worker_id': worker_id,
            'action': 'stop',
            'timestamp': datetime.now().isoformat()
        }
        
        response = requests.post("http://localhost:8080/worker/stop", json=payload, timeout=5)
        
        if response.status_code == 200:
            click.echo(f"✅ Worker {worker_id} stop initiated")
        else:
            click.echo(f"❌ Stop failed: {response.status_code}")
            
    except requests.RequestException as e:
        click.echo(f"❌ Stop error: {e}")

@worker_cli.command()
@click.argument('worker_id')
@click.option('--follow', '-f', is_flag=True, help='Follow log output')
@click.option('--lines', default=50, help='Number of lines to show')
def logs(worker_id, follow, lines):
    """Get worker logs"""
    click.echo(f"📋 Getting logs for worker {worker_id}")
    
    try:
        params = {'lines': lines}
        response = requests.get(f"http://localhost:8080/worker/{worker_id}/logs", params=params, timeout=10)
        
        if response.status_code == 200:
            logs_data = response.json()
            
            if isinstance(logs_data, list):
                for log_entry in logs_data:
                    timestamp = log_entry.get('timestamp', '')
                    level = log_entry.get('level', 'info').upper()
                    message = log_entry.get('message', '')
                    click.echo(f"[{timestamp}] {level}: {message}")
            else:
                click.echo(logs_data)
                
            if follow:
                click.echo("\n📡 Following logs (Ctrl+C to stop)...")
                follow_worker_logs(worker_id)
                
        else:
            click.echo(f"❌ Failed to get logs: {response.status_code}")
            
    except requests.RequestException as e:
        click.echo(f"❌ Logs error: {e}")
    except KeyboardInterrupt:
        click.echo("\n👋 Stopped following logs")

@worker_cli.command()
@click.argument('worker_id')
@click.option('--key', help='Configuration key to set')
@click.option('--value', help='Configuration value')
@click.option('--config-file', help='JSON configuration file')
def configure(worker_id, key, value, config_file):
    """Configure worker settings"""
    click.echo(f"⚙️  Configuring worker {worker_id}")
    
    config = {}
    if config_file:
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
        except Exception as e:
            click.echo(f"❌ Error loading config file: {e}")
            return
    elif key and value:
        config[key] = value
    else:
        click.echo("❌ Must specify either --key/--value or --config-file")
        return
    
    try:
        payload = {
            'worker_id': worker_id,
            'config': config,
            'timestamp': datetime.now().isoformat()
        }
        
        response = requests.post("http://localhost:8080/worker/configure", json=payload, timeout=5)
        
        if response.status_code == 200:
            click.echo(f"✅ Worker {worker_id} configured successfully")
            applied_config = response.json().get('applied_config', {})
            for k, v in applied_config.items():
                click.echo(f"   {k}: {v}")
        else:
            click.echo(f"❌ Configuration failed: {response.status_code}")
            
    except requests.RequestException as e:
        click.echo(f"❌ Configuration error: {e}")

@worker_cli.command()
@click.option('--port', default=8081, help='Port to listen on')
@click.option('--max-jobs', default=3, help='Maximum concurrent jobs')
def daemon(port, max_jobs):
    """Run worker in daemon mode"""
    click.echo(f"👹 Starting worker daemon on port {port}")
    click.echo(f"   Max concurrent jobs: {max_jobs}")
    
    try:
        # Start worker daemon
        cmd = [
            sys.executable, "-m", "hashmancer.worker.daemon",
            "--port", str(port),
            "--max-jobs", str(max_jobs)
        ]
        
        subprocess.run(cmd, cwd=project_root)
        
    except KeyboardInterrupt:
        click.echo("\n👋 Worker daemon stopped")
    except Exception as e:
        click.echo(f"❌ Failed to start daemon: {e}")

def get_worker_status(worker_id):
    """Get status information for a worker"""
    try:
        response = requests.get(f"http://localhost:8080/worker/{worker_id}/status", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    
    return {
        'id': worker_id,
        'status': 'unreachable',
        'active_jobs': 0,
        'max_jobs': 0,
        'speed': 'unknown',
        'uptime': 'unknown',
        'last_seen': 'unknown'
    }

def follow_worker_logs(worker_id):
    """Follow worker logs in real-time"""
    last_timestamp = None
    
    try:
        while True:
            params = {'since': last_timestamp} if last_timestamp else {}
            response = requests.get(f"http://localhost:8080/worker/{worker_id}/logs/stream", params=params, timeout=5)
            
            if response.status_code == 200:
                logs_data = response.json()
                if isinstance(logs_data, list) and logs_data:
                    for log_entry in logs_data:
                        timestamp = log_entry.get('timestamp', '')
                        level = log_entry.get('level', 'info').upper()
                        message = log_entry.get('message', '')
                        click.echo(f"[{timestamp}] {level}: {message}")
                        last_timestamp = timestamp
            
            time.sleep(2)  # Poll every 2 seconds
            
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    worker_cli()