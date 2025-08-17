#!/usr/bin/env python3
"""
Hashmancer Server CLI
Server management, broadcasting, and job control
"""

import click
import asyncio
import json
import socket
import subprocess
import time
import requests
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
def server_cli():
    """🖥️  Server management and broadcasting"""
    pass

@server_cli.command()
@click.option('--port', default=8080, help='Server port')
@click.option('--host', default='0.0.0.0', help='Server host')
@click.option('--debug', is_flag=True, help='Enable debug mode')
def start(port, host, debug):
    """Start the Hashmancer server"""
    click.echo(f"🚀 Starting Hashmancer server on {host}:{port}")
    
    # Check if port is available
    if is_port_in_use(port):
        click.echo(f"❌ Port {port} is already in use")
        return
    
    try:
        # Start the enhanced server
        cmd = [
            sys.executable, "-m", "hashmancer.server.main",
            "--host", host,
            "--port", str(port)
        ]
        
        if debug:
            cmd.append("--debug")
            
        click.echo(f"Executing: {' '.join(cmd)}")
        subprocess.run(cmd, cwd=project_root)
        
    except KeyboardInterrupt:
        click.echo("\n👋 Server stopped")
    except Exception as e:
        click.echo(f"❌ Failed to start server: {e}")

@server_cli.command()
@click.option('--port', default=8080, help='Server port to check')
def status(port):
    """Check server status"""
    try:
        response = requests.get(f"http://localhost:{port}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            click.echo("✅ Server is running")
            click.echo(f"   Status: {data.get('status', 'unknown')}")
            click.echo(f"   Service: {data.get('service', 'unknown')}")
            click.echo(f"   Timestamp: {data.get('timestamp', 'unknown')}")
        else:
            click.echo(f"❌ Server returned status {response.status_code}")
    except requests.RequestException:
        click.echo("❌ Server is not responding")

@server_cli.command()
@click.option('--port', default=8080, help='Server port')
def stop(port):
    """Stop the Hashmancer server"""
    click.echo(f"🛑 Stopping server on port {port}")
    
    # Find and kill process using the port
    try:
        cmd = ["lsof", "-t", f"-i:{port}"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                if pid:
                    subprocess.run(["kill", pid])
                    click.echo(f"✅ Stopped process {pid}")
        else:
            click.echo(f"No process found on port {port}")
            
    except Exception as e:
        click.echo(f"❌ Error stopping server: {e}")

@server_cli.command()
@click.option('--network', default='192.168.1.0/24', help='Network to scan for workers')
@click.option('--port', default=8081, help='Worker port to scan')
@click.option('--timeout', default=2, help='Scan timeout in seconds')
def discover(network, port, timeout):
    """Broadcast to discover local workers"""
    click.echo(f"🔍 Discovering workers on {network}:{port}")
    
    # Parse network CIDR
    import ipaddress
    try:
        net = ipaddress.IPv4Network(network, strict=False)
        discovered_workers = []
        
        with click.progressbar(list(net.hosts()), label="Scanning") as hosts:
            for host in hosts:
                if is_worker_responding(str(host), port, timeout):
                    worker_info = get_worker_info(str(host), port)
                    discovered_workers.append(worker_info)
                    click.echo(f"\n✅ Found worker: {host}:{port}")
        
        if discovered_workers:
            click.echo(f"\n🎯 Discovered {len(discovered_workers)} workers:")
            for worker in discovered_workers:
                click.echo(f"   • {worker['host']}:{worker['port']} - {worker.get('status', 'unknown')}")
        else:
            click.echo("\n❌ No workers found")
            
    except ValueError as e:
        click.echo(f"❌ Invalid network format: {e}")

@server_cli.command()
@click.option('--config-file', help='Configuration file to broadcast')
@click.option('--network', default='192.168.1.0/24', help='Network to broadcast to')
@click.option('--port', default=8081, help='Worker port')
def broadcast(config_file, network, port):
    """Broadcast configuration to workers"""
    click.echo(f"📡 Broadcasting configuration to workers on {network}:{port}")
    
    # Load configuration
    config = {}
    if config_file:
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
        except Exception as e:
            click.echo(f"❌ Error loading config file: {e}")
            return
    else:
        # Default configuration
        config = {
            "server_host": "localhost",
            "server_port": 8080,
            "max_concurrent_jobs": 3,
            "chunk_size": 600,
            "timestamp": datetime.now().isoformat()
        }
    
    # Broadcast to network
    import ipaddress
    try:
        net = ipaddress.IPv4Network(network, strict=False)
        success_count = 0
        
        with click.progressbar(list(net.hosts()), label="Broadcasting") as hosts:
            for host in hosts:
                if broadcast_config_to_worker(str(host), port, config):
                    success_count += 1
        
        click.echo(f"\n✅ Configuration broadcasted to {success_count} workers")
        
    except ValueError as e:
        click.echo(f"❌ Invalid network format: {e}")

@server_cli.command()
@click.option('--format', type=click.Choice(['table', 'json']), default='table')
def jobs(format):
    """List all jobs"""
    click.echo("📋 Listing jobs...")
    
    try:
        response = requests.get("http://localhost:8080/jobs", timeout=5)
        if response.status_code == 200:
            jobs_data = response.json()
            
            if format == 'json':
                click.echo(json.dumps(jobs_data, indent=2))
            else:
                if not jobs_data:
                    click.echo("No jobs found")
                    return
                
                from tabulate import tabulate
                headers = ['ID', 'Status', 'Algorithm', 'Progress', 'Created']
                rows = []
                
                for job in jobs_data:
                    rows.append([
                        job.get('id', 'unknown')[:12],
                        job.get('status', 'unknown'),
                        job.get('algorithm', 'unknown'),
                        f"{job.get('progress', 0):.1%}",
                        job.get('created_at', 'unknown')[:19]
                    ])
                
                click.echo(tabulate(rows, headers=headers, tablefmt='grid'))
        else:
            click.echo(f"❌ Failed to fetch jobs: {response.status_code}")
            
    except requests.RequestException as e:
        click.echo(f"❌ Error fetching jobs: {e}")

@server_cli.command()
@click.argument('gpu_type')
@click.option('--count', default=1, help='Number of workers to add')
@click.option('--max-price', type=float, default=1.0, help='Maximum price per hour')
@click.option('--vast-api-key', envvar='VAST_API_KEY', help='Vast.ai API key')
@click.option('--docker-image', default='hashmancer/worker:latest', help='Docker image to use')
def add_worker(gpu_type, count, max_price, vast_api_key, docker_image):
    """Add cheapest available worker on Vast.ai (e.g., 'add 3080')"""
    if not vast_api_key:
        click.echo("❌ Vast.ai API key required (set VAST_API_KEY env var)")
        return
    
    click.echo(f"🔍 Finding cheapest {gpu_type} workers on Vast.ai...")
    click.echo(f"   Count: {count}")
    click.echo(f"   Max price: ${max_price:.3f}/hr")
    
    try:
        # Search for available instances
        headers = {'Authorization': f'Bearer {vast_api_key}'}
        params = {
            'q': f'gpu_name:{gpu_type} rentable:true',
            'order': 'dph_total'  # Order by price
        }
        
        response = requests.get('https://console.vast.ai/api/v0/bundles/', 
                              headers=headers, params=params, timeout=10)
        
        if response.status_code != 200:
            click.echo(f"❌ Failed to search Vast.ai: {response.status_code}")
            return
        
        offers = response.json().get('offers', [])
        
        # Filter by price and GPU type
        suitable_offers = []
        for offer in offers:
            if (offer.get('dph_total', float('inf')) <= max_price and 
                gpu_type.lower() in offer.get('gpu_name', '').lower()):
                suitable_offers.append(offer)
        
        if not suitable_offers:
            click.echo(f"❌ No suitable {gpu_type} workers found under ${max_price:.3f}/hr")
            return
        
        # Display best offers
        click.echo(f"\n💰 Found {len(suitable_offers)} suitable offers:")
        for i, offer in enumerate(suitable_offers[:5]):  # Show top 5
            click.echo(f"   {i+1}. {offer.get('gpu_name')} - ${offer.get('dph_total'):.3f}/hr - {offer.get('cpu_cores')}CPU {offer.get('ram')}GB RAM")
        
        if click.confirm(f'\nLaunch {count} workers using cheapest offers?'):
            launched = 0
            
            for i in range(min(count, len(suitable_offers))):
                offer = suitable_offers[i]
                
                # Get public IP for setup script
                try:
                    public_ip = requests.get('http://ifconfig.me', timeout=5).text.strip()
                except:
                    public_ip = 'localhost'
                
                # Launch instance
                launch_data = {
                    'client_id': 'hashmancer',
                    'image': 'nvidia/cuda:11.8-devel-ubuntu20.04',
                    'env': {
                        'HASHMANCER_SERVER_IP': public_ip
                    },
                    'onstart': f'wget -O - http://{public_ip}:8888/setup | bash'
                }
                
                response = requests.put(f'https://console.vast.ai/api/v0/asks/{offer["id"]}/', 
                                      headers=headers, json=launch_data, timeout=30)
                
                if response.status_code == 200:
                    instance_data = response.json()
                    click.echo(f"✅ Launched worker {i+1}: {instance_data.get('new_contract')}")
                    launched += 1
                else:
                    click.echo(f"❌ Failed to launch worker {i+1}: {response.status_code}")
            
            click.echo(f"\n🎯 Successfully launched {launched}/{count} workers")
            
    except requests.RequestException as e:
        click.echo(f"❌ Network error: {e}")
    except Exception as e:
        click.echo(f"❌ Error: {e}")

@server_cli.command()
@click.option('--job-ids', help='Comma-separated job IDs to delete')
@click.option('--status', help='Delete all jobs with this status')
@click.option('--confirm', is_flag=True, help='Skip confirmation prompt')
def delete_jobs(job_ids, status, confirm):
    """Delete jobs in batch"""
    jobs_to_delete = []
    
    if job_ids:
        jobs_to_delete = [id.strip() for id in job_ids.split(',')]
    elif status:
        # Get jobs with specific status
        try:
            response = requests.get("http://localhost:8080/jobs", timeout=5)
            if response.status_code == 200:
                all_jobs = response.json()
                jobs_to_delete = [job['id'] for job in all_jobs if job.get('status') == status]
            else:
                click.echo(f"❌ Failed to fetch jobs: {response.status_code}")
                return
        except requests.RequestException as e:
            click.echo(f"❌ Error fetching jobs: {e}")
            return
    else:
        click.echo("❌ Must specify either --job-ids or --status")
        return
    
    if not jobs_to_delete:
        click.echo("No jobs to delete")
        return
    
    click.echo(f"🗑️  Will delete {len(jobs_to_delete)} jobs:")
    for job_id in jobs_to_delete:
        click.echo(f"   • {job_id}")
    
    if not confirm and not click.confirm('Continue with deletion?'):
        click.echo("Cancelled")
        return
    
    # Delete jobs
    success_count = 0
    for job_id in jobs_to_delete:
        try:
            response = requests.delete(f"http://localhost:8080/jobs/{job_id}", timeout=5)
            if response.status_code == 200:
                success_count += 1
                click.echo(f"✅ Deleted {job_id}")
            else:
                click.echo(f"❌ Failed to delete {job_id}: {response.status_code}")
        except requests.RequestException as e:
            click.echo(f"❌ Error deleting {job_id}: {e}")
    
    click.echo(f"\n✅ Successfully deleted {success_count}/{len(jobs_to_delete)} jobs")

@server_cli.command()
@click.option('--port', default=8888, help='Port to host setup script on')
def host_setup(port):
    """Host the worker setup script for Vast.ai instances"""
    click.echo(f"🌐 Hosting worker setup script on port {port}")
    
    try:
        # Get public IP
        public_ip = requests.get('http://ifconfig.me', timeout=5).text.strip()
        click.echo(f"📡 Public IP: {public_ip}")
        click.echo(f"📋 Vast.ai workers will use: wget -O - http://{public_ip}:{port}/setup | bash")
    except:
        click.echo("⚠️  Could not determine public IP")
    
    # Start the hosting server
    script_path = project_root / "scripts" / "host-setup-script.py"
    try:
        subprocess.run([sys.executable, str(script_path), str(port)])
    except KeyboardInterrupt:
        click.echo("\n👋 Setup script hosting stopped")

def is_port_in_use(port):
    """Check if a port is in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def is_worker_responding(host, port, timeout):
    """Check if a worker is responding on the given host:port"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            result = s.connect_ex((host, port))
            return result == 0
    except:
        return False

def get_worker_info(host, port):
    """Get worker information"""
    try:
        response = requests.get(f"http://{host}:{port}/status", timeout=2)
        if response.status_code == 200:
            data = response.json()
            data['host'] = host
            data['port'] = port
            return data
    except:
        pass
    
    return {'host': host, 'port': port, 'status': 'unknown'}

def broadcast_config_to_worker(host, port, config):
    """Broadcast configuration to a specific worker"""
    try:
        response = requests.post(
            f"http://{host}:{port}/config",
            json=config,
            timeout=2
        )
        return response.status_code == 200
    except:
        return False

if __name__ == "__main__":
    server_cli()