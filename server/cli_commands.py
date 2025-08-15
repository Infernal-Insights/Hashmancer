"""
CLI Commands for Vast.ai Worker Management
Provides comprehensive command-line interface for worker deployment and management
"""

import asyncio
import json
import sys
import argparse
from datetime import datetime
from typing import List, Optional
import click
from tabulate import tabulate

from vast_ai_manager import (
    VastAIManager, WorkerSpec, GPUType, WorkerStatus, 
    WorkerAutoScaler, CostOptimizer
)

class HashmancerCLI:
    """Main CLI interface for Hashmancer worker management"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        
    async def get_vast_manager(self):
        """Get async context manager for VastAI"""
        return VastAIManager(self.api_key)

@click.group()
@click.option('--api-key', envvar='VAST_API_KEY', required=True, help='Vast.ai API key')
@click.pass_context
def cli(ctx, api_key):
    """Hashmancer Vast.ai Worker Management CLI"""
    ctx.ensure_object(dict)
    ctx.obj['api_key'] = api_key
    ctx.obj['cli'] = HashmancerCLI(api_key)

@cli.group()
def workers():
    """Worker management commands"""
    pass

@workers.command('launch')
@click.option('--gpu-type', type=click.Choice([gpu.value for gpu in GPUType]), 
              default='rtx4090', help='GPU type to request')
@click.option('--gpu-count', type=int, default=1, help='Number of GPUs')
@click.option('--max-price', type=float, default=1.0, help='Maximum price per hour')
@click.option('--job-id', help='Job ID to assign to this worker')
@click.option('--cpu-cores', type=int, default=4, help='CPU cores required')
@click.option('--ram-gb', type=int, default=16, help='RAM in GB')
@click.option('--storage-gb', type=int, default=50, help='Storage in GB')
@click.option('--image', default='pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime', 
              help='Docker image to use')
@click.option('--startup-script', help='Custom startup script file')
@click.option('--env', multiple=True, help='Environment variables (KEY=VALUE)')
@click.pass_context
def launch_worker(ctx, gpu_type, gpu_count, max_price, job_id, cpu_cores, 
                 ram_gb, storage_gb, image, startup_script, env):
    """Launch a new worker on vast.ai"""
    
    async def _launch():
        # Parse environment variables
        env_vars = {}
        for env_var in env:
            if '=' in env_var:
                key, value = env_var.split('=', 1)
                env_vars[key] = value
                
        # Load custom startup script if provided
        script_content = None
        if startup_script:
            try:
                with open(startup_script, 'r') as f:
                    script_content = f.read()
            except FileNotFoundError:
                click.echo(f"Error: Startup script file '{startup_script}' not found", err=True)
                return
                
        # Create worker specification
        spec = WorkerSpec(
            gpu_type=GPUType(gpu_type),
            gpu_count=gpu_count,
            cpu_cores=cpu_cores,
            ram_gb=ram_gb,
            storage_gb=storage_gb,
            max_price_per_hour=max_price,
            image=image,
            startup_script=script_content,
            env_vars=env_vars
        )
        
        click.echo(f"Launching worker with {gpu_count}x {gpu_type} (max ${max_price:.3f}/hr)...")
        
        async with VastAIManager(ctx.obj['api_key']) as vast_manager:
            try:
                worker = await vast_manager.launch_worker(spec, job_id)
                
                click.echo(f"✅ Worker launched successfully!")
                click.echo(f"   Instance ID: {worker.instance_id}")
                click.echo(f"   Vast ID: {worker.vast_id}")
                click.echo(f"   Price: ${worker.price_per_hour:.3f}/hr")
                click.echo(f"   Status: {worker.status.value}")
                
                if job_id:
                    click.echo(f"   Assigned to job: {job_id}")
                    
            except Exception as e:
                click.echo(f"❌ Failed to launch worker: {e}", err=True)
                
    asyncio.run(_launch())

@workers.command('list')
@click.option('--status', type=click.Choice([s.value for s in WorkerStatus]), 
              help='Filter by worker status')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']), 
              default='table', help='Output format')
@click.pass_context
def list_workers(ctx, status, output_format):
    """List all workers"""
    
    async def _list():
        async with VastAIManager(ctx.obj['api_key']) as vast_manager:
            status_filter = WorkerStatus(status) if status else None
            workers = await vast_manager.list_workers(status_filter)
            
            if output_format == 'json':
                worker_data = []
                for worker in workers:
                    data = {
                        'instance_id': worker.instance_id,
                        'vast_id': worker.vast_id,
                        'status': worker.status.value,
                        'gpu_type': worker.gpu_type.value,
                        'gpu_count': worker.gpu_count,
                        'price_per_hour': worker.price_per_hour,
                        'total_cost': worker.total_cost,
                        'uptime_hours': worker.uptime_seconds / 3600,
                        'ip_address': worker.ip_address,
                        'job_assignments': worker.job_assignments or []
                    }
                    worker_data.append(data)
                click.echo(json.dumps(worker_data, indent=2))
            else:
                if not workers:
                    click.echo("No workers found")
                    return
                    
                headers = ['Instance ID', 'Vast ID', 'Status', 'GPU', 'Price/hr', 'Cost', 'Uptime', 'Jobs']
                rows = []
                
                for worker in workers:
                    uptime_str = f"{worker.uptime_seconds / 3600:.1f}h"
                    cost_str = f"${worker.total_cost:.2f}"
                    price_str = f"${worker.price_per_hour:.3f}"
                    gpu_str = f"{worker.gpu_count}x{worker.gpu_type.value}"
                    jobs_str = ','.join(worker.job_assignments or []) or 'None'
                    
                    # Truncate long values
                    if len(jobs_str) > 20:
                        jobs_str = jobs_str[:17] + '...'
                        
                    rows.append([
                        worker.instance_id[:12],  # Truncate ID
                        worker.vast_id,
                        worker.status.value,
                        gpu_str,
                        price_str,
                        cost_str,
                        uptime_str,
                        jobs_str
                    ])
                    
                click.echo(tabulate(rows, headers=headers, tablefmt='grid'))
                
    asyncio.run(_list())

@workers.command('status')
@click.argument('instance_id')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']), 
              default='table', help='Output format')
@click.pass_context
def worker_status(ctx, instance_id, output_format):
    """Get detailed status of a specific worker"""
    
    async def _status():
        async with VastAIManager(ctx.obj['api_key']) as vast_manager:
            worker = await vast_manager.get_worker_status(instance_id)
            
            if not worker:
                click.echo(f"Worker {instance_id} not found", err=True)
                return
                
            if output_format == 'json':
                data = {
                    'instance_id': worker.instance_id,
                    'vast_id': worker.vast_id,
                    'status': worker.status.value,
                    'gpu_type': worker.gpu_type.value,
                    'gpu_count': worker.gpu_count,
                    'price_per_hour': worker.price_per_hour,
                    'total_cost': worker.total_cost,
                    'uptime_seconds': worker.uptime_seconds,
                    'ip_address': worker.ip_address,
                    'ssh_port': worker.ssh_port,
                    'created_at': worker.created_at.isoformat(),
                    'job_assignments': worker.job_assignments or [],
                    'performance_metrics': worker.performance_metrics or {}
                }
                click.echo(json.dumps(data, indent=2))
            else:
                click.echo(f"Worker Details: {worker.instance_id}")
                click.echo(f"{'='*50}")
                click.echo(f"Vast ID:          {worker.vast_id}")
                click.echo(f"Status:           {worker.status.value}")
                click.echo(f"GPU:              {worker.gpu_count}x {worker.gpu_type.value}")
                click.echo(f"Price:            ${worker.price_per_hour:.3f}/hour")
                click.echo(f"Total Cost:       ${worker.total_cost:.2f}")
                click.echo(f"Uptime:           {worker.uptime_seconds / 3600:.1f} hours")
                click.echo(f"Created:          {worker.created_at}")
                
                if worker.ip_address:
                    click.echo(f"IP Address:       {worker.ip_address}:{worker.ssh_port}")
                    
                if worker.job_assignments:
                    click.echo(f"Job Assignments:  {', '.join(worker.job_assignments)}")
                    
                if worker.performance_metrics:
                    click.echo(f"Performance:")
                    for metric, value in worker.performance_metrics.items():
                        click.echo(f"  {metric}: {value}")
                        
    asyncio.run(_status())

@workers.command('stop')
@click.argument('instance_id')
@click.option('--reason', default='manual', help='Reason for stopping')
@click.option('--force', is_flag=True, help='Force stop without confirmation')
@click.pass_context
def stop_worker(ctx, instance_id, reason, force):
    """Stop a worker instance"""
    
    async def _stop():
        if not force:
            click.confirm(f'Are you sure you want to stop worker {instance_id}?', abort=True)
            
        async with VastAIManager(ctx.obj['api_key']) as vast_manager:
            success = await vast_manager.stop_worker(instance_id, reason)
            
            if success:
                click.echo(f"✅ Worker {instance_id} stop initiated")
            else:
                click.echo(f"❌ Failed to stop worker {instance_id}", err=True)
                
    asyncio.run(_stop())

@workers.command('logs')
@click.argument('instance_id')
@click.option('--follow', '-f', is_flag=True, help='Follow log output')
@click.pass_context
def worker_logs(ctx, instance_id, follow):
    """Get logs from a worker instance"""
    
    async def _logs():
        async with VastAIManager(ctx.obj['api_key']) as vast_manager:
            logs = await vast_manager.get_worker_logs(instance_id)
            
            if logs:
                click.echo(logs)
            else:
                click.echo(f"No logs available for worker {instance_id}")
                
            if follow:
                click.echo("Log following not yet implemented")
                
    asyncio.run(_logs())

@workers.command('assign')
@click.argument('instance_id')
@click.argument('job_id')
@click.pass_context
def assign_job(ctx, instance_id, job_id):
    """Assign a job to a worker"""
    
    async def _assign():
        async with VastAIManager(ctx.obj['api_key']) as vast_manager:
            success = await vast_manager.update_worker_assignment(instance_id, job_id)
            
            if success:
                click.echo(f"✅ Job {job_id} assigned to worker {instance_id}")
            else:
                click.echo(f"❌ Failed to assign job to worker", err=True)
                
    asyncio.run(_assign())

@cli.group()
def scaling():
    """Auto-scaling management commands"""
    pass

@scaling.command('status')
@click.pass_context
def scaling_status(ctx):
    """Show current scaling status and recommendations"""
    
    async def _status():
        async with VastAIManager(ctx.obj['api_key']) as vast_manager:
            scaler = WorkerAutoScaler(vast_manager)
            
            # Mock job queue data - in real implementation, this would come from job manager
            job_queue_size = 0
            worker_utilization = 0.5
            
            recommendation = await scaler.evaluate_scaling(job_queue_size, worker_utilization)
            
            click.echo("Auto-Scaling Status")
            click.echo("==================")
            click.echo(f"Current workers:     {len(await vast_manager.list_workers(WorkerStatus.RUNNING))}")
            click.echo(f"Job queue size:      {job_queue_size}")
            click.echo(f"Worker utilization:  {worker_utilization:.1%}")
            click.echo(f"Recommendation:      {recommendation['action']}")
            click.echo(f"Target workers:      {recommendation['target_workers']}")
            click.echo(f"Reason:              {recommendation['reason']}")
            
    asyncio.run(_status())

@scaling.command('execute')
@click.option('--dry-run', is_flag=True, help='Show what would be done without executing')
@click.pass_context
def execute_scaling(ctx, dry_run):
    """Execute auto-scaling recommendations"""
    
    async def _execute():
        async with VastAIManager(ctx.obj['api_key']) as vast_manager:
            scaler = WorkerAutoScaler(vast_manager)
            
            # Mock job queue data
            job_queue_size = 0
            worker_utilization = 0.5
            
            recommendation = await scaler.evaluate_scaling(job_queue_size, worker_utilization)
            
            if recommendation['action'] == 'none':
                click.echo("No scaling action needed")
                return
                
            click.echo(f"Scaling recommendation: {recommendation['action']}")
            click.echo(f"Target workers: {recommendation['target_workers']}")
            click.echo(f"Reason: {recommendation['reason']}")
            
            if dry_run:
                click.echo("(Dry run - no action taken)")
                return
                
            if click.confirm('Execute this scaling action?'):
                success = await scaler.execute_scaling(recommendation)
                if success:
                    click.echo("✅ Scaling action completed successfully")
                else:
                    click.echo("❌ Scaling action failed", err=True)
                    
    asyncio.run(_execute())

@cli.group()
def cost():
    """Cost optimization and reporting commands"""
    pass

@cost.command('summary')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']), 
              default='table', help='Output format')
@click.pass_context
def cost_summary(ctx, output_format):
    """Show cost summary for all workers"""
    
    async def _summary():
        async with VastAIManager(ctx.obj['api_key']) as vast_manager:
            summary = await vast_manager.get_cost_summary()
            
            if output_format == 'json':
                click.echo(json.dumps(summary, indent=2))
            else:
                click.echo("Cost Summary")
                click.echo("============")
                click.echo(f"Total cost:           ${summary['total_cost']:.2f}")
                click.echo(f"Estimated hourly:     ${summary['estimated_hourly_cost']:.2f}/hr")
                click.echo(f"Active workers:       {summary['active_workers']}")
                click.echo(f"Total workers:        {summary['total_workers']}")
                
                if summary['cost_by_gpu_type']:
                    click.echo("\nCost by GPU Type:")
                    for gpu_type, cost in summary['cost_by_gpu_type'].items():
                        click.echo(f"  {gpu_type:15} ${cost:.2f}")
                        
    asyncio.run(_summary())

@cost.command('optimize')
@click.option('--performance', type=click.Choice(['basic', 'standard', 'high', 'extreme']), 
              default='standard', help='Performance tier required')
@click.option('--budget', type=float, default=2.0, help='Maximum budget per hour')
@click.pass_context
def optimize_cost(ctx, performance, budget):
    """Find optimal worker configuration for given requirements"""
    
    async def _optimize():
        async with VastAIManager(ctx.obj['api_key']) as vast_manager:
            optimizer = CostOptimizer(vast_manager)
            
            spec = await optimizer.find_optimal_spec(performance, budget)
            
            if spec:
                click.echo("Optimal Worker Configuration")
                click.echo("============================")
                click.echo(f"GPU Type:         {spec.gpu_type.value}")
                click.echo(f"GPU Count:        {spec.gpu_count}")
                click.echo(f"CPU Cores:        {spec.cpu_cores}")
                click.echo(f"RAM:              {spec.ram_gb} GB")
                click.echo(f"Storage:          {spec.storage_gb} GB")
                click.echo(f"Max Price:        ${spec.max_price_per_hour:.3f}/hr")
                
                if click.confirm('Launch worker with this configuration?'):
                    try:
                        worker = await vast_manager.launch_worker(spec)
                        click.echo(f"✅ Worker launched: {worker.instance_id}")
                    except Exception as e:
                        click.echo(f"❌ Launch failed: {e}", err=True)
            else:
                click.echo("No suitable configuration found for requirements")
                
    asyncio.run(_optimize())

@cli.group()
def jobs():
    """Job management commands"""
    pass

@jobs.command('submit')
@click.option('--hash-file', required=True, help='Hash file to crack')
@click.option('--wordlist', required=True, help='Wordlist file')
@click.option('--rules', help='Rules file')
@click.option('--hash-type', default='md5', help='Hash type')
@click.option('--workers', type=int, default=1, help='Number of workers to use')
@click.option('--max-price', type=float, default=1.0, help='Maximum price per worker')
@click.option('--gpu-type', type=click.Choice([gpu.value for gpu in GPUType]), 
              default='rtx4090', help='GPU type preference')
@click.pass_context
def submit_job(ctx, hash_file, wordlist, rules, hash_type, workers, max_price, gpu_type):
    """Submit a new cracking job with automatic worker deployment"""
    
    async def _submit():
        job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        click.echo(f"Submitting job: {job_id}")
        click.echo(f"Hash file: {hash_file}")
        click.echo(f"Wordlist: {wordlist}")
        click.echo(f"Workers needed: {workers}")
        
        async with VastAIManager(ctx.obj['api_key']) as vast_manager:
            # Launch required workers
            launched_workers = []
            
            for i in range(workers):
                spec = WorkerSpec(
                    gpu_type=GPUType(gpu_type),
                    max_price_per_hour=max_price
                )
                
                try:
                    worker = await vast_manager.launch_worker(spec, job_id)
                    launched_workers.append(worker)
                    click.echo(f"✅ Launched worker {i+1}/{workers}: {worker.instance_id}")
                except Exception as e:
                    click.echo(f"❌ Failed to launch worker {i+1}: {e}", err=True)
                    
            if launched_workers:
                click.echo(f"\n✅ Job {job_id} submitted with {len(launched_workers)} workers")
                click.echo("Workers will automatically start processing when ready")
            else:
                click.echo("❌ Failed to launch any workers for job", err=True)
                
    asyncio.run(_submit())

if __name__ == '__main__':
    cli()