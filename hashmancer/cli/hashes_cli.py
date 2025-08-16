#!/usr/bin/env python3
"""
Hashmancer Hashes.com Integration CLI
Automated job pulling, filtering, and management
"""

import click
import requests
import json
import hashlib
from datetime import datetime, timedelta
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
def hashes_cli():
    """🌐 Hashes.com integration and job pulling"""
    pass

@hashes_cli.command()
@click.option('--api-key', required=True, help='Hashes.com API key')
@click.option('--exclude-md5', is_flag=True, help='Exclude MD5 hashes')
@click.option('--exclude-btc', is_flag=True, help='Exclude Bitcoin hashes')
@click.option('--max-jobs', default=10, help='Maximum jobs to pull')
@click.option('--min-reward', type=float, default=0.0, help='Minimum reward threshold')
@click.option('--auto-accept', is_flag=True, help='Automatically accept suitable jobs')
@click.option('--dry-run', is_flag=True, help='Show what would be pulled without executing')
def pull(api_key, exclude_md5, exclude_btc, max_jobs, min_reward, auto_accept, dry_run):
    """Pull recent jobs from hashes.com with filtering"""
    click.echo("🌐 Pulling jobs from hashes.com...")
    
    if dry_run:
        click.echo("🧪 DRY RUN MODE - No jobs will be actually pulled")
    
    try:
        # Get recent jobs from hashes.com
        headers = {'Authorization': f'Bearer {api_key}'}
        response = requests.get('https://hashes.com/api/v1/jobs/recent', headers=headers, timeout=10)
        
        if response.status_code != 200:
            click.echo(f"❌ Failed to fetch jobs: {response.status_code}")
            return
        
        jobs_data = response.json()
        jobs = jobs_data.get('jobs', [])
        
        click.echo(f"📋 Found {len(jobs)} recent jobs")
        
        # Filter jobs
        filtered_jobs = filter_jobs(jobs, exclude_md5, exclude_btc, min_reward)
        
        click.echo(f"✅ {len(filtered_jobs)} jobs passed filters")
        
        # Check for duplicates
        existing_jobs = get_existing_jobs()
        new_jobs = remove_duplicates(filtered_jobs, existing_jobs)
        
        click.echo(f"🔄 {len(new_jobs)} new jobs (excluding duplicates)")
        
        # Limit results
        limited_jobs = new_jobs[:max_jobs]
        
        if not limited_jobs:
            click.echo("ℹ️  No new jobs to process")
            return
        
        # Display jobs
        display_jobs_table(limited_jobs)
        
        if dry_run:
            click.echo(f"\n🧪 Would pull {len(limited_jobs)} jobs")
            return
        
        if auto_accept or click.confirm(f'\nPull {len(limited_jobs)} jobs?'):
            pulled_count = 0
            
            for job in limited_jobs:
                if pull_single_job(job, api_key):
                    pulled_count += 1
                    click.echo(f"✅ Pulled job: {job['id']}")
                else:
                    click.echo(f"❌ Failed to pull job: {job['id']}")
            
            click.echo(f"\n🎯 Successfully pulled {pulled_count}/{len(limited_jobs)} jobs")
            
            # Save job IDs to prevent duplicates
            save_pulled_jobs(limited_jobs)
        
    except requests.RequestException as e:
        click.echo(f"❌ Network error: {e}")
    except Exception as e:
        click.echo(f"❌ Error: {e}")

@hashes_cli.command()
@click.option('--api-key', required=True, help='Hashes.com API key')
@click.option('--format', type=click.Choice(['table', 'json']), default='table')
@click.option('--status', help='Filter by status')
def list(api_key, format, status):
    """List jobs from hashes.com"""
    click.echo("📋 Listing hashes.com jobs...")
    
    try:
        headers = {'Authorization': f'Bearer {api_key}'}
        params = {}
        if status:
            params['status'] = status
        
        response = requests.get('https://hashes.com/api/v1/jobs', headers=headers, params=params, timeout=10)
        
        if response.status_code != 200:
            click.echo(f"❌ Failed to fetch jobs: {response.status_code}")
            return
        
        jobs_data = response.json()
        jobs = jobs_data.get('jobs', [])
        
        if format == 'json':
            click.echo(json.dumps(jobs, indent=2))
        else:
            display_jobs_table(jobs)
            
    except requests.RequestException as e:
        click.echo(f"❌ Network error: {e}")

@hashes_cli.command()
@click.argument('job_id')
@click.option('--api-key', required=True, help='Hashes.com API key')
@click.option('--output-dir', default='./jobs', help='Output directory for job files')
def download(job_id, api_key, output_dir):
    """Download specific job from hashes.com"""
    click.echo(f"⬇️  Downloading job {job_id}...")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    try:
        headers = {'Authorization': f'Bearer {api_key}'}
        
        # Get job details
        response = requests.get(f'https://hashes.com/api/v1/jobs/{job_id}', headers=headers, timeout=10)
        
        if response.status_code != 200:
            click.echo(f"❌ Failed to fetch job details: {response.status_code}")
            return
        
        job_data = response.json()
        
        # Download hash file
        hash_file = output_path / f"job_{job_id}_hashes.txt"
        with open(hash_file, 'w') as f:
            f.write(job_data.get('hashes', ''))
        
        # Save job metadata
        metadata_file = output_path / f"job_{job_id}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(job_data, f, indent=2)
        
        click.echo(f"✅ Job downloaded:")
        click.echo(f"   Hashes: {hash_file}")
        click.echo(f"   Metadata: {metadata_file}")
        click.echo(f"   Hash count: {job_data.get('hash_count', 'unknown')}")
        click.echo(f"   Algorithm: {job_data.get('algorithm', 'unknown')}")
        click.echo(f"   Reward: {job_data.get('reward', 'unknown')}")
        
    except requests.RequestException as e:
        click.echo(f"❌ Network error: {e}")
    except Exception as e:
        click.echo(f"❌ Error: {e}")

@hashes_cli.command()
@click.option('--api-key', required=True, help='Hashes.com API key')
@click.option('--interval', default=300, help='Check interval in seconds')
@click.option('--exclude-md5', is_flag=True, help='Exclude MD5 hashes')
@click.option('--exclude-btc', is_flag=True, help='Exclude Bitcoin hashes')
@click.option('--min-reward', type=float, default=0.0, help='Minimum reward threshold')
def watch(api_key, interval, exclude_md5, exclude_btc, min_reward):
    """Watch for new jobs and auto-pull them"""
    click.echo(f"👁️  Watching for new jobs (checking every {interval}s)")
    click.echo("Press Ctrl+C to stop...")
    
    import time
    
    try:
        while True:
            click.echo(f"\n[{datetime.now().strftime('%H:%M:%S')}] Checking for new jobs...")
            
            # Use the pull command logic but with auto-accept
            ctx = click.get_current_context()
            ctx.invoke(pull, 
                      api_key=api_key,
                      exclude_md5=exclude_md5,
                      exclude_btc=exclude_btc,
                      min_reward=min_reward,
                      auto_accept=True,
                      max_jobs=5)  # Limit to 5 jobs per check
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        click.echo("\n👋 Stopped watching")

@hashes_cli.command()
@click.argument('job_id')
@click.option('--api-key', required=True, help='Hashes.com API key')
@click.option('--results-file', required=True, help='File containing cracked results')
def submit(job_id, api_key, results_file):
    """Submit cracked results for a job"""
    if not Path(results_file).exists():
        click.echo(f"❌ Results file not found: {results_file}")
        return
    
    click.echo(f"📤 Submitting results for job {job_id}...")
    
    try:
        # Read results file
        with open(results_file, 'r') as f:
            results = f.read().strip()
        
        # Parse results (assume hash:plain format)
        cracked_hashes = []
        for line in results.split('\n'):
            if ':' in line:
                hash_val, plain = line.split(':', 1)
                cracked_hashes.append({
                    'hash': hash_val.strip(),
                    'plain': plain.strip()
                })
        
        if not cracked_hashes:
            click.echo("❌ No valid results found in file")
            return
        
        click.echo(f"📊 Found {len(cracked_hashes)} cracked hashes")
        
        # Submit to hashes.com
        headers = {'Authorization': f'Bearer {api_key}'}
        payload = {
            'job_id': job_id,
            'results': cracked_hashes
        }
        
        response = requests.post('https://hashes.com/api/v1/jobs/submit', 
                               headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result_data = response.json()
            click.echo(f"✅ Results submitted successfully")
            click.echo(f"   Accepted: {result_data.get('accepted', 0)}")
            click.echo(f"   Rejected: {result_data.get('rejected', 0)}")
            click.echo(f"   Reward: {result_data.get('reward', 'TBD')}")
        else:
            click.echo(f"❌ Submission failed: {response.status_code}")
            
    except Exception as e:
        click.echo(f"❌ Error: {e}")

@hashes_cli.command()
@click.option('--api-key', required=True, help='Hashes.com API key')
def stats(api_key):
    """Show hashes.com account statistics"""
    click.echo("📊 Getting account statistics...")
    
    try:
        headers = {'Authorization': f'Bearer {api_key}'}
        response = requests.get('https://hashes.com/api/v1/account/stats', headers=headers, timeout=10)
        
        if response.status_code != 200:
            click.echo(f"❌ Failed to fetch stats: {response.status_code}")
            return
        
        stats_data = response.json()
        
        click.echo("📈 Account Statistics:")
        click.echo(f"   Total jobs completed: {stats_data.get('jobs_completed', 0)}")
        click.echo(f"   Total hashes cracked: {stats_data.get('hashes_cracked', 0)}")
        click.echo(f"   Success rate: {stats_data.get('success_rate', 0):.1%}")
        click.echo(f"   Total earnings: ${stats_data.get('total_earnings', 0):.2f}")
        click.echo(f"   Account level: {stats_data.get('level', 'Unknown')}")
        click.echo(f"   Reputation: {stats_data.get('reputation', 0)}")
        
    except requests.RequestException as e:
        click.echo(f"❌ Network error: {e}")

def filter_jobs(jobs, exclude_md5, exclude_btc, min_reward):
    """Filter jobs based on criteria"""
    filtered = []
    
    for job in jobs:
        # Check algorithm filters
        algorithm = job.get('algorithm', '').lower()
        
        if exclude_md5 and 'md5' in algorithm:
            continue
        
        if exclude_btc and ('bitcoin' in algorithm or 'btc' in algorithm):
            continue
        
        # Check reward threshold
        reward = float(job.get('reward', 0))
        if reward < min_reward:
            continue
        
        filtered.append(job)
    
    return filtered

def get_existing_jobs():
    """Get list of already pulled job IDs"""
    jobs_file = Path.home() / '.hashmancer' / 'pulled_jobs.txt'
    
    if not jobs_file.exists():
        return set()
    
    try:
        with open(jobs_file, 'r') as f:
            return set(line.strip() for line in f if line.strip())
    except:
        return set()

def remove_duplicates(jobs, existing_jobs):
    """Remove jobs that have already been pulled"""
    return [job for job in jobs if job.get('id') not in existing_jobs]

def save_pulled_jobs(jobs):
    """Save pulled job IDs to prevent duplicates"""
    jobs_file = Path.home() / '.hashmancer' / 'pulled_jobs.txt'
    jobs_file.parent.mkdir(exist_ok=True)
    
    try:
        with open(jobs_file, 'a') as f:
            for job in jobs:
                f.write(f"{job.get('id')}\n")
    except Exception as e:
        click.echo(f"⚠️  Warning: Could not save job IDs: {e}")

def display_jobs_table(jobs):
    """Display jobs in table format"""
    if not jobs:
        click.echo("No jobs found")
        return
    
    from tabulate import tabulate
    
    headers = ['ID', 'Algorithm', 'Hashes', 'Reward', 'Status', 'Created']
    rows = []
    
    for job in jobs:
        rows.append([
            job.get('id', 'unknown')[:12],
            job.get('algorithm', 'unknown'),
            job.get('hash_count', 'unknown'),
            f"${job.get('reward', 0):.2f}",
            job.get('status', 'unknown'),
            job.get('created_at', 'unknown')[:19]
        ])
    
    click.echo(tabulate(rows, headers=headers, tablefmt='grid'))

def pull_single_job(job, api_key):
    """Pull a single job and create local job entry"""
    try:
        # Download job data
        job_id = job.get('id')
        headers = {'Authorization': f'Bearer {api_key}'}
        
        response = requests.get(f'https://hashes.com/api/v1/jobs/{job_id}/download', 
                              headers=headers, timeout=30)
        
        if response.status_code != 200:
            return False
        
        # Create local job entry
        job_data = {
            'id': job_id,
            'source': 'hashes.com',
            'algorithm': job.get('algorithm'),
            'hash_count': job.get('hash_count'),
            'reward': job.get('reward'),
            'status': 'pending',
            'created_at': datetime.now().isoformat(),
            'hashes': response.text
        }
        
        # Save to local jobs directory
        jobs_dir = Path('./jobs')
        jobs_dir.mkdir(exist_ok=True)
        
        job_file = jobs_dir / f'hashes_com_{job_id}.json'
        with open(job_file, 'w') as f:
            json.dump(job_data, f, indent=2)
        
        return True
        
    except Exception:
        return False

if __name__ == "__main__":
    hashes_cli()