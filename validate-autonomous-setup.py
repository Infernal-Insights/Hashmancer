#!/usr/bin/env python3
"""
Hashmancer Autonomous Development Setup Validator
================================================

Validates that all components are properly configured for autonomous development.
"""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

def check_gpu_setup():
    """Check dual RTX 2080 Ti setup"""
    print("🎮 Checking GPU Setup...")
    
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total', '--format=csv,noheader'], 
                              capture_output=True, text=True)
        
        if result.returncode != 0:
            print("❌ NVIDIA drivers not available")
            return False
        
        gpus = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = [p.strip() for p in line.split(',')]
                gpus.append({
                    'index': int(parts[0]),
                    'name': parts[1],
                    'memory': parts[2]
                })
        
        if len(gpus) != 2:
            print(f"⚠️  Expected 2 GPUs, found {len(gpus)}")
            return False
        
        rtx_2080ti_count = sum(1 for gpu in gpus if '2080' in gpu['name'])
        if rtx_2080ti_count == 2:
            print("✅ Dual RTX 2080 Ti setup detected")
            for gpu in gpus:
                print(f"   GPU {gpu['index']}: {gpu['name']} ({gpu['memory']})")
            return True
        else:
            print(f"⚠️  Found {rtx_2080ti_count} RTX 2080 Ti GPUs")
            return False
            
    except Exception as e:
        print(f"❌ GPU check failed: {e}")
        return False

def check_docker_setup():
    """Check Docker and compose setup"""
    print("\n🐳 Checking Docker Setup...")
    
    try:
        # Check Docker
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Docker not available")
            return False
        
        # Check Docker daemon
        result = subprocess.run(['docker', 'ps'], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Docker daemon not running")
            return False
        
        # Check Docker Compose
        result = subprocess.run(['docker-compose', '--version'], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Docker Compose not available")
            return False
        
        # Check compose file
        if not Path('docker-compose.ultimate.yml').exists():
            print("❌ docker-compose.ultimate.yml not found")
            return False
        
        print("✅ Docker and Docker Compose available")
        print("✅ docker-compose.ultimate.yml found")
        return True
        
    except Exception as e:
        print(f"❌ Docker check failed: {e}")
        return False

def check_python_dependencies():
    """Check Python dependencies"""
    print("\n🐍 Checking Python Dependencies...")
    
    required_packages = [
        'asyncio', 'aiohttp', 'redis', 'psutil', 'yaml'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip3 install -r requirements-autonomous-dev.txt")
        return False
    
    return True

def check_configuration_files():
    """Check configuration files"""
    print("\n📋 Checking Configuration Files...")
    
    required_files = [
        'autonomous-dev-framework.py',
        'gpu-optimization-system.py',
        'autonomous-dev-config.yaml',
        'start-autonomous-development.sh'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - missing")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️  Missing files: {', '.join(missing_files)}")
        return False
    
    return True

def check_api_keys():
    """Check API key configuration"""
    print("\n🔑 Checking API Keys...")
    
    api_keys = {
        'ANTHROPIC_API_KEY': 'Claude Opus API',
        'HASHES_COM_API_KEY': 'hashes.com API',
        'VAST_AI_API_KEY': 'Vast.ai API'
    }
    
    configured_keys = 0
    
    for env_var, description in api_keys.items():
        if os.getenv(env_var):
            print(f"✅ {description} configured")
            configured_keys += 1
        else:
            print(f"⚠️  {description} not configured")
    
    if configured_keys == 0:
        print("❌ No API keys configured - limited functionality")
        return False
    elif configured_keys < len(api_keys):
        print(f"⚠️  {configured_keys}/{len(api_keys)} API keys configured")
        return True
    else:
        print("✅ All API keys configured")
        return True

def check_redis_connection():
    """Check Redis connection"""
    print("\n📊 Checking Redis Connection...")
    
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        r.ping()
        print("✅ Redis connection successful")
        
        # Test basic operations
        r.set('test_key', 'test_value')
        value = r.get('test_key')
        r.delete('test_key')
        
        if value == 'test_value':
            print("✅ Redis operations working")
            return True
        else:
            print("❌ Redis operations failed")
            return False
            
    except Exception as e:
        print(f"⚠️  Redis not available: {e}")
        print("   Redis will be started with Docker")
        return True  # Not critical, can start with Docker

def check_github_actions_setup():
    """Check GitHub Actions local testing setup"""
    print("\n🎭 Checking GitHub Actions Setup...")
    
    if Path('setup-act.sh').exists():
        print("✅ act setup script available")
    else:
        print("⚠️  act setup script missing")
    
    if Path('.github/workflows').exists():
        workflow_files = list(Path('.github/workflows').glob('*.yml'))
        print(f"✅ {len(workflow_files)} workflow files found")
        for workflow in workflow_files:
            print(f"   - {workflow.name}")
    else:
        print("❌ No GitHub workflows found")
        return False
    
    return True

async def test_autonomous_framework():
    """Test the autonomous development framework"""
    print("\n🤖 Testing Autonomous Framework...")
    
    try:
        # Import the framework
        sys.path.append('.')
        from autonomous_dev_framework import AutonomousDevFramework
        
        # Initialize framework
        framework = AutonomousDevFramework()
        await framework.initialize()
        
        print("✅ Framework initialization successful")
        
        # Test health check
        health_status = await framework.perform_health_check()
        print(f"✅ Health check completed: {health_status['containers_running']}")
        
        await framework.cleanup()
        print("✅ Framework cleanup successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Framework test failed: {e}")
        return False

def generate_setup_report(checks_results):
    """Generate setup validation report"""
    print("\n" + "="*50)
    print("📊 AUTONOMOUS DEVELOPMENT SETUP REPORT")
    print("="*50)
    
    passed_checks = sum(1 for result in checks_results.values() if result)
    total_checks = len(checks_results)
    
    print(f"\nOverall Status: {passed_checks}/{total_checks} checks passed")
    
    if passed_checks == total_checks:
        print("🎉 ✅ READY FOR AUTONOMOUS DEVELOPMENT!")
        print("\nNext steps:")
        print("1. Run: ./start-autonomous-development.sh start")
        print("2. Monitor: ./start-autonomous-development.sh status")
        print("3. View logs: tail -f /tmp/autonomous_dev_main.log")
        
    elif passed_checks >= total_checks - 2:
        print("⚠️  🟡 MOSTLY READY - Some optional components missing")
        print("\nYou can start autonomous development, but some features may be limited.")
        print("Run: ./start-autonomous-development.sh start")
        
    else:
        print("❌ 🔴 NOT READY - Critical components missing")
        print("\nPlease resolve the following issues:")
        
        for check_name, result in checks_results.items():
            if not result:
                print(f"  - Fix {check_name}")
    
    print("\n" + "="*50)

async def main():
    """Main validation function"""
    print("🔍 Hashmancer Autonomous Development Setup Validation")
    print("="*55)
    
    checks = {
        'GPU Setup': check_gpu_setup(),
        'Docker Setup': check_docker_setup(),
        'Python Dependencies': check_python_dependencies(),
        'Configuration Files': check_configuration_files(),
        'API Keys': check_api_keys(),
        'Redis Connection': check_redis_connection(),
        'GitHub Actions': check_github_actions_setup(),
        'Autonomous Framework': await test_autonomous_framework()
    }
    
    generate_setup_report(checks)

if __name__ == "__main__":
    asyncio.run(main())