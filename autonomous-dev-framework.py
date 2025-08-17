#!/usr/bin/env python3
"""
Hashmancer Autonomous Development Framework
==========================================

This system enables Claude to continuously improve Hashmancer with minimal human intervention.
It provides intelligent log analysis, automated testing, performance monitoring, and development coordination.
"""

import asyncio
import json
import logging
import os
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import aiohttp
import redis
import psutil
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/autonomous_dev.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('AutonomousDev')

class AutonomousDevFramework:
    """
    Main framework for autonomous Hashmancer development
    """
    
    def __init__(self, config_path: str = "autonomous-dev-config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
        self.redis_client = None
        self.session = None
        self.development_state = {
            'current_cycle': 0,
            'last_analysis': None,
            'identified_issues': [],
            'planned_improvements': [],
            'completed_tasks': [],
            'performance_metrics': {},
            'system_health': {}
        }
        
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        default_config = {
            'testing': {
                'docker_compose_file': 'docker-compose.ultimate.yml',
                'test_duration_minutes': 30,
                'performance_monitoring_interval': 60,
                'log_analysis_interval': 300
            },
            'development': {
                'max_opus_calls_per_day': 20,
                'development_cycles_per_day': 4,
                'auto_commit_changes': True,
                'run_github_actions_locally': True
            },
            'monitoring': {
                'gpu_monitoring': True,
                'redis_monitoring': True,
                'container_monitoring': True,
                'performance_thresholds': {
                    'cpu_usage_max': 80,
                    'memory_usage_max': 85,
                    'gpu_temp_max': 80,
                    'redis_latency_max': 10
                }
            },
            'claude_opus': {
                'api_key': '',  # Set via environment variable
                'model': 'claude-3-5-sonnet-20241022',
                'max_tokens': 4000,
                'temperature': 0.7
            }
        }
        
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                # Merge with defaults
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                    elif isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            if subkey not in config[key]:
                                config[key][subkey] = subvalue
                return config
        else:
            # Create default config file
            with open(self.config_path, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False)
            logger.info(f"Created default config file: {self.config_path}")
            return default_config
    
    async def initialize(self):
        """Initialize the autonomous development framework"""
        logger.info("Initializing Autonomous Development Framework")
        
        # Initialize Redis connection
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            self.redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            
        # Initialize HTTP session
        self.session = aiohttp.ClientSession()
        
        # Load previous development state if it exists
        await self.load_development_state()
        
        logger.info("Framework initialized successfully")
    
    async def load_development_state(self):
        """Load previous development state from file"""
        state_file = Path('/tmp/development_state.json')
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    self.development_state = json.load(f)
                logger.info("Loaded previous development state")
            except Exception as e:
                logger.warning(f"Failed to load development state: {e}")
    
    async def save_development_state(self):
        """Save current development state to file"""
        state_file = Path('/tmp/development_state.json')
        try:
            with open(state_file, 'w') as f:
                json.dump(self.development_state, f, indent=2, default=str)
            logger.info("Saved development state")
        except Exception as e:
            logger.error(f"Failed to save development state: {e}")
    
    async def run_development_cycle(self):
        """Run a complete development cycle"""
        cycle_start = datetime.now()
        cycle_num = self.development_state['current_cycle'] + 1
        
        logger.info(f"Starting development cycle {cycle_num}")
        
        try:
            # 1. System Health Check
            health_status = await self.perform_health_check()
            
            # 2. Deploy/Update Environment
            if not health_status['containers_running']:
                await self.deploy_environment()
            
            # 3. Run Performance Tests
            performance_results = await self.run_performance_tests()
            
            # 4. Analyze Logs and System State
            analysis_results = await self.analyze_system_state()
            
            # 5. Identify Issues and Improvements
            issues = await self.identify_issues(analysis_results, performance_results)
            
            # 6. Plan and Execute Improvements
            if issues:
                improvements = await self.plan_improvements(issues)
                await self.execute_improvements(improvements)
            
            # 7. Validate Changes
            validation_results = await self.validate_changes()
            
            # 8. Update Development State
            self.development_state.update({
                'current_cycle': cycle_num,
                'last_analysis': datetime.now().isoformat(),
                'cycle_duration': (datetime.now() - cycle_start).total_seconds(),
                'health_status': health_status,
                'performance_results': performance_results,
                'analysis_results': analysis_results,
                'validation_results': validation_results
            })
            
            await self.save_development_state()
            
            logger.info(f"Completed development cycle {cycle_num}")
            
        except Exception as e:
            logger.error(f"Error in development cycle {cycle_num}: {e}")
            
    async def perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive system health check"""
        logger.info("Performing system health check")
        
        health_status = {
            'timestamp': datetime.now().isoformat(),
            'containers_running': False,
            'redis_healthy': False,
            'gpus_available': False,
            'disk_space_ok': False,
            'memory_ok': False,
            'details': {}
        }
        
        try:
            # Check Docker containers
            result = subprocess.run(['docker', 'ps', '--format', 'json'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                containers = [json.loads(line) for line in result.stdout.strip().split('\n') if line]
                hashmancer_containers = [c for c in containers if 'hashmancer' in c.get('Names', '')]
                health_status['containers_running'] = len(hashmancer_containers) > 0
                health_status['details']['containers'] = hashmancer_containers
                
            # Check Redis
            if self.redis_client:
                try:
                    self.redis_client.ping()
                    health_status['redis_healthy'] = True
                except:
                    health_status['redis_healthy'] = False
            
            # Check GPUs
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,temperature.gpu', 
                                       '--format=csv,noheader,nounits'], capture_output=True, text=True)
                if result.returncode == 0:
                    gpu_info = []
                    for line in result.stdout.strip().split('\n'):
                        if line:
                            name, memory, temp = line.split(', ')
                            gpu_info.append({
                                'name': name.strip(),
                                'memory_total': int(memory),
                                'temperature': int(temp)
                            })
                    health_status['gpus_available'] = len(gpu_info) > 0
                    health_status['details']['gpus'] = gpu_info
            except Exception as e:
                logger.warning(f"GPU check failed: {e}")
            
            # Check system resources
            disk_usage = psutil.disk_usage('/')
            memory = psutil.virtual_memory()
            
            health_status['disk_space_ok'] = disk_usage.percent < 90
            health_status['memory_ok'] = memory.percent < 85
            health_status['details']['disk_usage_percent'] = disk_usage.percent
            health_status['details']['memory_usage_percent'] = memory.percent
            
        except Exception as e:
            logger.error(f"Health check error: {e}")
            
        return health_status
    
    async def deploy_environment(self):
        """Deploy or redeploy the Hashmancer environment"""
        logger.info("Deploying Hashmancer environment")
        
        try:
            # Stop existing containers
            subprocess.run(['docker-compose', '-f', self.config['testing']['docker_compose_file'], 
                          'down', '-v'], check=False)
            
            # Start new deployment
            result = subprocess.run(['./deploy-hashmancer.sh', 'quick'], 
                                  capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info("Environment deployed successfully")
                # Wait for services to be ready
                await asyncio.sleep(30)
            else:
                logger.error(f"Deployment failed: {result.stderr}")
                
        except Exception as e:
            logger.error(f"Deployment error: {e}")
    
    async def run_performance_tests(self) -> Dict[str, Any]:
        """Run comprehensive performance tests"""
        logger.info("Running performance tests")
        
        performance_results = {
            'timestamp': datetime.now().isoformat(),
            'redis_performance': {},
            'gpu_performance': {},
            'container_performance': {},
            'integration_test_results': {}
        }
        
        try:
            # Redis performance test
            if self.redis_client:
                start_time = time.time()
                for i in range(1000):
                    self.redis_client.set(f'perf_test_{i}', f'value_{i}')
                    self.redis_client.get(f'perf_test_{i}')
                end_time = time.time()
                
                performance_results['redis_performance'] = {
                    'operations_per_second': 2000 / (end_time - start_time),
                    'latency_ms': (end_time - start_time) * 1000 / 2000
                }
                
                # Cleanup test keys
                for i in range(1000):
                    self.redis_client.delete(f'perf_test_{i}')
            
            # GPU performance test
            try:
                result = subprocess.run(['python3', 'docker/worker/gpu-utils.py', 'benchmark'], 
                                      capture_output=True, text=True, timeout=120)
                if result.returncode == 0:
                    # Parse GPU benchmark results
                    performance_results['gpu_performance'] = {
                        'benchmark_completed': True,
                        'output': result.stdout
                    }
            except Exception as e:
                logger.warning(f"GPU benchmark failed: {e}")
            
            # Container performance test
            result = subprocess.run(['docker', 'stats', '--no-stream', '--format', 
                                   'table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                performance_results['container_performance'] = {
                    'stats_output': result.stdout
                }
                
        except Exception as e:
            logger.error(f"Performance test error: {e}")
            
        return performance_results
    
    async def analyze_system_state(self) -> Dict[str, Any]:
        """Analyze current system state and logs"""
        logger.info("Analyzing system state and logs")
        
        analysis_results = {
            'timestamp': datetime.now().isoformat(),
            'log_analysis': {},
            'error_patterns': [],
            'performance_issues': [],
            'optimization_opportunities': []
        }
        
        try:
            # Analyze Docker logs
            containers = ['hashmancer-server', 'hashmancer-worker-gpu', 'hashmancer-redis']
            
            for container in containers:
                try:
                    result = subprocess.run(['docker', 'logs', '--tail', '1000', container], 
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        logs = result.stdout
                        
                        # Count error patterns
                        error_count = logs.lower().count('error')
                        warning_count = logs.lower().count('warning')
                        exception_count = logs.lower().count('exception')
                        
                        analysis_results['log_analysis'][container] = {
                            'error_count': error_count,
                            'warning_count': warning_count,
                            'exception_count': exception_count,
                            'total_lines': len(logs.split('\n'))
                        }
                        
                        # Extract specific error patterns
                        if error_count > 0:
                            error_lines = [line for line in logs.split('\n') if 'error' in line.lower()]
                            analysis_results['error_patterns'].extend(error_lines[-10:])  # Last 10 errors
                            
                except Exception as e:
                    logger.warning(f"Failed to analyze logs for {container}: {e}")
            
            # Analyze Redis performance
            if self.redis_client:
                try:
                    info = self.redis_client.info()
                    analysis_results['redis_info'] = {
                        'connected_clients': info.get('connected_clients', 0),
                        'used_memory': info.get('used_memory', 0),
                        'keyspace_hits': info.get('keyspace_hits', 0),
                        'keyspace_misses': info.get('keyspace_misses', 0)
                    }
                except Exception as e:
                    logger.warning(f"Redis info analysis failed: {e}")
                    
        except Exception as e:
            logger.error(f"System analysis error: {e}")
            
        return analysis_results
    
    async def identify_issues(self, analysis_results: Dict, performance_results: Dict) -> List[Dict]:
        """Identify issues based on analysis and performance results"""
        logger.info("Identifying system issues")
        
        issues = []
        
        # Check for high error rates
        for container, log_data in analysis_results.get('log_analysis', {}).items():
            error_rate = log_data['error_count'] / max(log_data['total_lines'], 1)
            if error_rate > 0.05:  # More than 5% error rate
                issues.append({
                    'type': 'high_error_rate',
                    'severity': 'high',
                    'container': container,
                    'error_rate': error_rate,
                    'description': f"High error rate ({error_rate:.2%}) in {container}"
                })
        
        # Check Redis performance
        redis_perf = performance_results.get('redis_performance', {})
        if redis_perf.get('latency_ms', 0) > 10:
            issues.append({
                'type': 'high_redis_latency',
                'severity': 'medium',
                'latency': redis_perf['latency_ms'],
                'description': f"High Redis latency: {redis_perf['latency_ms']:.2f}ms"
            })
        
        # Check for GPU utilization issues
        if not performance_results.get('gpu_performance', {}).get('benchmark_completed', False):
            issues.append({
                'type': 'gpu_performance_issue',
                'severity': 'medium',
                'description': "GPU benchmark failed or not completed"
            })
        
        logger.info(f"Identified {len(issues)} issues")
        return issues
    
    async def plan_improvements(self, issues: List[Dict]) -> List[Dict]:
        """Plan improvements based on identified issues"""
        logger.info("Planning improvements")
        
        improvements = []
        
        for issue in issues:
            if issue['type'] == 'high_error_rate':
                improvements.append({
                    'type': 'error_investigation',
                    'priority': 'high',
                    'target_container': issue['container'],
                    'action': 'investigate_and_fix_errors',
                    'description': f"Investigate and fix errors in {issue['container']}"
                })
            
            elif issue['type'] == 'high_redis_latency':
                improvements.append({
                    'type': 'redis_optimization',
                    'priority': 'medium',
                    'action': 'optimize_redis_configuration',
                    'description': "Optimize Redis configuration for better performance"
                })
            
            elif issue['type'] == 'gpu_performance_issue':
                improvements.append({
                    'type': 'gpu_optimization',
                    'priority': 'medium',
                    'action': 'investigate_gpu_setup',
                    'description': "Investigate GPU setup and optimization"
                })
        
        logger.info(f"Planned {len(improvements)} improvements")
        return improvements
    
    async def execute_improvements(self, improvements: List[Dict]):
        """Execute planned improvements"""
        logger.info("Executing improvements")
        
        for improvement in improvements:
            try:
                if improvement['type'] == 'error_investigation':
                    await self.investigate_container_errors(improvement['target_container'])
                elif improvement['type'] == 'redis_optimization':
                    await self.optimize_redis_configuration()
                elif improvement['type'] == 'gpu_optimization':
                    await self.investigate_gpu_setup()
                    
                # Mark improvement as completed
                self.development_state['completed_tasks'].append({
                    'improvement': improvement,
                    'completed_at': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Failed to execute improvement {improvement['type']}: {e}")
    
    async def investigate_container_errors(self, container: str):
        """Investigate errors in a specific container"""
        logger.info(f"Investigating errors in {container}")
        
        # Get detailed logs
        result = subprocess.run(['docker', 'logs', '--tail', '200', container], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            logs = result.stdout
            
            # Save logs for analysis
            log_file = f"/tmp/{container}_error_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(log_file, 'w') as f:
                f.write(logs)
            
            logger.info(f"Saved error logs to {log_file}")
            
            # TODO: Call Claude Opus for complex error analysis if needed
            # This would be one of the strategic Opus API calls
    
    async def optimize_redis_configuration(self):
        """Optimize Redis configuration"""
        logger.info("Optimizing Redis configuration")
        
        # Run Redis optimization tools
        try:
            subprocess.run(['python3', 'redis_tool.py', 'optimize'], check=True)
            logger.info("Redis optimization completed")
        except Exception as e:
            logger.error(f"Redis optimization failed: {e}")
    
    async def investigate_gpu_setup(self):
        """Investigate GPU setup and optimization"""
        logger.info("Investigating GPU setup")
        
        try:
            # Run GPU diagnostics
            result = subprocess.run(['python3', 'docker/worker/gpu-utils.py', 'info'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                gpu_info = result.stdout
                logger.info(f"GPU info: {gpu_info}")
            
            # Test GPU worker container
            result = subprocess.run(['docker', 'exec', 'hashmancer-worker-gpu', 'nvidia-smi'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                logger.warning("GPU not accessible in worker container")
                
        except Exception as e:
            logger.error(f"GPU investigation failed: {e}")
    
    async def validate_changes(self) -> Dict[str, Any]:
        """Validate that changes improved the system"""
        logger.info("Validating changes")
        
        # Run tests to validate improvements
        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'tests_passed': 0,
            'tests_failed': 0,
            'performance_improved': False,
            'errors_reduced': False
        }
        
        try:
            # Run deployment test
            result = subprocess.run(['./test-deployment.sh'], capture_output=True, text=True, timeout=600)
            if result.returncode == 0:
                validation_results['tests_passed'] += 1
                validation_results['deployment_test'] = 'passed'
            else:
                validation_results['tests_failed'] += 1
                validation_results['deployment_test'] = 'failed'
            
            # Run GitHub Actions locally if configured
            if self.config['development']['run_github_actions_locally']:
                result = subprocess.run(['./test-workflows-local.sh', 'python'], 
                                      capture_output=True, text=True, timeout=600)
                if result.returncode == 0:
                    validation_results['tests_passed'] += 1
                    validation_results['github_actions_test'] = 'passed'
                else:
                    validation_results['tests_failed'] += 1
                    validation_results['github_actions_test'] = 'failed'
                    
        except Exception as e:
            logger.error(f"Validation error: {e}")
            validation_results['tests_failed'] += 1
            
        return validation_results
    
    async def call_claude_opus_for_complex_analysis(self, context: Dict) -> Optional[str]:
        """Call Claude Opus API for complex analysis tasks"""
        
        # Check daily Opus call limit
        today = datetime.now().date()
        opus_calls_today = len([task for task in self.development_state.get('completed_tasks', [])
                               if task.get('used_opus', False) and 
                               datetime.fromisoformat(task['completed_at']).date() == today])
        
        if opus_calls_today >= self.config['claude_opus']['max_opus_calls_per_day']:
            logger.warning("Daily Opus API call limit reached")
            return None
        
        api_key = os.getenv('ANTHROPIC_API_KEY') or self.config['claude_opus']['api_key']
        if not api_key:
            logger.warning("No Anthropic API key configured")
            return None
        
        try:
            prompt = self.create_opus_analysis_prompt(context)
            
            headers = {
                'Content-Type': 'application/json',
                'x-api-key': api_key,
                'anthropic-version': '2023-06-01'
            }
            
            data = {
                'model': self.config['claude_opus']['model'],
                'max_tokens': self.config['claude_opus']['max_tokens'],
                'temperature': self.config['claude_opus']['temperature'],
                'messages': [{'role': 'user', 'content': prompt}]
            }
            
            async with self.session.post('https://api.anthropic.com/v1/messages', 
                                       headers=headers, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    return result['content'][0]['text']
                else:
                    logger.error(f"Opus API call failed: {response.status}")
                    
        except Exception as e:
            logger.error(f"Opus API call error: {e}")
            
        return None
    
    def create_opus_analysis_prompt(self, context: Dict) -> str:
        """Create a focused prompt for Opus analysis"""
        
        prompt = f"""
        I'm Claude Code autonomously improving Hashmancer. I need your expert analysis for complex issues.
        
        Current Context:
        - System: Dual RTX 2080 Ti setup with Docker
        - Issues identified: {json.dumps(context.get('issues', []), indent=2)}
        - Performance data: {json.dumps(context.get('performance', {}), indent=2)}
        - Recent errors: {context.get('recent_errors', [])}
        
        Please provide:
        1. Root cause analysis of the most critical issues
        2. Specific implementation recommendations
        3. Code changes needed (if any)
        4. Priority order for fixes
        
        Focus on actionable insights for GPU hash cracking optimization and system reliability.
        """
        
        return prompt
    
    async def run_continuous_development(self):
        """Run continuous development cycles"""
        logger.info("Starting continuous development mode")
        
        cycles_per_day = self.config['development']['development_cycles_per_day']
        cycle_interval = 24 * 3600 / cycles_per_day  # seconds between cycles
        
        while True:
            try:
                await self.run_development_cycle()
                
                # Wait until next cycle
                logger.info(f"Waiting {cycle_interval/3600:.1f} hours until next cycle")
                await asyncio.sleep(cycle_interval)
                
            except KeyboardInterrupt:
                logger.info("Continuous development stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in continuous development: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour before retrying
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
        
        await self.save_development_state()
        logger.info("Framework cleanup completed")

async def main():
    """Main entry point"""
    framework = AutonomousDevFramework()
    
    try:
        await framework.initialize()
        await framework.run_continuous_development()
    finally:
        await framework.cleanup()

if __name__ == "__main__":
    asyncio.run(main())