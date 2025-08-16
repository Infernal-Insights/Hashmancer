#!/usr/bin/env python3
"""
Dual RTX 2080 Ti GPU Optimization System
========================================

Specialized system for optimizing Hashmancer performance on dual RTX 2080 Ti GPUs.
Handles load balancing, thermal management, memory optimization, and performance monitoring.
"""

import asyncio
import json
import logging
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import psutil
import yaml

logger = logging.getLogger('GPUOptimization')

@dataclass
class GPUStatus:
    """GPU status information"""
    gpu_id: int
    name: str
    temperature: int
    memory_used: int
    memory_total: int
    utilization: int
    power_draw: int
    fan_speed: int

@dataclass
class HashingJob:
    """Hashing job information"""
    job_id: str
    hash_type: str
    estimated_hashes: int
    priority: str
    assigned_gpu: Optional[int] = None

class DualGPUOptimizer:
    """
    Optimizer for dual RTX 2080 Ti setup
    """
    
    def __init__(self, config_path: str = "autonomous-dev-config.yaml"):
        self.config = self.load_config(config_path)
        self.gpu_configs = self.config['gpu_configuration']['dual_rtx_2080ti']
        self.monitoring_data = {
            'gpu_0': {'history': [], 'current': None},
            'gpu_1': {'history': [], 'current': None}
        }
        self.load_balancer = LoadBalancer(self.gpu_configs)
        self.thermal_manager = ThermalManager(self.gpu_configs)
        
    def load_config(self, config_path: str) -> Dict:
        """Load configuration"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    async def get_gpu_status(self) -> List[GPUStatus]:
        """Get current status of both GPUs"""
        try:
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=index,name,temperature.gpu,memory.used,memory.total,utilization.gpu,power.draw,fan.speed',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Failed to get GPU status: {result.stderr}")
                return []
            
            gpu_statuses = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 8:
                        gpu_status = GPUStatus(
                            gpu_id=int(parts[0]),
                            name=parts[1],
                            temperature=int(parts[2]) if parts[2] != '[Not Supported]' else 0,
                            memory_used=int(parts[3]) if parts[3] != '[Not Supported]' else 0,
                            memory_total=int(parts[4]) if parts[4] != '[Not Supported]' else 0,
                            utilization=int(parts[5]) if parts[5] != '[Not Supported]' else 0,
                            power_draw=int(float(parts[6])) if parts[6] != '[Not Supported]' else 0,
                            fan_speed=int(parts[7]) if parts[7] != '[Not Supported]' else 0
                        )
                        gpu_statuses.append(gpu_status)
            
            return gpu_statuses
            
        except Exception as e:
            logger.error(f"Error getting GPU status: {e}")
            return []
    
    async def optimize_gpu_settings(self):
        """Optimize GPU settings for dual 2080 Ti"""
        logger.info("Optimizing GPU settings for dual RTX 2080 Ti")
        
        try:
            # Set power limits
            target_power = self.gpu_configs['power_management']['target_power_limit']
            for gpu_id in [0, 1]:
                subprocess.run([
                    'nvidia-smi', '-i', str(gpu_id), '-pl', str(target_power)
                ], check=False)  # Don't fail if not supported
            
            # Set memory and GPU clocks for optimal hash cracking
            # These are conservative settings for stability
            for gpu_id in [0, 1]:
                # Memory clock offset (conservative)
                subprocess.run([
                    'nvidia-settings', '-a', f'[gpu:{gpu_id}]/GPUMemoryTransferRateOffset[3]=500'
                ], check=False)
                
                # GPU clock offset (conservative)
                subprocess.run([
                    'nvidia-settings', '-a', f'[gpu:{gpu_id}]/GPUGraphicsClockOffset[3]=50'
                ], check=False)
            
            logger.info("GPU optimization completed")
            
        except Exception as e:
            logger.error(f"GPU optimization failed: {e}")
    
    async def monitor_thermal_performance(self) -> Dict:
        """Monitor thermal performance and adjust as needed"""
        gpu_statuses = await self.get_gpu_status()
        thermal_report = {
            'timestamp': time.time(),
            'gpus': {},
            'actions_taken': []
        }
        
        for gpu in gpu_statuses:
            gpu_key = f'gpu_{gpu.gpu_id}'
            
            # Update monitoring data
            self.monitoring_data[gpu_key]['current'] = gpu
            self.monitoring_data[gpu_key]['history'].append({
                'timestamp': time.time(),
                'temperature': gpu.temperature,
                'utilization': gpu.utilization,
                'memory_usage': gpu.memory_used / gpu.memory_total * 100,
                'power_draw': gpu.power_draw
            })
            
            # Keep only last 100 readings
            if len(self.monitoring_data[gpu_key]['history']) > 100:
                self.monitoring_data[gpu_key]['history'] = self.monitoring_data[gpu_key]['history'][-100:]
            
            thermal_report['gpus'][gpu_key] = {
                'temperature': gpu.temperature,
                'utilization': gpu.utilization,
                'memory_usage_percent': gpu.memory_used / gpu.memory_total * 100,
                'power_draw': gpu.power_draw,
                'status': 'normal'
            }
            
            # Check thermal thresholds
            temp_warning = self.config['monitoring']['alerts']['gpu_temp_warning']
            temp_critical = self.config['monitoring']['alerts']['gpu_temp_critical']
            
            if gpu.temperature >= temp_critical:
                thermal_report['gpus'][gpu_key]['status'] = 'critical'
                action = await self.thermal_manager.handle_critical_temperature(gpu)
                thermal_report['actions_taken'].append(action)
                
            elif gpu.temperature >= temp_warning:
                thermal_report['gpus'][gpu_key]['status'] = 'warning'
                action = await self.thermal_manager.handle_high_temperature(gpu)
                thermal_report['actions_taken'].append(action)
        
        return thermal_report
    
    async def optimize_job_distribution(self, jobs: List[HashingJob]) -> Dict[int, List[HashingJob]]:
        """Optimize job distribution across GPUs"""
        return await self.load_balancer.distribute_jobs(jobs, await self.get_gpu_status())
    
    async def generate_performance_report(self) -> Dict:
        """Generate comprehensive performance report"""
        gpu_statuses = await self.get_gpu_status()
        thermal_report = await self.monitor_thermal_performance()
        
        report = {
            'timestamp': time.time(),
            'gpu_summary': {
                'total_gpus': len(gpu_statuses),
                'average_temperature': sum(gpu.temperature for gpu in gpu_statuses) / len(gpu_statuses) if gpu_statuses else 0,
                'total_memory_used': sum(gpu.memory_used for gpu in gpu_statuses),
                'total_memory_available': sum(gpu.memory_total for gpu in gpu_statuses),
                'average_utilization': sum(gpu.utilization for gpu in gpu_statuses) / len(gpu_statuses) if gpu_statuses else 0,
                'total_power_draw': sum(gpu.power_draw for gpu in gpu_statuses)
            },
            'individual_gpus': {},
            'optimization_recommendations': [],
            'thermal_status': thermal_report
        }
        
        for gpu in gpu_statuses:
            gpu_key = f'gpu_{gpu.gpu_id}'
            
            # Calculate efficiency metrics
            memory_efficiency = (gpu.memory_used / gpu.memory_total) * 100 if gpu.memory_total > 0 else 0
            thermal_efficiency = max(0, 100 - (gpu.temperature - 30) * 2)  # Efficiency decreases with temperature
            
            report['individual_gpus'][gpu_key] = {
                'name': gpu.name,
                'temperature': gpu.temperature,
                'memory_usage_percent': memory_efficiency,
                'utilization': gpu.utilization,
                'power_draw': gpu.power_draw,
                'thermal_efficiency': thermal_efficiency,
                'status': 'optimal' if gpu.temperature < 75 and gpu.utilization > 80 else 'suboptimal'
            }
            
            # Generate recommendations
            if gpu.temperature > 80:
                report['optimization_recommendations'].append(
                    f"GPU {gpu.gpu_id}: Reduce workload or improve cooling - temperature {gpu.temperature}°C"
                )
            
            if gpu.utilization < 50:
                report['optimization_recommendations'].append(
                    f"GPU {gpu.gpu_id}: Low utilization ({gpu.utilization}%) - consider increasing workload"
                )
            
            if memory_efficiency < 70:
                report['optimization_recommendations'].append(
                    f"GPU {gpu.gpu_id}: Low memory utilization ({memory_efficiency:.1f}%) - optimize batch sizes"
                )
        
        return report

class LoadBalancer:
    """
    Intelligent load balancer for dual GPU setup
    """
    
    def __init__(self, gpu_configs: Dict):
        self.gpu_configs = gpu_configs
        self.job_history = []
    
    async def distribute_jobs(self, jobs: List[HashingJob], gpu_statuses: List[GPUStatus]) -> Dict[int, List[HashingJob]]:
        """Distribute jobs optimally across GPUs"""
        if len(gpu_statuses) != 2:
            logger.warning(f"Expected 2 GPUs, found {len(gpu_statuses)}")
            return {0: jobs}  # Fallback to single GPU
        
        # Calculate GPU scores based on current state
        gpu_scores = {}
        for gpu in gpu_statuses:
            # Score based on utilization (lower is better), temperature (lower is better), memory availability
            utilization_score = 100 - gpu.utilization
            thermal_score = max(0, 100 - gpu.temperature)
            memory_score = ((gpu.memory_total - gpu.memory_used) / gpu.memory_total) * 100
            
            gpu_scores[gpu.gpu_id] = (utilization_score + thermal_score + memory_score) / 3
        
        # Sort jobs by priority and estimated workload
        priority_order = {'high': 3, 'medium': 2, 'low': 1}
        jobs.sort(key=lambda j: (priority_order.get(j.priority, 0), j.estimated_hashes), reverse=True)
        
        # Distribute jobs
        gpu_assignments = {0: [], 1: []}
        gpu_workloads = {0: 0, 1: 0}
        
        for job in jobs:
            # Choose GPU with better score and lower current workload
            if gpu_scores[0] + (100 - gpu_workloads[0]) > gpu_scores[1] + (100 - gpu_workloads[1]):
                assigned_gpu = 0
            else:
                assigned_gpu = 1
            
            job.assigned_gpu = assigned_gpu
            gpu_assignments[assigned_gpu].append(job)
            gpu_workloads[assigned_gpu] += job.estimated_hashes / 1000000  # Normalize workload
        
        logger.info(f"Distributed {len(jobs)} jobs: GPU0={len(gpu_assignments[0])}, GPU1={len(gpu_assignments[1])}")
        return gpu_assignments

class ThermalManager:
    """
    Thermal management for RTX 2080 Ti GPUs
    """
    
    def __init__(self, gpu_configs: Dict):
        self.gpu_configs = gpu_configs
        self.thermal_actions_history = []
    
    async def handle_critical_temperature(self, gpu: GPUStatus) -> str:
        """Handle critical temperature situation"""
        logger.warning(f"Critical temperature on GPU {gpu.gpu_id}: {gpu.temperature}°C")
        
        action = f"Critical temp GPU {gpu.gpu_id}: "
        
        try:
            # Reduce power limit
            new_power_limit = max(150, gpu.power_draw - 50)  # Reduce by 50W, minimum 150W
            subprocess.run([
                'nvidia-smi', '-i', str(gpu.gpu_id), '-pl', str(new_power_limit)
            ], check=False)
            action += f"Reduced power limit to {new_power_limit}W"
            
            # Set more aggressive fan curve if supported
            subprocess.run([
                'nvidia-settings', '-a', f'[gpu:{gpu.gpu_id}]/GPUFanControlState=1',
                '-a', f'[fan:{gpu.gpu_id}]/GPUTargetFanSpeed=90'
            ], check=False)
            action += ", Set fans to 90%"
            
        except Exception as e:
            action += f"Failed to apply thermal management: {e}"
        
        self.thermal_actions_history.append({
            'timestamp': time.time(),
            'gpu_id': gpu.gpu_id,
            'temperature': gpu.temperature,
            'action': action,
            'severity': 'critical'
        })
        
        return action
    
    async def handle_high_temperature(self, gpu: GPUStatus) -> str:
        """Handle high temperature situation"""
        logger.info(f"High temperature on GPU {gpu.gpu_id}: {gpu.temperature}°C")
        
        action = f"High temp GPU {gpu.gpu_id}: "
        
        try:
            # Moderate power reduction
            new_power_limit = max(200, gpu.power_draw - 25)  # Reduce by 25W, minimum 200W
            subprocess.run([
                'nvidia-smi', '-i', str(gpu.gpu_id), '-pl', str(new_power_limit)
            ], check=False)
            action += f"Reduced power limit to {new_power_limit}W"
            
            # Increase fan speed moderately
            subprocess.run([
                'nvidia-settings', '-a', f'[gpu:{gpu.gpu_id}]/GPUFanControlState=1',
                '-a', f'[fan:{gpu.gpu_id}]/GPUTargetFanSpeed=75'
            ], check=False)
            action += ", Set fans to 75%"
            
        except Exception as e:
            action += f"Failed to apply thermal management: {e}"
        
        self.thermal_actions_history.append({
            'timestamp': time.time(),
            'gpu_id': gpu.gpu_id,
            'temperature': gpu.temperature,
            'action': action,
            'severity': 'warning'
        })
        
        return action

class HashCrackingOptimizer:
    """
    Optimizer for hash cracking performance
    """
    
    def __init__(self, gpu_configs: Dict):
        self.gpu_configs = gpu_configs
        self.optimal_batch_sizes = gpu_configs['optimal_batch_sizes']
    
    def get_optimal_batch_size(self, hash_type: str, gpu_memory_available: int) -> int:
        """Get optimal batch size for hash type and available memory"""
        base_batch_size = self.optimal_batch_sizes.get(hash_type, 100000)
        
        # Adjust based on available memory (assuming 11GB total for 2080 Ti)
        memory_factor = gpu_memory_available / (11 * 1024)  # GB to MB conversion
        adjusted_batch_size = int(base_batch_size * memory_factor)
        
        # Ensure minimum and maximum bounds
        min_batch = 1000
        max_batch = base_batch_size * 2
        
        return max(min_batch, min(max_batch, adjusted_batch_size))
    
    def get_hashcat_optimizations(self, hash_type: str, gpu_id: int) -> List[str]:
        """Get Hashcat optimization flags for specific hash type and GPU"""
        base_flags = [
            '--optimized-kernel-enable',
            '--workload-profile', '4',  # Nightmare mode for maximum performance
            f'--opencl-device-types', '2',  # GPU only
            f'--opencl-devices', str(gpu_id + 1),  # Hashcat uses 1-based indexing
        ]
        
        # Hash-specific optimizations
        if hash_type in ['md5', 'sha1']:
            base_flags.extend([
                '--kernel-accel', '1024',  # High acceleration for fast hashes
                '--kernel-loops', '1024'
            ])
        elif hash_type in ['sha256', 'sha512']:
            base_flags.extend([
                '--kernel-accel', '512',   # Moderate acceleration for medium hashes
                '--kernel-loops', '512'
            ])
        elif hash_type in ['bcrypt', 'scrypt']:
            base_flags.extend([
                '--kernel-accel', '32',    # Low acceleration for slow hashes
                '--kernel-loops', '64'
            ])
        
        # RTX 2080 Ti specific optimizations
        base_flags.extend([
            '--bitmap-min', '24',
            '--bitmap-max', '24',
            '--spin-damp', '0',  # Disable spin damping for maximum performance
        ])
        
        return base_flags

async def main():
    """Main function for testing the GPU optimization system"""
    logging.basicConfig(level=logging.INFO)
    
    optimizer = DualGPUOptimizer()
    
    # Test GPU status
    gpu_statuses = await optimizer.get_gpu_status()
    print(f"Found {len(gpu_statuses)} GPUs:")
    for gpu in gpu_statuses:
        print(f"  GPU {gpu.gpu_id}: {gpu.name} - {gpu.temperature}°C, {gpu.utilization}% util")
    
    # Test performance report
    report = await optimizer.generate_performance_report()
    print(f"\nPerformance Report:")
    print(f"  Average Temperature: {report['gpu_summary']['average_temperature']:.1f}°C")
    print(f"  Average Utilization: {report['gpu_summary']['average_utilization']:.1f}%")
    print(f"  Total Power Draw: {report['gpu_summary']['total_power_draw']}W")
    
    if report['optimization_recommendations']:
        print(f"\nRecommendations:")
        for rec in report['optimization_recommendations']:
            print(f"  - {rec}")

if __name__ == "__main__":
    asyncio.run(main())