"""
Vast.ai Integration for On-Demand Worker Deployment
Provides comprehensive worker management, cost optimization, and auto-scaling
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import hashlib
import hmac
from urllib.parse import urlencode

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WorkerStatus(Enum):
    CREATING = "creating"
    STARTING = "starting" 
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"
    DESTROYED = "destroyed"

class GPUType(Enum):
    RTX_3060 = "RTX 3060"
    RTX_3070 = "RTX 3070" 
    RTX_3080 = "RTX 3080"
    RTX_3090 = "RTX 3090"
    RTX_4070 = "RTX 4070"
    RTX_4080 = "RTX 4080"
    RTX_4090 = "RTX 4090"
    RTX_A6000 = "RTX A6000"
    A100_40GB = "A100 PCIE"
    A100_80GB = "A100 SXM4"
    H100 = "H100"

@dataclass
class WorkerSpec:
    """Specification for launching a worker"""
    gpu_type: GPUType
    gpu_count: int = 1
    cpu_cores: int = 4
    ram_gb: int = 16
    storage_gb: int = 50
    max_price_per_hour: float = 1.0
    image: str = "pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime"
    startup_script: Optional[str] = None
    env_vars: Dict[str, str] = None
    
@dataclass  
class WorkerInstance:
    """Represents a running worker instance"""
    instance_id: str
    vast_id: int
    status: WorkerStatus
    gpu_type: GPUType
    gpu_count: int
    price_per_hour: float
    total_cost: float
    uptime_seconds: int
    ip_address: Optional[str]
    ssh_port: Optional[int]
    created_at: datetime
    last_heartbeat: Optional[datetime]
    job_assignments: List[str] = None
    performance_metrics: Dict[str, float] = None

class VastAIManager:
    """Manages worker deployment and lifecycle on vast.ai"""
    
    def __init__(self, api_key: str, base_url: str = "https://console.vast.ai/api/v0"):
        self.api_key = api_key
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
        self.workers: Dict[str, WorkerInstance] = {}
        self.deployment_templates: Dict[str, Dict] = {}
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={"Authorization": f"Bearer {self.api_key}"}
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            
    async def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """Make authenticated request to vast.ai API"""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            async with self.session.request(method, url, **kwargs) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"API Error {response.status}: {error_text}")
                    raise aiohttp.ClientError(f"HTTP {response.status}: {error_text}")
                return await response.json()
        except aiohttp.ClientError as e:
            logger.error(f"Vast.ai API request failed: {e}")
            raise
            
    async def get_available_offers(self, spec: WorkerSpec) -> List[Dict]:
        """Get available GPU offers matching the specification"""
        params = {
            "q": json.dumps({
                "verified": {"eq": True},
                "external": {"eq": False},
                "rentable": {"eq": True},
                "gpu_name": {"in": [spec.gpu_type.value]},
                "num_gpus": {"gte": spec.gpu_count},
                "cpu_cores": {"gte": spec.cpu_cores},
                "cpu_ram": {"gte": spec.ram_gb * 1024},  # Convert to MB
                "disk_space": {"gte": spec.storage_gb * 1024},  # Convert to MB
                "dph_total": {"lte": spec.max_price_per_hour}
            }),
            "order": json.dumps([["dph_total", "asc"]]),  # Sort by price ascending
            "limit": 20
        }
        
        logger.info(f"Searching for offers with params: {params}")
        offers = await self._make_request("GET", "/bundles", params=params)
        logger.info(f"Found {len(offers.get('offers', []))} offers")
        return offers.get("offers", [])
        
    async def launch_worker(self, spec: WorkerSpec, job_id: Optional[str] = None) -> WorkerInstance:
        """Launch a new worker instance"""
        logger.info(f"Launching worker with spec: {spec}")
        
        # Get available offers
        offers = await self.get_available_offers(spec)
        if not offers:
            raise Exception(f"No available offers for spec: {spec}")
            
        # Select best offer (cheapest that meets requirements)
        best_offer = offers[0]
        logger.info(f"Selected offer: {best_offer['id']} at ${best_offer['dph_total']:.3f}/hr")
        
        # Prepare launch parameters
        startup_script = spec.startup_script or self._get_default_startup_script()
        env_vars = spec.env_vars or {}
        env_vars.update({
            "HASHMANCER_SERVER_URL": "http://your-server-ip:8080",
            "HASHMANCER_JOB_ID": job_id or "",
            "HASHMANCER_WORKER_ID": f"vast_{int(time.time())}",
        })
        
        launch_params = {
            "client_id": "hashmancer",
            "image": spec.image,
            "env": env_vars,
            "args": [],
            "onstart": startup_script,
            "runtype": "ssh",
            "disk": spec.storage_gb,
        }
        
        # Launch instance
        result = await self._make_request(
            "PUT", 
            f"/asks/{best_offer['id']}/", 
            json=launch_params
        )
        
        if not result.get("success"):
            raise Exception(f"Failed to launch worker: {result}")
            
        vast_id = result["new_contract"]
        instance_id = f"vast_{vast_id}_{int(time.time())}"
        
        # Create worker instance record
        worker = WorkerInstance(
            instance_id=instance_id,
            vast_id=vast_id,
            status=WorkerStatus.CREATING,
            gpu_type=spec.gpu_type,
            gpu_count=spec.gpu_count,
            price_per_hour=best_offer["dph_total"],
            total_cost=0.0,
            uptime_seconds=0,
            ip_address=None,
            ssh_port=None,
            created_at=datetime.now(),
            last_heartbeat=None,
            job_assignments=[job_id] if job_id else [],
            performance_metrics={}
        )
        
        self.workers[instance_id] = worker
        logger.info(f"Worker {instance_id} launched with vast ID {vast_id}")
        
        return worker
        
    async def get_worker_status(self, instance_id: str) -> Optional[WorkerInstance]:
        """Get current status of a worker"""
        if instance_id not in self.workers:
            return None
            
        worker = self.workers[instance_id]
        
        # Query vast.ai for current status
        try:
            instances = await self._make_request("GET", "/instances")
            vast_instance = next(
                (inst for inst in instances if inst["id"] == worker.vast_id), 
                None
            )
            
            if vast_instance:
                # Update worker status from vast.ai data
                worker.status = self._map_vast_status(vast_instance["actual_status"])
                worker.ip_address = vast_instance.get("public_ipaddr")
                worker.ssh_port = vast_instance.get("ssh_port")
                worker.uptime_seconds = vast_instance.get("duration", 0)
                worker.total_cost = worker.price_per_hour * (worker.uptime_seconds / 3600)
                
        except Exception as e:
            logger.error(f"Failed to get worker status: {e}")
            
        return worker
        
    async def stop_worker(self, instance_id: str, reason: str = "manual") -> bool:
        """Stop and destroy a worker instance"""
        if instance_id not in self.workers:
            return False
            
        worker = self.workers[instance_id]
        logger.info(f"Stopping worker {instance_id} (vast ID: {worker.vast_id}) - {reason}")
        
        try:
            result = await self._make_request("DELETE", f"/instances/{worker.vast_id}/")
            
            if result.get("success"):
                worker.status = WorkerStatus.STOPPING
                logger.info(f"Worker {instance_id} stop initiated")
                return True
            else:
                logger.error(f"Failed to stop worker: {result}")
                return False
                
        except Exception as e:
            logger.error(f"Error stopping worker {instance_id}: {e}")
            worker.status = WorkerStatus.FAILED
            return False
            
    async def list_workers(self, status_filter: Optional[WorkerStatus] = None) -> List[WorkerInstance]:
        """List all workers, optionally filtered by status"""
        workers = list(self.workers.values())
        
        if status_filter:
            workers = [w for w in workers if w.status == status_filter]
            
        return workers
        
    async def get_worker_logs(self, instance_id: str) -> Optional[str]:
        """Get logs from a worker instance"""
        if instance_id not in self.workers:
            return None
            
        worker = self.workers[instance_id]
        
        try:
            # This would require SSH connection to the instance
            # For now, return placeholder
            return f"Logs for worker {instance_id} (vast ID: {worker.vast_id})"
        except Exception as e:
            logger.error(f"Failed to get logs for worker {instance_id}: {e}")
            return None
            
    async def update_worker_assignment(self, instance_id: str, job_id: str) -> bool:
        """Assign a job to a worker"""
        if instance_id not in self.workers:
            return False
            
        worker = self.workers[instance_id]
        if job_id not in worker.job_assignments:
            worker.job_assignments.append(job_id)
            
        logger.info(f"Assigned job {job_id} to worker {instance_id}")
        return True
        
    async def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost summary for all workers"""
        total_cost = 0.0
        active_workers = 0
        cost_by_gpu_type = {}
        
        for worker in self.workers.values():
            total_cost += worker.total_cost
            
            if worker.status == WorkerStatus.RUNNING:
                active_workers += 1
                
            gpu_type_str = worker.gpu_type.value
            if gpu_type_str not in cost_by_gpu_type:
                cost_by_gpu_type[gpu_type_str] = 0.0
            cost_by_gpu_type[gpu_type_str] += worker.total_cost
            
        return {
            "total_cost": total_cost,
            "active_workers": active_workers,
            "total_workers": len(self.workers),
            "cost_by_gpu_type": cost_by_gpu_type,
            "estimated_hourly_cost": sum(
                w.price_per_hour for w in self.workers.values() 
                if w.status == WorkerStatus.RUNNING
            )
        }
        
    def _map_vast_status(self, vast_status: str) -> WorkerStatus:
        """Map vast.ai status to our WorkerStatus enum"""
        status_mapping = {
            "created": WorkerStatus.CREATING,
            "starting": WorkerStatus.STARTING,
            "running": WorkerStatus.RUNNING,
            "stopping": WorkerStatus.STOPPING,
            "stopped": WorkerStatus.STOPPED,
            "exited": WorkerStatus.STOPPED,
            "destroyed": WorkerStatus.DESTROYED,
        }
        return status_mapping.get(vast_status, WorkerStatus.FAILED)
        
    def _get_default_startup_script(self) -> str:
        """Get default startup script for Hashmancer workers"""
        return """#!/bin/bash
set -e

echo "Starting Hashmancer worker setup..."

# Update system
apt-get update
apt-get install -y curl wget git python3 python3-pip

# Install CUDA if not present
if ! command -v nvcc &> /dev/null; then
    echo "Installing CUDA..."
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
    dpkg -i cuda-keyring_1.0-1_all.deb
    apt-get update
    apt-get -y install cuda-toolkit-12-3
fi

# Clone Hashmancer repository
if [ ! -d "/workspace/hashmancer" ]; then
    cd /workspace
    git clone https://github.com/Infernal-Insights/Hashmancer.git hashmancer
    cd hashmancer
else
    cd /workspace/hashmancer
    git pull origin main
fi

# Build Darkling
cd darkling
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Install Python dependencies
cd /workspace/hashmancer
pip3 install -r requirements.txt

# Setup worker configuration
cat > worker_config.json << EOF
{
    "server_url": "${HASHMANCER_SERVER_URL}",
    "worker_id": "${HASHMANCER_WORKER_ID}",
    "job_id": "${HASHMANCER_JOB_ID}",
    "gpu_devices": "auto",
    "heartbeat_interval": 30,
    "max_job_runtime": 3600
}
EOF

# Start worker
echo "Starting Hashmancer worker..."
python3 worker/worker_main.py --config worker_config.json

echo "Hashmancer worker startup complete"
"""

class WorkerAutoScaler:
    """Automatic scaling of workers based on job queue and performance"""
    
    def __init__(self, vast_manager: VastAIManager):
        self.vast_manager = vast_manager
        self.scaling_rules = {
            "min_workers": 0,
            "max_workers": 10,
            "target_queue_time": 300,  # 5 minutes
            "scale_up_threshold": 0.8,
            "scale_down_threshold": 0.3,
            "cooldown_period": 600,  # 10 minutes
        }
        self.last_scaling_action = None
        
    async def evaluate_scaling(self, job_queue_size: int, worker_utilization: float) -> Dict[str, Any]:
        """Evaluate if scaling action is needed"""
        current_workers = len(await self.vast_manager.list_workers(WorkerStatus.RUNNING))
        
        recommendation = {
            "action": "none",
            "target_workers": current_workers,
            "reason": "",
            "estimated_cost_change": 0.0
        }
        
        # Check cooldown period
        if (self.last_scaling_action and 
            datetime.now() - self.last_scaling_action < timedelta(seconds=self.scaling_rules["cooldown_period"])):
            recommendation["reason"] = "In cooldown period"
            return recommendation
            
        # Scale up if queue is large or workers are highly utilized
        if (job_queue_size > 5 or worker_utilization > self.scaling_rules["scale_up_threshold"]):
            if current_workers < self.scaling_rules["max_workers"]:
                recommendation["action"] = "scale_up"
                recommendation["target_workers"] = min(
                    current_workers + max(1, job_queue_size // 3),
                    self.scaling_rules["max_workers"]
                )
                recommendation["reason"] = f"High queue size ({job_queue_size}) or utilization ({worker_utilization:.1%})"
                
        # Scale down if workers are underutilized
        elif worker_utilization < self.scaling_rules["scale_down_threshold"] and job_queue_size == 0:
            if current_workers > self.scaling_rules["min_workers"]:
                recommendation["action"] = "scale_down" 
                recommendation["target_workers"] = max(
                    current_workers - 1,
                    self.scaling_rules["min_workers"]
                )
                recommendation["reason"] = f"Low utilization ({worker_utilization:.1%}) and empty queue"
                
        return recommendation
        
    async def execute_scaling(self, recommendation: Dict[str, Any]) -> bool:
        """Execute a scaling recommendation"""
        if recommendation["action"] == "none":
            return True
            
        current_workers = len(await self.vast_manager.list_workers(WorkerStatus.RUNNING))
        target_workers = recommendation["target_workers"]
        
        if recommendation["action"] == "scale_up":
            workers_to_add = target_workers - current_workers
            logger.info(f"Scaling up: adding {workers_to_add} workers")
            
            for _ in range(workers_to_add):
                spec = WorkerSpec(
                    gpu_type=GPUType.RTX_4090,  # Default to high-performance GPU
                    max_price_per_hour=1.5
                )
                try:
                    await self.vast_manager.launch_worker(spec)
                except Exception as e:
                    logger.error(f"Failed to launch worker during scale-up: {e}")
                    return False
                    
        elif recommendation["action"] == "scale_down":
            workers_to_remove = current_workers - target_workers
            logger.info(f"Scaling down: removing {workers_to_remove} workers")
            
            # Remove workers with least active jobs first
            workers = await self.vast_manager.list_workers(WorkerStatus.RUNNING)
            workers.sort(key=lambda w: len(w.job_assignments or []))
            
            for worker in workers[:workers_to_remove]:
                try:
                    await self.vast_manager.stop_worker(worker.instance_id, "auto_scale_down")
                except Exception as e:
                    logger.error(f"Failed to stop worker during scale-down: {e}")
                    return False
                    
        self.last_scaling_action = datetime.now()
        return True

class CostOptimizer:
    """Optimize costs by selecting best GPU offers and managing worker lifecycle"""
    
    def __init__(self, vast_manager: VastAIManager):
        self.vast_manager = vast_manager
        
    async def find_optimal_spec(self, performance_requirement: str, budget_per_hour: float) -> WorkerSpec:
        """Find optimal worker spec for given performance and budget requirements"""
        
        # Define performance tiers
        performance_tiers = {
            "basic": [GPUType.RTX_3060, GPUType.RTX_3070],
            "standard": [GPUType.RTX_3080, GPUType.RTX_3090, GPUType.RTX_4070],
            "high": [GPUType.RTX_4080, GPUType.RTX_4090, GPUType.RTX_A6000],
            "extreme": [GPUType.A100_40GB, GPUType.A100_80GB, GPUType.H100]
        }
        
        gpu_types = performance_tiers.get(performance_requirement, performance_tiers["standard"])
        
        best_spec = None
        best_value_score = 0
        
        for gpu_type in gpu_types:
            spec = WorkerSpec(
                gpu_type=gpu_type,
                max_price_per_hour=budget_per_hour
            )
            
            try:
                offers = await self.vast_manager.get_available_offers(spec)
                if offers:
                    cheapest_offer = offers[0]
                    
                    # Calculate value score (performance per dollar)
                    performance_score = self._get_gpu_performance_score(gpu_type)
                    value_score = performance_score / cheapest_offer["dph_total"]
                    
                    if value_score > best_value_score:
                        best_value_score = value_score
                        best_spec = spec
                        best_spec.max_price_per_hour = cheapest_offer["dph_total"]
                        
            except Exception as e:
                logger.error(f"Error evaluating {gpu_type}: {e}")
                
        return best_spec or WorkerSpec(gpu_type=GPUType.RTX_4090, max_price_per_hour=budget_per_hour)
        
    def _get_gpu_performance_score(self, gpu_type: GPUType) -> float:
        """Get relative performance score for GPU type"""
        performance_scores = {
            GPUType.RTX_3060: 100,
            GPUType.RTX_3070: 130,
            GPUType.RTX_3080: 170,
            GPUType.RTX_3090: 220,
            GPUType.RTX_4070: 150,
            GPUType.RTX_4080: 200,
            GPUType.RTX_4090: 280,
            GPUType.RTX_A6000: 250,
            GPUType.A100_40GB: 400,
            GPUType.A100_80GB: 450,
            GPUType.H100: 600,
        }
        return performance_scores.get(gpu_type, 100)