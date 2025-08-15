"""
Web Portal for Vast.ai Worker Management
Provides a comprehensive web interface for managing workers, jobs, and costs
"""

from fastapi import FastAPI, Request, Form, HTTPException, BackgroundTasks
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
import os
from datetime import datetime, timedelta
import json

from vast_ai_manager import (
    VastAIManager, WorkerSpec, GPUType, WorkerStatus, WorkerInstance,
    WorkerAutoScaler, CostOptimizer
)

app = FastAPI(title="Hashmancer Worker Portal", version="1.0.0")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Global instances
vast_manager: Optional[VastAIManager] = None
auto_scaler: Optional[WorkerAutoScaler] = None
cost_optimizer: Optional[CostOptimizer] = None

# Pydantic models for API
class LaunchWorkerRequest(BaseModel):
    gpu_type: str
    gpu_count: int = 1
    max_price_per_hour: float
    job_id: Optional[str] = None
    cpu_cores: int = 4
    ram_gb: int = 16
    storage_gb: int = 50
    startup_script: Optional[str] = None
    env_vars: Dict[str, str] = {}

class WorkerResponse(BaseModel):
    instance_id: str
    vast_id: int
    status: str
    gpu_type: str
    gpu_count: int
    price_per_hour: float
    total_cost: float
    uptime_hours: float
    ip_address: Optional[str]
    job_assignments: List[str]

class CostSummaryResponse(BaseModel):
    total_cost: float
    active_workers: int
    total_workers: int
    estimated_hourly_cost: float
    cost_by_gpu_type: Dict[str, float]

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    global vast_manager, auto_scaler, cost_optimizer
    
    api_key = os.getenv("VAST_API_KEY")
    if not api_key:
        raise Exception("VAST_API_KEY environment variable not set")
    
    vast_manager = VastAIManager(api_key)
    await vast_manager.__aenter__()
    
    auto_scaler = WorkerAutoScaler(vast_manager)
    cost_optimizer = CostOptimizer(vast_manager)

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources"""
    if vast_manager:
        await vast_manager.__aexit__(None, None, None)

# Web interface routes
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page"""
    try:
        workers = await vast_manager.list_workers()
        cost_summary = await vast_manager.get_cost_summary()
        
        # Calculate summary statistics
        running_workers = [w for w in workers if w.status == WorkerStatus.RUNNING]
        total_gpus = sum(w.gpu_count for w in running_workers)
        
        # Group workers by status
        workers_by_status = {}
        for worker in workers:
            status = worker.status.value
            if status not in workers_by_status:
                workers_by_status[status] = 0
            workers_by_status[status] += 1
        
        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "workers": workers,
            "cost_summary": cost_summary,
            "total_gpus": total_gpus,
            "workers_by_status": workers_by_status,
            "gpu_types": [gpu.value for gpu in GPUType]
        })
    except Exception as e:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": str(e)
        })

@app.get("/workers", response_class=HTMLResponse)
async def workers_page(request: Request):
    """Workers management page"""
    workers = await vast_manager.list_workers()
    return templates.TemplateResponse("workers.html", {
        "request": request,
        "workers": workers,
        "gpu_types": [gpu.value for gpu in GPUType]
    })

@app.get("/launch", response_class=HTMLResponse)
async def launch_page(request: Request):
    """Launch new worker page"""
    return templates.TemplateResponse("launch.html", {
        "request": request,
        "gpu_types": [gpu.value for gpu in GPUType]
    })

@app.post("/launch", response_class=HTMLResponse)
async def launch_worker_form(
    request: Request,
    gpu_type: str = Form(...),
    gpu_count: int = Form(1),
    max_price: float = Form(...),
    job_id: Optional[str] = Form(None),
    cpu_cores: int = Form(4),
    ram_gb: int = Form(16),
    storage_gb: int = Form(50)
):
    """Handle worker launch form submission"""
    try:
        spec = WorkerSpec(
            gpu_type=GPUType(gpu_type),
            gpu_count=gpu_count,
            cpu_cores=cpu_cores,
            ram_gb=ram_gb,
            storage_gb=storage_gb,
            max_price_per_hour=max_price
        )
        
        worker = await vast_manager.launch_worker(spec, job_id)
        
        return templates.TemplateResponse("launch_success.html", {
            "request": request,
            "worker": worker
        })
    except Exception as e:
        return templates.TemplateResponse("launch.html", {
            "request": request,
            "error": str(e),
            "gpu_types": [gpu.value for gpu in GPUType]
        })

@app.get("/costs", response_class=HTMLResponse)
async def costs_page(request: Request):
    """Cost analysis and optimization page"""
    cost_summary = await vast_manager.get_cost_summary()
    workers = await vast_manager.list_workers()
    
    # Calculate daily/monthly projections
    hourly_cost = cost_summary["estimated_hourly_cost"]
    daily_projection = hourly_cost * 24
    monthly_projection = daily_projection * 30
    
    return templates.TemplateResponse("costs.html", {
        "request": request,
        "cost_summary": cost_summary,
        "daily_projection": daily_projection,
        "monthly_projection": monthly_projection,
        "workers": workers
    })

@app.get("/scaling", response_class=HTMLResponse)
async def scaling_page(request: Request):
    """Auto-scaling management page"""
    workers = await vast_manager.list_workers(WorkerStatus.RUNNING)
    
    # Mock job queue data - in real implementation, get from job manager
    job_queue_size = 0
    worker_utilization = 0.5
    
    recommendation = await auto_scaler.evaluate_scaling(job_queue_size, worker_utilization)
    
    return templates.TemplateResponse("scaling.html", {
        "request": request,
        "current_workers": len(workers),
        "job_queue_size": job_queue_size,
        "worker_utilization": worker_utilization * 100,
        "recommendation": recommendation
    })

@app.get("/vast-settings", response_class=HTMLResponse)
async def vast_settings_page(request: Request):
    """Vast.AI settings and configuration page"""
    import subprocess
    import os
    
    # Check SSH key status
    ssh_key_path = os.path.expanduser("~/.ssh/id_rsa")
    ssh_public_key_path = os.path.expanduser("~/.ssh/id_rsa.pub")
    ssh_key_exists = os.path.exists(ssh_key_path)
    
    ssh_public_key = None
    if os.path.exists(ssh_public_key_path):
        with open(ssh_public_key_path, 'r') as f:
            ssh_public_key = f.read().strip()
    
    # Get current settings from environment
    api_key = os.getenv('VAST_API_KEY', '')
    server_url = os.getenv('HASHMANCER_SERVER_URL', 'http://localhost:8080')
    
    # Default worker settings
    default_gpu_type = os.getenv('DEFAULT_GPU_TYPE', 'rtx4090')
    default_max_price = float(os.getenv('DEFAULT_MAX_PRICE', '1.50'))
    default_cpu_cores = int(os.getenv('DEFAULT_CPU_CORES', '4'))
    default_ram_gb = int(os.getenv('DEFAULT_RAM_GB', '16'))
    default_storage_gb = int(os.getenv('DEFAULT_STORAGE_GB', '50'))
    auto_scaling_enabled = os.getenv('AUTO_SCALING_ENABLED', 'true').lower() == 'true'
    cost_monitoring_enabled = os.getenv('COST_MONITORING_ENABLED', 'true').lower() == 'true'
    
    return templates.TemplateResponse("vast_settings.html", {
        "request": request,
        "ssh_key_exists": ssh_key_exists,
        "ssh_public_key": ssh_public_key,
        "api_key": api_key,
        "server_url": server_url,
        "gpu_types": [gpu.value for gpu in GPUType],
        "default_gpu_type": default_gpu_type,
        "default_max_price": default_max_price,
        "default_cpu_cores": default_cpu_cores,
        "default_ram_gb": default_ram_gb,
        "default_storage_gb": default_storage_gb,
        "auto_scaling_enabled": auto_scaling_enabled,
        "cost_monitoring_enabled": cost_monitoring_enabled
    })

@app.get("/jobs", response_class=HTMLResponse)
async def jobs_page(request: Request):
    """Job management page"""
    # Mock job data - in real implementation, get from job manager
    jobs = [
        {
            "id": "job_001",
            "status": "running",
            "hash_file": "corporate_hashes.txt",
            "wordlist": "rockyou.txt", 
            "assigned_workers": 3,
            "progress": 45.2,
            "estimated_completion": "2h 15m"
        },
        {
            "id": "job_002", 
            "status": "queued",
            "hash_file": "leaked_hashes.txt",
            "wordlist": "custom_wordlist.txt",
            "assigned_workers": 0,
            "progress": 0.0,
            "estimated_completion": "Pending"
        }
    ]
    
    return templates.TemplateResponse("jobs.html", {
        "request": request,
        "jobs": jobs
    })

# API endpoints
@app.get("/api/workers", response_model=List[WorkerResponse])
async def api_list_workers(status: Optional[str] = None):
    """API endpoint to list workers"""
    status_filter = WorkerStatus(status) if status else None
    workers = await vast_manager.list_workers(status_filter)
    
    return [
        WorkerResponse(
            instance_id=w.instance_id,
            vast_id=w.vast_id,
            status=w.status.value,
            gpu_type=w.gpu_type.value,
            gpu_count=w.gpu_count,
            price_per_hour=w.price_per_hour,
            total_cost=w.total_cost,
            uptime_hours=w.uptime_seconds / 3600,
            ip_address=w.ip_address,
            job_assignments=w.job_assignments or []
        )
        for w in workers
    ]

@app.post("/api/workers/launch", response_model=WorkerResponse)
async def api_launch_worker(request: LaunchWorkerRequest):
    """API endpoint to launch a worker"""
    try:
        spec = WorkerSpec(
            gpu_type=GPUType(request.gpu_type),
            gpu_count=request.gpu_count,
            cpu_cores=request.cpu_cores,
            ram_gb=request.ram_gb,
            storage_gb=request.storage_gb,
            max_price_per_hour=request.max_price_per_hour,
            startup_script=request.startup_script,
            env_vars=request.env_vars
        )
        
        worker = await vast_manager.launch_worker(spec, request.job_id)
        
        return WorkerResponse(
            instance_id=worker.instance_id,
            vast_id=worker.vast_id,
            status=worker.status.value,
            gpu_type=worker.gpu_type.value,
            gpu_count=worker.gpu_count,
            price_per_hour=worker.price_per_hour,
            total_cost=worker.total_cost,
            uptime_hours=worker.uptime_seconds / 3600,
            ip_address=worker.ip_address,
            job_assignments=worker.job_assignments or []
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/api/workers/{instance_id}")
async def api_stop_worker(instance_id: str, reason: str = "api_request"):
    """API endpoint to stop a worker"""
    success = await vast_manager.stop_worker(instance_id, reason)
    if success:
        return {"message": f"Worker {instance_id} stop initiated"}
    else:
        raise HTTPException(status_code=400, detail="Failed to stop worker")

@app.get("/api/workers/{instance_id}/status")
async def api_worker_status(instance_id: str):
    """API endpoint to get worker status"""
    worker = await vast_manager.get_worker_status(instance_id)
    if not worker:
        raise HTTPException(status_code=404, detail="Worker not found")
    
    return WorkerResponse(
        instance_id=worker.instance_id,
        vast_id=worker.vast_id,
        status=worker.status.value,
        gpu_type=worker.gpu_type.value,
        gpu_count=worker.gpu_count,
        price_per_hour=worker.price_per_hour,
        total_cost=worker.total_cost,
        uptime_hours=worker.uptime_seconds / 3600,
        ip_address=worker.ip_address,
        job_assignments=worker.job_assignments or []
    )

@app.get("/api/costs/summary", response_model=CostSummaryResponse)
async def api_cost_summary():
    """API endpoint for cost summary"""
    summary = await vast_manager.get_cost_summary()
    return CostSummaryResponse(**summary)

@app.post("/api/scaling/evaluate")
async def api_evaluate_scaling():
    """API endpoint to evaluate scaling recommendations"""
    # Mock data - in real implementation, get from job manager
    job_queue_size = 0
    worker_utilization = 0.5
    
    recommendation = await auto_scaler.evaluate_scaling(job_queue_size, worker_utilization)
    return recommendation

@app.post("/api/scaling/execute")
async def api_execute_scaling(background_tasks: BackgroundTasks):
    """API endpoint to execute scaling recommendations"""
    # Mock data
    job_queue_size = 0
    worker_utilization = 0.5
    
    recommendation = await auto_scaler.evaluate_scaling(job_queue_size, worker_utilization)
    
    if recommendation["action"] == "none":
        return {"message": "No scaling action needed"}
    
    # Execute scaling in background
    background_tasks.add_task(auto_scaler.execute_scaling, recommendation)
    
    return {
        "message": f"Scaling action '{recommendation['action']}' initiated",
        "recommendation": recommendation
    }

# Vast.AI Settings API Endpoints
@app.post("/api/vast-settings/api-config")
async def api_update_api_config(
    vast_api_key: str = Form(...),
    server_url: str = Form(...)
):
    """Update Vast.AI API configuration"""
    import os
    import aiohttp
    
    try:
        # Test the API key
        headers = {'Authorization': f'Bearer {vast_api_key}'}
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get('https://console.vast.ai/api/v0/instances') as response:
                if response.status != 200:
                    return {"success": False, "message": "Invalid API key"}
        
        # Update environment variables (in a real app, update .env file)
        os.environ['VAST_API_KEY'] = vast_api_key
        os.environ['HASHMANCER_SERVER_URL'] = server_url
        
        # Update the vast manager with new API key
        global vast_manager
        if vast_manager:
            vast_manager.api_key = vast_api_key
        
        return {"success": True, "message": "API configuration updated successfully"}
    except Exception as e:
        return {"success": False, "message": f"Failed to update API config: {str(e)}"}

@app.post("/api/vast-settings/worker-defaults")
async def api_update_worker_defaults(
    default_gpu_type: str = Form(...),
    default_max_price: float = Form(...),
    default_cpu_cores: int = Form(...),
    default_ram_gb: int = Form(...),
    default_storage_gb: int = Form(...),
    auto_scaling_enabled: bool = Form(False),
    cost_monitoring_enabled: bool = Form(False)
):
    """Update default worker configuration"""
    import os
    
    try:
        # Update environment variables
        os.environ['DEFAULT_GPU_TYPE'] = default_gpu_type
        os.environ['DEFAULT_MAX_PRICE'] = str(default_max_price)
        os.environ['DEFAULT_CPU_CORES'] = str(default_cpu_cores)
        os.environ['DEFAULT_RAM_GB'] = str(default_ram_gb)
        os.environ['DEFAULT_STORAGE_GB'] = str(default_storage_gb)
        os.environ['AUTO_SCALING_ENABLED'] = str(auto_scaling_enabled).lower()
        os.environ['COST_MONITORING_ENABLED'] = str(cost_monitoring_enabled).lower()
        
        return {"success": True, "message": "Worker defaults updated successfully"}
    except Exception as e:
        return {"success": False, "message": f"Failed to update worker defaults: {str(e)}"}

@app.get("/api/vast-settings/test-connection")
async def api_test_connection():
    """Test Vast.AI API connection"""
    import os
    import aiohttp
    
    try:
        api_key = os.getenv('VAST_API_KEY')
        if not api_key:
            return {"success": False, "message": "No API key configured"}
        
        headers = {'Authorization': f'Bearer {api_key}'}
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get('https://console.vast.ai/api/v0/instances') as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "success": True, 
                        "message": f"✅ Connection successful! Found {len(data)} instances."
                    }
                else:
                    return {
                        "success": False, 
                        "message": f"❌ API connection failed: HTTP {response.status}"
                    }
    except Exception as e:
        return {"success": False, "message": f"❌ Connection error: {str(e)}"}

@app.post("/api/vast-settings/generate-ssh-key")
async def api_generate_ssh_key():
    """Generate SSH key for Vast.AI"""
    import subprocess
    import os
    
    try:
        ssh_key_path = os.path.expanduser("~/.ssh/id_rsa")
        if os.path.exists(ssh_key_path):
            return {"success": False, "message": "SSH key already exists"}
        
        # Generate SSH key
        result = subprocess.run([
            'ssh-keygen', '-t', 'rsa', '-b', '4096', 
            '-C', 'hashmancer@vast.ai', 
            '-f', ssh_key_path, 
            '-N', ''
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            return {
                "success": True, 
                "message": "✅ SSH key generated successfully! Please add it to your Vast.AI account."
            }
        else:
            return {"success": False, "message": f"Failed to generate SSH key: {result.stderr}"}
    except Exception as e:
        return {"success": False, "message": f"Error generating SSH key: {str(e)}"}

@app.get("/api/vast-settings/test-ssh")
async def api_test_ssh():
    """Test SSH connection to Vast.AI"""
    import subprocess
    import os
    
    try:
        ssh_key_path = os.path.expanduser("~/.ssh/id_rsa")
        if not os.path.exists(ssh_key_path):
            return {"success": False, "message": "No SSH key found. Please generate one first."}
        
        # Test SSH key by trying to scan vast.ai hosts
        result = subprocess.run([
            'ssh-keyscan', 'console.vast.ai'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0 and result.stdout:
            return {
                "success": True, 
                "message": "✅ SSH key setup appears correct. Can connect to Vast.AI hosts."
            }
        else:
            return {
                "success": False, 
                "message": "❌ Could not connect to Vast.AI hosts. Check your network connection."
            }
    except subprocess.TimeoutExpired:
        return {"success": False, "message": "❌ SSH test timed out"}
    except Exception as e:
        return {"success": False, "message": f"SSH test error: {str(e)}"}

@app.post("/api/vast-settings/refresh-instances")
async def api_refresh_instances():
    """Refresh worker instances from Vast.AI"""
    try:
        global vast_manager
        if not vast_manager:
            return {"success": False, "message": "Vast.AI manager not initialized"}
        
        workers = await vast_manager.list_workers()
        return {
            "success": True, 
            "message": f"✅ Refreshed instances. Found {len(workers)} workers."
        }
    except Exception as e:
        return {"success": False, "message": f"Failed to refresh instances: {str(e)}"}

@app.post("/api/vast-settings/stop-all-workers")
async def api_stop_all_workers():
    """Stop all running workers"""
    try:
        global vast_manager
        if not vast_manager:
            return {"success": False, "message": "Vast.AI manager not initialized"}
        
        workers = await vast_manager.list_workers(WorkerStatus.RUNNING)
        stopped_count = 0
        
        for worker in workers:
            try:
                await vast_manager.terminate_worker(worker.instance_id)
                stopped_count += 1
            except:
                continue  # Continue stopping others even if one fails
        
        return {
            "success": True, 
            "message": f"✅ Stopped {stopped_count} workers successfully."
        }
    except Exception as e:
        return {"success": False, "message": f"Failed to stop workers: {str(e)}"}

@app.post("/api/vast-settings/export-config")
async def api_export_config():
    """Export current configuration"""
    import os
    import json
    
    try:
        config = {
            "vast_api_key": os.getenv('VAST_API_KEY', '')[:10] + "..." if os.getenv('VAST_API_KEY') else "",
            "server_url": os.getenv('HASHMANCER_SERVER_URL', ''),
            "defaults": {
                "gpu_type": os.getenv('DEFAULT_GPU_TYPE', ''),
                "max_price": os.getenv('DEFAULT_MAX_PRICE', ''),
                "cpu_cores": os.getenv('DEFAULT_CPU_CORES', ''),
                "ram_gb": os.getenv('DEFAULT_RAM_GB', ''),
                "storage_gb": os.getenv('DEFAULT_STORAGE_GB', ''),
                "auto_scaling_enabled": os.getenv('AUTO_SCALING_ENABLED', ''),
                "cost_monitoring_enabled": os.getenv('COST_MONITORING_ENABLED', '')
            }
        }
        
        return {
            "success": True, 
            "message": "✅ Configuration exported",
            "config": config
        }
    except Exception as e:
        return {"success": False, "message": f"Failed to export config: {str(e)}"}

@app.get("/api/optimization/spec")
async def api_optimize_spec(performance: str = "standard", budget: float = 2.0):
    """API endpoint to get optimized worker specification"""
    spec = await cost_optimizer.find_optimal_spec(performance, budget)
    
    if spec:
        return {
            "gpu_type": spec.gpu_type.value,
            "gpu_count": spec.gpu_count,
            "cpu_cores": spec.cpu_cores,
            "ram_gb": spec.ram_gb,
            "storage_gb": spec.storage_gb,
            "max_price_per_hour": spec.max_price_per_hour
        }
    else:
        raise HTTPException(status_code=404, detail="No suitable configuration found")

@app.websocket("/ws/live-updates")
async def websocket_endpoint(websocket):
    """WebSocket endpoint for live updates"""
    await websocket.accept()
    
    try:
        while True:
            # Send live updates every 30 seconds
            workers = await vast_manager.list_workers()
            cost_summary = await vast_manager.get_cost_summary()
            
            update_data = {
                "timestamp": datetime.now().isoformat(),
                "workers_count": len(workers),
                "running_workers": len([w for w in workers if w.status == WorkerStatus.RUNNING]),
                "total_cost": cost_summary["total_cost"],
                "hourly_cost": cost_summary["estimated_hourly_cost"]
            }
            
            await websocket.send_json(update_data)
            await asyncio.sleep(30)
            
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, loop="asyncio")