# Hashmancer Worker

This directory contains a minimal Redis based worker implementation for the Hashmancer project.
Workers register their GPUs with Redis, consume jobs from a stream and update job status when
processing completes. Each GPU is handled by a lightweight sidecar thread.

## Components

- `hashmancer_worker/worker_agent.py` – registers the worker and spawns GPU sidecars
- `hashmancer_worker/gpu_sidecar.py` – reads jobs from a Redis stream and simulates GPU work

The worker expects a running Redis instance which can be configured through the environment
variables `REDIS_HOST`, `REDIS_PORT` and `JOBS_STREAM`.

Start a worker with:

```bash
python -m hashmancer_worker.worker_agent
```

The example implementation only simulates work but demonstrates how GPU registration and
stream consumption operate in the new architecture.
