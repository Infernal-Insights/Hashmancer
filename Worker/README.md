# Hashmancer Worker

This directory contains a lightweight worker that communicates with the server
over its HTTP API. Jobs are cached in a local Redis instance and for GPUs on
x1/x4 PCIe links a copy of the payload is stored in VRAM so hashcat loads data
faster.

## Components

- `hashmancer_worker/worker_agent.py` – registers with `/register_worker` and spawns GPU sidecars
- `hashmancer_worker/gpu_sidecar.py` – fetches batches via `/get_batch` and submits results

The worker expects a local Redis instance for caching. Configure `REDIS_HOST` and
`REDIS_PORT`. Point to the server with `SERVER_URL` and provide signing keys via
`PRIVATE_KEY_PATH` and `PUBLIC_KEY_PATH`. The status heartbeat interval can be
customized with `STATUS_INTERVAL` (seconds).

Start a worker with:

```bash
python -m hashmancer_worker.worker_agent
```

The example implementation only simulates work but demonstrates how GPU registration,
HTTP batch retrieval and result submission operate in the new architecture.
