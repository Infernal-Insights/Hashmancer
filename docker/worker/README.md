# Hashmancer Worker

This directory contains Docker resources for running the worker component.

## Requirements

The worker relies on the official `redis` client and `pydantic` for data
validation. Install dependencies with:

```bash
pip install -r hashmancer/worker/requirements.txt
```

## Environment variables

The worker is configured entirely through environment variables:

- `REDIS_HOST` – hostname for the Redis instance used for caching and logging.
- `REDIS_PORT` – port for the Redis server (default `6379`).
- `REDIS_PASSWORD` – optional password used when connecting to Redis.
- `REDIS_SSL` – set to `1` to enable TLS. Use `REDIS_SSL_CA_CERT`,
  `REDIS_SSL_CERT` and `REDIS_SSL_KEY` to provide certificate paths.
- `SERVER_URL` – base URL of the Hashmancer server API.
- `PRIVATE_KEY_PATH` – path to the worker's private RSA key.
- `PUBLIC_KEY_PATH` – path to the worker's public key.
- `STATUS_INTERVAL` – seconds between status heartbeats (default `30`).
- `HASHCAT_WORKLOAD` – workload profile passed to `hashcat` via `-w`.
- `HASHCAT_OPTIMIZED` – when set to `1`, adds `-O` to enable optimized kernels.
- `GPU_POWER_LIMIT` – power limit applied before starting hashcat.
- `DARKLING_GPU_POWER_LIMIT` – alternate power limit for the darkling engine.
- `GPU_POWER_TIMEOUT` – timeout in seconds for power limit commands.
- `DARKLING_ENGINE_URL` – URL for downloading a prebuilt darkling engine.
- `DARKLING_GPU_BACKEND` – choose `cuda`, `hip` or `opencl` for darkling.
- `WORKER_ID` – optional fixed identifier for the worker.

If the paths specified by `PRIVATE_KEY_PATH` and `PUBLIC_KEY_PATH` do not
exist the worker generates a new 4096‑bit key pair on first run.
