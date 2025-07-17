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

## Generating signing keys

To sign requests sent to the server you need an RSA key pair.  Create the keys
and store them under `~/.hashmancer` so the worker can load them automatically.

Using `ssh-keygen`:

```bash
ssh-keygen -t rsa -b 4096 -m PEM -f ~/.hashmancer/worker_private.pem
```

This writes the private key to `~/.hashmancer/worker_private.pem` and the
matching public key to `~/.hashmancer/worker_private.pem.pub`.  Alternatively
you can generate the pair with `openssl`:

```bash
openssl genpkey -algorithm RSA -out ~/.hashmancer/worker_private.pem \
    -pkeyopt rsa_keygen_bits:4096
openssl rsa -in ~/.hashmancer/worker_private.pem -pubout \
    -out ~/.hashmancer/worker_public.pem
```

Keep the private key secure and supply the public key when registering the
worker.  The worker reads `~/.hashmancer/worker_private.pem` when signing API
requests.
