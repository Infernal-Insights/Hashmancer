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

Run `python3 ../setup.py --worker` from the repository root to install
dependencies and configure the worker.  Passing `--server-ip` skips broadcast
discovery.

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

