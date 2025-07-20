# Hashmancer Worker

This directory contains a lightweight worker that communicates with the server
over its HTTP API. Jobs are cached in a local Redis instance and for GPUs on
x1/x4 PCIe links a copy of the payload is stored in VRAM so hashcat loads data
faster.
Both NVIDIA, AMD, and Intel GPUs are supported.

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

Each sidecar launches `hashcat` for incoming batches and streams per-GPU
hashrate statistics back to the server.  Restore files are uploaded if a job is
interrupted so the server can requeue work.  Wordlists are cached in VRAM on
low-bandwidth GPUs and can also be pulled from the server's Redis cache so
workers start faster.

## Performance tuning

Two optional environment variables allow you to tweak how `hashcat` runs:

- `HASHCAT_WORKLOAD` – passed to `hashcat` as `-w`. Set `4` for maximum GPU load.
- `HASHCAT_OPTIMIZED` – when `1`, adds `-O` to enable optimized kernels.
- `GPU_POWER_LIMIT` – if set, attempts to cap GPU power in watts using
  vendor tools (`nvidia-smi`, `rocm-smi`, or `intel_gpu_frequency`) before
  launching the cracking engine. Values ending in `%` apply AMD power
  overdrive using `rocm-smi`.
- `DARKLING_GPU_POWER_LIMIT` – similar to `GPU_POWER_LIMIT` but only applied
  when the darkling engine is used. This allows independent tuning of
  experimental kernels.

Each detected GPU is assigned its own sidecar thread so multiple devices run in
parallel.  Logging and watchdog tasks now execute in dedicated threads inside
the sidecars to avoid blocking GPU kernels.

Example:

```bash
HASHCAT_WORKLOAD=4 HASHCAT_OPTIMIZED=1 python -m hashmancer_worker.worker_agent
```


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

## BIOS flashing

When flash tasks are queued the worker will attempt to apply vendor specific
settings with `nvidia-smi` or `rocm-smi`.  If a `bios_rom` field is supplied in
the settings the worker first uses `nvflash_linux` (for NVIDIA) or `amdvbflash`
for AMD GPUs to dump the existing ROM.  Should flashing fail the backup ROM is
flashed back automatically.  PCI addresses are detected with `lspci` or vendor
utilities so the correct adapter is selected.

