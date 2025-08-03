# Hashmancer Worker

This directory contains a lightweight worker that communicates with the server
over its HTTP API. Jobs are cached in a local Redis instance and for GPUs on
x1/x4 PCIe links a copy of the payload is stored in VRAM so hashcat loads data
faster.
Both NVIDIA, AMD, and Intel GPUs are supported.

## Components

- `hashmancer_worker/worker_agent.py` – registers with `/register_worker` and spawns GPU sidecars
- `hashmancer_worker/gpu_sidecar.py` – fetches batches via `/get_batch` and submits results
- `utils/event_logger.py` – shared logger that records events to Redis

The worker expects a Redis instance for caching. Configure `REDIS_HOST` and
`REDIS_PORT`. When batches reference `wordlist_key`, `REDIS_HOST` must point to
the server's Redis host so the worker can retrieve the cached wordlist.
Provide a password with `REDIS_PASSWORD` and set `REDIS_SSL=1` to enable TLS.
Certificate paths can be specified with `REDIS_SSL_CA_CERT`, `REDIS_SSL_CERT`,
and `REDIS_SSL_KEY`. Point to the server with `SERVER_URL` and provide signing
keys via `PRIVATE_KEY_PATH` and `PUBLIC_KEY_PATH`. If these files do not exist
the worker will create a new 4096-bit RSA pair on first use. The status
heartbeat interval can be customized with `STATUS_INTERVAL` (seconds).

## Setup

Run `python3 ../setup.py --worker` from the repository root to install
dependencies and configure the worker. This command installs and starts a
`hashmancer-worker` systemd service so the worker launches automatically on
boot. You can combine `--server --worker` to set up both components at once.
Passing `--server-ip` skips broadcast discovery.
Use `python3 ../setup.py --upgrade` anytime to pull the latest code and
update dependencies. If `DARKLING_ENGINE_URL` is set the setup script will
fetch a prebuilt `darkling-engine` binary from the provided URL. Set
`DARKLING_GPU_BACKEND` to `cuda`, `hip`, or `opencl` so the matching vendor
build is downloaded. A full toolchain isn't required on the worker.

Minimal `redis.conf` for a password-protected TLS instance:

```conf
port 0
tls-port 6379
requirepass s3cret
tls-cert-file /etc/redis/server.pem
tls-key-file /etc/redis/server.key
tls-ca-cert-file /etc/redis/ca.pem
```

Start a worker with:

```bash
python -m hashmancer.worker.hashmancer_worker.worker_agent
```

Include a PIN with `--pin` or the `WORKER_PIN` environment variable if the
server requires one:

```bash
WORKER_PIN=1234 python -m hashmancer.worker.hashmancer_worker.worker_agent --pin 1234
```

Each sidecar launches `hashcat` for incoming batches and streams per-GPU
hashrate statistics back to the server.  Restore files are uploaded if a job is
interrupted so the server can requeue work.  Wordlists are cached in VRAM on
low-bandwidth GPUs and can also be pulled from the server's Redis cache so
workers start faster. When batches include a `wordlist_key` instead of a local
wordlist path, make sure the worker connects to the server's Redis instance by
setting `REDIS_HOST` to the server's host:

```bash
REDIS_HOST=redis.example.com REDIS_PORT=6379 \
    python -m hashmancer.worker.hashmancer_worker.worker_agent
```

On startup the worker benchmarks each GPU and submits the results to the
server's `/submit_benchmark` endpoint. The orchestrator aggregates these
metrics so batch ranges can scale with device performance.

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
- `DARKLING_AUTOTUNE` – when set, the worker runs a short tuning pass for
  `darkling-engine` to adjust grid and block sizes.
- `DARKLING_TARGET_POWER_LIMIT` – desired power draw in watts used during
  autotuning and as a fallback power cap for darkling jobs.

### Darkling autotuning

When `DARKLING_AUTOTUNE` is enabled the worker launches `darkling-engine`
on a small range without grid or block overrides and checks GPU power draw.
If consumption exceeds `DARKLING_TARGET_POWER_LIMIT` the grid and block values
are halved until the reading falls below the target.  Tuned values are stored
per GPU and exported via `DARKLING_GRID`/`DARKLING_BLOCK` for subsequent runs.

Grid and block sizes for the darkling engine can also be set per GPU model in
`~/.hashmancer/worker_config.json`.  Add a `darkling_tuning` object mapping
model names to `grid` and `block` values.  The worker exports these as the
`DARKLING_GRID` and `DARKLING_BLOCK` environment variables when launching
`darkling-engine`.

Example configuration:

```json
{
  "darkling_tuning": {
    "GeForce RTX 3060": {"grid": 256, "block": 256}
  }
}
```

Each detected GPU is assigned its own sidecar thread so multiple devices run in
parallel.  Logging and watchdog tasks now execute in dedicated threads inside
the sidecars to avoid blocking GPU kernels.

Example:

```bash
HASHCAT_WORKLOAD=4 HASHCAT_OPTIMIZED=1 python -m hashmancer.worker.hashmancer_worker.worker_agent
```


## Generating signing keys

The worker automatically generates a 4096-bit RSA key pair on first run if
`PRIVATE_KEY_PATH` and `PUBLIC_KEY_PATH` do not exist.  You can also create the
keys manually and store them under `~/.hashmancer` so the worker loads them
at startup.

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
The server reads baseline clock/power presets from
`hashmancer/server/flash_presets.json`. Edit this file to add or modify GPU models as
needed. A helper script `flash_tuner.py` can be used to test flashing locally
and reduce the power limit in small steps while verifying stability. Run it with
`python -m hashmancer.worker.hashmancer_worker.flash_tuner --index 0 --vendor nvidia --model "RTX
3080"` for example.

