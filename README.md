# Hashmancer

This project hosts the server side code as well as a lightweight worker
implementation.  The server exposes a FastAPI application and uses Redis to
queue hash cracking jobs.  Workers register their GPUs with Redis and consume
jobs from a Redis stream.  Sidecars now invoke `hashcat` directly and report
per-GPU hashrates and progress back to the server. Wordlists can be served
from a Redis cache for faster startup on remote nodes.
GPU specs include a `pci_link_width` field for bandwidth-aware scheduling. The worker automatically detects NVIDIA, AMD, and Intel GPUs.
Each worker record now stores the configured engine for low-bandwidth GPUs so the
orchestrator knows whether to dispatch the special `darkling` engine. When
batches are prefetched the orchestrator checks all workers and routes jobs into
separate Redis streams:

- `jobs` – normal queue for hashcat-based workers
- `darkling-jobs` – mask tasks for workers running the `darkling` engine
- Benchmark results are posted via `/submit_benchmark` so batch sizes scale
  with GPU speed

Dictionary or hybrid batches are duplicated for `darkling-jobs` as simple mask
attacks so low-bandwidth nodes can contribute work.

## Terminology

In the code a **batch** refers to the high level unit of work pulled from the
`batch:queue`. Each batch becomes a short lived **job** when dispatched to a
worker. The job ID only exists while the task sits in a Redis stream, whereas
the batch ID follows the work through result submission. Earlier revisions used
the terms interchangeably which led to confusion. The worker now stores results
under the batch ID and only uses the job ID to acknowledge the stream entry.


* `Server/` – FastAPI server and orchestration tools
* `Worker/` – HTTP-based worker with GPU sidecar threads

Run `python3 setup.py` from the repository root to configure either a server or
worker.  Use `--server` or `--worker` flags to skip the prompt.  A worker can
also supply `--server-ip` to skip auto-discovery. After setup, run
`python3 setup.py --upgrade` to pull the latest version and update dependencies.

## Thread Safety

Both the server and worker load their private signing keys once at module import
time.  The `cryptography` library returns immutable key objects, so concurrent
threads can safely call the signing helpers without additional locking.

## Example configuration

Settings for the server are loaded from `~/.hashmancer/server_config.json`.
To enable the optional language model orchestrator and provide the model path
add these fields along with default training parameters:

```json
{
  "llm_enabled": true,
  "llm_model_path": "/opt/models/distilgpt2",
  "llm_train_epochs": 1,
  "llm_train_learning_rate": 0.0001
}
```

### Redis security

Redis should only listen on the loopback interface. Add `bind 127.0.0.1` to
`redis.conf` so the service is not exposed to the LAN. When remote workers need
access, restrict the port to their IPs with firewall rules (`iptables` or
`ufw`). It's best to place Redis behind a VPN or require TLS with client
certificates when exposing it over the internet.
