# Hashmancer

This project hosts the server side code as well as a lightweight worker
implementation.  The server exposes a FastAPI application and uses Redis to
queue hash cracking jobs.  Workers register their GPUs with Redis and consume
jobs from a Redis stream.  Sidecars now invoke `hashcat` directly and report
per-GPU hashrates and progress back to the server. Wordlists can be served
from a Redis cache for faster startup on remote nodes.
GPU specs include a `pci_link_width` field for bandwidth-aware scheduling. The worker automatically detects NVIDIA, AMD, and Intel GPUs.


* `Server/` – FastAPI server and orchestration tools
* `Worker/` – HTTP-based worker with GPU sidecar threads

Run `python3 setup.py` from the repository root to configure either a server or
worker.  Use `--server` or `--worker` flags to skip the prompt.  A worker can
also supply `--server-ip` to skip auto-discovery.
