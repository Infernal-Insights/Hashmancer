# Hashmancer

This project hosts the server side code as well as a lightweight worker
implementation. The server exposes a FastAPI application backed by Redis. The
worker communicates with the server over HTTP, caching batches in its local
Redis instance for fast GPU access. Heartbeat updates can be tuned with the
`STATUS_INTERVAL` environment variable.

* `Server/` – FastAPI server and orchestration tools
* `Worker/` – HTTP-based worker with GPU sidecar threads
