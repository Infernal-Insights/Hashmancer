# Hashmancer

This project hosts the server side code as well as a lightweight worker
implementation.  The server exposes a FastAPI application and uses Redis to
queue hash cracking jobs.  Workers register their GPUs with Redis and consume
jobs from a Redis stream.

* `Server/` – FastAPI server and orchestration tools
* `Worker/` – basic Redis worker with GPU sidecar threads
