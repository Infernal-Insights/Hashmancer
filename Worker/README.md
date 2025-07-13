# Hashmancer-Worker

This repository contains the worker agent used with the Hashmancer-Server project. It registers GPUs, fetches tasks, runs hashcat, reports results, and sends health updates.

## Modules
- `task_fetcher.py` – fetch tasks from Redis according to GPU bandwidth
- `hashcat_runner.py` – execute hashcat based on task details
- `results_client.py` – send found hashes or completion reports
- `watchdog_agent.py` – report GPU metrics and restart stalled tasks
- `worker_manager.py` – coordinate fetching and running tasks on all GPUs
- `setup_agent.py` – interactive setup and worker registration
- `simple_worker.py` – spawn mask-only workers for GPUs with four or fewer PCIe lanes
- `advanced_worker.py` – spawn workers for GPUs with x8/x16 links for dictionary and hybrid tasks

## Requirements

See [REQUIREMENTS.md](REQUIREMENTS.md) for a list of required packages. The
`setup_agent.py` script will attempt to install them automatically.

The scripts rely on environment variables for server URL, API keys, and Redis connection details, making deployment flexible across various systems.
