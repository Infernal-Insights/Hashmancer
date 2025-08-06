# Hashmancer

This project hosts the server side code as well as a lightweight worker
implementation.  The server exposes a FastAPI application and uses Redis to
queue hash cracking jobs.  Workers register their GPUs with Redis and consume
jobs from a Redis stream.  Sidecars now invoke `hashcat` directly and report
per-GPU hashrates and progress back to the server. Wordlists can be served
from a Redis cache for faster startup on remote nodes.
GPU specs include a `pci_link_width` field for bandwidth-aware scheduling. The worker automatically detects NVIDIA, AMD, and Intel GPUs. Detection is
handled through platform specific helpers selected via `sys.platform` so Windows and macOS support can be added without touching the core logic.
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


* `hashmancer/server` – FastAPI server and orchestration tools
* `hashmancer/worker` – HTTP-based worker with GPU sidecar threads

Run `python3 setup.py` from the repository root to configure either a server or
worker.  Use `--server` or `--worker` flags to skip the prompt, or pass both to
install the services in one step.  A worker can also supply `--server-ip` to
skip auto-discovery. After setup, run `python3 setup.py --upgrade` to pull the
latest version and update dependencies. Both components generate their RSA
signing keys automatically the first time they run if no key files are present.

You can also use the helper script `scripts/install_all.sh` for a non-
interactive install:

```bash
./scripts/install_all.sh --server --worker
```
The script prints commands for starting the services once installation
finishes.

## Requirements

Hashmancer requires the real `redis` and `pydantic` packages along with
FastAPI and related dependencies. The simplest way to install everything is:

```bash
pip install -r hashmancer/server/requirements.txt \
    -r hashmancer/worker/requirements.txt \
    -r hashmancer/server/requirements-dev.txt
```

GPU detection works across platforms. Linux uses `nvidia-smi`, `rocm-smi` and
sysfs while Windows relies on WMI queries. On macOS the worker parses
`system_profiler` output. The appropriate detector is chosen automatically via
`sys.platform`.

Workers can enable probabilistic candidate ordering using `--probabilistic-order`.
Use `--inverse-prob-order` to iterate from least likely candidates first. The
server portal exposes matching checkboxes under the Markov Training section.

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

## Docker Compose

The repository includes a `docker-compose.yml` for running Redis, the server
and a worker. Start everything with one command:

```bash
docker compose up --build
```

Additional workers can be launched using the scale flag:

```bash
docker compose up --scale worker=3
```

## Packaging

Build and install the wheel using `python -m build` then install from `dist`:

```bash
python -m build
pip install dist/hashmancer-<version>-py3-none-any.whl
```

GitHub releases trigger a publish workflow. Configure a `PYPI_API_TOKEN`
repository secret so the job can upload the wheel to PyPI using
`pypa/gh-action-pypi-publish`. If the secret is omitted the release job only
attaches the wheel artifact.

## Tests

The unit tests cover both the server and worker components. Install their
dependencies with:

```bash
pip install -r hashmancer/server/requirements.txt -r hashmancer/worker/requirements.txt -r hashmancer/server/requirements-dev.txt
```

You can also use the provided `requirements-dev.txt` which includes the
packages from all three files. Once installed, run `pytest` from the repository
root.

The project uses [flake8](https://flake8.pycqa.org) for linting. Legacy files do not
fully conform to the default style yet, so ``.flake8`` extends the ignore list
(E203, E225, E301, E302, E303, E305, E306, E401, E402, E501, E741,
F401, F541, F811, F824, W291, W292, W391). This allows the current codebase to
pass the lint step while work continues on modernization.

## GitHub Issue helper

Hashmancer can open issues on GitHub using a personal access token stored in
`GITHUB_TOKEN`. The helper function `hashmancer.utils.github_client.create_issue`
wraps the API call:

```python
from hashmancer.utils.github_client import create_issue

create_issue("Infernal-Insights/Hashmancer", "Bug title", "Details about the bug")
```
