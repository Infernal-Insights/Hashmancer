# 🔥 Hashmancer Server

Hashmancer is a high-performance, distributed hash cracking orchestration system designed for maximum efficiency, extensibility, and control. This server component is responsible for managing workers, distributing batches, processing cracked hashes, and integrating with external APIs like [Hashes.com](https://hashes.com).

---

## 🚀 Features

- FastAPI-based server with secure key-based authentication
- Redis-backed batch queue, worker registry, and result logging
- Automated worker registration with unique anime-style naming
- Support for hybrid, mask, and dictionary attack types
- Intelligent batch dispatching with N+2 prefetch
- Orchestration tools for AWS, Vast.ai, and on-prem deployments
- Self-healing logic with watchdog and error reporting
- Systemd service setup and optional cloud-init support
- Worker and GPU agent code moved to [Hashmancer-Agent](https://github.com/infernal-Insights/hashmancer-agent)
- Agents handle PCIe-aware mask, dictionary, and hybrid attacks
- GPU specs are stored in Redis for tuning
- Redis-based orchestrator balances batches between high- and low-bandwidth queues
- Optional UDP broadcast so workers on the local network can auto-discover the server

---

## 🧱 Architecture

```
             +-------------+
             |  Workers    |
             +------+------+
                    |
           HTTPS (FastAPI)
                    |
            +-------v--------+
            |   Hashmancer   |
            |     Server     |
            +-------+--------+
                    |
        +-----------+-----------+
        | Redis (batches, logs) |
        +----------------------+
```

---

## ⚙️ Setup

1. Clone this repo:

```bash
git clone https://github.com/infernal-Insights/hashmancer-server.git
cd hashmancer-server
```

2. Run the interactive setup script:

```bash
python3 setup.py
```

This will:
- Prompt for base directories for wordlists, masks, and rules. A Hashes.com API key is required for registration
- Configure Redis and logging
- Install a systemd service
- Optionally enable a UDP broadcast so workers can auto-discover the server

3. Start the server (if not done via systemd):

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## 🔧 Systemd Service

To install manually:

```bash
sudo cp hashmancer-server.service /etc/systemd/system/
sudo systemctl daemon-reexec
sudo systemctl enable hashmancer-server
sudo systemctl start hashmancer-server
```

---

## 📊 Glyph Dashboard

`main.py` now includes a tiny dashboard called **Glyph** that you can view from
any device on your network. Start the server with `uvicorn` as shown above and
open your browser to `http://<server-ip>:8000/glyph`.

The page refreshes every few seconds and displays worker counts, queue length,
found results, and GPU temperatures. It also lists all connected workers with a
drop-down to change their status (for example `idle`, `maintenance`, or
`offline`). The layout is landscape friendly so you can mount an old Android
phone or tablet on your rack and use it to interact with and monitor the
server. A simple hashrate chart lets you switch between total rate or a
specific worker so you can keep an eye on performance over time.

---

## 📡 API Endpoints (Overview)

| Method | Path                | Description                         |
|--------|---------------------|-------------------------------------|
| POST   | `/register_worker`  | Register a new worker               |
| GET    | `/get_batch`        | Fetch a job from the Redis stream   |
| POST   | `/submit_founds`    | Submit cracked hashes               |
| POST   | `/submit_no_founds` | Report a finished batch with none   |
| GET    | `/wordlists`        | List available wordlists            |
| GET    | `/masks`            | List available masks                |
| GET    | `/rules`            | List available hashcat rules        |
| GET    | `/workers`          | List registered workers             |
| GET    | `/hashrate`         | Aggregate hashrate of all workers   |

---

## 📁 File Tree (Core)

```
hashmancer-server/
├── main.py
├── event_logger.py
├── redis_manager.py
├── waifus.py
├── hashescom_client.py
├── ...
└── setup.py
```

---

## 🛠 Orchestrator

`orchestrator_agent.py` now keeps things simple. It pulls batches from
Redis, decides the required attack type, and enqueues a lightweight task
into either a high- or low-bandwidth queue.  Queue lengths and the number
of workers are used to keep load balanced so tasks flow smoothly to
available GPUs.

---

## 🚀 Hashmancer Agent

Worker and GPU agent code has moved to the [Hashmancer-Agent](https://github.com/infernal-Insights/hashmancer-agent) repository. Refer to that repo for worker setup and environment variables.

### Auto-discovery snippet

The server can periodically broadcast its URL over UDP. Set
`"broadcast_enabled": false` in `~/.hashmancer/server_config.json` to disable it.
To make use of the broadcast in your worker repository, add a small helper to
listen for the message before
prompting for `SERVER_URL`:

```python
def discover_server(timeout: int = 5, port: int = 50000) -> str | None:
    import socket, json
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("", port))
        s.settimeout(timeout)
        try:
            data, _ = s.recvfrom(1024)
            info = json.loads(data.decode())
            return info.get("server_url")
        except Exception:
            return None

# Use it in your setup script
server = discover_server() or prompt("SERVER_URL", SERVER_URL)
```


---

## 🧪 Tests

Install development requirements and run the test suite with `pytest`:

```bash
pip install -r requirements-dev.txt
pytest
```

---

## 📜 License

MIT License — use it, break it, improve it, share it.

---

## 👋 Contributions

PRs, issues, and feedback welcome! Join the mission to make password recovery faster, smarter, and cooler.
