# 🔥 Hashmancer Server

Hashmancer is a high-performance, distributed hash cracking orchestration system designed for maximum efficiency, extensibility, and control. This server component is responsible for managing workers, distributing batches, processing cracked hashes, and integrating with external APIs like [Hashes.com](https://hashes.com).

---

## 🚀 Features

- FastAPI-based server with secure key-based authentication
- Redis-backed batch queue, worker registry, and result logging
- Automated worker registration with unique anime-style naming
- Support for hybrid, mask, and dictionary attack types
- Intelligent batch dispatching with N+2 prefetch
- Larger backlog for high-bandwidth GPUs
- Wordlists cached in Redis for quick distribution
- Cracked hashes cached in Redis to skip duplicates
- Backlog scales with reported GPU load
- Orchestration tools for AWS, Vast.ai, and on-prem deployments
- Self-healing logic with watchdog and error reporting
- Systemd service setup and optional cloud-init support
- Worker and GPU agent code moved to [Hashmancer-Agent](https://github.com/infernal-Insights/hashmancer-agent)
- Agents handle PCIe-aware mask, dictionary, and hybrid attacks
- GPU specs are stored in Redis for tuning
- Each GPU spec includes a `pci_link_width` field used to route work
- `hashescom_client.upload_founds` returns `True` when the API confirms receipt
- Redis-based orchestrator balances batches between high- and low-bandwidth queues
- Optional UDP broadcast so workers on the local network can auto-discover the server
- Portal page includes an **Open Glyph** button and Glyph reports server load, backlog size, and queued batches

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

2. Run the interactive setup script from the repository root:

```bash
python3 ../setup.py --server
```

This will:
- Prompt for base directories for wordlists, masks, and rules. A Hashes.com API key is required for registration
- Configure Redis and logging
- Install a systemd service
- Optionally enable a UDP broadcast so workers can auto-discover the server
- Generate a random portal passkey and show it at the end

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
open your browser to `http://<server-ip>:8000/glyph`. The `/portal` page also
contains an **Open Glyph** button so you can pop the dashboard into its own tab
for fullscreen monitoring.

The page refreshes every few seconds and displays worker counts, queue length,
found results, GPU temperatures, and additional metrics such as server load,
backlog target size, and queued batches. Recent cracked hashes are shown in a
small list so you can quickly verify progress. Worker statuses are now
color-coded (`idle` in green, `maintenance` in orange, `offline` in red) for
easier at-a-glance monitoring. The footer shows the last update time.

All connected workers are listed with a drop-down to change their status
(for example `idle`, `maintenance`, or `offline`). The layout is landscape
friendly so you can mount an old Android phone or tablet on your rack and use
it to interact with and monitor the server. A simple hashrate chart lets you
switch between total rate or a specific worker so you can keep an eye on
performance over time.

---

## 📡 API Endpoints (Overview)

| Method | Path                | Description                         |
|--------|---------------------|-------------------------------------|
| POST   | `/register_worker`  | Register a new worker (optional signature) |
| GET    | `/get_batch`        | Worker requests a batch             |
| POST   | `/submit_founds`    | Submit cracked hashes (cached)      |
| POST   | `/submit_no_founds` | Report a finished batch with none   |
| POST   | `/upload_restore`   | Upload a `.restore` file from a worker |
| POST   | `/import_hashes`    | Import a CSV of hashes to queue      |
| GET    | `/wordlists`        | List available wordlists            |
| GET    | `/masks`            | List available masks                |
| GET    | `/rules`            | List available hashcat rules        |
| GET    | `/workers`          | List registered workers             |
| GET    | `/hashrate`         | Aggregate hashrate of all workers   |
| POST   | `/worker_status`    | Update a worker's status (optional signature) |
| POST   | `/submit_benchmark` | Submit benchmark results            |

---

### CSV format for `/import_hashes`

Upload a CSV containing the following columns:
`hash`, `mask`, `wordlist`, `target`, and `hash_mode`.
`hash_mode` numbers follow [hashcat's list](https://hashcat.net/wiki/doku.php?id=example_hashes)
(also available on the portal).

Example:

```csv
hash,mask,wordlist,target,hash_mode
d41d8cd98f00b204e9800998ecf8427e,?a?a?a?a,rockyou.txt,example.com,0
```

---

### Benchmark submission

Workers can post benchmark results to `/submit_benchmark`:

```json
{
  "worker_id": "worker1",
  "gpu_uuid": "GPU-UUID",
  "engine": "hashcat",
  "hashrates": {"MD5": 1000, "SHA1": 2000, "NTLM": 3000},
  "signature": "<signature>"
}
```

Results are stored under `benchmark:<gpu_uuid>` and aggregated totals are kept in `benchmark_total:<worker_id>`.


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

An optional `LLMOrchestrator` component can load a local language model via
`transformers` to suggest batch sizes and choose between the high and low
bandwidth queues. Set `LLM_MODEL_PATH` in the environment to enable it.

To enable it via configuration instead, add these fields to
`~/.hashmancer/server_config.json`:

```json
{
  "llm_enabled": true,
  "llm_model_path": "/opt/models/distilgpt2",
  "llm_train_epochs": 1,
  "llm_train_learning_rate": 0.0001
}
```

Fine-tuning can be initiated through `/train_llm`. Provide a JSON body
containing `dataset`, `base_model`, `epochs`, `learning_rate` and
`output_dir` to run a local `transformers.Trainer` job.

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

### Restore directories

The server periodically checks `RESTORE_DIR` for any `.restore` files and moves
processed files to `BACKUP_DIR`. Both paths can be configured through
environment variables or `~/.hashmancer/server_config.json`:

```json
{
  "restore_dir": "/opt/hashmancer/restores",
  "backup_dir": "/opt/hashmancer/restore_backups"
}
```
If not provided, the defaults are `./` for `RESTORE_DIR` and
`./restore_backups` for `BACKUP_DIR`.

### Portal API key

To restrict access to the web dashboard set `"portal_key"` in
`~/.hashmancer/server_config.json`. Requests to `/portal`, `/glyph` and
`/admin` must then include an `X-API-Key` header with the same value.

### Portal passkey

Dashboard logins also require a `"portal_passkey"` in `~/.hashmancer/server_config.json`.
The interactive setup script now creates one automatically and prints it when
setup finishes. If you ever need to generate a new key manually run:

```bash
python3 -c 'from Server.setup import generate_passkey; generate_passkey()'
```

After setup, send the passkey to `/login` to obtain a session token:

```bash
curl -X POST -H 'Content-Type: application/json' \
     -d '{"passkey": "<your key>"}' http://localhost:8000/login
```

Use the returned `cookie` value as the `session` cookie for subsequent requests.

You can also login through the built-in HTML form by visiting `/login_page` in a
browser. Enter the passkey and you will be redirected to the portal once the
`session` cookie is set.

## 🌐 Reverse Proxy Setup

When exposing Hashmancer to the internet it's best to place a TLS-enabled
reverse proxy like **Nginx** or **Caddy** in front of the FastAPI server. Set
`"server_url"` in `~/.hashmancer/server_config.json` to the public domain so
workers report the correct address. Be sure to choose a `"portal_key"` as shown
above for basic protection of the dashboard.

### Nginx example

```nginx
server {
    listen 443 ssl;
    server_name hashmancer.win;

    ssl_certificate /etc/letsencrypt/live/hashmancer.win/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/hashmancer.win/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### Caddy example

```caddy
hashmancer.win {
    reverse_proxy localhost:8000
    tls /etc/letsencrypt/live/hashmancer.win/fullchain.pem \
        /etc/letsencrypt/live/hashmancer.win/privkey.pem
}
```

## 📈 Learning Password Trends

Run `learn_trends.py <wordlist_dir>` from the `Server` folder to scan every
wordlist in the directory. Each word is converted to a simplified hashcat-style
pattern and the frequency is incremented in the `dictionary:patterns` sorted
set in Redis. These counts can be used to generate smarter masks or analyze
the prevalence of different password formats.

### Converting existing pattern counts

If your statistics were generated before the ``$c`` and ``$e`` tokens were
introduced you can migrate the data with ``convert_patterns.py``:

```bash
python3 convert_patterns.py --src dictionary:patterns --dest dictionary:patterns:v2
```

This reads each stored pattern, reclassifies characters using the current
token rules and writes the results to the destination key.


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
