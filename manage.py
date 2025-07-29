import argparse
import json
import subprocess
import shutil
import socket
import os
from pathlib import Path
import requests
from hashmancer.ascii_logo import print_logo
from hashmancer.utils import event_logger

CONFIG_DIR = Path.home() / ".hashmancer"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
WORKER_CONFIG = CONFIG_DIR / "worker_config.json"
WORKER_SERVICE_FILE = "/etc/systemd/system/hashmancer-worker.service"


def discover_server(timeout: int = 5, port: int = 50000) -> str | None:
    """Listen for a UDP broadcast announcing the server URL."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("", port))
        s.settimeout(timeout)
        try:
            data, _ = s.recvfrom(1024)
            info = json.loads(data.decode())
            return info.get("server_url")
        except (socket.timeout, json.JSONDecodeError, OSError) as e:
            event_logger.log_error("manage", "system", "M001", "Server discovery failed", e)
            return None


def run_server_setup(pin: str | None = None):
    from hashmancer.server import setup as srv_setup

    srv_setup.install_dependencies()
    srv_setup.configure(pin)


def upgrade_repo() -> None:
    """Pull the latest code from the git repository."""
    print("\n🔄 Pulling latest updates from GitHub...")
    subprocess.run(["git", "pull"], cwd=os.path.dirname(__file__), check=False)


def run_server_upgrade():
    upgrade_repo()
    from hashmancer.server import setup as srv_setup

    srv_setup.install_dependencies()


def run_worker_upgrade():
    upgrade_repo()
    worker_install_deps()


def download_prebuilt_engine() -> None:
    """Download a vendor specific darkling-engine if DARKLING_ENGINE_URL is set."""
    base = os.getenv("DARKLING_ENGINE_URL")
    if not base:
        return

    backend = os.getenv("DARKLING_GPU_BACKEND")
    if not backend:
        if shutil.which("nvidia-smi"):
            backend = "cuda"
        elif shutil.which("rocm-smi"):
            backend = "hip"
        else:
            backend = "opencl"

    url = f"{base}-{backend}"

    dest_dir = CONFIG_DIR / "bin"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "darkling-engine"

    try:
        print(f"\U0001F53D Downloading prebuilt darkling-engine ({backend})...")
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        dest.chmod(0o755)
        print(f"Downloaded darkling-engine to {dest}")
    except (requests.RequestException, OSError) as e:
        event_logger.log_error("manage", "system", "M002", "Download prebuilt engine failed", e)
        print(f"⚠️  Failed to download prebuilt engine: {e}")


def worker_install_deps():
    print("📦 Installing worker dependencies...")
    subprocess.run(["pip3", "install", "-r", "hashmancer/worker/requirements.txt"], check=False)
    subprocess.run(["sudo", "apt", "install", "-y", "redis-server", "hashcat"], check=False)
    download_prebuilt_engine()


def worker_configure(server_url: str):
    cfg = {"server_url": server_url}
    with open(WORKER_CONFIG, "w") as f:
        json.dump(cfg, f, indent=2)
    print("Worker configured for server:", server_url)
    print("Config saved to", WORKER_CONFIG)
    print("Run 'python -m hashmancer.worker.hashmancer_worker.worker_agent' to start the worker")


def configure_worker_systemd() -> None:
    python_path = subprocess.getoutput("which python3")
    working_dir = os.path.abspath("docker/worker")
    bin_path = CONFIG_DIR / "bin"
    service = f"""[Unit]
Description=Hashmancer Worker
After=network.target

[Service]
ExecStart={python_path} -m hashmancer.worker.hashmancer_worker.worker_agent
WorkingDirectory={working_dir}
Restart=always
Environment=PYTHONUNBUFFERED=1
Environment=PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:{bin_path}
User={os.getenv('USER')}
Group={os.getenv('USER')}

[Install]
WantedBy=multi-user.target
"""

    with open("/tmp/hashmancer-worker.service", "w") as f:
        f.write(service)

    subprocess.run(["sudo", "mv", "/tmp/hashmancer-worker.service", WORKER_SERVICE_FILE])
    subprocess.run(["sudo", "systemctl", "daemon-reexec"])
    subprocess.run([
        "sudo",
        "systemctl",
        "enable",
        "--now",
        "hashmancer-worker.service",
    ])
    print("✅ Worker systemd service installed and started.")


def run_worker_setup(server_ip: str | None):
    worker_install_deps()
    server_url = None
    if server_ip:
        if server_ip.startswith("http"):
            server_url = server_ip
        else:
            server_url = f"http://{server_ip}:8000"
    else:
        print("🔎 Listening for server broadcast...")
        server_url = discover_server()
        if not server_url:
            server_url = input("Server URL (e.g. http://1.2.3.4:8000): ").strip()
    worker_configure(server_url)
    configure_worker_systemd()


def main():
    print_logo()
    parser = argparse.ArgumentParser(description="Hashmancer setup")
    parser.add_argument("--server", action="store_true", help="setup a server")
    parser.add_argument("--worker", action="store_true", help="setup a worker")
    parser.add_argument("--server-ip", help="server IP or URL for worker setup")
    parser.add_argument("--pin", help="worker registration PIN")
    parser.add_argument(
        "--upgrade",
        action="store_true",
        help="pull the latest code and update dependencies",
    )
    args = parser.parse_args()

    if args.upgrade:
        target = None
        if args.server:
            target = "server"
        elif args.worker:
            target = "worker"
        else:
            if (CONFIG_DIR / "server_config.json").exists():
                target = "server"
            else:
                target = "worker"

        if target == "server":
            run_server_upgrade()
        else:
            run_worker_upgrade()
        return

    if not args.server and not args.worker:
        choice = input("Configure this machine as [server/worker]?: ").strip().lower()
        if choice.startswith("s"):
            args.server = True
        else:
            args.worker = True

    if args.server and args.worker:
        run_server_setup(args.pin)
        try:
            with open(CONFIG_DIR / "server_config.json") as f:
                conf = json.load(f)
            url = conf.get("server_url", "http://127.0.0.1")
            port = conf.get("server_port", "8000")
            server_url = url if url.startswith("http") else f"http://{url}"
            server_url = f"{server_url.rstrip('/')}:{port}"
        except (OSError, json.JSONDecodeError) as e:
            event_logger.log_error("manage", "system", "M003", "Failed to read server config", e)
            server_url = None
        run_worker_setup(server_url)
    elif args.server:
        run_server_setup(args.pin)
    else:
        run_worker_setup(args.server_ip)


if __name__ == "__main__":
    main()
