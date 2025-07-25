import argparse
import json
import subprocess
import socket
import os
from pathlib import Path
from ascii_logo import print_logo

CONFIG_DIR = Path.home() / ".hashmancer"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
WORKER_CONFIG = CONFIG_DIR / "worker_config.json"


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
        except Exception:
            return None


def run_server_setup():
    import Server.setup as srv_setup

    srv_setup.install_dependencies()
    srv_setup.configure()


def upgrade_repo() -> None:
    """Pull the latest code from the git repository."""
    print("\n🔄 Pulling latest updates from GitHub...")
    subprocess.run(["git", "pull"], cwd=os.path.dirname(__file__), check=False)


def run_server_upgrade():
    upgrade_repo()
    import Server.setup as srv_setup

    srv_setup.install_dependencies()


def run_worker_upgrade():
    upgrade_repo()
    worker_install_deps()


def worker_install_deps():
    print("📦 Installing worker dependencies...")
    subprocess.run(["pip3", "install", "-r", "Worker/requirements.txt"], check=False)
    subprocess.run(["sudo", "apt", "install", "-y", "redis-server", "hashcat"], check=False)


def worker_configure(server_url: str):
    cfg = {"server_url": server_url}
    with open(WORKER_CONFIG, "w") as f:
        json.dump(cfg, f, indent=2)
    print("Worker configured for server:", server_url)
    print("Config saved to", WORKER_CONFIG)
    print("Run 'python -m hashmancer_worker.worker_agent' to start the worker")


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


def main():
    print_logo()
    parser = argparse.ArgumentParser(description="Hashmancer setup")
    parser.add_argument("--server", action="store_true", help="setup a server")
    parser.add_argument("--worker", action="store_true", help="setup a worker")
    parser.add_argument("--server-ip", help="server IP or URL for worker setup")
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

    if args.server:
        run_server_setup()
    else:
        run_worker_setup(args.server_ip)


if __name__ == "__main__":
    main()
