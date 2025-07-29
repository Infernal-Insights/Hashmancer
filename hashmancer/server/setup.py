import os
import json
import subprocess
import socket
import getpass
import secrets
from pathlib import Path

CONFIG_DIR = Path.home() / ".hashmancer"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
from .app.config import CONFIG_FILE
ENV_FILE = CONFIG_DIR / ".env"
SERVICE_FILE = "/etc/systemd/system/hashmancer-server.service"
DEFAULT_BASE_DIR = "/opt/hashmancer"


def prompt(prompt_text, default=None, secret=False):
    if secret:
        return getpass.getpass(prompt_text + ": ")
    val = input(f"{prompt_text}{f' [{default}]' if default else ''}: ").strip()
    return val or default


def detect_local_ip():
    try:
        return socket.gethostbyname(socket.gethostname())
    except Exception:
        return "127.0.0.1"


def generate_passkey() -> str:
    """Generate and store a portal passkey in server_config.json."""
    key = secrets.token_hex(16)
    config = {}
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                config = json.load(f)
        except Exception:
            pass
    config["portal_passkey"] = key
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Generated portal passkey stored in {CONFIG_FILE}")
    return key


def install_dependencies():
    print("📦 Installing required Python and system packages...")
    subprocess.run(["pip3", "install", "-r", "requirements.txt"], check=False)
    subprocess.run(["sudo", "apt", "install", "-y", "redis-server"], check=False)


def write_env(api_key):
    with open(ENV_FILE, "w") as f:
        f.write(f"HASHES_COM_API_KEY={api_key}\n")


def configure_systemd(bind_ip, port):
    python_path = subprocess.getoutput("which python3")
    app_path = os.path.abspath("main.py")

    service = f"""[Unit]
Description=Hashmancer Server
After=network.target

[Service]
ExecStart={python_path} {app_path}
WorkingDirectory={os.getcwd()}
Restart=always
Environment=PYTHONUNBUFFERED=1
User={os.getenv('USER')}
Group={os.getenv('USER')}

[Install]
WantedBy=multi-user.target
"""

    with open("/tmp/hashmancer-server.service", "w") as f:
        f.write(service)

    subprocess.run(["sudo", "mv", "/tmp/hashmancer-server.service", SERVICE_FILE])
    subprocess.run(["sudo", "systemctl", "daemon-reexec"])
    subprocess.run([
        "sudo",
        "systemctl",
        "enable",
        "--now",
        "hashmancer-server.service",
    ])
    print("✅ Systemd service installed and started.")


def configure(pin: str | None = None):
    print("🔧 Hashmancer Server Setup")

    api_key = prompt("Enter your hashes.com API key", secret=True)
    if not api_key:
        print("❌ Hashes.com API key is required for registration. Aborting.")
        return
    public_url = prompt(
        "Public URL for cloud workers (leave blank for private/local only)", ""
    )
    local_ip = detect_local_ip()

    server_url = public_url or f"http://{local_ip}"
    bind_ip = "0.0.0.0"

    port = prompt("FastAPI port to bind", "8000")

    base_dir = prompt(
        "Base directory for wordlists, masks, and rules",
        DEFAULT_BASE_DIR,
    )
    wordlists_dir = prompt(
        "Folder to store wordlists",
        os.path.join(base_dir, "wordlists"),
    )
    masks_dir = prompt(
        "Folder to store mask files",
        os.path.join(base_dir, "masks"),
    )
    rules_dir = prompt(
        "Folder to store hashcat rules",
        os.path.join(base_dir, "rules"),
    )

    wordlist_db_path = prompt(
        "SQLite DB path for uploaded wordlists",
        os.path.join(base_dir, "wordlists.db"),
    )

    storage_path = prompt("Storage location for logs, exports, etc", "/data/hashmancer")

    system_role = prompt(
        "Role of this server (full / batch-only / cloud-gateway)", "full"
    )

    redis_memory = prompt("Redis memory cap (MB)", "4096")
    redis_persistence = (
        prompt("Enable Redis AOF persistence? (yes/no)", "no").lower() == "yes"
    )
    broadcast = (
        prompt(
            "Enable UDP broadcast for worker auto-discovery? (yes/no)", "yes"
        ).lower()
        == "yes"
    )
    broadcast_port = prompt("Broadcast UDP port", "50000")

    config = {
        "api_key_set": bool(api_key),
        "server_url": server_url,
        "server_port": port,
        "wordlists_dir": wordlists_dir,
        "masks_dir": masks_dir,
        "rules_dir": rules_dir,
        "wordlist_db_path": wordlist_db_path,
        "storage_path": storage_path,
        "role": system_role,
        "redis_memory_mb": int(redis_memory),
        "redis_aof": redis_persistence,
        "broadcast_enabled": broadcast,
        "broadcast_port": int(broadcast_port),
    }
    if pin:
        config["worker_pin"] = pin

    passkey = generate_passkey()
    config["portal_passkey"] = passkey
    initial_token = secrets.token_hex(16)
    config["initial_admin_token"] = initial_token

    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)

    write_env(api_key)

    print("\n🚀 Installing systemd service...")
    configure_systemd(bind_ip, port)

    print("\n🎉 Setup complete.")
    print(f"🔑 Config: {CONFIG_FILE}")
    print(f"🔐 API key: {ENV_FILE}")
    print(f"🔑 Portal passkey: {passkey}")
    print(f"🔑 Initial admin token: {initial_token}")
    print("   Use this key when logging in to the dashboard for the first time.")
    print("🧠 Server URL:", server_url)
    print("🟢 Service: hashmancer-server (enabled & running)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pin", help="worker registration PIN")
    args = parser.parse_args()

    install_dependencies()
    configure(args.pin)
