import os
import glob
import shutil
import logging
import json
from pathlib import Path
import redis
from redis_utils import get_redis
from event_logger import log_error

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s"
)
r = get_redis()

CONFIG_FILE = Path.home() / ".hashmancer" / "server_config.json"
try:
    with CONFIG_FILE.open() as f:
        CONFIG = json.load(f)
except Exception:
    CONFIG = {}

RESTORE_DIR = Path(os.getenv("RESTORE_DIR", CONFIG.get("restore_dir", "./")))
BACKUP_DIR = Path(
    os.getenv("BACKUP_DIR", CONFIG.get("backup_dir", "./restore_backups"))
)


def scan_restore_files():
    """Return a list of available restore files."""
    return [str(p) for p in RESTORE_DIR.glob("*.restore")]


def move_to_backup(file_path: str | Path):
    """Move a processed restore file to BACKUP_DIR."""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    src = Path(file_path)
    shutil.move(str(src), str(BACKUP_DIR / src.name))


def requeue_from_restore(file_path: str | Path):
    """Push batch from restore file back onto the queue."""
    try:
        src = Path(file_path)
        batch_id = src.stem
        batch_data = r.hgetall(f"batch:{batch_id}")
        if not batch_data:
            logging.warning(f"No Redis data for {batch_id}")
            return

        r.lpush("batch:queue", batch_id)
        logging.info(f"Requeued batch {batch_id}")
        move_to_backup(src)
    except redis.exceptions.RedisError as e:
        log_error(
            "restore_manager",
            "server",
            "RRED",
            "Redis unavailable",
            e,
        )
    except Exception as e:
        log_error(
            "restore_manager", "server", "R001", "Failed to requeue restore file", e
        )


def process_restore_files():
    restores = scan_restore_files()
    logging.info(f"Found {len(restores)} restore files.")
    for f in restores:
        requeue_from_restore(f)


if __name__ == "__main__":
    process_restore_files()
