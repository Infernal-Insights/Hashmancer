import os
import glob
import shutil
import logging
import redis
from redis_utils import get_redis
from event_logger import log_error

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s"
)
r = get_redis()

RESTORE_DIR = "./"
BACKUP_DIR = "./restore_backups"


def scan_restore_files():
    return glob.glob(os.path.join(RESTORE_DIR, "*.restore"))


def move_to_backup(file_path):
    os.makedirs(BACKUP_DIR, exist_ok=True)
    shutil.move(file_path, os.path.join(BACKUP_DIR, os.path.basename(file_path)))


def requeue_from_restore(file_path):
    try:
        base = os.path.basename(file_path)
        batch_id = base.replace(".restore", "")
        batch_data = r.hgetall(f"batch:{batch_id}")
        if not batch_data:
            logging.warning(f"No Redis data for {batch_id}")
            return

        r.lpush("batch:queue", batch_id)
        logging.info(f"Requeued batch {batch_id}")
        move_to_backup(file_path)
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
