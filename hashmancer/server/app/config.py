"""Central configuration for the Hashmancer server."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

CONFIG_FILE = Path.home() / ".hashmancer" / "server_config.json"
CONFIG: dict[str, Any] = {}


def load_config() -> dict[str, Any]:
    """Load configuration from ``CONFIG_FILE`` into ``CONFIG``."""
    global CONFIG_FILE, CONFIG
    CONFIG_FILE = Path.home() / ".hashmancer" / "server_config.json"
    try:
        with CONFIG_FILE.open() as f:
            CONFIG = json.load(f)
    except Exception:
        CONFIG = {}
    return CONFIG


def save_config() -> None:
    """Persist ``CONFIG`` to ``CONFIG_FILE`` with current runtime settings."""
    try:
        CONFIG["llm_enabled"] = bool(LLM_ENABLED)
        CONFIG["llm_model_path"] = LLM_MODEL_PATH
        CONFIG["llm_train_epochs"] = int(LLM_TRAIN_EPOCHS)
        CONFIG["llm_train_learning_rate"] = float(LLM_TRAIN_LEARNING_RATE)
        CONFIG["inverse_prob_order"] = bool(INVERSE_PROB_ORDER)

        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with CONFIG_FILE.open("w") as f:
            json.dump(CONFIG, f, indent=2)

        if LLM_ENABLED and LLM_MODEL_PATH:
            os.environ["LLM_MODEL_PATH"] = LLM_MODEL_PATH
        else:
            os.environ.pop("LLM_MODEL_PATH", None)
    except Exception:
        pass


# load configuration at module import
load_config()

WORDLISTS_DIR = Path(CONFIG.get("wordlists_dir", "/opt/hashmancer/wordlists"))
MASKS_DIR = Path(CONFIG.get("masks_dir", "/opt/hashmancer/masks"))
RULES_DIR = Path(CONFIG.get("rules_dir", "/opt/hashmancer/rules"))
RESTORE_DIR = Path(CONFIG.get("restore_dir", "/opt/hashmancer/restores"))
STORAGE_DIR = Path(CONFIG.get("storage_path", "/opt/hashmancer"))
WORDLIST_DB_PATH = Path(
    CONFIG.get(
        "wordlist_db_path",
        str(Path.home() / ".hashmancer" / "wordlists.db"),
    )
)
TRUSTED_KEYS_FILE = CONFIG.get("trusted_keys_file")
TRUSTED_KEY_FINGERPRINTS: set[str] = set()
if TRUSTED_KEYS_FILE:
    try:
        with open(TRUSTED_KEYS_FILE) as f:
            TRUSTED_KEY_FINGERPRINTS = {line.strip() for line in f if line.strip()}
    except Exception:
        TRUSTED_KEY_FINGERPRINTS = set()

FOUNDS_FILE = STORAGE_DIR / "founds.txt"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

PORTAL_KEY = CONFIG.get("portal_key")
PORTAL_PASSKEY = CONFIG.get("portal_passkey")
WORKER_PIN = CONFIG.get("worker_pin")
SESSION_TTL = 3600

MAX_IMPORT_SIZE = int(CONFIG.get("max_import_size", 1_048_576))
LOW_BW_ENGINE = CONFIG.get("low_bw_engine", "hashcat")

BROADCAST_ENABLED = bool(CONFIG.get("broadcast_enabled", True))
BROADCAST_PORT = int(CONFIG.get("broadcast_port", 50000))
BROADCAST_INTERVAL = int(CONFIG.get("broadcast_interval", 30))

WATCHDOG_TOKEN = CONFIG.get("watchdog_token")

HASHES_SETTINGS: dict[str, Any] = dict(CONFIG.get("hashes_settings", {}))
HASHES_SETTINGS.setdefault(
    "hashes_poll_interval", int(CONFIG.get("hashes_poll_interval", 1800))
)
HASHES_SETTINGS.setdefault(
    "algo_params", dict(CONFIG.get("hashes_algo_params", {}))
)
HASHES_POLL_INTERVAL = int(HASHES_SETTINGS.get("hashes_poll_interval", 1800))
HASHES_ALGORITHMS = [a.lower() for a in CONFIG.get("hashes_algorithms", [])]
HASHES_DEFAULT_PRIORITY = int(CONFIG.get("hashes_default_priority", 0))
PREDEFINED_MASKS = list(CONFIG.get("predefined_masks", []))
HASHES_ALGO_PARAMS: dict[str, dict[str, Any]] = dict(
    HASHES_SETTINGS.get("algo_params", {})
)

PROBABILISTIC_ORDER = bool(CONFIG.get("probabilistic_order", False))
INVERSE_PROB_ORDER = bool(CONFIG.get("inverse_prob_order", False))
MARKOV_LANG = CONFIG.get("markov_lang", "english")

LLM_ENABLED = bool(CONFIG.get("llm_enabled", False))
LLM_MODEL_PATH = CONFIG.get("llm_model_path", "")
LLM_TRAIN_EPOCHS = int(CONFIG.get("llm_train_epochs", 1))
LLM_TRAIN_LEARNING_RATE = float(CONFIG.get("llm_train_learning_rate", 0.0001))

# thresholds for worker monitoring
TEMP_THRESHOLD = int(CONFIG.get("temp_threshold", 90))
POWER_THRESHOLD = float(CONFIG.get("power_threshold", 250.0))
CRASH_THRESHOLD = int(CONFIG.get("crash_threshold", 3))

