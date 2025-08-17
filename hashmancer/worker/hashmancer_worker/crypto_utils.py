"""Utility helpers for signing messages from the worker.

The worker keeps the private key loaded at module import time so that each
signing operation doesn't repeatedly read the key file.  The key object is
immutable in the ``cryptography`` library, so sharing a single instance across
threads is safe as long as the key is not modified.
"""

import os
import base64
import time
from pathlib import Path
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

PRIVATE_KEY_PATH = os.getenv("PRIVATE_KEY_PATH", "./worker_private_key.pem")
PUBLIC_KEY_PATH = os.getenv("PUBLIC_KEY_PATH", "./worker_public_key.pem")

# Load the private key once so repeated signing doesn't hit the filesystem.
# The returned key object from the cryptography library is immutable and safe
# to use across threads for sign operations.


def generate_keypair() -> rsa.RSAPrivateKey:
    """Generate a 4096-bit RSA key pair and write it to disk."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=4096)
    priv_bytes = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    pub_bytes = key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    priv_path = Path(PRIVATE_KEY_PATH)
    pub_path = Path(PUBLIC_KEY_PATH)
    priv_path.parent.mkdir(parents=True, exist_ok=True)
    pub_path.parent.mkdir(parents=True, exist_ok=True)
    priv_path.write_bytes(priv_bytes)
    pub_path.write_bytes(pub_bytes)
    return key


def load_private_key():
    try:
        with open(PRIVATE_KEY_PATH, "rb") as f:
            key_data = f.read()
    except FileNotFoundError:
        return generate_keypair()
    except Exception:
        raise
    return serialization.load_pem_private_key(key_data, password=None)


# Cache the private key at import time for reuse.
try:
    _PRIVATE_KEY = load_private_key()
except FileNotFoundError:
    # Allow import in environments without the private key. The key will be
    # loaded on first use of ``sign_message``.
    _PRIVATE_KEY = None


def load_public_key_pem() -> str:
    with open(PUBLIC_KEY_PATH, "r") as f:
        return f.read()


def sign_message(message: str, timestamp: int | None = None) -> str:
    """Return a base64 signature for ``message`` and ``timestamp``.

    If ``timestamp`` is ``None`` the current UNIX time is used.  The payload
    signed is ``"{message}|{timestamp}"``.
    """
    global _PRIVATE_KEY
    if timestamp is None:
        timestamp = int(time.time())
    key = _PRIVATE_KEY
    if key is None:
        _PRIVATE_KEY = key = load_private_key()
    payload = f"{message}|{timestamp}"
    signature = key.sign(
        payload.encode(), padding.PKCS1v15(), hashes.SHA256()
    )
    return base64.b64encode(signature).decode()
