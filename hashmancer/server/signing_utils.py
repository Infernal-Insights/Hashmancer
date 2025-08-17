"""Helpers for signing server responses.

The server loads the private key once at import time.  The key object returned
by ``cryptography`` is immutable and safe to share between threads for signing
operations.
"""

import base64
import time
from pathlib import Path
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

KEY_PATH = "./keys/private_key.pem"


def generate_private_key() -> rsa.RSAPrivateKey:
    """Generate a 4096-bit RSA private key and save it to ``KEY_PATH``."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=4096)
    priv_bytes = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    path = Path(KEY_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(priv_bytes)
    return key


def load_private_key():
    try:
        with open(KEY_PATH, "rb") as f:
            key_data = f.read()
    except FileNotFoundError:
        return generate_private_key()
    return serialization.load_pem_private_key(key_data, password=None)


# Cache the server private key at import time.
try:
    _PRIVATE_KEY = load_private_key()
except FileNotFoundError:
    # Importing the module shouldn't fail in environments without the key.
    # The key will be loaded on first call to ``sign_message``.
    _PRIVATE_KEY = None


def sign_message(message: str, timestamp: int | None = None) -> str:
    """Return a base64 signature for ``message`` at ``timestamp`` using the cached key."""
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
