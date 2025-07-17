import os
import base64
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

PRIVATE_KEY_PATH = os.getenv("PRIVATE_KEY_PATH", "./worker_private_key.pem")
PUBLIC_KEY_PATH = os.getenv("PUBLIC_KEY_PATH", "./worker_public_key.pem")


def load_private_key():
    with open(PRIVATE_KEY_PATH, "rb") as f:
        key_data = f.read()
    return serialization.load_pem_private_key(key_data, password=None)


def load_public_key_pem() -> str:
    with open(PUBLIC_KEY_PATH, "r") as f:
        return f.read()


def sign_message(message: str) -> str:
    key = load_private_key()
    signature = key.sign(message.encode(), padding.PKCS1v15(), hashes.SHA256())
    return base64.b64encode(signature).decode()
