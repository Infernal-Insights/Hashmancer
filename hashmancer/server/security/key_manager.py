#!/usr/bin/env python3
"""
Secure Key Management System
Encrypted storage and management of cryptographic keys
"""

import os
import time
import hmac
import hashlib
import secrets
import logging
from typing import Optional, Dict, Tuple
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)

class SecureKeyManager:
    """Secure key management with encrypted storage."""
    
    def __init__(self, key_dir: Optional[Path] = None):
        self.key_dir = key_dir or (Path.home() / ".hashmancer" / "secure_keys")
        self.key_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        
        # Derive master key from system entropy and stored salt
        self.master_key = self._get_or_create_master_key()
        
        # Server keys cache
        self._server_private_key = None
        self._server_public_key = None
    
    def _get_or_create_master_key(self) -> bytes:
        """Get or create master encryption key."""
        salt_file = self.key_dir / "master.salt"
        
        if salt_file.exists():
            with open(salt_file, "rb") as f:
                salt = f.read()
        else:
            salt = secrets.token_bytes(32)
            with open(salt_file, "wb") as f:
                f.write(salt)
            os.chmod(salt_file, 0o600)
        
        # Derive key from system characteristics + salt
        # This provides a reasonable level of security for local key encryption
        system_info = (
            os.uname().nodename + 
            str(os.getuid()) + 
            str(os.getgid()) + 
            str(self.key_dir)
        ).encode()
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        return kdf.derive(system_info)
    
    def _encrypt_data(self, data: bytes) -> bytes:
        """Encrypt data with master key."""
        # Generate random IV
        iv = secrets.token_bytes(16)
        
        # Encrypt data
        cipher = Cipher(algorithms.AES(self.master_key), modes.CBC(iv))
        encryptor = cipher.encryptor()
        
        # Pad data to block size
        padding_length = 16 - (len(data) % 16)
        padded_data = data + bytes([padding_length] * padding_length)
        
        encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
        
        # Return IV + encrypted data
        return iv + encrypted_data
    
    def _decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt data with master key."""
        # Extract IV and encrypted data
        iv = encrypted_data[:16]
        ciphertext = encrypted_data[16:]
        
        # Decrypt data
        cipher = Cipher(algorithms.AES(self.master_key), modes.CBC(iv))
        decryptor = cipher.decryptor()
        
        padded_data = decryptor.update(ciphertext) + decryptor.finalize()
        
        # Remove padding
        padding_length = padded_data[-1]
        return padded_data[:-padding_length]
    
    def _store_encrypted_key(self, key_name: str, key_data: bytes) -> None:
        """Store encrypted key to disk."""
        key_file = self.key_dir / f"{key_name}.key"
        
        # Encrypt key data
        encrypted_data = self._encrypt_data(key_data)
        
        # Calculate HMAC for integrity
        hmac_key = hmac.new(self.master_key, encrypted_data, hashlib.sha256).digest()
        
        # Store encrypted data + HMAC
        with open(key_file, "wb") as f:
            f.write(hmac_key + encrypted_data)
        
        os.chmod(key_file, 0o600)
        logger.info(f"Stored encrypted key: {key_name}")
    
    def _load_encrypted_key(self, key_name: str) -> Optional[bytes]:
        """Load encrypted key from disk."""
        key_file = self.key_dir / f"{key_name}.key"
        
        if not key_file.exists():
            return None
        
        try:
            with open(key_file, "rb") as f:
                stored_data = f.read()
            
            # Extract HMAC and encrypted data
            stored_hmac = stored_data[:32]
            encrypted_data = stored_data[32:]
            
            # Verify HMAC
            expected_hmac = hmac.new(self.master_key, encrypted_data, hashlib.sha256).digest()
            if not hmac.compare_digest(stored_hmac, expected_hmac):
                logger.error(f"HMAC verification failed for key: {key_name}")
                return None
            
            # Decrypt key data
            key_data = self._decrypt_data(encrypted_data)
            logger.info(f"Loaded encrypted key: {key_name}")
            return key_data
            
        except Exception as e:
            logger.error(f"Failed to load key {key_name}: {e}")
            return None
    
    def generate_server_key_pair(self, force_regenerate: bool = False) -> Tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey]:
        """Generate or load server RSA key pair."""
        if not force_regenerate and self._server_private_key:
            return self._server_private_key, self._server_public_key
        
        # Try to load existing key
        if not force_regenerate:
            private_key_data = self._load_encrypted_key("server_private")
            if private_key_data:
                try:
                    private_key = serialization.load_pem_private_key(
                        private_key_data, 
                        password=None
                    )
                    public_key = private_key.public_key()
                    
                    self._server_private_key = private_key
                    self._server_public_key = public_key
                    
                    logger.info("Loaded existing server key pair")
                    return private_key, public_key
                except Exception as e:
                    logger.warning(f"Failed to load server key, generating new: {e}")
        
        # Generate new key pair
        logger.info("Generating new server RSA key pair...")
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=4096
        )
        public_key = private_key.public_key()
        
        # Store encrypted private key
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        self._store_encrypted_key("server_private", private_pem)
        
        # Store public key (unencrypted for easy access)
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        public_key_file = self.key_dir / "server_public.pem"
        with open(public_key_file, "wb") as f:
            f.write(public_pem)
        os.chmod(public_key_file, 0o644)
        
        self._server_private_key = private_key
        self._server_public_key = public_key
        
        logger.info("Generated and stored new server key pair")
        return private_key, public_key
    
    def get_server_private_key(self) -> rsa.RSAPrivateKey:
        """Get server private key."""
        if not self._server_private_key:
            self.generate_server_key_pair()
        return self._server_private_key
    
    def get_server_public_key(self) -> rsa.RSAPublicKey:
        """Get server public key."""
        if not self._server_public_key:
            self.generate_server_key_pair()
        return self._server_public_key
    
    def get_server_public_key_pem(self) -> str:
        """Get server public key in PEM format."""
        public_key = self.get_server_public_key()
        pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        return pem.decode('utf-8')
    
    def store_symmetric_key(self, key_name: str, key_data: bytes) -> None:
        """Store encrypted symmetric key."""
        self._store_encrypted_key(f"sym_{key_name}", key_data)
    
    def load_symmetric_key(self, key_name: str) -> Optional[bytes]:
        """Load encrypted symmetric key."""
        return self._load_encrypted_key(f"sym_{key_name}")
    
    def generate_symmetric_key(self, key_name: str, key_size: int = 32) -> bytes:
        """Generate and store new symmetric key."""
        key_data = secrets.token_bytes(key_size)
        self.store_symmetric_key(key_name, key_data)
        logger.info(f"Generated symmetric key: {key_name}")
        return key_data
    
    def store_api_key(self, service_name: str, api_key: str) -> None:
        """Store encrypted API key."""
        api_key_data = api_key.encode('utf-8')
        self._store_encrypted_key(f"api_{service_name}", api_key_data)
    
    def load_api_key(self, service_name: str) -> Optional[str]:
        """Load encrypted API key."""
        api_key_data = self._load_encrypted_key(f"api_{service_name}")
        if api_key_data:
            return api_key_data.decode('utf-8')
        return None
    
    def store_database_credentials(self, db_name: str, username: str, password: str) -> None:
        """Store encrypted database credentials."""
        creds = f"{username}:{password}".encode('utf-8')
        self._store_encrypted_key(f"db_{db_name}", creds)
    
    def load_database_credentials(self, db_name: str) -> Optional[Tuple[str, str]]:
        """Load encrypted database credentials."""
        creds_data = self._load_encrypted_key(f"db_{db_name}")
        if creds_data:
            creds_str = creds_data.decode('utf-8')
            if ':' in creds_str:
                username, password = creds_str.split(':', 1)
                return username, password
        return None
    
    def rotate_master_key(self) -> None:
        """Rotate master encryption key (re-encrypt all stored keys)."""
        logger.info("Starting master key rotation...")
        
        # Load all existing keys
        existing_keys = {}
        for key_file in self.key_dir.glob("*.key"):
            key_name = key_file.stem
            key_data = self._load_encrypted_key(key_name)
            if key_data:
                existing_keys[key_name] = key_data
        
        # Generate new master key
        salt_file = self.key_dir / "master.salt"
        if salt_file.exists():
            # Backup old salt
            backup_salt = self.key_dir / f"master.salt.backup.{int(time.time())}"
            salt_file.rename(backup_salt)
        
        # This will generate a new master key
        self.master_key = self._get_or_create_master_key()
        
        # Re-encrypt all keys with new master key
        for key_name, key_data in existing_keys.items():
            self._store_encrypted_key(key_name, key_data)
        
        # Clear cached keys to force reload
        self._server_private_key = None
        self._server_public_key = None
        
        logger.info(f"Master key rotation complete. Re-encrypted {len(existing_keys)} keys.")
    
    def list_stored_keys(self) -> Dict[str, Dict[str, any]]:
        """List all stored keys with metadata."""
        keys_info = {}
        
        for key_file in self.key_dir.glob("*.key"):
            key_name = key_file.stem
            stat = key_file.stat()
            
            # Determine key type
            key_type = "unknown"
            if key_name.startswith("sym_"):
                key_type = "symmetric"
            elif key_name.startswith("api_"):
                key_type = "api_key"
            elif key_name.startswith("db_"):
                key_type = "database"
            elif key_name == "server_private":
                key_type = "rsa_private"
            
            keys_info[key_name] = {
                "type": key_type,
                "size": stat.st_size,
                "created": stat.st_ctime,
                "modified": stat.st_mtime,
                "accessible": self._load_encrypted_key(key_name) is not None
            }
        
        return keys_info
    
    def delete_key(self, key_name: str) -> bool:
        """Securely delete a stored key."""
        key_file = self.key_dir / f"{key_name}.key"
        
        if not key_file.exists():
            return False
        
        try:
            # Overwrite file with random data before deletion
            file_size = key_file.stat().st_size
            with open(key_file, "r+b") as f:
                f.write(secrets.token_bytes(file_size))
                f.flush()
                os.fsync(f.fileno())
            
            # Delete file
            key_file.unlink()
            
            # Clear from cache if it's the server key
            if key_name == "server_private":
                self._server_private_key = None
                self._server_public_key = None
            
            logger.info(f"Securely deleted key: {key_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete key {key_name}: {e}")
            return False
    
    def backup_keys(self, backup_path: Path) -> bool:
        """Create encrypted backup of all keys."""
        try:
            backup_path.mkdir(parents=True, exist_ok=True)
            
            # Generate backup encryption key
            backup_key = secrets.token_bytes(32)
            backup_salt = secrets.token_bytes(32)
            
            # Copy all key files to backup location
            for key_file in self.key_dir.glob("*.key"):
                backup_file = backup_path / key_file.name
                
                # Read and re-encrypt with backup key
                with open(key_file, "rb") as f:
                    original_data = f.read()
                
                # Decrypt with current master key, re-encrypt with backup key
                stored_hmac = original_data[:32]
                encrypted_data = original_data[32:]
                
                # Verify and decrypt
                expected_hmac = hmac.new(self.master_key, encrypted_data, hashlib.sha256).digest()
                if hmac.compare_digest(stored_hmac, expected_hmac):
                    decrypted_data = self._decrypt_data(encrypted_data)
                    
                    # Re-encrypt with backup key
                    cipher = Cipher(algorithms.AES(backup_key), modes.CBC(backup_salt[:16]))
                    encryptor = cipher.encryptor()
                    
                    padding_length = 16 - (len(decrypted_data) % 16)
                    padded_data = decrypted_data + bytes([padding_length] * padding_length)
                    
                    backup_encrypted = encryptor.update(padded_data) + encryptor.finalize()
                    backup_hmac = hmac.new(backup_key, backup_encrypted, hashlib.sha256).digest()
                    
                    with open(backup_file, "wb") as f:
                        f.write(backup_hmac + backup_encrypted)
                    
                    os.chmod(backup_file, 0o600)
            
            # Store backup key info
            backup_info = {
                "backup_key": backup_key.hex(),
                "backup_salt": backup_salt.hex(),
                "timestamp": time.time()
            }
            
            info_file = backup_path / "backup_info.txt"
            with open(info_file, "w") as f:
                f.write(f"# Hashmancer Key Backup - {time.ctime()}\n")
                f.write(f"# Store this information securely!\n")
                f.write(f"BACKUP_KEY={backup_info['backup_key']}\n")
                f.write(f"BACKUP_SALT={backup_info['backup_salt']}\n")
            
            os.chmod(info_file, 0o600)
            
            logger.info(f"Created encrypted key backup at {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create key backup: {e}")
            return False

# Global instance
_secure_key_manager = None

def get_secure_key_manager() -> SecureKeyManager:
    """Get global secure key manager."""
    global _secure_key_manager
    if _secure_key_manager is None:
        _secure_key_manager = SecureKeyManager()
    return _secure_key_manager