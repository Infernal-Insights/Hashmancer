#!/usr/bin/env python3
"""
Secure Worker Authentication System
Automated RSA key exchange and challenge-response authentication
"""

import os
import time
import uuid
import base64
import hashlib
import secrets
import logging
from typing import Dict, Optional, Tuple
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.exceptions import InvalidSignature

from ..redis_utils import get_redis
from .rate_limiter import get_rate_limiter

logger = logging.getLogger(__name__)

class WorkerAuthManager:
    """Manages secure worker authentication with automated key exchange."""
    
    def __init__(self):
        self.redis = get_redis()
        self.rate_limiter = get_rate_limiter()
        
        # Server key pair for worker authentication
        self.server_private_key = self._load_or_generate_server_key()
        self.server_public_key = self.server_private_key.public_key()
        
        # Authentication timeouts
        self.challenge_timeout = 300  # 5 minutes
        self.registration_timeout = 600  # 10 minutes
        
    def _load_or_generate_server_key(self) -> rsa.RSAPrivateKey:
        """Load or generate server RSA key pair."""
        key_dir = Path.home() / ".hashmancer" / "keys"
        key_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        
        server_key_path = key_dir / "server_auth_key.pem"
        
        if server_key_path.exists():
            try:
                with open(server_key_path, "rb") as f:
                    private_key = serialization.load_pem_private_key(
                        f.read(), 
                        password=None
                    )
                logger.info("Loaded existing server authentication key")
                return private_key
            except Exception as e:
                logger.warning(f"Failed to load server key, generating new one: {e}")
        
        # Generate new key pair
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=4096
        )
        
        # Save private key securely
        pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        with open(server_key_path, "wb") as f:
            f.write(pem)
        
        # Set secure permissions
        os.chmod(server_key_path, 0o600)
        
        logger.info("Generated new server authentication key")
        return private_key
    
    def get_server_public_key_pem(self) -> str:
        """Get server public key in PEM format for workers."""
        pem = self.server_public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        return pem.decode('utf-8')
    
    async def initiate_worker_registration(self, client_ip: str) -> Dict[str, str]:
        """Initiate worker registration process with challenge."""
        # Rate limiting check
        if not (await self.rate_limiter.allow_request(
            f"worker_reg:{client_ip}", 
            max_requests=5, 
            window_seconds=300
        )):
            raise Exception("Rate limit exceeded for worker registration")
        
        # Generate registration session
        session_id = secrets.token_urlsafe(32)
        challenge = secrets.token_bytes(32)
        
        # Store challenge in Redis with expiration
        challenge_key = f"worker_auth:challenge:{session_id}"
        self.redis.setex(
            challenge_key, 
            self.challenge_timeout, 
            base64.b64encode(challenge).decode()
        )
        
        # Store client IP for validation
        ip_key = f"worker_auth:ip:{session_id}"
        self.redis.setex(ip_key, self.challenge_timeout, client_ip)
        
        logger.info(f"Initiated worker registration for {client_ip}, session: {session_id}")
        
        return {
            "session_id": session_id,
            "challenge": base64.b64encode(challenge).decode(),
            "server_public_key": self.get_server_public_key_pem(),
            "expires_in": self.challenge_timeout
        }
    
    async def complete_worker_registration(
        self, 
        session_id: str, 
        worker_public_key: str,
        signed_challenge: str,
        client_ip: str,
        worker_metadata: Dict
    ) -> Dict[str, str]:
        """Complete worker registration with public key exchange."""
        
        # Validate session exists and get challenge
        challenge_key = f"worker_auth:challenge:{session_id}"
        stored_challenge = self.redis.get(challenge_key)
        if not stored_challenge:
            raise Exception("Invalid or expired registration session")
        
        # Validate client IP consistency
        ip_key = f"worker_auth:ip:{session_id}"
        stored_ip = self.redis.get(ip_key)
        if stored_ip != client_ip:
            raise Exception("IP address mismatch during registration")
        
        # Validate worker public key format
        try:
            worker_key = serialization.load_pem_public_key(worker_public_key.encode())
        except Exception as e:
            raise Exception(f"Invalid worker public key: {e}")
        
        # Verify signed challenge
        try:
            challenge_bytes = base64.b64decode(stored_challenge)
            signature_bytes = base64.b64decode(signed_challenge)
            
            worker_key.verify(
                signature_bytes,
                challenge_bytes,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
        except (InvalidSignature, Exception) as e:
            raise Exception(f"Challenge verification failed: {e}")
        
        # Generate unique worker ID
        worker_id = f"worker_{uuid.uuid4().hex[:16]}"
        
        # Generate worker authentication token
        auth_token = secrets.token_urlsafe(64)
        
        # Store worker credentials in Redis
        worker_data = {
            "worker_id": worker_id,
            "public_key": worker_public_key,
            "auth_token": auth_token,
            "registered_at": int(time.time()),
            "registered_ip": client_ip,
            "status": "registered",
            "metadata": base64.b64encode(str(worker_metadata).encode()).decode()
        }
        
        worker_key_redis = f"worker:auth:{worker_id}"
        self.redis.hset(worker_key_redis, mapping=worker_data)
        
        # Index by auth token for fast lookup
        token_key = f"worker:token:{auth_token}"
        self.redis.setex(token_key, 86400 * 30, worker_id)  # 30 days
        
        # Clean up registration session
        self.redis.delete(challenge_key, ip_key)
        
        logger.info(f"Worker {worker_id} registered successfully from {client_ip}")
        
        return {
            "worker_id": worker_id,
            "auth_token": auth_token,
            "status": "registered"
        }
    
    async def authenticate_worker(self, auth_token: str, client_ip: str) -> Optional[Dict]:
        """Authenticate worker using auth token."""
        if not auth_token:
            return None
        
        # Rate limiting
        if not (await self.rate_limiter.allow_request(
            f"worker_auth:{client_ip}", 
            max_requests=60, 
            window_seconds=60
        )):
            logger.warning(f"Rate limit exceeded for worker auth from {client_ip}")
            return None
        
        # Look up worker by token
        token_key = f"worker:token:{auth_token}"
        worker_id = self.redis.get(token_key)
        if not worker_id:
            logger.warning(f"Invalid auth token from {client_ip}")
            return None
        
        # Get worker data
        worker_key = f"worker:auth:{worker_id}"
        worker_data = self.redis.hgetall(worker_key)
        if not worker_data:
            logger.warning(f"Worker data not found for {worker_id}")
            return None
        
        # Update last seen
        self.redis.hset(worker_key, "last_seen", int(time.time()))
        self.redis.hset(worker_key, "last_ip", client_ip)
        
        return {
            "worker_id": worker_id,
            "status": worker_data.get("status"),
            "public_key": worker_data.get("public_key"),
            "registered_at": int(worker_data.get("registered_at", 0)),
            "metadata": worker_data.get("metadata")
        }
    
    async def create_worker_challenge(self, worker_id: str) -> str:
        """Create a signed challenge for worker to verify server identity."""
        challenge_data = f"{worker_id}:{int(time.time())}:{secrets.token_hex(16)}"
        
        signature = self.server_private_key.sign(
            challenge_data.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        return base64.b64encode(signature).decode()
    
    async def verify_worker_signature(
        self, 
        worker_id: str, 
        data: str, 
        signature: str
    ) -> bool:
        """Verify a signature from a worker."""
        try:
            # Get worker's public key
            worker_key = f"worker:auth:{worker_id}"
            public_key_pem = self.redis.hget(worker_key, "public_key")
            if not public_key_pem:
                return False
            
            public_key = serialization.load_pem_public_key(public_key_pem.encode())
            signature_bytes = base64.b64decode(signature)
            
            public_key.verify(
                signature_bytes,
                data.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
            
        except Exception as e:
            logger.warning(f"Signature verification failed for {worker_id}: {e}")
            return False
    
    async def revoke_worker(self, worker_id: str, reason: str = "revoked"):
        """Revoke worker authentication."""
        worker_key = f"worker:auth:{worker_id}"
        worker_data = self.redis.hgetall(worker_key)
        
        if worker_data:
            # Remove token mapping
            auth_token = worker_data.get("auth_token")
            if auth_token:
                token_key = f"worker:token:{auth_token}"
                self.redis.delete(token_key)
            
            # Mark as revoked
            self.redis.hset(worker_key, "status", "revoked")
            self.redis.hset(worker_key, "revoked_at", int(time.time()))
            self.redis.hset(worker_key, "revoke_reason", reason)
            
            logger.info(f"Worker {worker_id} revoked: {reason}")
    
    async def list_workers(self) -> Dict[str, Dict]:
        """List all registered workers."""
        workers = {}
        
        # Find all worker keys
        worker_keys = self.redis.keys("worker:auth:*")
        
        for key in worker_keys:
            worker_data = self.redis.hgetall(key)
            if worker_data:
                worker_id = worker_data.get("worker_id")
                if worker_id:
                    workers[worker_id] = {
                        "status": worker_data.get("status"),
                        "registered_at": int(worker_data.get("registered_at", 0)),
                        "last_seen": int(worker_data.get("last_seen", 0)),
                        "registered_ip": worker_data.get("registered_ip"),
                        "last_ip": worker_data.get("last_ip")
                    }
        
        return workers

# Global instance
_worker_auth_manager = None

def get_worker_auth_manager() -> WorkerAuthManager:
    """Get global worker authentication manager."""
    global _worker_auth_manager
    if _worker_auth_manager is None:
        _worker_auth_manager = WorkerAuthManager()
    return _worker_auth_manager