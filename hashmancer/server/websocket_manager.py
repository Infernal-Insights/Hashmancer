"""
Improved WebSocket Manager with Memory Leak Fixes
Addresses connection lifecycle, cleanup, and resource management
"""

import asyncio
import json
import logging
import time
import weakref
from typing import Dict, List, Set, Optional, Any, Callable
from fastapi import WebSocket, WebSocketDisconnect
from dataclasses import dataclass, field
from enum import Enum
import gc

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """WebSocket connection states."""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"


@dataclass
class WebSocketConnection:
    """Enhanced WebSocket connection wrapper with lifecycle management."""
    websocket: WebSocket
    connection_id: str
    connected_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    state: ConnectionState = ConnectionState.CONNECTING
    subscription_types: Set[str] = field(default_factory=set)
    ping_failures: int = 0
    max_ping_failures: int = 3
    
    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = time.time()
    
    def is_alive(self) -> bool:
        """Check if connection is alive."""
        return self.state == ConnectionState.CONNECTED
    
    def is_stale(self, timeout: float = 300) -> bool:
        """Check if connection is stale (no activity for timeout seconds)."""
        return time.time() - self.last_activity > timeout


class ImprovedWebSocketManager:
    """Memory-efficient WebSocket manager with proper cleanup."""
    
    def __init__(self, max_connections: int = 100, cleanup_interval: float = 60):
        # Use WeakSet to allow garbage collection of disconnected websockets
        self._connections: Dict[str, WebSocketConnection] = {}
        self._connection_counter = 0
        self._max_connections = max_connections
        self._cleanup_interval = cleanup_interval
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        # Message rate limiting
        self._message_counts: Dict[str, int] = {}
        self._rate_limit_window = 60  # seconds
        self._max_messages_per_window = 100
        
        # Statistics
        self.stats = {
            "total_connections": 0,
            "active_connections": 0,
            "messages_sent": 0,
            "messages_failed": 0,
            "cleanup_runs": 0
        }
        
        # Start cleanup task
        self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """Start the periodic cleanup task."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
    
    async def _periodic_cleanup(self):
        """Periodic cleanup of stale connections."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self._cleanup_interval)
                await self._cleanup_stale_connections()
                await self._cleanup_rate_limits()
                self.stats["cleanup_runs"] += 1
                
                # Force garbage collection if memory usage is high
                if len(self._connections) > self._max_connections * 0.8:
                    gc.collect()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup task error: {e}")
    
    async def _cleanup_stale_connections(self):
        """Remove stale and disconnected connections."""
        stale_connections = []
        
        for conn_id, connection in self._connections.items():
            if connection.is_stale() or connection.state == ConnectionState.DISCONNECTED:
                stale_connections.append(conn_id)
            elif connection.state == ConnectionState.CONNECTED:
                # Ping test for connected but potentially dead connections
                try:
                    await self._ping_connection(connection)
                except Exception:
                    stale_connections.append(conn_id)
        
        # Remove stale connections
        for conn_id in stale_connections:
            await self._force_disconnect(conn_id, "cleanup")
        
        if stale_connections:
            logger.info(f"Cleaned up {len(stale_connections)} stale WebSocket connections")
    
    async def _cleanup_rate_limits(self):
        """Clean up old rate limit entries."""
        current_time = time.time()
        expired_keys = [
            conn_id for conn_id, timestamp in self._message_counts.items()
            if current_time - timestamp > self._rate_limit_window
        ]
        
        for key in expired_keys:
            del self._message_counts[key]
    
    async def _ping_connection(self, connection: WebSocketConnection):
        """Send ping to test connection health."""
        try:
            await connection.websocket.send_json({
                "type": "ping",
                "timestamp": time.time()
            })
            connection.update_activity()
            connection.ping_failures = 0
        except Exception as e:
            connection.ping_failures += 1
            if connection.ping_failures >= connection.max_ping_failures:
                raise e
    
    def _generate_connection_id(self) -> str:
        """Generate unique connection ID."""
        self._connection_counter += 1
        return f"ws_{self._connection_counter}_{int(time.time())}"
    
    def _check_rate_limit(self, connection_id: str) -> bool:
        """Check if connection is rate limited."""
        current_time = time.time()
        
        # Clean old entries for this connection
        if connection_id in self._message_counts:
            if current_time - self._message_counts[connection_id] > self._rate_limit_window:
                del self._message_counts[connection_id]
        
        # Check rate limit
        message_count = self._message_counts.get(connection_id, 0)
        if message_count >= self._max_messages_per_window:
            return False
        
        self._message_counts[connection_id] = message_count + 1
        return True
    
    async def connect(self, websocket: WebSocket) -> str:
        """Connect a new WebSocket with proper lifecycle management."""
        # Check connection limits
        if len(self._connections) >= self._max_connections:
            logger.warning("Maximum WebSocket connections reached")
            await websocket.close(code=1013, reason="Server overloaded")
            raise Exception("Maximum connections exceeded")
        
        try:
            await websocket.accept()
            
            connection_id = self._generate_connection_id()
            connection = WebSocketConnection(
                websocket=websocket,
                connection_id=connection_id,
                state=ConnectionState.CONNECTED
            )
            
            self._connections[connection_id] = connection
            self.stats["total_connections"] += 1
            self.stats["active_connections"] = len(self._connections)
            
            logger.info(f"WebSocket connected: {connection_id} (total: {len(self._connections)})")
            return connection_id
            
        except Exception as e:
            logger.error(f"Failed to connect WebSocket: {e}")
            raise
    
    async def disconnect(self, connection_id: str, reason: str = "client_disconnect"):
        """Properly disconnect a WebSocket connection."""
        if connection_id not in self._connections:
            return
        
        connection = self._connections[connection_id]
        connection.state = ConnectionState.DISCONNECTING
        
        try:
            # Try graceful close
            if connection.websocket.client_state.name != "DISCONNECTED":
                await connection.websocket.close()
        except Exception as e:
            logger.debug(f"Error during WebSocket close: {e}")
        finally:
            await self._cleanup_connection(connection_id, reason)
    
    async def _force_disconnect(self, connection_id: str, reason: str = "force_disconnect"):
        """Force disconnect without graceful close."""
        await self._cleanup_connection(connection_id, reason)
    
    async def _cleanup_connection(self, connection_id: str, reason: str):
        """Clean up connection resources."""
        if connection_id in self._connections:
            connection = self._connections[connection_id]
            connection.state = ConnectionState.DISCONNECTED
            
            # Remove from active connections
            del self._connections[connection_id]
            
            # Clean up rate limiting
            if connection_id in self._message_counts:
                del self._message_counts[connection_id]
            
            self.stats["active_connections"] = len(self._connections)
            logger.info(f"WebSocket disconnected: {connection_id} ({reason})")
    
    async def send_to_connection(self, connection_id: str, message: Dict[str, Any]) -> bool:
        """Send message to specific connection with error handling."""
        if connection_id not in self._connections:
            return False
        
        connection = self._connections[connection_id]
        
        # Check rate limit
        if not self._check_rate_limit(connection_id):
            logger.warning(f"Rate limit exceeded for connection {connection_id}")
            return False
        
        try:
            await connection.websocket.send_json(message)
            connection.update_activity()
            self.stats["messages_sent"] += 1
            return True
            
        except WebSocketDisconnect:
            await self._force_disconnect(connection_id, "websocket_disconnect")
            return False
        except Exception as e:
            logger.error(f"Failed to send message to {connection_id}: {e}")
            self.stats["messages_failed"] += 1
            await self._force_disconnect(connection_id, "send_error")
            return False
    
    async def broadcast(self, message: Dict[str, Any], subscription_filter: Optional[str] = None) -> int:
        """Broadcast message to all or filtered connections."""
        if not self._connections:
            return 0
        
        success_count = 0
        failed_connections = []
        
        for connection_id, connection in self._connections.items():
            # Apply subscription filter
            if subscription_filter and subscription_filter not in connection.subscription_types:
                continue
            
            success = await self.send_to_connection(connection_id, message)
            if success:
                success_count += 1
            else:
                failed_connections.append(connection_id)
        
        # Clean up failed connections
        for conn_id in failed_connections:
            await self._force_disconnect(conn_id, "broadcast_failure")
        
        return success_count
    
    async def subscribe(self, connection_id: str, subscription_type: str):
        """Subscribe connection to specific message types."""
        if connection_id in self._connections:
            self._connections[connection_id].subscription_types.add(subscription_type)
    
    async def unsubscribe(self, connection_id: str, subscription_type: str):
        """Unsubscribe connection from specific message types."""
        if connection_id in self._connections:
            self._connections[connection_id].subscription_types.discard(subscription_type)
    
    def get_connection_count(self) -> int:
        """Get current active connection count."""
        return len(self._connections)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        return {
            **self.stats,
            "active_connections": len(self._connections),
            "memory_usage_mb": self._estimate_memory_usage()
        }
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB."""
        # Rough estimate based on connection count
        base_size = 1024  # bytes per connection estimate
        return (len(self._connections) * base_size) / 1024 / 1024
    
    async def shutdown(self):
        """Gracefully shutdown the WebSocket manager."""
        logger.info("Shutting down WebSocket manager...")
        self._shutdown_event.set()
        
        # Cancel cleanup task
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Disconnect all connections
        connection_ids = list(self._connections.keys())
        for conn_id in connection_ids:
            await self.disconnect(conn_id, "server_shutdown")
        
        logger.info("WebSocket manager shutdown complete")


# Global instance
websocket_manager = ImprovedWebSocketManager()


async def handle_websocket_connection(
    websocket: WebSocket,
    message_handler: Optional[Callable] = None,
    subscription_types: Optional[List[str]] = None
) -> str:
    """
    Handle WebSocket connection with proper error handling and cleanup.
    
    Args:
        websocket: FastAPI WebSocket instance
        message_handler: Optional function to handle incoming messages
        subscription_types: List of message types to subscribe to
    
    Returns:
        Connection ID
    """
    connection_id = None
    
    try:
        connection_id = await websocket_manager.connect(websocket)
        
        # Subscribe to message types
        if subscription_types:
            for sub_type in subscription_types:
                await websocket_manager.subscribe(connection_id, sub_type)
        
        # Handle incoming messages
        while True:
            try:
                data = await websocket.receive_json()
                
                # Update activity
                if connection_id in websocket_manager._connections:
                    websocket_manager._connections[connection_id].update_activity()
                
                # Handle ping/pong
                if data.get("type") == "ping":
                    await websocket_manager.send_to_connection(connection_id, {
                        "type": "pong",
                        "timestamp": time.time()
                    })
                    continue
                
                # Custom message handler
                if message_handler:
                    await message_handler(connection_id, data)
                    
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket message handling error: {e}")
                break
    
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
    
    finally:
        if connection_id:
            await websocket_manager.disconnect(connection_id, "connection_end")
    
    return connection_id or ""


# Backwards compatibility functions
async def broadcast_message(message: Dict[str, Any], message_type: Optional[str] = None) -> int:
    """Broadcast message to WebSocket clients."""
    return await websocket_manager.broadcast(message, message_type)


async def get_websocket_stats() -> Dict[str, Any]:
    """Get WebSocket statistics."""
    return websocket_manager.get_stats()