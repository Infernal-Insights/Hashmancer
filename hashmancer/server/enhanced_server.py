#!/usr/bin/env python3
"""
Enhanced Hashmancer Server Launcher
Comprehensive server improvements with advanced monitoring and management
"""
import asyncio
import logging
import sys
import os
from pathlib import Path
import signal
import uvicorn
from contextlib import asynccontextmanager

# Add the hashmancer directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hashmancer.server.app.enhanced_app import app
from hashmancer.server.performance.monitor import get_performance_monitor, start_performance_monitoring
from hashmancer.server.security.rate_limiter import get_rate_limiter
from hashmancer.server.workers.enhanced_worker_manager import EnhancedWorkerManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/tmp/hashmancer_server.log')
    ]
)

logger = logging.getLogger(__name__)

class EnhancedHashmancerServer:
    """Enhanced Hashmancer Server with comprehensive improvements."""
    
    def __init__(self):
        self.performance_monitor = get_performance_monitor()
        self.rate_limiter = get_rate_limiter()
        self.worker_manager = EnhancedWorkerManager()
        self.is_running = False
        
        # Configuration
        self.host = os.getenv("HASHMANCER_HOST", "0.0.0.0")
        self.port = int(os.getenv("HASHMANCER_PORT", "8001"))
        self.workers = int(os.getenv("HASHMANCER_WORKERS", "4"))
        self.debug = os.getenv("HASHMANCER_DEBUG", "false").lower() == "true"
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.is_running = False
    
    async def start(self):
        """Start the enhanced server."""
        logger.info("🚀 Starting Enhanced Hashmancer Server...")
        
        try:
            # Initialize components
            await self.performance_monitor.start_monitoring()
            # await self.rate_limiter.initialize()  # RateLimiter doesn't have initialize method
            await self.worker_manager.initialize()
            
            self.is_running = True
            
            # Print startup banner
            self._print_startup_banner()
            
            # Configure uvicorn
            config = uvicorn.Config(
                app,
                host=self.host,
                port=self.port,
                workers=1,  # Use single worker for async compatibility
                loop="asyncio",
                log_level="info" if not self.debug else "debug",
                access_log=True,
                reload=self.debug
            )
            
            server = uvicorn.Server(config)
            
            # Start server
            await server.serve()
            
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            raise
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Shutdown the server gracefully."""
        if not self.is_running:
            return
        
        logger.info("🛑 Shutting down Enhanced Hashmancer Server...")
        
        try:
            # Shutdown components
            await self.worker_manager.cleanup()
            # await self.rate_limiter.cleanup()  # RateLimiter doesn't have cleanup method
            await self.performance_monitor.stop_monitoring()
            
            self.is_running = False
            logger.info("✅ Server shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    def _print_startup_banner(self):
        """Print startup banner with server information."""
        banner = f"""
╔══════════════════════════════════════════════════════════════════╗
║                    ENHANCED HASHMANCER SERVER                   ║
╠══════════════════════════════════════════════════════════════════╣
║ 🚀 Server URL: http://{self.host}:{self.port}                                 ║
║ 📊 Enhanced Portal: http://{self.host}:{self.port}/portal                     ║
║ 📈 Metrics: http://{self.host}:{self.port}/metrics                           ║
║ 🔌 WebSocket: ws://{self.host}:{self.port}/ws/{{client_id}}                   ║
║                                                                  ║
║ 🔧 Features:                                                     ║
║   • Real-time performance monitoring                            ║
║   • Advanced rate limiting & DDoS protection                    ║
║   • Intelligent worker management                               ║
║   • WebSocket real-time updates                                 ║
║   • Comprehensive benchmarking system                           ║
║   • Enhanced security & authentication                          ║
║                                                                  ║
║ 📁 Log file: /tmp/hashmancer_server.log                         ║
╚══════════════════════════════════════════════════════════════════╝
        """
        print(banner)
        
        # Print component status
        logger.info("✅ Performance Monitor: Active")
        logger.info("✅ Rate Limiter: Active")
        logger.info("✅ Worker Manager: Active")
        logger.info("✅ Enhanced API: Active")
        logger.info("✅ WebSocket Support: Active")

def main():
    """Main entry point."""
    server = EnhancedHashmancerServer()
    
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error(f"Server failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()