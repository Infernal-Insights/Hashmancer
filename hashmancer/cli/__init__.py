#!/usr/bin/env python3
"""
Hashmancer Comprehensive CLI Interface
Provides command-line access to server, worker, and darkling functionality
"""

import sys
import asyncio
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from .server_cli import server_cli
from .worker_cli import worker_cli  
from .darkling_cli import darkling_cli
from .hashes_cli import hashes_cli

__all__ = ['server_cli', 'worker_cli', 'darkling_cli', 'hashes_cli']