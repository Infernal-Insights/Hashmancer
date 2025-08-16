#!/usr/bin/env python3
"""
Hashmancer Main CLI Entry Point
Comprehensive command-line interface for server, worker, and darkling operations
"""

import click
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from .server_cli import server_cli
from .worker_cli import worker_cli
from .darkling_cli import darkling_cli
from .hashes_cli import hashes_cli

@click.group()
@click.version_option("1.0.0")
def cli():
    """🔓 Hashmancer - Comprehensive Password Cracking Platform
    
    Commands:
    • server    - Server management and broadcasting
    • worker    - Worker management and control
    • darkling  - Direct hash cracking interface
    • hashes    - Hashes.com integration and job pulling
    """
    pass

# Register command groups
cli.add_command(server_cli, name="server")
cli.add_command(worker_cli, name="worker") 
cli.add_command(darkling_cli, name="darkling")
cli.add_command(hashes_cli, name="hashes")

if __name__ == "__main__":
    cli()