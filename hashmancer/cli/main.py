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
from .sshkey_cli import sshkey_cli

@click.group()
@click.version_option("1.0.0")
def cli():
    """🔓 Hashmancer - Comprehensive Password Cracking Platform
    
    Commands:
    • server    - Server management and broadcasting
    • worker    - Worker management and control
    • darkling  - Direct hash cracking interface
    • hashes    - Hashes.com integration and job pulling
    • sshkey    - SSH key management for Vast.ai integration
    """
    pass

# Register command groups
cli.add_command(server_cli, name="server")
cli.add_command(worker_cli, name="worker") 
cli.add_command(darkling_cli, name="darkling")
cli.add_command(hashes_cli, name="hashes")
cli.add_command(sshkey_cli, name="sshkey")

# Add convenient shortcut commands as requested
@cli.command(name="generate-sshkey")
@click.option('--vastai', is_flag=True, 
              help='Automatically upload to Vast.ai (requires VASTAI_API_KEY)')
@click.option('--type', 'key_type', default='ed25519',
              type=click.Choice(['ed25519', 'rsa']),
              help='SSH key type (ed25519 recommended)')
def generate_sshkey(vastai, key_type):
    """🔑 Quick SSH key generation for Vast.ai workers (shortcut)
    
    This is a shortcut for the most common SSH key operations:
    hashmancer generate-sshkey           # Generate key, display for manual copy/paste  
    hashmancer generate-sshkey --vastai  # Generate and auto-upload to Vast.ai
    
    For full SSH key management, use: hashmancer sshkey <command>
    """
    from .sshkey_cli import SSHKeyManager, VastAIClient
    
    if vastai:
        # Use the upload functionality
        ctx = click.get_current_context()
        ctx.invoke(sshkey_cli.commands['upload'], key_type=key_type, 
                   name='hashmancer_vastai', key_name=None, api_key=None)
    else:
        # Use the generate functionality
        ctx = click.get_current_context()
        ctx.invoke(sshkey_cli.commands['generate'], key_type=key_type,
                   name='hashmancer_vastai', show_private=False)

if __name__ == "__main__":
    cli()