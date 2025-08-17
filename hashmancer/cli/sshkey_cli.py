#!/usr/bin/env python3
"""
SSH Key Management CLI for Hashmancer
Generates SSH keys for Vast.ai integration and VPN access
"""

import click
import os
import sys
import subprocess
import tempfile
import requests
import json
from pathlib import Path
from datetime import datetime
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa, ed25519
from cryptography.hazmat.backends import default_backend

class SSHKeyManager:
    """Manages SSH key generation and Vast.ai integration"""
    
    def __init__(self):
        self.ssh_dir = Path.home() / ".ssh"
        self.ssh_dir.mkdir(mode=0o700, exist_ok=True)
        
    def generate_ssh_key(self, key_type="ed25519", key_name="hashmancer_vastai"):
        """Generate SSH key pair"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        key_name_with_timestamp = f"{key_name}_{timestamp}"
        
        private_key_path = self.ssh_dir / key_name_with_timestamp
        public_key_path = self.ssh_dir / f"{key_name_with_timestamp}.pub"
        
        try:
            if key_type == "ed25519":
                # Generate Ed25519 key (modern, secure, fast)
                private_key = ed25519.Ed25519PrivateKey.generate()
                public_key = private_key.public_key()
                
                # Serialize private key
                private_pem = private_key.private_key(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.OpenSSH,
                    encryption_algorithm=serialization.NoEncryption()
                )
                
                # Serialize public key
                public_ssh = public_key.public_key(
                    encoding=serialization.Encoding.OpenSSH,
                    format=serialization.PublicFormat.OpenSSH
                )
                
            else:  # RSA fallback
                # Generate RSA key (compatibility)
                private_key = rsa.generate_private_key(
                    public_exponent=65537,
                    key_size=4096,
                    backend=default_backend()
                )
                public_key = private_key.public_key()
                
                # Serialize private key
                private_pem = private_key.private_key(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.OpenSSH,
                    encryption_algorithm=serialization.NoEncryption()
                )
                
                # Serialize public key
                public_ssh = public_key.public_key(
                    encoding=serialization.Encoding.OpenSSH,
                    format=serialization.PublicFormat.OpenSSH
                )
            
            # Write private key
            with open(private_key_path, 'wb') as f:
                f.write(private_pem)
            os.chmod(private_key_path, 0o600)
            
            # Write public key with comment
            comment = f"hashmancer-vastai-{timestamp}"
            public_key_content = f"{public_ssh.decode()} {comment}\n"
            with open(public_key_path, 'w') as f:
                f.write(public_key_content)
            os.chmod(public_key_path, 0o644)
            
            return {
                'private_key_path': str(private_key_path),
                'public_key_path': str(public_key_path),
                'public_key_content': public_key_content.strip(),
                'comment': comment
            }
            
        except Exception as e:
            raise click.ClickException(f"Failed to generate SSH key: {e}")

class VastAIClient:
    """Vast.ai API client for SSH key management"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('VASTAI_API_KEY')
        self.base_url = "https://console.vast.ai/api/v0"
        
        if not self.api_key:
            raise click.ClickException(
                "Vast.ai API key not found. Set VASTAI_API_KEY environment variable or use 'vastai set api-key'"
            )
    
    def _make_request(self, method, endpoint, data=None):
        """Make authenticated request to Vast.ai API"""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        url = f"{self.base_url}/{endpoint}"
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers)
            elif method == 'POST':
                response = requests.post(url, headers=headers, json=data)
            elif method == 'PUT':
                response = requests.put(url, headers=headers, json=data)
            elif method == 'DELETE':
                response = requests.delete(url, headers=headers)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            raise click.ClickException(f"Vast.ai API request failed: {e}")
    
    def get_ssh_keys(self):
        """Get list of SSH keys from Vast.ai"""
        return self._make_request('GET', 'users/current/ssh-keys')
    
    def add_ssh_key(self, name, public_key):
        """Add SSH key to Vast.ai account"""
        data = {
            'name': name,
            'key': public_key
        }
        return self._make_request('POST', 'users/current/ssh-keys', data)
    
    def delete_ssh_key(self, key_id):
        """Delete SSH key from Vast.ai account"""
        return self._make_request('DELETE', f'users/current/ssh-keys/{key_id}')

@click.group()
def sshkey_cli():
    """🔑 SSH Key Management for Vast.ai Integration
    
    Generate and manage SSH keys for secure access to Vast.ai workers
    and VPN connections to your local Hashmancer server.
    """
    pass

@sshkey_cli.command()
@click.option('--type', 'key_type', default='ed25519', 
              type=click.Choice(['ed25519', 'rsa']),
              help='SSH key type (ed25519 recommended)')
@click.option('--name', default='hashmancer_vastai',
              help='Base name for the key files')
@click.option('--show-private', is_flag=True,
              help='Also display the private key (use with caution)')
def generate(key_type, name, show_private):
    """Generate SSH key pair and display public key for manual copy/paste"""
    
    click.echo("🔑 Generating SSH key pair...")
    
    try:
        manager = SSHKeyManager()
        key_info = manager.generate_ssh_key(key_type=key_type, key_name=name)
        
        click.echo(f"✅ SSH key generated successfully!")
        click.echo(f"📁 Private key: {key_info['private_key_path']}")
        click.echo(f"📁 Public key:  {key_info['public_key_path']}")
        click.echo()
        
        click.echo("📋 Copy this public key to Vast.ai SSH Keys settings:")
        click.echo("=" * 80)
        click.secho(key_info['public_key_content'], fg='green', bold=True)
        click.echo("=" * 80)
        click.echo()
        
        click.echo("🌐 To add to Vast.ai manually:")
        click.echo("1. Go to https://console.vast.ai/account/")
        click.echo("2. Click 'SSH Keys' tab")
        click.echo("3. Click 'Add SSH Key'")
        click.echo("4. Paste the public key above")
        click.echo("5. Give it a name like 'Hashmancer Worker Key'")
        
        if show_private:
            click.echo()
            click.echo("🔐 Private key content (keep secure!):")
            click.echo("-" * 40)
            with open(key_info['private_key_path'], 'r') as f:
                click.secho(f.read(), fg='red')
            click.echo("-" * 40)
        
        click.echo()
        click.echo("💡 Pro tip: Use 'hashmancer sshkey upload --vastai' to add automatically!")
        
    except Exception as e:
        raise click.ClickException(f"Failed to generate SSH key: {e}")

@sshkey_cli.command()
@click.option('--type', 'key_type', default='ed25519',
              type=click.Choice(['ed25519', 'rsa']),
              help='SSH key type (ed25519 recommended)')
@click.option('--name', default='hashmancer_vastai',
              help='Base name for the key files')
@click.option('--key-name', help='Name for the key in Vast.ai (defaults to comment)')
@click.option('--api-key', help='Vast.ai API key (or set VASTAI_API_KEY env var)')
def upload(key_type, name, key_name, api_key):
    """Generate SSH key and automatically upload to Vast.ai account"""
    
    click.echo("🔑 Generating SSH key pair...")
    
    try:
        # Generate SSH key
        manager = SSHKeyManager()
        key_info = manager.generate_ssh_key(key_type=key_type, key_name=name)
        
        click.echo(f"✅ SSH key generated successfully!")
        click.echo(f"📁 Files saved to: {Path(key_info['private_key_path']).parent}")
        
        # Upload to Vast.ai
        click.echo("🌐 Uploading to Vast.ai...")
        
        vastai = VastAIClient(api_key)
        
        # Use custom name or generate from comment
        upload_name = key_name or f"Hashmancer-{key_info['comment']}"
        
        result = vastai.add_ssh_key(upload_name, key_info['public_key_content'])
        
        click.echo(f"🚀 Successfully uploaded SSH key to Vast.ai!")
        click.echo(f"📝 Key name: {upload_name}")
        click.echo(f"🆔 Key ID: {result.get('id', 'Unknown')}")
        
        click.echo()
        click.echo("🎯 Ready to deploy workers! Use this in your Vast.ai instances:")
        click.echo(f"Private key: {key_info['private_key_path']}")
        
        # Show deployment example
        click.echo()
        click.echo("📖 Example Vast.ai deployment command:")
        click.echo("-" * 50)
        click.secho(f"""docker run -d \\
  --gpus all \\
  --name hashmancer-worker \\
  -v {key_info['private_key_path']}:/root/.ssh/id_ed25519:ro \\
  -e SERVER_URL=http://YOUR_SERVER_IP:8080 \\
  -e REDIS_URL=redis://YOUR_REDIS_IP:6379 \\
  yourusername/hashmancer-worker-vpn:latest""", fg='cyan')
        click.echo("-" * 50)
        
    except Exception as e:
        raise click.ClickException(f"Failed to upload SSH key: {e}")

@sshkey_cli.command()
@click.option('--api-key', help='Vast.ai API key (or set VASTAI_API_KEY env var)')
def list(api_key):
    """List SSH keys in your Vast.ai account"""
    
    try:
        vastai = VastAIClient(api_key)
        keys = vastai.get_ssh_keys()
        
        if not keys:
            click.echo("📭 No SSH keys found in your Vast.ai account")
            return
        
        click.echo(f"🔑 Found {len(keys)} SSH key(s) in your Vast.ai account:")
        click.echo()
        
        for key in keys:
            key_id = key.get('id', 'Unknown')
            key_name = key.get('name', 'Unnamed')
            key_fingerprint = key.get('fingerprint', 'Unknown')
            created = key.get('created_on', 'Unknown')
            
            click.echo(f"📌 {key_name}")
            click.echo(f"   ID: {key_id}")
            click.echo(f"   Fingerprint: {key_fingerprint}")
            click.echo(f"   Created: {created}")
            click.echo()
            
    except Exception as e:
        raise click.ClickException(f"Failed to list SSH keys: {e}")

@sshkey_cli.command()
@click.argument('key_id', type=int)
@click.option('--api-key', help='Vast.ai API key (or set VASTAI_API_KEY env var)')
@click.confirmation_option(prompt='Are you sure you want to delete this SSH key?')
def delete(key_id, api_key):
    """Delete SSH key from Vast.ai account by ID"""
    
    try:
        vastai = VastAIClient(api_key)
        vastai.delete_ssh_key(key_id)
        
        click.echo(f"🗑️  Successfully deleted SSH key ID {key_id} from Vast.ai")
        
    except Exception as e:
        raise click.ClickException(f"Failed to delete SSH key: {e}")

@sshkey_cli.command()
def setup():
    """Interactive setup for Vast.ai integration"""
    
    click.echo("🛠️  Hashmancer Vast.ai SSH Key Setup")
    click.echo("=" * 50)
    
    # Check if vastai CLI is installed
    try:
        result = subprocess.run(['vastai', '--version'], 
                              capture_output=True, text=True, check=True)
        click.echo(f"✅ Vast.ai CLI found: {result.stdout.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        click.echo("⚠️  Vast.ai CLI not found. Install with: pip install vastai")
        if click.confirm("Continue anyway?"):
            pass
        else:
            return
    
    # Check for API key
    api_key = os.getenv('VASTAI_API_KEY')
    if not api_key:
        click.echo()
        click.echo("🔑 No Vast.ai API key found in environment")
        click.echo("To get your API key:")
        click.echo("1. Go to https://console.vast.ai/account/")
        click.echo("2. Look for 'API Key' section")
        click.echo("3. Copy your key")
        click.echo()
        
        api_key = click.prompt("Enter your Vast.ai API key (or press Enter to skip)", 
                             default="", show_default=False)
        
        if api_key:
            # Offer to save to environment
            if click.confirm("Save API key to ~/.bashrc?"):
                bashrc = Path.home() / ".bashrc"
                with open(bashrc, "a") as f:
                    f.write(f"\n# Vast.ai API key for Hashmancer\n")
                    f.write(f"export VASTAI_API_KEY='{api_key}'\n")
                click.echo(f"✅ API key added to {bashrc}")
                click.echo("💡 Run 'source ~/.bashrc' or restart terminal to use")
    else:
        click.echo(f"✅ Vast.ai API key found in environment")
    
    click.echo()
    
    # Generate and upload key
    if click.confirm("Generate new SSH key and upload to Vast.ai?"):
        key_type = click.prompt("Key type", 
                               type=click.Choice(['ed25519', 'rsa']),
                               default='ed25519')
        
        name = click.prompt("Key name", default="hashmancer_vastai")
        
        try:
            # Use the upload command
            ctx = click.get_current_context()
            ctx.invoke(upload, key_type=key_type, name=name, api_key=api_key)
            
        except Exception as e:
            click.echo(f"❌ Setup failed: {e}")
            return
    
    click.echo()
    click.echo("🎉 Setup complete! You're ready to deploy Hashmancer workers on Vast.ai")
    click.echo()
    click.echo("📚 Next steps:")
    click.echo("1. Build your worker Docker image: docker build -f Dockerfile.worker-vpn ...")
    click.echo("2. Push to Docker Hub: docker push yourusername/hashmancer-worker-vpn")
    click.echo("3. Deploy on Vast.ai with your new SSH key")

if __name__ == "__main__":
    sshkey_cli()