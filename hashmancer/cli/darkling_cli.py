#!/usr/bin/env python3
"""
Hashmancer Darkling CLI
Direct hash cracking interface similar to hashcat
"""

import click
import subprocess
import json
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

@click.group()
def darkling_cli():
    """⚡ Direct hash cracking (hashcat-like interface)"""
    pass

@darkling_cli.command()
@click.argument('hash_file')
@click.argument('wordlist')
@click.option('-a', '--attack-mode', type=int, default=0, help='Attack mode (0=straight, 1=combinator, 3=mask)')
@click.option('-m', '--hash-type', type=int, default=0, help='Hash type (0=MD5, 100=SHA1, 1000=NTLM, etc.)')
@click.option('-r', '--rules', help='Rules file')
@click.option('-o', '--outfile', help='Output file for cracked hashes')
@click.option('--outfile-format', type=int, default=2, help='Output format (1=hash, 2=plain, 3=hash:plain)')
@click.option('-w', '--workload-profile', type=int, default=3, help='Workload profile (1-4)')
@click.option('-O', '--optimized-kernel', is_flag=True, help='Enable optimized kernel')
@click.option('-S', '--slow-candidates', is_flag=True, help='Enable slow candidate generators')
@click.option('--gpu-temp-abort', type=int, default=90, help='Abort if GPU temp exceeds this')
@click.option('--runtime', type=int, help='Runtime limit in seconds')
@click.option('--status', is_flag=True, help='Enable status output')
@click.option('--status-timer', type=int, default=10, help='Status update interval')
@click.option('--machine-readable', is_flag=True, help='Machine readable output')
@click.option('-d', '--device', help='GPU devices to use (e.g., 1,2,3)')
@click.option('--force', is_flag=True, help='Force operation')
@click.option('--quiet', is_flag=True, help='Suppress non-essential output')
@click.option('--debug', is_flag=True, help='Enable debug mode')
def crack(hash_file, wordlist, attack_mode, hash_type, rules, outfile, outfile_format,
          workload_profile, optimized_kernel, slow_candidates, gpu_temp_abort, runtime,
          status, status_timer, machine_readable, device, force, quiet, debug):
    """Crack hashes using darkling engine (hashcat-compatible)"""
    
    # Validate input files
    if not Path(hash_file).exists():
        click.echo(f"❌ Hash file not found: {hash_file}")
        return
    
    if attack_mode == 0 and not Path(wordlist).exists():
        click.echo(f"❌ Wordlist not found: {wordlist}")
        return
    
    # Build darkling command
    cmd = [
        str(project_root / "darkling" / "build" / "darkling"),
        "-m", str(hash_type),
        "-a", str(attack_mode),
        hash_file
    ]
    
    # Add wordlist for dictionary attacks
    if attack_mode == 0:
        cmd.append(wordlist)
    elif attack_mode == 3:  # Mask attack
        cmd.append(wordlist)  # This would be the mask
    
    # Add optional parameters
    if rules:
        if not Path(rules).exists():
            click.echo(f"❌ Rules file not found: {rules}")
            return
        cmd.extend(["-r", rules])
    
    if outfile:
        cmd.extend(["-o", outfile])
        cmd.extend(["--outfile-format", str(outfile_format)])
    
    cmd.extend(["-w", str(workload_profile)])
    
    if optimized_kernel:
        cmd.append("-O")
    
    if slow_candidates:
        cmd.append("-S")
    
    cmd.extend(["--gpu-temp-abort", str(gpu_temp_abort)])
    
    if runtime:
        cmd.extend(["--runtime", str(runtime)])
    
    if status:
        cmd.append("--status")
        cmd.extend(["--status-timer", str(status_timer)])
    
    if machine_readable:
        cmd.append("--machine-readable")
    
    if device:
        cmd.extend(["-d", device])
    
    if force:
        cmd.append("--force")
    
    if quiet:
        cmd.append("--quiet")
    
    if debug:
        cmd.append("--debug")
    
    # Display command info
    if not quiet:
        click.echo("⚡ Darkling Hash Cracking Engine")
        click.echo(f"   Hash file: {hash_file}")
        click.echo(f"   Attack mode: {get_attack_mode_name(attack_mode)}")
        click.echo(f"   Hash type: {get_hash_type_name(hash_type)}")
        if attack_mode == 0:
            click.echo(f"   Wordlist: {wordlist}")
        elif attack_mode == 3:
            click.echo(f"   Mask: {wordlist}")
        if rules:
            click.echo(f"   Rules: {rules}")
        if outfile:
            click.echo(f"   Output: {outfile}")
        click.echo(f"   Command: {' '.join(cmd)}")
        click.echo("=" * 60)
    
    try:
        # Execute darkling
        result = subprocess.run(cmd, cwd=project_root)
        
        if result.returncode == 0:
            click.echo("\n✅ Cracking completed successfully")
        else:
            click.echo(f"\n❌ Cracking failed with return code {result.returncode}")
            
    except FileNotFoundError:
        click.echo("❌ Darkling binary not found. Please build darkling first:")
        click.echo("   cd darkling && mkdir -p build && cd build && cmake .. && make")
    except KeyboardInterrupt:
        click.echo("\n⏹️ Cracking interrupted by user")
    except Exception as e:
        click.echo(f"\n❌ Error: {e}")

@darkling_cli.command()
@click.option('-m', '--hash-type', type=int, help='Hash type to benchmark')
@click.option('-b', '--benchmark', is_flag=True, help='Run benchmark mode')
@click.option('-d', '--device', help='GPU devices to use')
@click.option('--runtime', type=int, default=30, help='Benchmark runtime in seconds')
def benchmark(hash_type, benchmark, device, runtime):
    """Run performance benchmark"""
    click.echo("🏃 Running Darkling benchmark...")
    
    cmd = [
        str(project_root / "darkling" / "build" / "darkling"),
        "-b"
    ]
    
    if hash_type is not None:
        cmd.extend(["-m", str(hash_type)])
    
    if device:
        cmd.extend(["-d", device])
    
    cmd.extend(["--runtime", str(runtime)])
    
    try:
        subprocess.run(cmd, cwd=project_root)
    except FileNotFoundError:
        click.echo("❌ Darkling binary not found. Please build darkling first.")
    except KeyboardInterrupt:
        click.echo("\n⏹️ Benchmark interrupted")

@darkling_cli.command()
@click.argument('hash_file')
@click.option('-m', '--hash-type', type=int, required=True, help='Hash type')
def identify(hash_file, hash_type):
    """Identify hash type and validate format"""
    if not Path(hash_file).exists():
        click.echo(f"❌ Hash file not found: {hash_file}")
        return
    
    click.echo(f"🔍 Analyzing hash file: {hash_file}")
    click.echo(f"   Expected hash type: {get_hash_type_name(hash_type)}")
    
    try:
        with open(hash_file, 'r') as f:
            hashes = [line.strip() for line in f if line.strip()]
        
        click.echo(f"   Total hashes: {len(hashes)}")
        
        # Analyze hash lengths and patterns
        hash_lengths = {}
        for hash_line in hashes[:10]:  # Analyze first 10 hashes
            hash_part = hash_line.split(':')[0] if ':' in hash_line else hash_line
            length = len(hash_part)
            hash_lengths[length] = hash_lengths.get(length, 0) + 1
        
        click.echo("   Hash length distribution (first 10):")
        for length, count in sorted(hash_lengths.items()):
            click.echo(f"     {length} chars: {count} hashes")
        
        # Show sample hashes
        click.echo("   Sample hashes:")
        for i, hash_line in enumerate(hashes[:3]):
            click.echo(f"     {i+1}: {hash_line}")
            
    except Exception as e:
        click.echo(f"❌ Error analyzing hash file: {e}")

@darkling_cli.command()
def list_algorithms():
    """List supported hash algorithms"""
    algorithms = {
        0: "MD5",
        100: "SHA1", 
        1000: "NTLM",
        1400: "SHA256",
        1700: "SHA512",
        2500: "WPA/WPA2",
        3200: "bcrypt",
        5600: "NetNTLMv2",
        13100: "Kerberos 5 TGS-REP",
        18200: "Kerberos 5 AS-REP",
        22000: "WPA-PBKDF2-PMKID+EAPOL"
    }
    
    click.echo("📋 Supported Hash Algorithms:")
    click.echo("=" * 40)
    
    for code, name in algorithms.items():
        click.echo(f"   {code:5d} | {name}")
    
    click.echo("\nUse -m <code> to specify hash type")

@darkling_cli.command()
@click.option('-a', '--attack-mode', type=int, help='Show info for specific attack mode')
def modes():
    """List attack modes and their descriptions"""
    modes = {
        0: ("Straight", "Dictionary attack"),
        1: ("Combination", "Combinator attack"),
        3: ("Mask", "Brute-force/mask attack"),
        6: ("Hybrid Wordlist + Mask", "Wordlist + mask"),
        7: ("Hybrid Mask + Wordlist", "Mask + wordlist")
    }
    
    click.echo("⚔️  Attack Modes:")
    click.echo("=" * 50)
    
    if attack_mode is not None:
        if attack_mode in modes:
            name, desc = modes[attack_mode]
            click.echo(f"Mode {attack_mode}: {name}")
            click.echo(f"Description: {desc}")
            
            if attack_mode == 3:
                click.echo("\nMask characters:")
                click.echo("  ?l = lowercase letters")
                click.echo("  ?u = uppercase letters") 
                click.echo("  ?d = digits")
                click.echo("  ?s = special characters")
                click.echo("  ?a = all characters")
                click.echo("\nExample: ?u?l?l?l?l?l?d?d")
        else:
            click.echo(f"Unknown attack mode: {attack_mode}")
    else:
        for code, (name, desc) in modes.items():
            click.echo(f"   {code} | {name:20} | {desc}")
    
    click.echo("\nUse -a <mode> to specify attack mode")

@darkling_cli.command()
@click.argument('hash_file')
@click.argument('mask')
@click.option('-m', '--hash-type', type=int, default=0, help='Hash type')
@click.option('-o', '--outfile', help='Output file')
@click.option('--increment', is_flag=True, help='Enable increment mode')
@click.option('--increment-min', type=int, default=1, help='Minimum increment length')
@click.option('--increment-max', type=int, default=8, help='Maximum increment length')
@click.option('-d', '--device', help='GPU devices to use')
def mask(hash_file, mask, hash_type, outfile, increment, increment_min, increment_max, device):
    """Mask attack (brute-force) mode"""
    if not Path(hash_file).exists():
        click.echo(f"❌ Hash file not found: {hash_file}")
        return
    
    click.echo(f"🎭 Mask Attack: {mask}")
    click.echo(f"   Hash file: {hash_file}")
    click.echo(f"   Hash type: {get_hash_type_name(hash_type)}")
    
    cmd = [
        str(project_root / "darkling" / "build" / "darkling"),
        "-a", "3",
        "-m", str(hash_type),
        hash_file,
        mask
    ]
    
    if outfile:
        cmd.extend(["-o", outfile])
    
    if increment:
        cmd.append("--increment")
        cmd.extend(["--increment-min", str(increment_min)])
        cmd.extend(["--increment-max", str(increment_max)])
    
    if device:
        cmd.extend(["-d", device])
    
    try:
        subprocess.run(cmd, cwd=project_root)
    except FileNotFoundError:
        click.echo("❌ Darkling binary not found. Please build darkling first.")
    except KeyboardInterrupt:
        click.echo("\n⏹️ Attack interrupted")

@darkling_cli.command()
def build():
    """Build the darkling binary"""
    click.echo("🔨 Building Darkling...")
    
    darkling_dir = project_root / "darkling"
    build_dir = darkling_dir / "build"
    
    try:
        # Create build directory
        build_dir.mkdir(exist_ok=True)
        
        # Run cmake
        click.echo("Running cmake...")
        result = subprocess.run(["cmake", ".."], cwd=build_dir, capture_output=True, text=True)
        if result.returncode != 0:
            click.echo(f"❌ CMake failed: {result.stderr}")
            return
        
        # Run make
        click.echo("Running make...")
        result = subprocess.run(["make", "-j4"], cwd=build_dir, capture_output=True, text=True)
        if result.returncode != 0:
            click.echo(f"❌ Make failed: {result.stderr}")
            return
        
        click.echo("✅ Darkling built successfully!")
        
        # Check if binary exists
        binary_path = build_dir / "darkling"
        if binary_path.exists():
            click.echo(f"   Binary location: {binary_path}")
        
    except Exception as e:
        click.echo(f"❌ Build error: {e}")

def get_attack_mode_name(mode):
    """Get human-readable attack mode name"""
    modes = {
        0: "Straight",
        1: "Combination", 
        3: "Mask",
        6: "Hybrid Wordlist + Mask",
        7: "Hybrid Mask + Wordlist"
    }
    return modes.get(mode, f"Mode {mode}")

def get_hash_type_name(hash_type):
    """Get human-readable hash type name"""
    types = {
        0: "MD5",
        100: "SHA1",
        1000: "NTLM", 
        1400: "SHA256",
        1700: "SHA512",
        2500: "WPA/WPA2",
        3200: "bcrypt",
        5600: "NetNTLMv2"
    }
    return types.get(hash_type, f"Type {hash_type}")

if __name__ == "__main__":
    darkling_cli()