#!/usr/bin/env python3
"""
Redis Management Tool for Hashmancer
Command-line interface for Redis diagnostics, monitoring, and maintenance.
"""

import asyncio
import argparse
import json
import sys
import os
from pathlib import Path

# Add the hashmancer directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "hashmancer"))

from hashmancer.server.redis_diagnostics import (
    RedisHealthMonitor,
    RedisMaintenanceTools,
    quick_health_check,
    run_full_diagnostics,
    format_health_report
)
from hashmancer.server.unified_redis import get_redis_stats, redis_health_check


async def cmd_health(args):
    """Run health check command."""
    if args.quick:
        print("Running quick health check...")
        result = await quick_health_check()
        
        if result.get("healthy"):
            print("✅ Redis is healthy")
        else:
            print("❌ Redis has issues")
            
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"Connection test: {'✅' if result.get('connection_test', {}).get('sync', {}).get('success') else '❌'}")
            
    else:
        print("Running comprehensive health check...")
        report = await run_full_diagnostics()
        
        if args.json:
            # Convert to dict for JSON serialization
            report_dict = {
                "timestamp": report.timestamp,
                "overall_status": report.overall_status,
                "connection_status": report.connection_status,
                "performance_metrics": report.performance_metrics,
                "memory_usage": report.memory_usage,
                "connection_stats": report.connection_stats,
                "key_analysis": report.key_analysis,
                "recommendations": report.recommendations,
                "errors": report.errors,
            }
            print(json.dumps(report_dict, indent=2))
        else:
            print(format_health_report(report))


async def cmd_stats(args):
    """Show Redis statistics."""
    try:
        stats = get_redis_stats()
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print("Redis Statistics:")
            print("=" * 40)
            
            # Connection stats
            if "connections" in stats:
                conn = stats["connections"]
                print(f"Active connections: {conn.get('active', 'N/A')}")
                print(f"Failed connections: {conn.get('failed', 'N/A')}")
            
            # Operation stats
            if "operations" in stats:
                ops = stats["operations"]
                print(f"Total operations: {ops.get('total', 'N/A')}")
                print(f"Failed operations: {ops.get('failed', 'N/A')}")
                print(f"Success rate: {ops.get('success_rate', 'N/A'):.1f}%")
            
            # Performance stats
            if "performance" in stats:
                perf = stats["performance"]
                print(f"Avg response time: {perf.get('avg_response_time_ms', 'N/A'):.2f}ms")
                print(f"Pool hit ratio: {perf.get('hit_ratio', 'N/A'):.1f}%")
            
            # Config
            if "config" in stats:
                conf = stats["config"]
                print(f"Host: {conf.get('host', 'N/A')}:{conf.get('port', 'N/A')}")
                print(f"SSL: {conf.get('ssl', 'N/A')}")
                print(f"Max connections: {conf.get('max_connections', 'N/A')}")
                
    except Exception as e:
        print(f"Error getting stats: {e}")
        sys.exit(1)


async def cmd_cleanup(args):
    """Run cleanup operations."""
    tools = RedisMaintenanceTools()
    
    print(f"Running cleanup (dry_run={args.dry_run})...")
    result = await tools.cleanup_expired_data(dry_run=args.dry_run)
    
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Cleaned up {result['cleaned_keys']} keys")
        if result['errors']:
            print("Errors:")
            for error in result['errors']:
                print(f"  - {error}")
        
        if args.dry_run:
            print("\n⚠️  This was a dry run. Use --no-dry-run to actually clean up data.")


async def cmd_optimize(args):
    """Run optimization operations."""
    tools = RedisMaintenanceTools()
    
    print("Running memory optimization...")
    result = await tools.optimize_memory()
    
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        if result.get("success"):
            print(f"✅ Optimization completed")
            print(f"Memory saved: {result.get('memory_saved_human', 'N/A')}")
        else:
            print(f"❌ Optimization failed: {result.get('error', 'Unknown error')}")


async def cmd_backup(args):
    """Create backup of Redis data."""
    tools = RedisMaintenanceTools()
    
    backup_path = args.path or f"redis_backup_{int(asyncio.get_event_loop().time())}.pkl"
    
    print(f"Creating backup: {backup_path}")
    result = await tools.backup_critical_data(backup_path)
    
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        if result.get("success"):
            print(f"✅ Backup created successfully")
            print(f"File: {result.get('backup_file')}")
            print(f"Keys backed up: {result.get('keys_backed_up')}")
            print(f"File size: {result.get('file_size')} bytes")
        else:
            print(f"❌ Backup failed: {result.get('error', 'Unknown error')}")


async def cmd_slowlog(args):
    """Analyze slow queries."""
    tools = RedisMaintenanceTools()
    
    print(f"Analyzing {args.count} slow queries...")
    result = await tools.analyze_slow_queries(args.count)
    
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        if "error" in result:
            print(f"❌ Analysis failed: {result['error']}")
        else:
            queries = result.get("slow_queries", [])
            if not queries:
                print("✅ No slow queries found")
            else:
                print(f"Found {len(queries)} slow queries:")
                print("-" * 60)
                for query in queries:
                    duration_ms = query.get("duration_microseconds", 0) / 1000
                    print(f"Duration: {duration_ms:.2f}ms")
                    print(f"Command: {query.get('command', 'N/A')}")
                    print(f"Client: {query.get('client_addr', 'N/A')}")
                    print("-" * 60)


def cmd_test(args):
    """Test Redis connection."""
    print("Testing Redis connection...")
    
    try:
        # Use sync health check for simplicity
        result = redis_health_check()
        
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            sync_ok = result.get("sync", {}).get("success", False)
            async_ok = result.get("async", {}).get("success", False)
            
            print(f"Sync connection: {'✅' if sync_ok else '❌'}")
            print(f"Async connection: {'✅' if async_ok else '❌'}")
            
            if sync_ok and async_ok:
                print("✅ All connections working")
                sys.exit(0)
            else:
                print("❌ Connection issues detected")
                sys.exit(1)
                
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Redis Management Tool for Hashmancer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  redis_tool.py test                    # Test Redis connection
  redis_tool.py health --quick          # Quick health check
  redis_tool.py health                  # Full health report
  redis_tool.py stats                   # Show statistics
  redis_tool.py cleanup --dry-run       # Preview cleanup (safe)
  redis_tool.py cleanup --no-dry-run    # Actually cleanup data
  redis_tool.py optimize               # Optimize memory usage
  redis_tool.py backup                 # Create backup
  redis_tool.py slowlog                # Analyze slow queries
        """
    )
    
    parser.add_argument("--json", action="store_true", help="Output in JSON format")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test Redis connection")
    
    # Health command
    health_parser = subparsers.add_parser("health", help="Check Redis health")
    health_parser.add_argument("--quick", action="store_true", help="Quick health check only")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show Redis statistics")
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up expired data")
    cleanup_parser.add_argument("--dry-run", action="store_true", default=True, help="Preview cleanup without making changes")
    cleanup_parser.add_argument("--no-dry-run", dest="dry_run", action="store_false", help="Actually perform cleanup")
    
    # Optimize command
    optimize_parser = subparsers.add_parser("optimize", help="Optimize Redis memory usage")
    
    # Backup command
    backup_parser = subparsers.add_parser("backup", help="Create backup of Redis data")
    backup_parser.add_argument("--path", help="Backup file path")
    
    # Slowlog command
    slowlog_parser = subparsers.add_parser("slowlog", help="Analyze slow queries")
    slowlog_parser.add_argument("--count", type=int, default=10, help="Number of slow queries to analyze")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Set up environment
    os.environ.setdefault("PYTHONPATH", str(Path(__file__).parent))
    
    try:
        if args.command == "test":
            cmd_test(args)
        elif args.command == "health":
            asyncio.run(cmd_health(args))
        elif args.command == "stats":
            asyncio.run(cmd_stats(args))
        elif args.command == "cleanup":
            asyncio.run(cmd_cleanup(args))
        elif args.command == "optimize":
            asyncio.run(cmd_optimize(args))
        elif args.command == "backup":
            asyncio.run(cmd_backup(args))
        elif args.command == "slowlog":
            asyncio.run(cmd_slowlog(args))
        else:
            parser.print_help()
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.json:
            print(json.dumps({"error": str(e)}, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()