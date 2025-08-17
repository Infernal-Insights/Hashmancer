"""
Redis Diagnostics and Maintenance Tool for Hashmancer
Provides comprehensive Redis health monitoring, performance analysis, and maintenance utilities.
"""

import time
import json
import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

from .unified_redis import (
    get_redis_manager, 
    redis_connection, 
    redis_async_connection,
    get_redis_stats
)

logger = logging.getLogger(__name__)


@dataclass
class RedisHealthReport:
    """Comprehensive Redis health report."""
    timestamp: str
    overall_status: str
    connection_status: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    memory_usage: Dict[str, Any]
    connection_stats: Dict[str, Any]
    key_analysis: Dict[str, Any]
    recommendations: List[str]
    errors: List[str]


class RedisHealthMonitor:
    """Redis health monitoring and diagnostics."""
    
    def __init__(self):
        self.manager = get_redis_manager()
    
    async def comprehensive_health_check(self) -> RedisHealthReport:
        """Perform comprehensive Redis health check."""
        timestamp = datetime.now().isoformat()
        errors = []
        recommendations = []
        
        try:
            # Test connections
            connection_status = await self._test_connections()
            
            # Get performance metrics
            performance_metrics = await self._get_performance_metrics()
            
            # Analyze memory usage
            memory_usage = await self._analyze_memory_usage()
            
            # Get connection statistics
            connection_stats = self.manager.get_stats()
            
            # Analyze key patterns
            key_analysis = await self._analyze_key_patterns()
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                connection_status, performance_metrics, memory_usage, key_analysis
            )
            
            # Determine overall status
            overall_status = self._determine_overall_status(
                connection_status, performance_metrics, memory_usage
            )
            
        except Exception as e:
            errors.append(f"Health check failed: {str(e)}")
            overall_status = "critical"
            connection_status = {"error": str(e)}
            performance_metrics = {}
            memory_usage = {}
            connection_stats = {}
            key_analysis = {}
        
        return RedisHealthReport(
            timestamp=timestamp,
            overall_status=overall_status,
            connection_status=connection_status,
            performance_metrics=performance_metrics,
            memory_usage=memory_usage,
            connection_stats=connection_stats,
            key_analysis=key_analysis,
            recommendations=recommendations,
            errors=errors
        )
    
    async def _test_connections(self) -> Dict[str, Any]:
        """Test Redis connections."""
        try:
            return await self.manager.test_connection()
        except Exception as e:
            return {"error": str(e), "sync": {"success": False}, "async": {"success": False}}
    
    async def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get Redis performance metrics."""
        try:
            async with redis_async_connection() as conn:
                info = await conn.info()
                
                return {
                    "ops_per_sec": info.get("instantaneous_ops_per_sec", 0),
                    "hit_rate": info.get("keyspace_hit_rate", 0),
                    "evicted_keys": info.get("evicted_keys", 0),
                    "expired_keys": info.get("expired_keys", 0),
                    "connected_clients": info.get("connected_clients", 0),
                    "blocked_clients": info.get("blocked_clients", 0),
                    "total_commands_processed": info.get("total_commands_processed", 0),
                    "rejected_connections": info.get("rejected_connections", 0),
                    "uptime_in_seconds": info.get("uptime_in_seconds", 0),
                    "redis_version": info.get("redis_version", "unknown"),
                }
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            return {"error": str(e)}
    
    async def _analyze_memory_usage(self) -> Dict[str, Any]:
        """Analyze Redis memory usage."""
        try:
            async with redis_async_connection() as conn:
                info = await conn.info("memory")
                
                used_memory = info.get("used_memory", 0)
                used_memory_human = info.get("used_memory_human", "0B")
                used_memory_peak = info.get("used_memory_peak", 0)
                used_memory_peak_human = info.get("used_memory_peak_human", "0B")
                max_memory = info.get("maxmemory", 0)
                max_memory_human = info.get("maxmemory_human", "0B")
                
                # Calculate memory usage percentage
                memory_usage_percent = 0
                if max_memory > 0:
                    memory_usage_percent = (used_memory / max_memory) * 100
                
                return {
                    "used_memory": used_memory,
                    "used_memory_human": used_memory_human,
                    "used_memory_peak": used_memory_peak,
                    "used_memory_peak_human": used_memory_peak_human,
                    "max_memory": max_memory,
                    "max_memory_human": max_memory_human,
                    "memory_usage_percent": round(memory_usage_percent, 2),
                    "mem_fragmentation_ratio": info.get("mem_fragmentation_ratio", 0),
                    "used_memory_rss": info.get("used_memory_rss", 0),
                    "used_memory_dataset": info.get("used_memory_dataset", 0),
                    "used_memory_overhead": info.get("used_memory_overhead", 0),
                }
        except Exception as e:
            logger.error(f"Failed to analyze memory usage: {e}")
            return {"error": str(e)}
    
    async def _analyze_key_patterns(self) -> Dict[str, Any]:
        """Analyze Redis key patterns and distribution."""
        try:
            async with redis_async_connection() as conn:
                # Get database info
                info = await conn.info("keyspace")
                db_info = {}
                total_keys = 0
                
                for key, value in info.items():
                    if key.startswith("db"):
                        # Parse db info: "keys=123,expires=45,avg_ttl=67890"
                        db_stats = {}
                        for part in value.split(","):
                            k, v = part.split("=")
                            db_stats[k] = int(v)
                        db_info[key] = db_stats
                        total_keys += db_stats.get("keys", 0)
                
                # Sample key patterns
                key_patterns = await self._sample_key_patterns(conn)
                
                return {
                    "total_keys": total_keys,
                    "databases": db_info,
                    "key_patterns": key_patterns,
                    "sample_time": datetime.now().isoformat(),
                }
        except Exception as e:
            logger.error(f"Failed to analyze key patterns: {e}")
            return {"error": str(e)}
    
    async def _sample_key_patterns(self, conn, sample_size: int = 100) -> Dict[str, int]:
        """Sample Redis keys to analyze patterns."""
        try:
            patterns = {}
            count = 0
            
            async for key in conn.scan_iter(count=sample_size):
                # Extract pattern (prefix before first colon)
                if isinstance(key, bytes):
                    key = key.decode('utf-8')
                
                pattern = key.split(':')[0] if ':' in key else 'no_prefix'
                patterns[pattern] = patterns.get(pattern, 0) + 1
                count += 1
                
                if count >= sample_size:
                    break
            
            # Sort by frequency
            return dict(sorted(patterns.items(), key=lambda x: x[1], reverse=True))
            
        except Exception as e:
            logger.warning(f"Failed to sample key patterns: {e}")
            return {}
    
    def _generate_recommendations(
        self, 
        connection_status: Dict[str, Any], 
        performance_metrics: Dict[str, Any], 
        memory_usage: Dict[str, Any], 
        key_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate optimization recommendations based on metrics."""
        recommendations = []
        
        # Connection recommendations
        if not connection_status.get("sync", {}).get("success"):
            recommendations.append("❌ Sync Redis connection is failing - check network and Redis server status")
        
        if not connection_status.get("async", {}).get("success"):
            recommendations.append("❌ Async Redis connection is failing - check async Redis configuration")
        
        # Performance recommendations
        ops_per_sec = performance_metrics.get("ops_per_sec", 0)
        if ops_per_sec > 1000:
            recommendations.append("⚡ High operations per second detected - consider Redis clustering or read replicas")
        
        hit_rate = performance_metrics.get("hit_rate", 0)
        if hit_rate < 0.8:
            recommendations.append("📊 Low cache hit rate - review caching strategy and TTL settings")
        
        evicted_keys = performance_metrics.get("evicted_keys", 0)
        if evicted_keys > 0:
            recommendations.append("💾 Key evictions detected - consider increasing Redis memory or optimizing data retention")
        
        connected_clients = performance_metrics.get("connected_clients", 0)
        if connected_clients > 100:
            recommendations.append("🔗 High number of connected clients - review connection pooling configuration")
        
        # Memory recommendations
        memory_usage_percent = memory_usage.get("memory_usage_percent", 0)
        if memory_usage_percent > 80:
            recommendations.append("🚨 High memory usage (>80%) - consider scaling Redis or implementing data cleanup")
        elif memory_usage_percent > 60:
            recommendations.append("⚠️ Moderate memory usage (>60%) - monitor memory growth trends")
        
        fragmentation_ratio = memory_usage.get("mem_fragmentation_ratio", 0)
        if fragmentation_ratio > 1.5:
            recommendations.append("🔧 High memory fragmentation - consider Redis restart or memory defragmentation")
        
        # Key pattern recommendations
        total_keys = key_analysis.get("total_keys", 0)
        if total_keys > 100000:
            recommendations.append("🔑 Large number of keys - consider data partitioning or archival strategies")
        
        # General recommendations
        if not recommendations:
            recommendations.append("✅ Redis appears to be running optimally")
        else:
            recommendations.insert(0, f"Found {len(recommendations)} optimization opportunities:")
        
        return recommendations
    
    def _determine_overall_status(
        self, 
        connection_status: Dict[str, Any], 
        performance_metrics: Dict[str, Any], 
        memory_usage: Dict[str, Any]
    ) -> str:
        """Determine overall Redis health status."""
        # Check for critical issues
        if not connection_status.get("sync", {}).get("success"):
            return "critical"
        
        # Check for major issues
        memory_usage_percent = memory_usage.get("memory_usage_percent", 0)
        if memory_usage_percent > 90:
            return "critical"
        
        evicted_keys = performance_metrics.get("evicted_keys", 0)
        if evicted_keys > 1000:
            return "warning"
        
        # Check for minor issues
        if memory_usage_percent > 70:
            return "warning"
        
        if not connection_status.get("async", {}).get("success"):
            return "warning"
        
        # All good
        return "healthy"


class RedisMaintenanceTools:
    """Redis maintenance and optimization tools."""
    
    def __init__(self):
        self.manager = get_redis_manager()
    
    async def cleanup_expired_data(self, dry_run: bool = True) -> Dict[str, Any]:
        """Clean up expired and old data from Redis."""
        cleaned_keys = 0
        errors = []
        
        try:
            async with redis_async_connection() as conn:
                # Find and clean expired batch data
                batch_keys = []
                async for key in conn.scan_iter("batch:*"):
                    batch_keys.append(key)
                
                current_time = int(time.time())
                cutoff_time = current_time - (24 * 3600)  # 24 hours ago
                
                for key in batch_keys:
                    try:
                        created = await conn.hget(key, "created")
                        if created and int(created) < cutoff_time:
                            if not dry_run:
                                batch_id = key.split(":", 1)[1]
                                
                                # Clean up related keys
                                pipe = conn.pipeline()
                                pipe.delete(key)
                                pipe.delete(f"keyspace:queued:{batch_id}")
                                pipe.delete(f"keyspace:done:{batch_id}")
                                pipe.lrem("batch:queue", 0, batch_id)
                                pipe.zrem("batch:prio", batch_id)
                                await pipe.execute()
                                
                            cleaned_keys += 1
                            
                    except Exception as e:
                        errors.append(f"Error processing {key}: {str(e)}")
                        continue
                
        except Exception as e:
            errors.append(f"Cleanup failed: {str(e)}")
        
        return {
            "cleaned_keys": cleaned_keys,
            "dry_run": dry_run,
            "errors": errors,
            "timestamp": datetime.now().isoformat(),
        }
    
    async def optimize_memory(self) -> Dict[str, Any]:
        """Optimize Redis memory usage."""
        try:
            async with redis_async_connection() as conn:
                # Get memory info before optimization
                before_info = await conn.info("memory")
                before_memory = before_info.get("used_memory", 0)
                
                # Run memory optimization commands
                await conn.memory_doctor()
                
                # Force garbage collection (if supported)
                try:
                    await conn.debug_gc()
                except:
                    pass  # Not all Redis versions support this
                
                # Get memory info after optimization
                after_info = await conn.info("memory")
                after_memory = after_info.get("used_memory", 0)
                
                memory_saved = before_memory - after_memory
                
                return {
                    "success": True,
                    "before_memory": before_memory,
                    "after_memory": after_memory,
                    "memory_saved": memory_saved,
                    "memory_saved_human": f"{memory_saved / 1024 / 1024:.2f} MB",
                    "timestamp": datetime.now().isoformat(),
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
    
    async def analyze_slow_queries(self, count: int = 10) -> Dict[str, Any]:
        """Analyze Redis slow queries."""
        try:
            async with redis_async_connection() as conn:
                slow_log = await conn.slowlog_get(count)
                
                analyzed_queries = []
                for entry in slow_log:
                    analyzed_queries.append({
                        "id": entry.get("id"),
                        "timestamp": entry.get("start_time"),
                        "duration_microseconds": entry.get("duration"),
                        "command": " ".join(entry.get("command", [])),
                        "client_addr": entry.get("client_addr", "unknown"),
                    })
                
                return {
                    "slow_queries": analyzed_queries,
                    "total_count": len(analyzed_queries),
                    "timestamp": datetime.now().isoformat(),
                }
                
        except Exception as e:
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
    
    async def backup_critical_data(self, backup_path: str) -> Dict[str, Any]:
        """Create a backup of critical Redis data."""
        try:
            import pickle
            from pathlib import Path
            
            backup_data = {}
            backup_file = Path(backup_path)
            
            async with redis_async_connection() as conn:
                # Backup critical configuration
                config = await conn.config_get("*")
                backup_data["config"] = config
                
                # Backup key patterns and counts
                info = await conn.info("keyspace")
                backup_data["keyspace_info"] = info
                
                # Sample key data (first 1000 keys)
                sampled_keys = {}
                count = 0
                async for key in conn.scan_iter():
                    if count >= 1000:
                        break
                    
                    try:
                        key_type = await conn.type(key)
                        ttl = await conn.ttl(key)
                        
                        sampled_keys[key] = {
                            "type": key_type,
                            "ttl": ttl,
                        }
                        
                        # Store actual data for small keys
                        if key_type == "string":
                            value = await conn.get(key)
                            if value and len(str(value)) < 1000:  # Only small values
                                sampled_keys[key]["value"] = value
                        
                        count += 1
                        
                    except Exception:
                        continue
                
                backup_data["sampled_keys"] = sampled_keys
                backup_data["timestamp"] = datetime.now().isoformat()
                backup_data["total_sampled"] = count
            
            # Save backup
            backup_file.parent.mkdir(parents=True, exist_ok=True)
            with open(backup_file, "wb") as f:
                pickle.dump(backup_data, f)
            
            return {
                "success": True,
                "backup_file": str(backup_file),
                "keys_backed_up": len(backup_data["sampled_keys"]),
                "file_size": backup_file.stat().st_size,
                "timestamp": datetime.now().isoformat(),
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }


# Convenience functions
async def quick_health_check() -> Dict[str, Any]:
    """Perform a quick Redis health check."""
    monitor = RedisHealthMonitor()
    try:
        connection_test = await monitor._test_connections()
        stats = get_redis_stats()
        
        return {
            "healthy": connection_test.get("sync", {}).get("success", False),
            "connection_test": connection_test,
            "basic_stats": stats,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        return {
            "healthy": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


async def run_full_diagnostics() -> RedisHealthReport:
    """Run comprehensive Redis diagnostics."""
    monitor = RedisHealthMonitor()
    return await monitor.comprehensive_health_check()


def format_health_report(report: RedisHealthReport) -> str:
    """Format health report for display."""
    output = []
    output.append("=" * 60)
    output.append("REDIS HEALTH REPORT")
    output.append("=" * 60)
    output.append(f"Timestamp: {report.timestamp}")
    output.append(f"Overall Status: {report.overall_status.upper()}")
    output.append("")
    
    # Connection Status
    output.append("CONNECTION STATUS:")
    output.append("-" * 20)
    for conn_type, status in report.connection_status.items():
        if isinstance(status, dict):
            success = "✅" if status.get("success") else "❌"
            latency = status.get("latency_ms", "N/A")
            output.append(f"  {conn_type}: {success} ({latency}ms)")
        else:
            output.append(f"  {conn_type}: {status}")
    output.append("")
    
    # Performance Metrics
    if report.performance_metrics and "error" not in report.performance_metrics:
        output.append("PERFORMANCE METRICS:")
        output.append("-" * 20)
        metrics = report.performance_metrics
        output.append(f"  Operations/sec: {metrics.get('ops_per_sec', 'N/A')}")
        output.append(f"  Hit rate: {metrics.get('hit_rate', 'N/A')}")
        output.append(f"  Connected clients: {metrics.get('connected_clients', 'N/A')}")
        output.append(f"  Redis version: {metrics.get('redis_version', 'N/A')}")
        output.append("")
    
    # Memory Usage
    if report.memory_usage and "error" not in report.memory_usage:
        output.append("MEMORY USAGE:")
        output.append("-" * 20)
        memory = report.memory_usage
        output.append(f"  Used memory: {memory.get('used_memory_human', 'N/A')}")
        output.append(f"  Memory usage: {memory.get('memory_usage_percent', 'N/A')}%")
        output.append(f"  Fragmentation ratio: {memory.get('mem_fragmentation_ratio', 'N/A')}")
        output.append("")
    
    # Key Analysis
    if report.key_analysis and "error" not in report.key_analysis:
        output.append("KEY ANALYSIS:")
        output.append("-" * 20)
        analysis = report.key_analysis
        output.append(f"  Total keys: {analysis.get('total_keys', 'N/A')}")
        patterns = analysis.get('key_patterns', {})
        if patterns:
            output.append("  Key patterns:")
            for pattern, count in list(patterns.items())[:5]:  # Top 5 patterns
                output.append(f"    {pattern}: {count}")
        output.append("")
    
    # Recommendations
    if report.recommendations:
        output.append("RECOMMENDATIONS:")
        output.append("-" * 20)
        for rec in report.recommendations:
            output.append(f"  {rec}")
        output.append("")
    
    # Errors
    if report.errors:
        output.append("ERRORS:")
        output.append("-" * 20)
        for error in report.errors:
            output.append(f"  ❌ {error}")
        output.append("")
    
    output.append("=" * 60)
    return "\n".join(output)